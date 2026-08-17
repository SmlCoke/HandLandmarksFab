from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .formats import make_hand_id, merge_label_with_manifest, resolve_path
from .hamer_worker import (
    HAMER_SOURCE,
    HAMER_TFLITE_RESCUE_SOURCE,
    HaMeRWorkerClient,
)
from .handedness_classifier import HandClassifierONNX
from .image_io import read_image
from .mediapipe_tflite_rescue import (
    MEDIAPIPE_TFLITE_MODEL_ID,
    MediaPipeTFLiteRescueClient,
    mediapipe_tflite_rescue_enabled,
)
from .onnx_runtime import onnx_provider_for, onnx_runtime_settings
from .progress import track_progress
from .projection import landmark_dicts_from_norm
from .quality_checks import rtmpose_geometry_gate_errors


HAMER_INPUT_SIZE = (256, 256)
HAMER_KEYPOINTS = 21


def _is_negative_candidate(manifest: Mapping[str, Any]) -> bool:
    return (
        manifest.get("palm_valid") is False
        or manifest.get("proposal_kind") == "negative_candidate"
    )


def _unassessed_candidate_label(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "crop_id": manifest["crop_id"],
        "image": manifest["image"],
        "palm_det_id": manifest["palm_det_id"],
        "hand_id": None,
        "hand_presence": {"present": False},
        "handedness": {"label": "unknown", "score": None},
        "handedness_teacher_model_id": None,
        "hand_presence_teacher_model_id": None,
        "human_modified_handedness": False,
        "human_modified_presence": False,
        "landmarks_crop_norm": [],
        "landmarks_crop_px": [],
        "landmarks_image_px": [],
        "source": "eos_negative_candidate_unassessed",
    }


def _label_hamer_output(
    manifest: Mapping[str, Any],
    prediction: Mapping[str, Any],
    classification: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    hand_classifier_model_id: str,
    hamer_model_id: str,
    hamer_device: str,
    hamer_rescale: float,
) -> Dict[str, Any]:
    output_size = manifest.get("output_size") or [
        cfg["hand_roi"]["output_width"],
        cfg["hand_roi"]["output_height"],
    ]
    crop_width, crop_height = int(output_size[0]), int(output_size[1])
    if (crop_width, crop_height) != HAMER_INPUT_SIZE:
        raise ValueError(
            f"HaMeR requires a 256x256 Hand ROI, got {crop_width}x{crop_height} "
            f"for {manifest.get('crop_id')}"
        )
    coordinates = np.asarray(prediction.get("keypoints_2d"), dtype=np.float32)
    if coordinates.shape != (HAMER_KEYPOINTS, 2) or not np.isfinite(coordinates).all():
        raise ValueError(
            f"HaMeR must return 21 finite keypoints, got {coordinates.shape}"
        )
    clipped = coordinates.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(crop_width - 1))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(crop_height - 1))
    clipped_coordinate_values = int(np.count_nonzero(clipped != coordinates))
    crop_px = [
        {"id": index, "x": float(point[0]), "y": float(point[1])}
        for index, point in enumerate(clipped)
    ]
    crop_norm = [
        {
            "id": point["id"],
            "x": point["x"] / float(crop_width - 1),
            "y": point["y"] / float(crop_height - 1),
        }
        for point in crop_px
    ]
    image_px = landmark_dicts_from_norm(crop_norm, manifest["roi_corners_px"])
    row = {
        "crop_id": manifest["crop_id"],
        "image": manifest["image"],
        "palm_det_id": manifest["palm_det_id"],
        "hand_id": make_hand_id(str(manifest["crop_id"])),
        "hand_presence": classification["hand_presence"],
        "handedness": classification["handedness"],
        "handedness_teacher_model_id": hand_classifier_model_id,
        "hand_presence_teacher_model_id": hand_classifier_model_id,
        "human_modified_handedness": False,
        "human_modified_presence": False,
        "landmarks_crop_norm": crop_norm,
        "landmarks_crop_px": crop_px,
        "landmarks_image_px": image_px,
        "source": HAMER_SOURCE,
        "teacher_model_id": hamer_model_id,
        "mediapipe_num_hands_detected": 1,
        "hamer_inference": {
            "model_id": hamer_model_id,
            "device": hamer_device,
            "rescale": float(hamer_rescale),
            "flipped": bool(prediction.get("flipped")),
            "bbox_size": float(prediction.get("bbox_size")),
            "clipped_coordinate_values": clipped_coordinate_values,
            "handedness_source": "hand_classifier",
        },
    }
    return merge_label_with_manifest(row, manifest, cfg)


def label_roi_manifest_hamer(
    manifest_rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    root: Path,
    *,
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    classifier_cfg = cfg.get("hamer") or {}
    if not isinstance(classifier_cfg, Mapping):
        raise ValueError("hamer must be a mapping")
    classifier_model_path = resolve_path(
        root, classifier_cfg.get("hand_classifier_model_onnx_path", "")
    )
    _default_provider, batch_size = onnx_runtime_settings(cfg)
    classifier_provider = onnx_provider_for(cfg, "hand_classifier")
    hand_classifier: HandClassifierONNX | None = None
    hamer_client: HaMeRWorkerClient | None = None
    manifests = list(manifest_rows)
    optional_rows: List[Dict[str, Any] | None] = [None] * len(manifests)
    runtime_items: List[tuple[int, Mapping[str, Any]]] = []
    candidates_skipped = 0
    for index, manifest in enumerate(manifests):
        if _is_negative_candidate(manifest):
            optional_rows[index] = merge_label_with_manifest(
                _unassessed_candidate_label(manifest), manifest, cfg
            )
            candidates_skipped += 1
            continue
        runtime_items.append((index, manifest))

    classifications: Dict[int, Dict[str, Any]] = {}
    if runtime_items:
        hand_classifier = HandClassifierONNX(
            classifier_model_path, classifier_provider
        )
        for offset in track_progress(
            range(0, len(runtime_items), batch_size),
            enabled=show_progress,
            description="HaMeR HCF inference",
            unit="batch",
        ):
            chunk = runtime_items[offset : offset + batch_size]
            images = []
            for _, manifest in chunk:
                crop_path = resolve_path(root, manifest["crop_path"])
                image = read_image(crop_path)
                if image is None:
                    raise RuntimeError(
                        f"unreadable runtime Hand ROI for HaMeR: {crop_path}"
                    )
                images.append(image)
            chunk_classifications = hand_classifier.classify_batch(images)
            if len(chunk_classifications) != len(chunk):
                raise RuntimeError(
                    "batched Hand Classifier output count does not match the input count"
                )
            for (index, _manifest), classification in zip(
                chunk, chunk_classifications
            ):
                classifications[index] = classification

        hamer_client = HaMeRWorkerClient(cfg, root)
        predictions = hamer_client.predict(
            {
                "crop_id": manifest["crop_id"],
                "crop_path": manifest["crop_path"],
                "handedness": classifications[index]["handedness"]["label"],
            }
            for index, manifest in runtime_items
        )
        for index, manifest in runtime_items:
            crop_id = str(manifest["crop_id"])
            labeled_row = _label_hamer_output(
                manifest,
                predictions[crop_id],
                classifications[index],
                cfg,
                hand_classifier_model_id=hand_classifier.model_id,
                hamer_model_id=hamer_client.model_id,
                hamer_device=hamer_client.device,
                hamer_rescale=hamer_client.rescale,
            )
            for key in ("capture_source_id", "split", "proposal_kind"):
                labeled_row.setdefault(key, manifest.get(key))
            optional_rows[index] = labeled_row

    rows: List[Dict[str, Any]] = []
    for row in optional_rows:
        if row is None:
            raise RuntimeError("HaMeR batching left an unlabeled manifest row")
        rows.append(row)

    rescue_enabled = mediapipe_tflite_rescue_enabled(cfg)
    rescue_candidates: List[tuple[int, Dict[str, Any], List[str]]] = []
    if rescue_enabled:
        for index, row in enumerate(rows):
            geometry_errors = rtmpose_geometry_gate_errors(row, cfg)
            if geometry_errors:
                rescue_candidates.append((index, row, geometry_errors))

    rescue_accepted = 0
    rescue_rejected = 0
    if rescue_candidates:
        rescue_client = MediaPipeTFLiteRescueClient(cfg, root)
        rescue_predictions = rescue_client.predict(
            {
                "crop_id": row.get("crop_id"),
                "crop_path": row.get("crop_path"),
            }
            for _, row, _ in rescue_candidates
        )
        for index, original_row, trigger_errors in rescue_candidates:
            crop_id = str(original_row.get("crop_id"))
            prediction = rescue_predictions[crop_id]
            candidate = dict(original_row)
            candidate["landmarks_crop_px"] = list(
                prediction["landmarks_crop_px"]
            )
            candidate["landmarks_crop_norm"] = list(
                prediction["landmarks_crop_norm"]
            )
            candidate["landmarks_image_px"] = landmark_dicts_from_norm(
                candidate["landmarks_crop_norm"], candidate["roi_corners_px"]
            )
            result_errors = rtmpose_geometry_gate_errors(candidate, cfg)
            rescue_metadata = {
                "attempted": True,
                "accepted": not result_errors,
                "trigger_errors": list(trigger_errors),
                "result_errors": list(result_errors),
                "model_id": MEDIAPIPE_TFLITE_MODEL_ID,
            }
            if result_errors:
                original_row["hamer_geometry_rescue"] = rescue_metadata
                rescue_rejected += 1
                continue
            candidate["source"] = HAMER_TFLITE_RESCUE_SOURCE
            candidate["teacher_model_id"] = MEDIAPIPE_TFLITE_MODEL_ID
            candidate["hamer_geometry_rescue"] = rescue_metadata
            rows[index] = candidate
            rescue_accepted += 1

    runtime_labeled = len(runtime_items)
    return rows, {
        "backend": "hamer",
        "mode": "hamer",
        "provider": hamer_client.device if hamer_client is not None else None,
        "provider_fallback_reason": None,
        "hand_classifier_provider": (
            hand_classifier.provider if hand_classifier is not None else None
        ),
        "hand_classifier_provider_fallback_reason": (
            getattr(hand_classifier, "fallback_reason", None)
            if hand_classifier is not None
            else None
        ),
        "onnx_batch_size": batch_size,
        "hand_classifier_model_id": (
            hand_classifier.model_id if hand_classifier is not None else None
        ),
        "hand_classifier_runtime_rois_labeled": runtime_labeled,
        "runtime_rois_labeled": runtime_labeled,
        "negative_candidates_skipped": candidates_skipped,
        "mediapipe_tflite_rescue_enabled": rescue_enabled,
        "mediapipe_tflite_rescue_model_id": MEDIAPIPE_TFLITE_MODEL_ID,
        "mediapipe_tflite_rescue_attempted": len(rescue_candidates),
        "mediapipe_tflite_rescue_accepted": rescue_accepted,
        "mediapipe_tflite_rescue_rejected": rescue_rejected,
        "hamer_model_id": hamer_client.model_id if hamer_client else None,
        "hamer_repository_path": (
            str(hamer_client.repository_path) if hamer_client else None
        ),
        "hamer_repository_commit": (
            hamer_client.repository_commit if hamer_client else None
        ),
        "hamer_checkpoint_path": (
            str(hamer_client.checkpoint_path) if hamer_client else None
        ),
        "hamer_device": hamer_client.device if hamer_client else None,
        "hamer_rescale": hamer_client.rescale if hamer_client else None,
    }
