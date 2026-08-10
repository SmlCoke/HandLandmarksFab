from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np

from .handedness_classifier import HandClassifierONNX
from .formats import make_hand_id, merge_label_with_manifest, resolve_path
from .image_io import read_image, to_uint8_gray
from .mediapipe_tflite_rescue import (
    MEDIAPIPE_TFLITE_MODEL_ID,
    MEDIAPIPE_TFLITE_RESCUE_SOURCE,
    MediaPipeTFLiteRescueClient,
    mediapipe_tflite_rescue_enabled,
)
from .onnx_runtime import create_onnx_session, onnx_provider_for, onnx_runtime_settings
from .progress import track_progress
from .projection import landmark_dicts_from_norm
from .quality_checks import rtmpose_geometry_gate_errors


RTMPOSE_INPUT_NAME = "input"
RTMPOSE_OUTPUT_NAMES = ("simcc_x", "simcc_y")
RTMPOSE_INPUT_SIZE = (256, 256)
RTMPOSE_KEYPOINTS = 21
RTMPOSE_SIMCC_BINS = 512
RTMPOSE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
RTMPOSE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_rtmpose_image(image: np.ndarray) -> np.ndarray:
    """Create the official MMPose RGB/mean/std NCHW input tensor."""

    gray = to_uint8_gray(image)
    width, height = RTMPOSE_INPUT_SIZE
    if gray.shape != (height, width):
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    tensor = (rgb - RTMPOSE_MEAN) / RTMPOSE_STD
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    if tensor.shape != (1, 3, height, width) or not np.isfinite(tensor).all():
        raise ValueError("RTMPose preprocessing produced an invalid input tensor")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def preprocess_rtmpose_images(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("RTMPose batch must contain at least one image")
    return np.concatenate([preprocess_rtmpose_image(image) for image in images], axis=0)


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    *,
    split_ratio: float,
    input_size: Sequence[int] = RTMPOSE_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw SimCC logits as MMPose does: argmax, then divide by ratio."""

    x = np.asarray(simcc_x)
    y = np.asarray(simcc_y)
    expected_x = (1, RTMPOSE_KEYPOINTS, RTMPOSE_SIMCC_BINS)
    expected_y = (1, RTMPOSE_KEYPOINTS, RTMPOSE_SIMCC_BINS)
    if x.shape != expected_x or y.shape != expected_y:
        raise ValueError(
            f"unexpected RTMPose output shapes: simcc_x={x.shape}, simcc_y={y.shape}; "
            f"expected {expected_x} and {expected_y}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("RTMPose output contains non-finite SimCC logits")
    if not np.isfinite(split_ratio) or split_ratio <= 0:
        raise ValueError("rtmpose.simcc_split_ratio must be a positive finite number")

    x_indices = np.argmax(x[0], axis=1).astype(np.float32)
    y_indices = np.argmax(y[0], axis=1).astype(np.float32)
    scores = np.minimum(np.max(x[0], axis=1), np.max(y[0], axis=1)).astype(np.float32)

    coordinates = np.stack((x_indices, y_indices), axis=1) / float(split_ratio)
    width, height = (int(input_size[0]), int(input_size[1]))
    if width < 2 or height < 2:
        raise ValueError("RTMPose input dimensions must be at least 2 pixels")
    # The last SimCC bin decodes to 255.5 at ratio 2.0. Clamp only this
    # half-pixel border overshoot before normalization to the ROI contract.
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0.0, float(width - 1))
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, float(height - 1))
    if coordinates.shape != (RTMPOSE_KEYPOINTS, 2) or not np.isfinite(coordinates).all():
        raise ValueError("RTMPose decoding produced invalid keypoint coordinates")
    return coordinates, scores


def decode_simcc_batch(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    *,
    split_ratio: float,
    input_size: Sequence[int] = RTMPOSE_INPUT_SIZE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    x = np.asarray(simcc_x)
    y = np.asarray(simcc_y)
    if (
        x.ndim != 3
        or y.ndim != 3
        or x.shape[1:] != (RTMPOSE_KEYPOINTS, RTMPOSE_SIMCC_BINS)
        or y.shape != x.shape
        or x.shape[0] < 1
    ):
        raise ValueError(
            "unexpected RTMPose batch output shapes: "
            f"simcc_x={x.shape}, simcc_y={y.shape}"
        )
    return [
        decode_simcc(
            x[index : index + 1],
            y[index : index + 1],
            split_ratio=split_ratio,
            input_size=input_size,
        )
        for index in range(x.shape[0])
    ]


def _shape_matches(actual: Sequence[Any], expected: Sequence[int | None]) -> bool:
    if len(actual) != len(expected):
        return False
    for value, wanted in zip(actual, expected):
        if wanted is None:
            continue
        # ONNX exports may retain a symbolic dimension even when every runtime
        # output is fixed. Reject contradictory static dimensions here; the
        # decoder validates the concrete output tensor exactly on every call.
        if isinstance(value, int) and value != wanted:
            return False
    return True


class RTMPoseONNXHandLabeler:
    def __init__(
        self,
        model_path: Path,
        split_ratio: float,
        provider_preference: str = "auto",
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"RTMPose ONNX model does not exist: {model_path}")
        self.session, self.provider, self.fallback_reason = create_onnx_session(
            model_path, provider_preference
        )
        self.split_ratio = float(split_ratio)
        self._validate_model_interface()

    def _validate_model_interface(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != RTMPOSE_INPUT_NAME:
            raise ValueError("RTMPose ONNX must expose exactly one input named 'input'")
        if not _shape_matches(inputs[0].shape, (None, 3, 256, 256)):
            raise ValueError(f"unexpected RTMPose input shape: {inputs[0].shape}")
        by_name = {output.name: output for output in outputs}
        if set(by_name) != set(RTMPOSE_OUTPUT_NAMES):
            raise ValueError("RTMPose ONNX outputs must be exactly 'simcc_x' and 'simcc_y'")
        for name in RTMPOSE_OUTPUT_NAMES:
            if not _shape_matches(by_name[name].shape, (None, 21, 512)):
                raise ValueError(f"unexpected RTMPose output shape for {name}: {by_name[name].shape}")

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.detect_batch([image])[0]

    def detect_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        tensor = preprocess_rtmpose_images(images)
        outputs = self.session.run(list(RTMPOSE_OUTPUT_NAMES), {RTMPOSE_INPUT_NAME: tensor})
        if len(outputs) != 2:
            raise ValueError(f"RTMPose ONNX returned {len(outputs)} outputs, expected 2")
        return decode_simcc_batch(
            outputs[0],
            outputs[1],
            split_ratio=self.split_ratio,
            input_size=RTMPOSE_INPUT_SIZE,
        )


def _is_negative_candidate(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("palm_valid") is False or manifest.get("proposal_kind") == "negative_candidate"


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


def label_one_roi_rtmpose(
    manifest: Mapping[str, Any],
    image: np.ndarray,
    detector: Any,
    hand_classifier: Any,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    if _is_negative_candidate(manifest):
        return merge_label_with_manifest(_unassessed_candidate_label(manifest), manifest, cfg)

    coordinates, _scores = detector.detect(image)
    classification = hand_classifier.classify(image)
    hand_classifier_model_id = str(
        getattr(hand_classifier, "model_id", "")
    ).strip()
    if not hand_classifier_model_id:
        raise ValueError("Hand classifier runtime must expose a non-empty model_id")
    return _label_rtmpose_outputs(
        manifest,
        coordinates,
        classification,
        cfg,
        hand_classifier_model_id=hand_classifier_model_id,
    )


def _label_rtmpose_outputs(
    manifest: Mapping[str, Any],
    coordinates: np.ndarray,
    classification: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    hand_classifier_model_id: str,
) -> Dict[str, Any]:
    output_size = manifest.get("output_size") or [
        cfg["hand_roi"]["output_width"],
        cfg["hand_roi"]["output_height"],
    ]
    crop_width, crop_height = int(output_size[0]), int(output_size[1])
    if (crop_width, crop_height) != RTMPOSE_INPUT_SIZE:
        raise ValueError(
            f"RTMPose requires a 256x256 Hand ROI, got {crop_width}x{crop_height} "
            f"for {manifest.get('crop_id')}"
        )
    if coordinates.shape != (RTMPOSE_KEYPOINTS, 2):
        raise ValueError(f"RTMPose must return 21 keypoints, got {coordinates.shape}")
    crop_px = [
        {"id": idx, "x": float(point[0]), "y": float(point[1])}
        for idx, point in enumerate(coordinates)
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
        "source": "rtmpose_m_hand5_onnx",
        # Legacy draft/release schema field. For this backend the value means
        # one runtime ROI was labeled, while provenance remains RTMPose.
        "mediapipe_num_hands_detected": 1,
    }
    return merge_label_with_manifest(row, manifest, cfg)


def label_roi_manifest_rtmpose(
    manifest_rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    root: Path,
    *,
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rtmpose_cfg = cfg.get("rtmpose") or {}
    split_ratio = float(rtmpose_cfg.get("simcc_split_ratio", 2.0))
    if split_ratio != 2.0:
        raise ValueError("rtmpose.simcc_split_ratio is fixed at 2.0 for this model")
    model_path = resolve_path(root, rtmpose_cfg.get("model_onnx_path", ""))
    classifier_cfg = cfg.get("hand_classifier") or {}
    classifier_model_path = resolve_path(root, classifier_cfg.get("model_onnx_path", ""))
    _default_provider, batch_size = onnx_runtime_settings(cfg)
    rtmpose_provider = onnx_provider_for(cfg, "rtmpose")
    classifier_provider = onnx_provider_for(cfg, "hand_classifier")
    detector: RTMPoseONNXHandLabeler | None = None
    hand_classifier: HandClassifierONNX | None = None
    manifests = list(manifest_rows)
    optional_rows: List[Dict[str, Any] | None] = [None] * len(manifests)
    runtime_items: List[tuple[int, Mapping[str, Any]]] = []
    runtime_labeled = 0
    candidates_skipped = 0
    rescue_enabled = mediapipe_tflite_rescue_enabled(cfg)
    for index, manifest in enumerate(manifests):
        if _is_negative_candidate(manifest):
            optional_rows[index] = (
                merge_label_with_manifest(_unassessed_candidate_label(manifest), manifest, cfg)
            )
            candidates_skipped += 1
            continue
        runtime_items.append((index, manifest))

    if runtime_items:
        detector = RTMPoseONNXHandLabeler(
            model_path, split_ratio, rtmpose_provider
        )
        hand_classifier = HandClassifierONNX(
            classifier_model_path, classifier_provider
        )
        for offset in track_progress(
            range(0, len(runtime_items), batch_size),
            enabled=show_progress,
            description="RTMPose/HCF inference",
            unit="batch",
        ):
            chunk = runtime_items[offset : offset + batch_size]
            images: List[np.ndarray] = []
            for _, manifest in chunk:
                crop_path = resolve_path(root, manifest["crop_path"])
                image = read_image(crop_path)
                if image is None:
                    raise RuntimeError(
                        f"unreadable runtime Hand ROI for RTMPose: {crop_path}"
                    )
                images.append(image)
            pose_outputs = (
                detector.detect_batch(images)
                if hasattr(detector, "detect_batch")
                else [detector.detect(image) for image in images]
            )
            classifications = (
                hand_classifier.classify_batch(images)
                if hasattr(hand_classifier, "classify_batch")
                else [hand_classifier.classify(image) for image in images]
            )
            if len(pose_outputs) != len(chunk) or len(classifications) != len(chunk):
                raise RuntimeError("batched ONNX output count does not match the input count")
            for (index, manifest), (coordinates, _scores), classification in zip(
                chunk, pose_outputs, classifications
            ):
                labeled_row = _label_rtmpose_outputs(
                    manifest,
                    coordinates,
                    classification,
                    cfg,
                    hand_classifier_model_id=hand_classifier.model_id,
                )
                # The shared geometry gates route on these manifest fields. The public
                # pipeline attaches them again after labeling, but rescue must run first.
                for key in ("capture_source_id", "split", "proposal_kind"):
                    labeled_row.setdefault(key, manifest.get(key))
                optional_rows[index] = labeled_row
                runtime_labeled += 1

    rows: List[Dict[str, Any]] = []
    for row in optional_rows:
        if row is None:
            raise RuntimeError("RTMPose batching left an unlabeled manifest row")
        rows.append(row)

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
        predictions = rescue_client.predict(
            {
                "crop_id": row.get("crop_id"),
                "crop_path": row.get("crop_path"),
            }
            for _, row, _ in rescue_candidates
        )
        for index, original_row, trigger_errors in rescue_candidates:
            crop_id = str(original_row.get("crop_id"))
            prediction = predictions[crop_id]
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
                original_row["rtmpose_geometry_rescue"] = rescue_metadata
                rescue_rejected += 1
                continue
            candidate["source"] = MEDIAPIPE_TFLITE_RESCUE_SOURCE
            candidate["rtmpose_geometry_rescue"] = rescue_metadata
            rows[index] = candidate
            rescue_accepted += 1
    return rows, {
        "backend": "rtmpose_onnx",
        "mode": "rtmpose_onnx",
        "provider": detector.provider if detector is not None else None,
        "provider_fallback_reason": (
            getattr(detector, "fallback_reason", None) if detector is not None else None
        ),
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
    }
