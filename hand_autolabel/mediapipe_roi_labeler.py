from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .formats import make_hand_id, merge_label_with_manifest, resolve_path
from .image_io import gray_to_rgb, read_image
from .palm_mediapipe import _category_label_and_score, create_mediapipe_detector
from .progress import track_progress
from .projection import landmark_dicts_from_norm


def _landmarks_to_rows(landmarks: Sequence, crop_width: int, crop_height: int) -> tuple[List[Dict], List[Dict]]:
    norm_rows: List[Dict] = []
    px_rows: List[Dict] = []
    for idx, lm in enumerate(landmarks):
        x = float(getattr(lm, "x"))
        y = float(getattr(lm, "y"))
        norm_rows.append({"id": idx, "x": x, "y": y})
        px_rows.append({"id": idx, "x": x * float(max(1, crop_width - 1)), "y": y * float(max(1, crop_height - 1))})
    return norm_rows, px_rows


def _empty_label(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "crop_id": manifest["crop_id"],
        "image": manifest["image"],
        "palm_det_id": manifest["palm_det_id"],
        "hand_id": None,
        "hand_presence": {"present": False},
        "handedness": {"label": "unknown", "score": None},
        "landmarks_crop_norm": [],
        "landmarks_crop_px": [],
        "landmarks_image_px": [],
        "source": "mediapipe_hand_landmarker",
    }


def label_one_roi(manifest: Mapping[str, Any], image, detector, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    output_size = manifest.get("output_size") or [cfg["hand_roi"]["output_width"], cfg["hand_roi"]["output_height"]]
    crop_width = int(output_size[0])
    crop_height = int(output_size[1])
    landmarks_list, handedness_list = detector.detect(gray_to_rgb(image))
    if not landmarks_list:
        return merge_label_with_manifest(_empty_label(manifest), manifest, cfg)

    landmarks = landmarks_list[0]
    handedness = handedness_list[0] if handedness_list else []
    label, score = _category_label_and_score(handedness)
    crop_norm, crop_px = _landmarks_to_rows(landmarks, crop_width, crop_height)
    image_px = landmark_dicts_from_norm(crop_norm, manifest["roi_corners_px"])
    row = {
        "crop_id": manifest["crop_id"],
        "image": manifest["image"],
        "palm_det_id": manifest["palm_det_id"],
        "hand_id": make_hand_id(str(manifest["crop_id"])),
        "hand_presence": {"present": True},
        "handedness": {"label": label, "score": score},
        "landmarks_crop_norm": crop_norm,
        "landmarks_crop_px": crop_px,
        "landmarks_image_px": image_px,
        "source": "mediapipe_hand_landmarker",
        "mediapipe_num_hands_detected": len(landmarks_list),
    }
    return merge_label_with_manifest(row, manifest, cfg)


def label_roi_manifest(
    manifest_rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    root: Path,
    *,
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], str]:
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on user environment.
        raise RuntimeError("mediapipe is required for script 03_run_mediapipe_on_rois.py.") from exc

    detector, mode = create_mediapipe_detector(cfg, num_hands=int(cfg["mediapipe"].get("num_hands", 1)))
    rows: List[Dict[str, Any]] = []
    try:
        for manifest in track_progress(
            manifest_rows,
            enabled=show_progress,
            description="Hand landmarks",
            unit="roi",
        ):
            crop_path = resolve_path(root, manifest["crop_path"])
            img = read_image(crop_path)
            if img is None:
                row = _empty_label(manifest)
                row["error"] = "unreadable_crop"
                rows.append(merge_label_with_manifest(row, manifest, cfg))
                continue
            rows.append(label_one_roi(manifest, img, detector, cfg))
    finally:
        detector.close()
    return rows, mode
