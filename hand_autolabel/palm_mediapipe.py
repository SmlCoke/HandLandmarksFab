from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .formats import clamp01, make_palm_det_id, normalize_detection_schema
from .image_io import gray_to_rgb, read_image
from .nms import nms_indices
from .progress import track_progress


PALM_BBOX_LANDMARK_IDS = (0, 1, 5, 9, 13, 17)


def _category_label_and_score(categories: Sequence) -> tuple[str, Optional[float]]:
    if not categories:
        return "unknown", None
    cat = categories[0]
    label = getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or getattr(cat, "label", None) or "unknown"
    score = getattr(cat, "score", None)
    return str(label), None if score is None else float(score)


def _bbox_from_landmarks(landmarks: Sequence, expand: float) -> List[float]:
    xs = [float(landmarks[i].x) for i in PALM_BBOX_LANDMARK_IDS if i < len(landmarks)]
    ys = [float(landmarks[i].y) for i in PALM_BBOX_LANDMARK_IDS if i < len(landmarks)]
    if not xs or not ys:
        xs = [float(lm.x) for lm in landmarks]
        ys = [float(lm.y) for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    x1 -= w * float(expand)
    x2 += w * float(expand)
    y1 -= h * float(expand)
    y2 += h * float(expand)
    return [clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2)]


def _detection_from_landmarks(image_name: str, idx: int, landmarks: Sequence, handedness: Sequence, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    label, score = _category_label_and_score(handedness)
    expand = float(cfg["palm"].get("compatible_bbox_expand", 0.25))
    det = {
        "palm_det_id": make_palm_det_id(image_name, idx, "palm"),
        "valid": True,
        "score": 1.0 if score is None else score,
        "score_source": "mediapipe_handedness_score" if score is not None else "mediapipe_detection_available",
        "bbox_norm": _bbox_from_landmarks(landmarks, expand=expand),
        "bbox_source": "mediapipe_palm_keypoints_0_1_5_9_13_17_expanded",
        "keypoints_norm": {
            "p0": [clamp01(float(landmarks[0].x)), clamp01(float(landmarks[0].y))],
            "p9": [clamp01(float(landmarks[9].x)), clamp01(float(landmarks[9].y))],
        },
        "source": "mediapipe_official",
        "head": "mediapipe_compatible",
        "handedness_hint": {"label": label, "score": score},
    }
    return det


def _tile_windows(width: int, height: int, tile_size: int, overlap: float) -> Iterable[tuple[int, int, int, int]]:
    tile_size = int(tile_size)
    if tile_size <= 0 or tile_size > width or tile_size > height:
        return
    step = max(64, int(round(tile_size * (1.0 - float(overlap)))))
    max_x = max(0, width - tile_size)
    max_y = max(0, height - tile_size)
    xs = list(range(0, max_x + 1, step))
    ys = list(range(0, max_y + 1, step))
    if not xs or xs[-1] != max_x:
        xs.append(max_x)
    if not ys or ys[-1] != max_y:
        ys.append(max_y)
    for y in ys:
        for x in xs:
            yield x, y, tile_size, tile_size


def _landmarks_from_tile_to_image(landmarks: Sequence, x0: int, y0: int, tile_width: int, tile_height: int, image_width: int, image_height: int) -> List[Any]:
    mapped = []
    for lm in landmarks:
        mapped.append(
            SimpleNamespace(
                x=(float(x0) + float(lm.x) * float(tile_width)) / float(image_width),
                y=(float(y0) + float(lm.y) * float(tile_height)) / float(image_height),
            )
        )
    return mapped


def _schema_from_landmarks(
    image_name: str,
    idx: int,
    landmarks: Sequence,
    handedness: Sequence,
    cfg: Mapping[str, Any],
    width: int,
    height: int,
    source_suffix: str,
) -> Dict[str, Any]:
    det = _detection_from_landmarks(image_name, idx, landmarks, handedness, cfg)
    det["source"] = f"mediapipe_official_{source_suffix}"
    schema = normalize_detection_schema(det, image_name, idx, width, height)
    schema["mediapipe_detection_strategy"] = source_suffix
    return schema


def _select_palm_candidates(candidates: List[Dict[str, Any]], image_name: str, cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    boxes = np.asarray([det["bbox_norm"] for det in candidates], dtype=np.float32)
    scores = np.asarray([float(det["score"]) for det in candidates], dtype=np.float32)
    keep = nms_indices(boxes, scores, float(cfg["palm"].get("nms_iou_threshold", 0.3)))
    selected = []
    for out_idx, cand_idx in enumerate(keep[: int(cfg["palm"].get("max_detections", 2))]):
        det = dict(candidates[cand_idx])
        det["palm_det_id"] = make_palm_det_id(image_name, out_idx, "palm")
        selected.append(det)
    return selected


class _TasksHandDetector:
    def __init__(self, cfg: Mapping[str, Any], num_hands: int):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        asset_path = str(cfg["mediapipe"].get("model_asset_path") or "").strip()
        if not asset_path:
            raise ValueError("mediapipe.model_asset_path is empty")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=asset_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(cfg["mediapipe"].get("min_hand_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(cfg["mediapipe"].get("min_hand_presence_confidence", 0.5)),
            min_tracking_confidence=float(cfg["mediapipe"].get("min_tracking_confidence", 0.5)),
        )
        self.mp = mp
        self.detector = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self.detector.close()

    def detect(self, rgb: np.ndarray) -> tuple[List[Sequence], List[Sequence]]:
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect(mp_image)
        return list(result.hand_landmarks), list(result.handedness)


def create_mediapipe_detector(cfg: Mapping[str, Any], num_hands: int):
    try:
        return _TasksHandDetector(cfg, num_hands), "tasks"
    except Exception as exc:
        raise RuntimeError(
            "Could not create MediaPipe Tasks HandLandmarker. Configure mediapipe.model_asset_path "
            "with an official hand_landmarker.task model file."
        ) from exc


def run_mediapipe_palm_detector(
    image_paths: Iterable[Path],
    cfg: Mapping[str, Any],
    *,
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], str]:
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on user environment.
        raise RuntimeError("mediapipe is required for palm.backend=mediapipe_official.") from exc

    image_cfg = cfg["image"]
    palm_cfg = cfg["palm"]
    width = int(image_cfg["width"])
    height = int(image_cfg["height"])
    detector, mode = create_mediapipe_detector(cfg, num_hands=int(palm_cfg.get("max_detections", 2)))
    rows: List[Dict[str, Any]] = []
    tiled_mode: Optional[str] = None
    try:
        for image_path in track_progress(
            image_paths,
            enabled=show_progress,
            description="Palm inference",
            unit="image",
        ):
            img = read_image(image_path)
            if img is None:
                rows.append({"image": image_path.name, "width": width, "height": height, "detections": [], "negative_candidates": [], "error": "unreadable_image"})
                continue
            candidates = []
            negatives = []
            tile_sizes = [int(v) for v in palm_cfg.get("mediapipe_official_tile_sizes", [512]) or []]
            full_image_first = bool(palm_cfg.get("mediapipe_official_full_image_first", not tile_sizes))
            if full_image_first:
                rgb = gray_to_rgb(img)
                landmarks, handedness = detector.detect(rgb)
                for idx, lms in enumerate(landmarks):
                    cats = handedness[idx] if idx < len(handedness) else []
                    schema = _schema_from_landmarks(image_path.name, idx, lms, cats, cfg, width, height, "full_image")
                    if float(schema["score"]) >= float(palm_cfg["score_threshold"]):
                        candidates.append(schema)
                    elif bool(palm_cfg.get("keep_low_score_candidates_for_negatives", True)) and float(schema["score"]) >= float(palm_cfg.get("negative_candidate_threshold", 0.15)):
                        schema["valid"] = False
                        schema["palm_det_id"] = make_palm_det_id(image_path.name, len(negatives), "neg")
                        negatives.append(schema)

            if not candidates and tile_sizes:
                tiled_mode = "fallback" if full_image_first else "primary"
                tile_overlap = float(palm_cfg.get("mediapipe_official_tile_overlap", 0.5))
                image_h, image_w = img.shape[:2]
                for tile_size in tile_sizes:
                    for x0, y0, tile_w, tile_h in _tile_windows(image_w, image_h, tile_size, tile_overlap):
                        tile = img[y0 : y0 + tile_h, x0 : x0 + tile_w]
                        tile_landmarks, tile_handedness = detector.detect(gray_to_rgb(tile))
                        for idx, lms in enumerate(tile_landmarks):
                            cats = tile_handedness[idx] if idx < len(tile_handedness) else []
                            mapped = _landmarks_from_tile_to_image(lms, x0, y0, tile_w, tile_h, image_w, image_h)
                            schema = _schema_from_landmarks(image_path.name, len(candidates), mapped, cats, cfg, width, height, f"tiled_{tile_size}")
                            if float(schema["score"]) >= float(palm_cfg["score_threshold"]):
                                candidates.append(schema)
                            elif bool(palm_cfg.get("keep_low_score_candidates_for_negatives", True)) and float(schema["score"]) >= float(palm_cfg.get("negative_candidate_threshold", 0.15)):
                                schema["valid"] = False
                                schema["palm_det_id"] = make_palm_det_id(image_path.name, len(negatives), "neg")
                                negatives.append(schema)

            detections = _select_palm_candidates(candidates, image_path.name, cfg)
            rows.append({"image": image_path.name, "width": width, "height": height, "detections": detections, "negative_candidates": negatives})
    finally:
        detector.close()
    if tiled_mode == "primary":
        return rows, f"{mode}_tiled"
    if tiled_mode == "fallback":
        return rows, f"{mode}_tiled_fallback"
    return rows, mode
