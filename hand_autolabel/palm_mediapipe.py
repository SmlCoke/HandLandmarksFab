from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .formats import clamp01, make_palm_det_id, normalize_detection_schema
from .image_io import gray_to_rgb, read_image


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


def run_mediapipe_palm_detector(image_paths: Iterable[Path], cfg: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], str]:
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
    try:
        for image_path in image_paths:
            img = read_image(image_path)
            if img is None:
                rows.append({"image": image_path.name, "width": width, "height": height, "detections": [], "negative_candidates": [], "error": "unreadable_image"})
                continue
            rgb = gray_to_rgb(img)
            landmarks, handedness = detector.detect(rgb)
            detections = []
            negatives = []
            for idx, lms in enumerate(landmarks):
                cats = handedness[idx] if idx < len(handedness) else []
                det = _detection_from_landmarks(image_path.name, idx, lms, cats, cfg)
                schema = normalize_detection_schema(det, image_path.name, idx, width, height)
                if float(schema["score"]) >= float(palm_cfg["score_threshold"]):
                    detections.append(schema)
                elif bool(palm_cfg.get("keep_low_score_candidates_for_negatives", True)) and float(schema["score"]) >= float(palm_cfg.get("negative_candidate_threshold", 0.15)):
                    schema["valid"] = False
                    schema["palm_det_id"] = make_palm_det_id(image_path.name, len(negatives), "neg")
                    negatives.append(schema)
            detections = sorted(detections, key=lambda d: float(d["score"]), reverse=True)[: int(palm_cfg.get("max_detections", 2))]
            rows.append({"image": image_path.name, "width": width, "height": height, "detections": detections, "negative_candidates": negatives})
    finally:
        detector.close()
    return rows, mode
