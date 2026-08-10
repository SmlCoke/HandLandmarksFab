from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .formats import clamp01
from .nms import nms_indices


VALUES_PER_ANCHOR = 8


def normalize_feature_levels(palm_cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_levels = palm_cfg.get("feature_levels")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError("palm.feature_levels must be a non-empty list")
    levels: List[Dict[str, Any]] = []
    names: set[str] = set()
    shapes: set[tuple[int, int]] = set()
    for index, raw in enumerate(raw_levels):
        if not isinstance(raw, Mapping):
            raise ValueError(f"palm.feature_levels[{index}] must be a mapping")
        name = str(raw.get("name") or "").strip()
        if not name or name in names:
            raise ValueError("palm feature-level names must be non-empty and unique")
        height = raw.get("height")
        width = raw.get("width")
        if (
            isinstance(height, bool)
            or not isinstance(height, int)
            or height < 1
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
        ):
            raise ValueError(f"palm.feature_levels[{index}] height/width must be positive integers")
        shape = (height, width)
        if shape in shapes:
            raise ValueError("palm feature-level shapes must be unique")
        raw_anchors = raw.get("anchor_sizes")
        if not isinstance(raw_anchors, list) or len(raw_anchors) != 2:
            raise ValueError(f"palm.feature_levels[{index}].anchor_sizes must contain exactly two sizes")
        anchors: List[Tuple[float, float]] = []
        for anchor_index, raw_anchor in enumerate(raw_anchors):
            if not isinstance(raw_anchor, (list, tuple)) or len(raw_anchor) != 2:
                raise ValueError(
                    f"palm.feature_levels[{index}].anchor_sizes[{anchor_index}] must be [width,height]"
                )
            anchor_width, anchor_height = [float(value) for value in raw_anchor]
            if not all(math.isfinite(value) and value > 0.0 for value in (anchor_width, anchor_height)):
                raise ValueError("palm anchor sizes must be finite and positive")
            anchors.append((anchor_width, anchor_height))
        levels.append(
            {
                "name": name,
                "height": height,
                "width": width,
                "anchor_sizes": tuple(anchors),
                "reg_channels": len(anchors) * VALUES_PER_ANCHOR,
                "cls_channels": len(anchors),
            }
        )
        names.add(name)
        shapes.add(shape)
    return levels


def feature_level_anchor_count(levels: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        int(level["height"]) * int(level["width"]) * len(level["anchor_sizes"])
        for level in levels
    )


def generate_anchors(
    feature_height: int,
    feature_width: int,
    sizes: Sequence[Tuple[float, float]],
) -> np.ndarray:
    anchors: List[List[float]] = []
    step_x = 1.0 / float(feature_width)
    step_y = 1.0 / float(feature_height)
    for y in range(feature_height):
        for x in range(feature_width):
            center_x = x * step_x + step_x * 0.5
            center_y = y * step_y + step_y * 0.5
            for width, height in sizes:
                anchors.append([center_x, center_y, width, height])
    return np.asarray(anchors, dtype=np.float32)


def _reshape_output(
    arr: np.ndarray,
    feature_height: int,
    feature_width: int,
    channels: int,
    layout: str,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape == (channels, feature_height, feature_width):
            return arr.transpose(1, 2, 0).reshape(-1, channels)
        if arr.shape == (feature_height, feature_width, channels):
            return arr.reshape(-1, channels)
    expected = feature_height * feature_width * channels
    flat = arr.reshape(-1)
    if flat.size != expected:
        raise ValueError(f"Unexpected head tensor size: got {flat.size}, expected {expected}")
    if layout == "hwc":
        return flat.reshape(feature_height, feature_width, channels).reshape(-1, channels)
    return flat.reshape(channels, feature_height, feature_width).transpose(1, 2, 0).reshape(-1, channels)


def _find_output(outputs: Sequence[np.ndarray], expected_size: int, used: set[int]) -> np.ndarray:
    matches = [
        index
        for index, output in enumerate(outputs)
        if index not in used and int(np.asarray(output).size) == int(expected_size)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one unused ONNX output with {expected_size} elements, got {len(matches)}"
        )
    used.add(matches[0])
    return np.asarray(outputs[matches[0]])


def split_onnx_outputs(
    outputs: Sequence[np.ndarray],
    feature_levels: Sequence[Mapping[str, Any]],
) -> List[Tuple[Mapping[str, Any], np.ndarray, np.ndarray]]:
    if len(outputs) != len(feature_levels) * 2:
        raise ValueError(
            f"Palm ONNX must expose {len(feature_levels) * 2} outputs, got {len(outputs)}"
        )
    used: set[int] = set()
    split: List[Tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = []
    for level in feature_levels:
        cells = int(level["height"]) * int(level["width"])
        regression = _find_output(outputs, cells * int(level["reg_channels"]), used)
        classification = _find_output(outputs, cells * int(level["cls_channels"]), used)
        split.append((level, regression, classification))
    if len(used) != len(outputs):
        raise ValueError("Palm ONNX contains unrecognized outputs")
    return split


def _infer_layout(
    arr: np.ndarray,
    feature_height: int,
    feature_width: int,
    channels: int,
    default_layout: str,
) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape == (feature_height, feature_width, channels):
            return "hwc"
        if arr.shape == (channels, feature_height, feature_width):
            return "nchw"
    return default_layout


def decode_head(
    regression: np.ndarray,
    classification: np.ndarray,
    level: Mapping[str, Any],
    score_threshold: float,
    output_layout: str = "auto",
) -> List[Dict[str, Any]]:
    feature_height = int(level["height"])
    feature_width = int(level["width"])
    anchor_sizes = level["anchor_sizes"]
    anchor_count = len(anchor_sizes)
    default_layout = output_layout if output_layout in {"nchw", "hwc"} else "nchw"
    reg_layout = _infer_layout(
        regression,
        feature_height,
        feature_width,
        int(level["reg_channels"]),
        default_layout,
    )
    cls_layout = _infer_layout(
        classification,
        feature_height,
        feature_width,
        int(level["cls_channels"]),
        default_layout,
    )
    reg = _reshape_output(
        regression,
        feature_height,
        feature_width,
        int(level["reg_channels"]),
        reg_layout,
    ).reshape(-1, anchor_count, VALUES_PER_ANCHOR)
    cls = _reshape_output(
        classification,
        feature_height,
        feature_width,
        int(level["cls_channels"]),
        cls_layout,
    )
    anchors = generate_anchors(feature_height, feature_width, anchor_sizes)
    candidates: List[Dict[str, Any]] = []
    for cell_index in range(feature_height * feature_width):
        for anchor_id in range(anchor_count):
            anchor_index = cell_index * anchor_count + anchor_id
            score = float(cls[cell_index, anchor_id])
            if not math.isfinite(score) or score < float(score_threshold):
                continue
            anchor_x, anchor_y, anchor_width, anchor_height = [
                float(value) for value in anchors[anchor_index]
            ]
            dx, dy, dw, dh = [float(value) for value in reg[cell_index, anchor_id, :4]]
            if not all(math.isfinite(value) for value in (dx, dy, dw, dh)):
                continue
            center_x = anchor_x + dx * anchor_width
            center_y = anchor_y + dy * anchor_height
            box_width = anchor_width * math.exp(max(-10.0, min(10.0, dw)))
            box_height = anchor_height * math.exp(max(-10.0, min(10.0, dh)))
            keypoints = []
            for keypoint_index in range(2):
                base = 4 + keypoint_index * 2
                keypoints.append(
                    [
                        clamp01(anchor_x + float(reg[cell_index, anchor_id, base]) * anchor_width),
                        clamp01(anchor_y + float(reg[cell_index, anchor_id, base + 1]) * anchor_height),
                    ]
                )
            candidates.append(
                {
                    "score": score,
                    "bbox_norm": [
                        clamp01(center_x - box_width * 0.5),
                        clamp01(center_y - box_height * 0.5),
                        clamp01(center_x + box_width * 0.5),
                        clamp01(center_y + box_height * 0.5),
                    ],
                    "keypoints_norm": {"p0": keypoints[0], "p9": keypoints[1]},
                    "head": str(level["name"]),
                    "anchor_index": anchor_index,
                }
            )
    return candidates


def select_detections(
    candidates: Sequence[Mapping[str, Any]],
    nms_iou_threshold: float,
    max_detections: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    boxes = np.asarray([candidate["bbox_norm"] for candidate in candidates], dtype=np.float32)
    scores = np.asarray([float(candidate["score"]) for candidate in candidates], dtype=np.float32)
    selected = [dict(candidates[index]) for index in nms_indices(boxes, scores, nms_iou_threshold)]
    selected.sort(key=lambda candidate: float(candidate["score"]), reverse=True)
    if max_detections > 0:
        selected = selected[: int(max_detections)]
    return selected


def decode_onnx_outputs(
    outputs: Sequence[np.ndarray],
    feature_levels: Sequence[Mapping[str, Any]],
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    negative_candidate_threshold: float = 0.15,
    output_layout: str = "auto",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    minimum_threshold = min(float(score_threshold), float(negative_candidate_threshold))
    candidates: List[Dict[str, Any]] = []
    for level, regression, classification in split_onnx_outputs(outputs, feature_levels):
        candidates.extend(
            decode_head(
                regression,
                classification,
                level,
                minimum_threshold,
                output_layout,
            )
        )
    positive_pool = [
        candidate
        for candidate in candidates
        if float(candidate["score"]) >= float(score_threshold)
    ]
    detections = select_detections(positive_pool, nms_iou_threshold, max_detections)
    negatives = [
        dict(candidate)
        for candidate in sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        if float(negative_candidate_threshold) <= float(candidate["score"]) < float(score_threshold)
    ][: max(0, int(max_detections) * 5 or 10)]
    return detections, negatives


def candidate_to_schema(candidate: Mapping[str, Any], source: str, valid: bool = True) -> Dict[str, Any]:
    return {
        "valid": bool(valid),
        "score": float(candidate.get("score", 0.0)),
        "score_source": "palm_score",
        "bbox_norm": [float(value) for value in candidate["bbox_norm"]],
        "bbox_source": f"{source}_palm_bbox",
        "keypoints_norm": {
            "p0": [float(value) for value in candidate["keypoints_norm"]["p0"]],
            "p9": [float(value) for value in candidate["keypoints_norm"]["p9"]],
        },
        "source": source,
        "head": candidate.get("head"),
    }
