from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .formats import clamp01
from .nms import bbox_iou, nms_indices


ANCHORS_14 = ((0.10, 0.10), (0.18, 0.18))
ANCHORS_7 = ((0.25, 0.25), (0.40, 0.40))
VALUES_PER_ANCHOR = 8
REG_CHANNELS = 16
CLS_CHANNELS = 2


def generate_anchors(feature_size: int, sizes: Sequence[Tuple[float, float]]) -> np.ndarray:
    anchors: List[List[float]] = []
    step = 1.0 / float(feature_size)
    for y in range(feature_size):
        for x in range(feature_size):
            cx = x * step + step * 0.5
            cy = y * step + step * 0.5
            for w, h in sizes:
                anchors.append([cx, cy, w, h])
    return np.asarray(anchors, dtype=np.float32)


def _reshape_output(arr: np.ndarray, feature_size: int, channels: int, layout: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim == 3:
        if arr.shape == (channels, feature_size, feature_size):
            return arr.transpose(1, 2, 0).reshape(-1, channels)
        if arr.shape == (feature_size, feature_size, channels):
            return arr.reshape(-1, channels)

    flat = arr.reshape(-1)
    expected = feature_size * feature_size * channels
    if flat.size != expected:
        raise ValueError(f"Unexpected head tensor size: got {flat.size}, expected {expected}")
    if layout == "hwc":
        return flat.reshape(feature_size, feature_size, channels).reshape(-1, channels)
    return flat.reshape(channels, feature_size, feature_size).transpose(1, 2, 0).reshape(-1, channels)


def _find_output(outputs: Sequence[np.ndarray], expected_size: int, used: set[int]) -> Tuple[int, np.ndarray]:
    for idx, arr in enumerate(outputs):
        if idx in used:
            continue
        if int(np.asarray(arr).size) == int(expected_size):
            used.add(idx)
            return idx, np.asarray(arr)
    raise ValueError(f"Could not find ONNX output with {expected_size} elements")


def split_onnx_outputs(outputs: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    used: set[int] = set()
    _, reg14 = _find_output(outputs, 14 * 14 * REG_CHANNELS, used)
    _, cls14 = _find_output(outputs, 14 * 14 * CLS_CHANNELS, used)
    _, reg7 = _find_output(outputs, 7 * 7 * REG_CHANNELS, used)
    _, cls7 = _find_output(outputs, 7 * 7 * CLS_CHANNELS, used)
    return reg14, cls14, reg7, cls7


def _infer_layout(arr: np.ndarray, feature_size: int, channels: int, default_layout: str) -> str:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape == (feature_size, feature_size, channels):
            return "hwc"
        if arr.shape == (channels, feature_size, feature_size):
            return "nchw"
    return default_layout


def decode_head(
    reg_pred: np.ndarray,
    cls_pred: np.ndarray,
    feature_size: int,
    anchor_sizes: Sequence[Tuple[float, float]],
    score_threshold: float,
    head_name: str,
    output_layout: str = "auto",
) -> List[Dict]:
    default_layout = "nchw"
    if output_layout in {"nchw", "hwc"}:
        default_layout = output_layout
    reg_layout = _infer_layout(reg_pred, feature_size, REG_CHANNELS, default_layout)
    cls_layout = _infer_layout(cls_pred, feature_size, CLS_CHANNELS, default_layout)
    reg = _reshape_output(reg_pred, feature_size, REG_CHANNELS, reg_layout).reshape(-1, 2, VALUES_PER_ANCHOR)
    cls = _reshape_output(cls_pred, feature_size, CLS_CHANNELS, cls_layout)
    anchors = generate_anchors(feature_size, anchor_sizes)
    candidates: List[Dict] = []
    cell_count = feature_size * feature_size
    for cell_idx in range(cell_count):
        for anchor_id in range(2):
            anchor_index = cell_idx * 2 + anchor_id
            score = float(cls[cell_idx, anchor_id])
            if not math.isfinite(score) or score < float(score_threshold):
                continue
            anc_cx, anc_cy, anc_w, anc_h = [float(v) for v in anchors[anchor_index]]
            dx, dy, dw, dh = [float(v) for v in reg[cell_idx, anchor_id, :4]]
            if not all(math.isfinite(v) for v in (dx, dy, dw, dh)):
                continue
            dw = max(-10.0, min(10.0, dw))
            dh = max(-10.0, min(10.0, dh))
            cx = anc_cx + dx * anc_w
            cy = anc_cy + dy * anc_h
            w_box = anc_w * math.exp(dw)
            h_box = anc_h * math.exp(dh)
            bbox = [
                clamp01(cx - w_box * 0.5),
                clamp01(cy - h_box * 0.5),
                clamp01(cx + w_box * 0.5),
                clamp01(cy + h_box * 0.5),
            ]
            keypoints = []
            for kp_idx in range(2):
                base = 4 + kp_idx * 2
                kx = clamp01(anc_cx + float(reg[cell_idx, anchor_id, base]) * anc_w)
                ky = clamp01(anc_cy + float(reg[cell_idx, anchor_id, base + 1]) * anc_h)
                keypoints.append([kx, ky])
            candidates.append(
                {
                    "score": score,
                    "bbox_norm": bbox,
                    "keypoints_norm": {"p0": keypoints[0], "p9": keypoints[1]},
                    "head": head_name,
                    "anchor_index": int(anchor_index),
                }
            )
    return candidates


def _nms_candidates(candidates: Sequence[Mapping], threshold: float) -> List[Mapping]:
    if not candidates:
        return []
    boxes = np.asarray([c["bbox_norm"] for c in candidates], dtype=np.float32)
    scores = np.asarray([float(c["score"]) for c in candidates], dtype=np.float32)
    keep = nms_indices(boxes, scores, threshold)
    return [candidates[i] for i in keep]


def select_detections(
    candidates: Sequence[Mapping],
    nms_iou_threshold: float,
    cross_head_suppress_iou: float,
    max_detections: int,
) -> List[Dict]:
    head14 = [c for c in candidates if c.get("head") == "head14"]
    head7 = [c for c in candidates if c.get("head") == "head7"]
    selected: List[Mapping] = []
    selected.extend(_nms_candidates(head14, nms_iou_threshold))
    for cand in sorted(_nms_candidates(head7, nms_iou_threshold), key=lambda c: float(c["score"]), reverse=True):
        if any(bbox_iou(cand["bbox_norm"], prev["bbox_norm"]) > float(cross_head_suppress_iou) for prev in selected):
            continue
        selected.append(cand)
    selected = sorted(selected, key=lambda c: float(c["score"]), reverse=True)
    if max_detections > 0:
        selected = selected[: int(max_detections)]
    return [dict(c) for c in selected]


def decode_onnx_outputs(
    outputs: Sequence[np.ndarray],
    score_threshold: float,
    nms_iou_threshold: float,
    cross_head_suppress_iou: float,
    max_detections: int,
    negative_candidate_threshold: float = 0.15,
    output_layout: str = "auto",
) -> Tuple[List[Dict], List[Dict]]:
    reg14, cls14, reg7, cls7 = split_onnx_outputs(outputs)
    min_threshold = min(float(score_threshold), float(negative_candidate_threshold))
    candidates: List[Dict] = []
    candidates.extend(decode_head(reg14, cls14, 14, ANCHORS_14, min_threshold, "head14", output_layout))
    candidates.extend(decode_head(reg7, cls7, 7, ANCHORS_7, min_threshold, "head7", output_layout))

    positive_pool = [c for c in candidates if float(c["score"]) >= float(score_threshold)]
    detections = select_detections(positive_pool, nms_iou_threshold, cross_head_suppress_iou, max_detections)

    negatives = [
        dict(c)
        for c in sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        if float(negative_candidate_threshold) <= float(c["score"]) < float(score_threshold)
    ][: max(0, int(max_detections) * 5 or 10)]
    return detections, negatives


def candidate_to_schema(candidate: Mapping, source: str, valid: bool = True) -> Dict:
    return {
        "valid": bool(valid),
        "score": float(candidate.get("score", 0.0)),
        "score_source": "palm_score",
        "bbox_norm": [float(v) for v in candidate["bbox_norm"]],
        "bbox_source": f"{source}_palm_bbox",
        "keypoints_norm": {
            "p0": [float(v) for v in candidate["keypoints_norm"]["p0"]],
            "p9": [float(v) for v in candidate["keypoints_norm"]["p9"]],
        },
        "source": source,
        "head": candidate.get("head"),
    }
