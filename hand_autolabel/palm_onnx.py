from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import cv2
import numpy as np

from .formats import make_palm_det_id, normalize_detection_schema
from .image_io import read_image, to_uint8_gray
from .onnx_runtime import create_onnx_session, onnx_provider_for
from .palm_decode import (
    candidate_to_schema,
    decode_onnx_outputs,
    feature_level_anchor_count,
    normalize_feature_levels,
)
from .progress import track_progress


def preprocess_for_onnx(
    image: np.ndarray,
    input_width: int,
    input_height: int,
    input_type: str = "tensor(float)",
) -> np.ndarray:
    gray = to_uint8_gray(image)
    resized = cv2.resize(
        gray,
        (int(input_width), int(input_height)),
        interpolation=cv2.INTER_AREA,
    )
    if "uint8" in input_type:
        return resized[np.newaxis, np.newaxis, :, :].astype(np.uint8)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]


def _static_shape(value: Any, field: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} must be one static NCHW rank-4 shape")
    shape: List[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError(f"{field} must contain only positive integer dimensions")
        shape.append(dimension)
    return shape


def palm_model_contract(session: Any, cfg: Mapping[str, Any], model_path: Path) -> Dict[str, Any]:
    palm_cfg = cfg.get("palm") or {}
    if not isinstance(palm_cfg, Mapping):
        raise ValueError("palm must be a mapping")
    feature_levels = normalize_feature_levels(palm_cfg)
    model_id = str(palm_cfg.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("palm.model_id must be a non-empty string")
    raw_input_width = palm_cfg.get("input_width")
    raw_input_height = palm_cfg.get("input_height")
    if (
        isinstance(raw_input_width, bool)
        or not isinstance(raw_input_width, int)
        or raw_input_width < 1
        or isinstance(raw_input_height, bool)
        or not isinstance(raw_input_height, int)
        or raw_input_height < 1
    ):
        raise ValueError("palm.input_width and palm.input_height must be positive integers")
    input_width = raw_input_width
    input_height = raw_input_height
    inputs = list(session.get_inputs())
    if len(inputs) != 1:
        raise ValueError(f"Palm ONNX must expose exactly one input, got {len(inputs)}")
    input_meta = inputs[0]
    input_shape = _static_shape(input_meta.shape, "Palm ONNX input")
    expected_input_shape = [1, 1, input_height, input_width]
    if input_shape != expected_input_shape:
        raise ValueError(
            f"Palm ONNX input shape {input_shape} does not match configured {expected_input_shape}"
        )
    input_type = str(getattr(input_meta, "type", ""))
    if input_type not in {"tensor(float)", "tensor(uint8)"}:
        raise ValueError(f"Palm ONNX input type must be float32 or uint8, got {input_type!r}")
    output_layout = str(palm_cfg.get("onnx_output_layout", "nchw")).strip().lower()
    if output_layout not in {"nchw", "hwc"}:
        raise ValueError("palm.onnx_output_layout must be nchw or hwc")
    expected_shapes: List[List[int]] = []
    for level in feature_levels:
        height = int(level["height"])
        width = int(level["width"])
        for channels in (int(level["reg_channels"]), int(level["cls_channels"])):
            expected_shapes.append(
                [1, channels, height, width]
                if output_layout == "nchw"
                else [1, height, width, channels]
            )
    outputs = list(session.get_outputs())
    output_shapes = [_static_shape(meta.shape, f"Palm ONNX output {meta.name}") for meta in outputs]
    if sorted(output_shapes) != sorted(expected_shapes):
        raise ValueError(
            f"Palm ONNX output shapes {output_shapes} do not match configured {expected_shapes}"
        )
    if any(str(getattr(meta, "type", "")) != "tensor(float)" for meta in outputs):
        raise ValueError("Palm ONNX outputs must all be float32")
    score_threshold = float(palm_cfg["score_threshold"])
    negative_threshold = float(palm_cfg.get("negative_candidate_threshold", 0.15))
    nms_threshold = float(palm_cfg["nms_iou_threshold"])
    raw_max_detections = palm_cfg["max_detections"]
    if isinstance(raw_max_detections, bool) or not isinstance(raw_max_detections, int):
        raise ValueError("palm.max_detections must be a positive integer")
    max_detections = raw_max_detections
    if not (
        math.isfinite(negative_threshold)
        and math.isfinite(score_threshold)
        and 0.0 <= negative_threshold < score_threshold <= 1.0
    ):
        raise ValueError("Palm thresholds must satisfy 0 <= negative < score <= 1")
    if not math.isfinite(nms_threshold) or not 0.0 <= nms_threshold <= 1.0:
        raise ValueError("palm.nms_iou_threshold must be finite and in [0,1]")
    if max_detections < 1:
        raise ValueError("palm.max_detections must be a positive integer")
    return {
        "model_id": model_id,
        "model_path": str((cfg.get("paths") or {}).get("palm_model_onnx") or model_path),
        "input_name": str(input_meta.name),
        "input_shape": input_shape,
        "input_type": input_type,
        "output_shapes": output_shapes,
        "output_names": [str(meta.name) for meta in outputs],
        "output_layout": output_layout,
        "preprocess": "uint8_gray_INTER_AREA_float32_div255_nchw",
        "anchor_count": feature_level_anchor_count(feature_levels),
        "feature_levels": [
            {
                "name": str(level["name"]),
                "height": int(level["height"]),
                "width": int(level["width"]),
                "anchor_sizes": [list(anchor) for anchor in level["anchor_sizes"]],
            }
            for level in feature_levels
        ],
        "score_threshold": score_threshold,
        "nms_iou_threshold": nms_threshold,
        "max_detections": max_detections,
        "negative_candidate_threshold": negative_threshold,
    }


def run_onnx_palm_detector(
    image_paths: Iterable[Path],
    cfg: Mapping[str, Any],
    model_path: Path,
    *,
    show_progress: bool = False,
    runtime_info: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Palm ONNX model not found: {model_path}")
    provider_preference = onnx_provider_for(cfg, "palm")
    session, provider, fallback_reason = create_onnx_session(
        model_path, provider_preference
    )
    contract = palm_model_contract(session, cfg, Path(model_path))
    if runtime_info is not None:
        runtime_info.update(
            {
                "provider": provider,
                "provider_fallback_reason": fallback_reason,
                "batch_size": 1,
                "batch_size_reason": "Palm ONNX input has a fixed batch dimension of 1",
                "model_contract": contract,
            }
        )
    input_name = contract["input_name"]
    input_type = contract["input_type"]
    palm_cfg = cfg["palm"]
    feature_levels = normalize_feature_levels(palm_cfg)
    image_cfg = cfg["image"]
    width = int(image_cfg["width"])
    height = int(image_cfg["height"])
    rows: List[Dict[str, Any]] = []
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
        inp = preprocess_for_onnx(
            img,
            int(palm_cfg["input_width"]),
            int(palm_cfg["input_height"]),
            input_type,
        )
        outputs = session.run(None, {input_name: inp})
        detections_raw, negatives_raw = decode_onnx_outputs(
            outputs,
            feature_levels=feature_levels,
            score_threshold=float(palm_cfg["score_threshold"]),
            nms_iou_threshold=float(palm_cfg["nms_iou_threshold"]),
            max_detections=int(palm_cfg["max_detections"]),
            negative_candidate_threshold=float(palm_cfg.get("negative_candidate_threshold", 0.15)),
            output_layout=str(palm_cfg.get("onnx_output_layout", "nchw")).lower(),
        )
        detections = []
        for idx, cand in enumerate(detections_raw):
            det = candidate_to_schema(cand, "aethersign_onnx", valid=True)
            det["palm_det_id"] = make_palm_det_id(image_path.name, idx, "palm")
            detections.append(normalize_detection_schema(det, image_path.name, idx, width, height))
        negatives = []
        if bool(palm_cfg.get("keep_low_score_candidates_for_negatives", True)):
            for idx, cand in enumerate(negatives_raw):
                det = candidate_to_schema(cand, "aethersign_onnx", valid=False)
                det["palm_det_id"] = make_palm_det_id(image_path.name, idx, "neg")
                negatives.append(normalize_detection_schema(det, image_path.name, idx, width, height))
        rows.append({"image": image_path.name, "width": width, "height": height, "detections": detections, "negative_candidates": negatives})
    return rows
