from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import cv2
import numpy as np

from .formats import make_palm_det_id, normalize_detection_schema
from .image_io import read_image, to_uint8_gray
from .onnx_runtime import create_onnx_session, onnx_provider_for
from .palm_decode import candidate_to_schema, decode_onnx_outputs
from .progress import track_progress


def preprocess_for_onnx(image: np.ndarray, input_size: int, input_type: str = "tensor(float)") -> np.ndarray:
    gray = to_uint8_gray(image)
    resized = cv2.resize(gray, (int(input_size), int(input_size)), interpolation=cv2.INTER_LINEAR)
    if "uint8" in input_type:
        return resized[np.newaxis, np.newaxis, :, :].astype(np.uint8)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]


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
    if runtime_info is not None:
        runtime_info.update(
            {
                "provider": provider,
                "provider_fallback_reason": fallback_reason,
                "batch_size": 1,
                "batch_size_reason": "Palm ONNX input has a fixed batch dimension of 1",
            }
        )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_type = getattr(input_meta, "type", "tensor(float)")
    palm_cfg = cfg["palm"]
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
        inp = preprocess_for_onnx(img, int(palm_cfg["input_size"]), input_type)
        outputs = session.run(None, {input_name: inp})
        detections_raw, negatives_raw = decode_onnx_outputs(
            outputs,
            score_threshold=float(palm_cfg["score_threshold"]),
            nms_iou_threshold=float(palm_cfg["nms_iou_threshold"]),
            cross_head_suppress_iou=float(palm_cfg["cross_head_suppress_iou"]),
            max_detections=int(palm_cfg["max_detections"]),
            negative_candidate_threshold=float(palm_cfg.get("negative_candidate_threshold", 0.15)),
            output_layout=str(palm_cfg.get("onnx_output_layout", "auto")).lower(),
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
