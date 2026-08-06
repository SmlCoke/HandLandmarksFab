from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .image_io import image_shape_info, read_image
from .roi_geometry import corners_all_far_from_image


def validate_image_file(path: Path, expected_width: int, expected_height: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {"image": Path(path).name, "path": str(path), "ok": False, "warnings": [], "errors": []}
    img = read_image(path)
    if img is None:
        row["errors"].append("unreadable_image")
        return row
    info = image_shape_info(img)
    row.update(info)
    if info["width"] != int(expected_width) or info["height"] != int(expected_height):
        row["errors"].append(f"unexpected_size:{info['width']}x{info['height']}")
    if info["channels"] not in {1, 3, 4}:
        row["errors"].append(f"unsupported_channels:{info['channels']}")
    elif info["channels"] != 1:
        row["warnings"].append(f"convertible_to_gray_channels:{info['channels']}")
    row["ok"] = not row["errors"]
    return row


def palm_record_issues(record: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    max_det = int(cfg["palm"].get("max_detections", 2))
    detections = list(record.get("detections") or [])
    if len(detections) > max_det:
        errors.append(f"detections_exceed_max:{len(detections)}>{max_det}")
    for det in detections + list(record.get("negative_candidates") or []):
        bbox = det.get("bbox_norm") or []
        if len(bbox) != 4:
            errors.append(f"{det.get('palm_det_id')}:missing_bbox")
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x1 < 0.0 or y1 < 0.0 or x2 > 1.0 or y2 > 1.0:
            warnings.append(f"{det.get('palm_det_id')}:bbox_out_of_bounds")
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < 0.0003 or area > 0.8:
            warnings.append(f"{det.get('palm_det_id')}:bbox_area_abnormal:{area:.6f}")
        kps = det.get("keypoints_norm") or {}
        if "p0" not in kps or "p9" not in kps:
            errors.append(f"{det.get('palm_det_id')}:missing_p0_or_p9")
    return warnings, errors


def roi_manifest_issues(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    corners = row.get("roi_corners_px") or []
    if len(corners) != 4:
        errors.append("roi_corners_not_4")
    elif corners_all_far_from_image(corners, int(cfg["image"]["width"]), int(cfg["image"]["height"])):
        warnings.append("roi_corners_far_from_image")
    rect = row.get("roi_rect") or {}
    if float(rect.get("width", 0.0)) < 2.0 or float(rect.get("height", 0.0)) < 2.0:
        errors.append("roi_too_small")
    return warnings, errors


def _points_out_of_bounds(points: Iterable[Mapping[str, Any]], width: int, height: int) -> int:
    count = 0
    for p in points:
        x = float(p.get("x", 0.0))
        y = float(p.get("y", 0.0))
        if x < 0.0 or y < 0.0 or x > float(width - 1) or y > float(height - 1):
            count += 1
    return count


def _rtmpose_boundary_coordinate_count(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> int:
    if str(row.get("split")) != "train":
        return 0
    if str(row.get("proposal_kind")) != "runtime":
        return 0
    if str(row.get("source")) != "rtmpose_m_hand5_onnx":
        return 0
    width = int(row.get("width", cfg["hand_roi"]["output_width"]))
    height = int(row.get("height", cfg["hand_roi"]["output_height"]))
    count = 0
    for point in row.get("landmarks_crop_px") or []:
        for axis, maximum in (("x", float(width - 1)), ("y", float(height - 1))):
            try:
                value = float(point[axis])
            except (KeyError, TypeError, ValueError):
                continue
            if value == 0.0 or value == maximum:
                count += 1
    return count


def label_issues(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[List[str], List[str], bool]:
    warnings: List[str] = []
    errors: List[str] = []
    needs_review = False
    present = bool((row.get("hand_presence") or {}).get("present", False))
    landmarks = list(row.get("landmarks_crop_norm") or [])
    if present and len(landmarks) != 21:
        errors.append(f"present_landmark_count_not_21:{len(landmarks)}")
        needs_review = True
    if not present and landmarks:
        errors.append("negative_sample_has_landmarks")
        needs_review = True
    crop_px = list(row.get("landmarks_crop_px") or [])
    out_count = _points_out_of_bounds(crop_px, int(row.get("width", cfg["hand_roi"]["output_width"])), int(row.get("height", cfg["hand_roi"]["output_height"])))
    if out_count:
        warnings.append(f"crop_points_out_of_bounds:{out_count}")
        needs_review = True
    try:
        boundary_threshold = int(
            cfg.get("quality", {}).get(
                "rtmpose_train_boundary_coordinate_reject_threshold", 3
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "quality.rtmpose_train_boundary_coordinate_reject_threshold must be an integer"
        ) from exc
    if boundary_threshold < 1:
        raise ValueError(
            "quality.rtmpose_train_boundary_coordinate_reject_threshold must be >= 1"
        )
    boundary_count = _rtmpose_boundary_coordinate_count(row, cfg)
    if boundary_count >= boundary_threshold:
        errors.append(
            f"rtmpose_boundary_coordinate_values:{boundary_count}>={boundary_threshold}"
        )
        needs_review = True
    if int(row.get("mediapipe_num_hands_detected", 0)) > 1:
        warnings.append("multiple_hands_in_one_crop")
        needs_review = True
    handedness = row.get("handedness") or {}
    hs = handedness.get("score")
    if present and hs is not None and float(hs) < float(cfg.get("quality", {}).get("handedness_review_threshold", 0.7)):
        warnings.append(f"low_handedness_score:{float(hs):.3f}")
        needs_review = True
    palm_score = row.get("palm_score")
    if not present and palm_score is not None and float(palm_score) >= float(cfg.get("quality", {}).get("high_palm_score_review_threshold", 0.8)):
        warnings.append(f"negative_with_high_palm_score:{float(palm_score):.3f}")
        needs_review = True
    return warnings, errors, needs_review


def summarize_label_rows(rows: Iterable[Mapping[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "left": 0,
        "right": 0,
        "unknown_handedness": 0,
        "errors": [],
        "warnings": [],
        "needs_review": 0,
        "out_of_bounds_points": 0,
    }
    for row in rows:
        stats["total"] += 1
        warnings, errors, needs_review = label_issues(row, cfg)
        if warnings:
            stats["warnings"].append({"crop_id": row.get("crop_id"), "warnings": warnings})
        if errors:
            stats["errors"].append({"crop_id": row.get("crop_id"), "errors": errors})
        if needs_review:
            stats["needs_review"] += 1
        present = bool((row.get("hand_presence") or {}).get("present", False))
        if present:
            stats["positive"] += 1
            label = str((row.get("handedness") or {}).get("label", "unknown")).lower()
            if label == "left":
                stats["left"] += 1
            elif label == "right":
                stats["right"] += 1
            else:
                stats["unknown_handedness"] += 1
        else:
            stats["negative"] += 1
        stats["out_of_bounds_points"] += _points_out_of_bounds(
            row.get("landmarks_crop_px") or [],
            int(row.get("width", cfg["hand_roi"]["output_width"])),
            int(row.get("height", cfg["hand_roi"]["output_height"])),
        )
    return stats


def histogram(values: Iterable[int]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for value in values:
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return hist
