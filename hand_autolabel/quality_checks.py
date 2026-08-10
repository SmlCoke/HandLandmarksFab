from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from .dataset_v3 import DatasetContractError, parse_capture_source_id
from .image_io import image_shape_info, read_image
from .roi_geometry import corners_all_far_from_image


RTMPOSE_CONNECTION_PAIRS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
RTMPOSE_CONNECTION_DISTANCES = ("near", "mid", "far")
RTMPOSE_TRAIN_RUNTIME_SOURCES = {
    "rtmpose_m_hand5_onnx",
    "mediapipe_hand_landmarker_full_tflite_rtmpose_rescue",
}


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
        try:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
        except (AttributeError, TypeError, ValueError):
            count += 1
            continue
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or x < 0.0
            or y < 0.0
            or x > float(width - 1)
            or y > float(height - 1)
        ):
            count += 1
    return count


def _is_rtmpose_train_runtime(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("split")) == "train"
        and str(row.get("proposal_kind")) == "runtime"
        and str(row.get("source")) in RTMPOSE_TRAIN_RUNTIME_SOURCES
    )


def validate_rtmpose_boundary_threshold(cfg: Mapping[str, Any]) -> int:
    try:
        threshold = int(
            cfg.get("quality", {}).get(
                "rtmpose_train_boundary_coordinate_reject_threshold", 3
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "quality.rtmpose_train_boundary_coordinate_reject_threshold must be an integer"
        ) from exc
    if threshold < 1:
        raise ValueError(
            "quality.rtmpose_train_boundary_coordinate_reject_threshold must be >= 1"
        )
    return threshold


def _rtmpose_boundary_coordinate_count(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> int:
    if not _is_rtmpose_train_runtime(row):
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


def _rtmpose_connection_gate_enabled(cfg: Mapping[str, Any]) -> bool:
    raw = cfg.get("quality", {}).get(
        "rtmpose_train_connection_length_gate_enabled", True
    )
    if not isinstance(raw, bool):
        raise ValueError(
            "quality.rtmpose_train_connection_length_gate_enabled must be a boolean"
        )
    return raw


def validate_rtmpose_connection_thresholds(
    cfg: Mapping[str, Any],
) -> Dict[str, Dict[tuple[int, int], float]]:
    raw = cfg.get("quality", {}).get(
        "rtmpose_train_connection_length_thresholds_px"
    )
    if not isinstance(raw, Mapping):
        raise ValueError(
            "quality.rtmpose_train_connection_length_thresholds_px must be a mapping"
        )
    actual_distances = {str(key) for key in raw}
    expected_distances = set(RTMPOSE_CONNECTION_DISTANCES)
    if actual_distances != expected_distances:
        raise ValueError(
            "quality.rtmpose_train_connection_length_thresholds_px must define "
            "exactly near, mid and far"
        )

    expected_keys = {f"{start}-{end}" for start, end in RTMPOSE_CONNECTION_PAIRS}
    validated: Dict[str, Dict[tuple[int, int], float]] = {}
    for distance in RTMPOSE_CONNECTION_DISTANCES:
        distance_values = raw.get(distance)
        if not isinstance(distance_values, Mapping):
            raise ValueError(
                "quality.rtmpose_train_connection_length_thresholds_px."
                f"{distance} must be a mapping"
            )
        actual_keys = {str(key) for key in distance_values}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                "quality.rtmpose_train_connection_length_thresholds_px."
                f"{distance} must define exactly the 20 configured connections; "
                f"missing={missing}, extra={extra}"
            )
        validated[distance] = {}
        for pair in RTMPOSE_CONNECTION_PAIRS:
            key = f"{pair[0]}-{pair[1]}"
            threshold = distance_values.get(key)
            if isinstance(threshold, bool):
                raise ValueError(
                    f"connection threshold {distance}.{key} must be a finite number > 0"
                )
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"connection threshold {distance}.{key} must be a finite number > 0"
                ) from exc
            if not np.isfinite(threshold_value) or threshold_value <= 0.0:
                raise ValueError(
                    f"connection threshold {distance}.{key} must be a finite number > 0"
                )
            validated[distance][pair] = threshold_value
    return validated


def rtmpose_connection_lengths_px(
    points: Iterable[Mapping[str, Any]],
) -> Dict[tuple[int, int], float]:
    raw_points = list(points)
    if len(raw_points) != 21:
        raise ValueError("RTMPose connection gate requires exactly 21 landmarks")
    coordinates: Dict[int, tuple[float, float]] = {}
    for point in raw_points:
        if not isinstance(point, Mapping):
            raise ValueError("RTMPose connection landmark must be a mapping")
        raw_id = point.get("id")
        if isinstance(raw_id, bool):
            raise ValueError("RTMPose connection landmark id must be an integer")
        try:
            point_id = int(raw_id)
            if float(raw_id) != float(point_id):
                raise ValueError
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("RTMPose connection landmark is invalid") from exc
        if point_id in coordinates or point_id < 0 or point_id > 20:
            raise ValueError("RTMPose connection landmark ids must be unique 0..20")
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("RTMPose connection coordinates must be finite")
        coordinates[point_id] = (x, y)
    if set(coordinates) != set(range(21)):
        raise ValueError("RTMPose connection landmark ids must be exactly 0..20")
    return {
        pair: math.hypot(
            coordinates[pair[0]][0] - coordinates[pair[1]][0],
            coordinates[pair[0]][1] - coordinates[pair[1]][1],
        )
        for pair in RTMPOSE_CONNECTION_PAIRS
    }


def _rtmpose_connection_length_gate_errors(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> List[str]:
    if not _is_rtmpose_train_runtime(row):
        return []
    if not _rtmpose_connection_gate_enabled(cfg):
        return []
    thresholds = validate_rtmpose_connection_thresholds(cfg)
    capture_source_id = row.get("capture_source_id")
    try:
        distance = parse_capture_source_id(str(capture_source_id))["distance"]
    except DatasetContractError as exc:
        raise ValueError(
            "RTMPose connection gate requires a valid capture_source_id"
        ) from exc
    if distance not in thresholds:
        raise ValueError(
            f"RTMPose connection gate has no thresholds for distance {distance!r}"
        )
    try:
        lengths = rtmpose_connection_lengths_px(row.get("landmarks_crop_px") or [])
    except ValueError:
        return ["rtmpose_connection_length_landmarks_invalid"]
    errors: List[str] = []
    for pair in RTMPOSE_CONNECTION_PAIRS:
        length = lengths[pair]
        threshold = thresholds[distance][pair]
        if length > threshold:
            errors.append(
                "rtmpose_connection_length_exceeded:"
                f"{pair[0]}-{pair[1]}:{length:.6f}>{threshold:.6f}:"
                f"distance={distance}"
            )
    return errors


def rtmpose_geometry_gate_errors(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> List[str]:
    """Return the unchanged boundary/connection errors for an RTMPose Train row."""

    if not _is_rtmpose_train_runtime(row):
        return []
    errors: List[str] = []
    boundary_threshold = validate_rtmpose_boundary_threshold(cfg)
    boundary_count = _rtmpose_boundary_coordinate_count(row, cfg)
    if boundary_count >= boundary_threshold:
        errors.append(
            f"rtmpose_boundary_coordinate_values:{boundary_count}>={boundary_threshold}"
        )
    errors.extend(_rtmpose_connection_length_gate_errors(row, cfg))
    return errors


def _rtmpose_hand_presence_gate_error(
    row: Mapping[str, Any], cfg: Mapping[str, Any]
) -> str | None:
    if not _is_rtmpose_train_runtime(row):
        return None
    raw_threshold = cfg.get("quality", {}).get(
        "rtmpose_train_hand_presence_threshold", 0.5
    )
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "quality.rtmpose_train_hand_presence_threshold must be a finite number"
        ) from exc
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            "quality.rtmpose_train_hand_presence_threshold must be within [0, 1]"
        )
    score = (row.get("hand_presence") or {}).get("score")
    if score is None:
        return "rtmpose_hand_presence_score_missing"
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return "rtmpose_hand_presence_score_non_finite"
    if not np.isfinite(score_value):
        return "rtmpose_hand_presence_score_non_finite"
    if score_value < threshold:
        return (
            f"rtmpose_hand_presence_score_below_threshold:"
            f"{score_value:.6f}<{threshold:.6f}"
        )
    return None


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
    # Preserve the existing eager validation even for rows outside the RTMPose
    # Train route, while sharing the exact gate implementation with rescue.
    validate_rtmpose_boundary_threshold(cfg)
    geometry_errors = rtmpose_geometry_gate_errors(row, cfg)
    if geometry_errors:
        errors.extend(geometry_errors)
        needs_review = True
    presence_gate_error = _rtmpose_hand_presence_gate_error(row, cfg)
    if presence_gate_error is not None:
        errors.append(presence_gate_error)
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
