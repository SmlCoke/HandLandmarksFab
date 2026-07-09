from __future__ import annotations

import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .image_io import to_uint8_gray


def normalize_radians(angle: float) -> float:
    return float(angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi)))


def build_roi_rect_from_palm(
    palm_detection: Mapping,
    image_width: int,
    image_height: int,
    scale_x: float = 1.8,
    scale_y: float = 1.8,
    shift_x: float = 0.0,
    shift_y: float = -0.1,
) -> dict:
    bbox = palm_detection.get("bbox_px") or []
    if len(bbox) != 4:
        raise ValueError(f"Palm detection has no bbox_px: {palm_detection}")
    x1 = max(0.0, min(float(image_width - 1), min(float(bbox[0]), float(bbox[2]))))
    y1 = max(0.0, min(float(image_height - 1), min(float(bbox[1]), float(bbox[3]))))
    x2 = max(0.0, min(float(image_width - 1), max(float(bbox[0]), float(bbox[2]))))
    y2 = max(0.0, min(float(image_height - 1), max(float(bbox[1]), float(bbox[3]))))
    raw_width = max(1.0, x2 - x1)
    raw_height = max(1.0, y2 - y1)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    keypoints = palm_detection.get("keypoints_px") or {}
    p0 = keypoints.get("p0")
    p9 = keypoints.get("p9")
    if p0 is None or p9 is None:
        raise ValueError(f"Palm detection must contain keypoints_px.p0 and keypoints_px.p9: {palm_detection}")
    dx = float(p9[0]) - float(p0[0])
    dy = float(p9[1]) - float(p0[1])
    rotation = normalize_radians((math.pi * 0.5) - math.atan2(-dy, dx))
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    center_x += raw_width * float(shift_x) * cos_r - raw_height * float(shift_y) * sin_r
    center_y += raw_width * float(shift_x) * sin_r + raw_height * float(shift_y) * cos_r

    long_side = max(raw_width, raw_height)
    roi_width = long_side * float(scale_x)
    roi_height = long_side * float(scale_y)
    return {
        "x_center": float(center_x),
        "y_center": float(center_y),
        "width": float(roi_width),
        "height": float(roi_height),
        "rotation_rad": float(rotation),
    }


def roi_corners_px(roi_rect: Mapping[str, float]) -> np.ndarray:
    center_x = float(roi_rect["x_center"])
    center_y = float(roi_rect["y_center"])
    width = float(roi_rect["width"])
    height = float(roi_rect["height"])
    rotation = float(roi_rect.get("rotation_rad", roi_rect.get("rotation", 0.0)))
    vx = np.array([math.cos(rotation) * width * 0.5, math.sin(rotation) * width * 0.5], dtype=np.float32)
    vy = np.array([-math.sin(rotation) * height * 0.5, math.cos(rotation) * height * 0.5], dtype=np.float32)
    center = np.array([center_x, center_y], dtype=np.float32)
    return np.stack([center - vx - vy, center + vx - vy, center + vx + vy, center - vx + vy]).astype(np.float32)


def crop_image_by_roi(image: np.ndarray, roi_rect: Mapping[str, float], output_width: int, output_height: int) -> tuple[np.ndarray, np.ndarray]:
    gray = to_uint8_gray(image)
    corners = roi_corners_px(roi_rect)
    src = np.array([corners[0], corners[1], corners[3]], dtype=np.float32)
    dst = np.array(
        [[0.0, 0.0], [float(output_width - 1), 0.0], [0.0, float(output_height - 1)]],
        dtype=np.float32,
    )
    transform = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(
        gray,
        transform,
        (int(output_width), int(output_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return crop, corners


def corners_all_far_from_image(corners: Sequence[Sequence[float]], width: int, height: int, margin: float = 64.0) -> bool:
    arr = np.asarray(corners, dtype=np.float32)
    inside_x = (arr[:, 0] >= -float(margin)) & (arr[:, 0] <= float(width) + float(margin))
    inside_y = (arr[:, 1] >= -float(margin)) & (arr[:, 1] <= float(height) + float(margin))
    return not bool(np.any(inside_x & inside_y))


def self_check() -> None:
    det = {
        "bbox_px": [500.0, 300.0, 620.0, 420.0],
        "keypoints_px": {"p0": [540.0, 410.0], "p9": [560.0, 330.0]},
    }
    rect = build_roi_rect_from_palm(det, 1280, 720)
    corners = roi_corners_px(rect)
    assert corners.shape == (4, 2)
    img = np.zeros((720, 1280), dtype=np.uint8)
    crop, crop_corners = crop_image_by_roi(img, rect, 256, 256)
    assert crop.shape == (256, 256)
    assert crop_corners.shape == (4, 2)


if __name__ == "__main__":
    self_check()
    print("roi_geometry self_check OK")
