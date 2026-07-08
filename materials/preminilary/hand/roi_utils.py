# File purpose: Provide ROI geometry, cropping, and projection helpers shared by inference scripts.
import math

import cv2
import numpy as np


def clamp01(x):
    return max(0.0, min(1.0, float(x)))


def normalize_radians(angle):
    return angle - 2.0 * math.pi * math.floor((angle - (-math.pi)) / (2.0 * math.pi))


def rect_from_bbox(bbox):
    xmin, ymin, xmax, ymax = [float(v) for v in bbox]
    return {
        "x_center": 0.5 * (xmin + xmax),
        "y_center": 0.5 * (ymin + ymax),
        "width": xmax - xmin,
        "height": ymax - ymin,
        "rotation": 0.0,
    }


def compute_rotation_from_palm_keypoints(palm_keypoints, image_width, image_height):
    wrist_x, wrist_y, middle_x, middle_y = [float(v) for v in palm_keypoints]
    x0 = wrist_x * image_width
    y0 = wrist_y * image_height
    x1 = middle_x * image_width
    y1 = middle_y * image_height
    return normalize_radians((math.pi / 2.0) - math.atan2(-(y1 - y0), x1 - x0))


def compute_rotation_from_points(point_a, point_b, image_width, image_height):
    x0 = float(point_a[0]) * image_width
    y0 = float(point_a[1]) * image_height
    x1 = float(point_b[0]) * image_width
    y1 = float(point_b[1]) * image_height
    return normalize_radians((math.pi / 2.0) - math.atan2(-(y1 - y0), x1 - x0))


def transform_normalized_rect(
    rect,
    image_width,
    image_height,
    *,
    scale_x=1.0,
    scale_y=1.0,
    shift_x=0.0,
    shift_y=0.0,
    square_long=False,
    square_short=False,
):
    width = float(rect["width"])
    height = float(rect["height"])
    rotation = float(rect.get("rotation", 0.0))
    x_center = float(rect["x_center"])
    y_center = float(rect["y_center"])

    if rotation == 0.0:
        x_center += width * shift_x
        y_center += height * shift_y
    else:
        x_shift = (
            image_width * width * shift_x * math.cos(rotation)
            - image_height * height * shift_y * math.sin(rotation)
        ) / image_width
        y_shift = (
            image_width * width * shift_x * math.sin(rotation)
            + image_height * height * shift_y * math.cos(rotation)
        ) / image_height
        x_center += x_shift
        y_center += y_shift

    if square_long:
        long_side = max(width * image_width, height * image_height)
        width = long_side / image_width
        height = long_side / image_height
    elif square_short:
        short_side = min(width * image_width, height * image_height)
        width = short_side / image_width
        height = short_side / image_height

    return {
        "x_center": x_center,
        "y_center": y_center,
        "width": width * scale_x,
        "height": height * scale_y,
        "rotation": rotation,
    }


def blend_rects(base_rect, refined_rect, alpha):
    alpha = float(alpha)
    beta = 1.0 - alpha
    return {
        "x_center": beta * float(base_rect["x_center"]) + alpha * float(refined_rect["x_center"]),
        "y_center": beta * float(base_rect["y_center"]) + alpha * float(refined_rect["y_center"]),
        "width": beta * float(base_rect["width"]) + alpha * float(refined_rect["width"]),
        "height": beta * float(base_rect["height"]) + alpha * float(refined_rect["height"]),
        "rotation": beta * float(base_rect.get("rotation", 0.0))
        + alpha * float(refined_rect.get("rotation", 0.0)),
    }


def rect_from_hand_points(hand_points, bbox=None):
    usable_order = [0, 5, 9, 13, 17]
    pts = [hand_points[idx] for idx in usable_order if idx in hand_points]
    if len(pts) < 3:
        return rect_from_bbox(bbox) if bbox is not None else None

    xs = [float(pt[0]) for pt in pts]
    ys = [float(pt[1]) for pt in pts]
    xmin = min(xs)
    ymin = min(ys)
    xmax = max(xs)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin
    if bbox is not None:
        bbox_rect = rect_from_bbox(bbox)
        width = max(width, 0.55 * float(bbox_rect["width"]))
        height = max(height, 0.55 * float(bbox_rect["height"]))

    return {
        "x_center": float(np.mean(xs)),
        "y_center": float(np.mean(ys)),
        "width": width,
        "height": height,
        "rotation": 0.0,
    }


def rect_from_bbox_and_hand_points(
    bbox,
    hand_points,
    image_width,
    image_height,
    *,
    scale_x=2.6,
    scale_y=2.6,
    shift_y=-0.5,
):
    # Strict MP-like chain:
    # bbox -> rotation(from keypoints) -> normalized-rect transform.
    rect = rect_from_bbox(bbox)

    rotation = 0.0
    if 0 in hand_points and 9 in hand_points:
        rotation = compute_rotation_from_points(
            hand_points[0], hand_points[9], image_width, image_height
        )

    rect["rotation"] = rotation

    return transform_normalized_rect(
        rect,
        image_width,
        image_height,
        scale_x=scale_x,
        scale_y=scale_y,
        shift_y=shift_y,
        square_long=True,
    )


def palm_bbox_to_rect(
    bbox,
    image_width,
    image_height,
    *,
    palm_keypoints=None,
    rotation_deg=None,
    scale_x=2.6,
    scale_y=2.6,
    shift_y=-0.5,
):
    rect = rect_from_bbox(bbox)

    if palm_keypoints is not None:
        rect["rotation"] = compute_rotation_from_palm_keypoints(
            palm_keypoints, image_width, image_height
        )
    elif rotation_deg is not None:
        rect["rotation"] = math.radians(float(rotation_deg))

    return transform_normalized_rect(
        rect,
        image_width,
        image_height,
        scale_x=scale_x,
        scale_y=scale_y,
        shift_y=shift_y,
        square_long=True,
    )


def rect_corners_px(rect, image_width, image_height):
    cx = float(rect["x_center"]) * image_width
    cy = float(rect["y_center"]) * image_height
    width = float(rect["width"]) * image_width
    height = float(rect["height"]) * image_height
    rotation = float(rect.get("rotation", 0.0))

    vx = np.array([math.cos(rotation) * width * 0.5, math.sin(rotation) * width * 0.5])
    vy = np.array([-math.sin(rotation) * height * 0.5, math.cos(rotation) * height * 0.5])
    center = np.array([cx, cy], dtype=np.float32)

    top_left = center - vx - vy
    top_right = center + vx - vy
    bottom_right = center + vx + vy
    bottom_left = center - vx + vy
    return np.stack([top_left, top_right, bottom_right, bottom_left]).astype(np.float32)


def crop_image_by_rect(
    img,
    rect,
    output_width,
    output_height,
    *,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
):
    corners = rect_corners_px(rect, img.shape[1], img.shape[0])
    src = np.array([corners[0], corners[1], corners[3]], dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [float(output_width - 1), 0.0],
            [0.0, float(output_height - 1)],
        ],
        dtype=np.float32,
    )
    transform = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(
        img,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )
    return crop, corners


def project_points_from_rect(points, rect, image_width, image_height):
    corners = rect_corners_px(rect, image_width, image_height)
    top_left = corners[0]
    top_right = corners[1]
    bottom_left = corners[3]
    results = []
    for x, y in points:
        px = top_left + x * (top_right - top_left) + y * (bottom_left - top_left)
        results.append((float(px[0]), float(px[1])))
    return results
