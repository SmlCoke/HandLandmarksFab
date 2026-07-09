from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def project_norm_points_to_image(points_norm: Iterable[Tuple[float, float]], roi_corners_px: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    corners = np.asarray(roi_corners_px, dtype=np.float32)
    top_left = corners[0]
    top_right = corners[1]
    bottom_left = corners[3]
    result: List[Tuple[float, float]] = []
    for x, y in points_norm:
        x = float(x)
        y = float(y)
        p = top_left + x * (top_right - top_left) + y * (bottom_left - top_left)
        result.append((float(p[0]), float(p[1])))
    return result


def project_px_points_to_image(points_px: Iterable[Tuple[float, float]], roi_corners_px: Sequence[Sequence[float]], crop_width: int, crop_height: int) -> List[Tuple[float, float]]:
    denom_x = float(max(1, int(crop_width) - 1))
    denom_y = float(max(1, int(crop_height) - 1))
    norm = [(float(x) / denom_x, float(y) / denom_y) for x, y in points_px]
    return project_norm_points_to_image(norm, roi_corners_px)


def landmark_dicts_from_norm(points: Iterable[Mapping[str, float]], roi_corners_px: Sequence[Sequence[float]]) -> List[dict]:
    point_list = list(points)
    pairs = [(float(p["x"]), float(p["y"])) for p in point_list]
    projected = project_norm_points_to_image(pairs, roi_corners_px)
    return [
        {"id": int(src.get("id", idx)), "x": x, "y": y, "visible": int(src.get("visible", 1))}
        for idx, (src, (x, y)) in enumerate(zip(point_list, projected))
    ]


def self_check() -> None:
    corners = [[10.0, 20.0], [110.0, 20.0], [110.0, 120.0], [10.0, 120.0]]
    pts = project_norm_points_to_image([(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)], corners)
    assert pts[0] == (10.0, 20.0)
    assert pts[1] == (110.0, 120.0)
    assert pts[2] == (60.0, 70.0)


if __name__ == "__main__":
    self_check()
    print("projection self_check OK")
