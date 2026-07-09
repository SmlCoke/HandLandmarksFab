from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import cv2
import numpy as np

from .formats import index_by, relpath, resolve_path
from .image_io import ensure_bgr, read_image, write_image


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _draw_text(img, text: str, org: tuple[int, int], color=(0, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_palm(out, det: Mapping[str, Any]) -> None:
    bbox = [int(round(float(v))) for v in det.get("bbox_px", [])]
    if len(bbox) == 4:
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
    kps = det.get("keypoints_px") or {}
    for name, color in (("p0", (0, 0, 255)), ("p9", (255, 0, 0))):
        if name in kps:
            x, y = [int(round(float(v))) for v in kps[name]]
            cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)
            _draw_text(out, name, (x + 4, y - 4), color)
    if len(bbox) == 4:
        _draw_text(out, f"palm {float(det.get('score', 0.0)):.2f}", (bbox[0], max(16, bbox[1] - 6)))


def _draw_roi(out, manifest: Mapping[str, Any]) -> None:
    corners = np.asarray(manifest.get("roi_corners_px") or [], dtype=np.float32)
    if corners.shape != (4, 2):
        return
    poly = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)


def _draw_landmarks(out, label: Mapping[str, Any]) -> None:
    pts = label.get("landmarks_image_px") or []
    if len(pts) != 21:
        return
    coords = [(int(round(float(p["x"]))), int(round(float(p["y"])))) for p in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(out, coords[a], coords[b], (0, 180, 255), 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(coords):
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if idx in {0, 4, 8, 12, 16, 20}:
            _draw_text(out, str(idx), (x + 3, y + 3), (255, 255, 255))


def render_overlays(
    images_dir: Path,
    palm_rows: List[Mapping[str, Any]],
    manifest_rows: List[Mapping[str, Any]],
    label_rows: List[Mapping[str, Any]],
    root: Path,
    output_dir: Path,
    review_index_csv: Path,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    palm_by_image = {str(r["image"]): r for r in palm_rows}
    manifests_by_image: Dict[str, List[Mapping[str, Any]]] = {}
    for row in manifest_rows:
        manifests_by_image.setdefault(str(row.get("image")), []).append(row)
    labels_by_crop = index_by(label_rows, "crop_id")
    index_rows: List[Dict[str, Any]] = []
    saved = 0
    missing_images = 0

    for image_name, manifests in sorted(manifests_by_image.items()):
        image_path = Path(images_dir) / image_name
        img = read_image(image_path)
        if img is None:
            missing_images += 1
            continue
        out = ensure_bgr(img)
        palm_record = palm_by_image.get(image_name, {})
        if cfg.get("review", {}).get("draw_palm_bbox", True):
            for det in palm_record.get("detections", []) + palm_record.get("negative_candidates", []):
                _draw_palm(out, det)
        for manifest in manifests:
            if cfg.get("review", {}).get("draw_hand_roi", True):
                _draw_roi(out, manifest)
            label = labels_by_crop.get(str(manifest.get("crop_id")), {})
            if cfg.get("review", {}).get("draw_landmarks", True):
                _draw_landmarks(out, label)
            present = bool((label.get("hand_presence") or {}).get("present", False))
            handed = label.get("handedness") or {}
            text = f"{manifest.get('crop_id')} present={int(present)} {handed.get('label', 'unknown')} src={label.get('source', '')}"
            rect = manifest.get("roi_rect") or {}
            x = int(round(float(rect.get("x_center", 8))))
            y = int(round(float(rect.get("y_center", 24))))
            _draw_text(out, text, (max(8, min(x, out.shape[1] - 320)), max(18, min(y, out.shape[0] - 8))))
            index_rows.append(
                {
                    "image": image_name,
                    "crop_id": manifest.get("crop_id"),
                    "crop_path": manifest.get("crop_path"),
                    "hand_presence": present,
                    "handedness": handed.get("label", "unknown"),
                    "source": label.get("source", ""),
                    "overlay_path": relpath(output_dir / (Path(image_name).stem + "_overlay.png"), root),
                }
            )
        out_path = output_dir / (Path(image_name).stem + "_overlay.png")
        if write_image(out_path, out):
            saved += 1

    review_index_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "crop_id", "crop_path", "hand_presence", "handedness", "source", "overlay_path"])
        writer.writeheader()
        writer.writerows(index_rows)
    return {"overlay_images": saved, "index_rows": len(index_rows), "missing_images": missing_images, "review_index_csv": relpath(review_index_csv, root)}
