from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .formats import basename_index_by_path, index_by, merge_label_with_manifest, relpath, resolve_path
from .projection import project_px_points_to_image


def _points_to_cvat(points: Sequence[Mapping[str, Any]]) -> str:
    return ";".join(f"{float(p['x']):.3f},{float(p['y']):.3f}" for p in points)


def _parse_cvat_points(text: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    if not text:
        return points
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        x_str, y_str = part.split(",", 1)
        points.append((float(x_str), float(y_str)))
    return points


def prepare_cvat_upload_images(manifest_rows: Iterable[Mapping[str, Any]], root: Path, upload_dir: Path) -> int:
    upload_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in manifest_rows:
        src = resolve_path(root, row["crop_path"])
        dst = upload_dir / src.name
        if src.exists():
            shutil.copy2(src, dst)
            count += 1
    return count


def export_cvat_xml(
    manifest_rows: List[Mapping[str, Any]],
    label_rows: List[Mapping[str, Any]],
    root: Path,
    upload_dir: Path,
    xml_path: Path,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    label_by_crop = index_by(label_rows, "crop_id")
    label_name = str(cfg["cvat"].get("label_name", "hand_landmarks"))
    no_hand_label = str(cfg["cvat"].get("no_hand_label_name", "no_hand"))
    copied = prepare_cvat_upload_images(manifest_rows, root, upload_dir)

    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = str(cfg["cvat"].get("xml_version", "1.1"))
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = "hand_landmarker_autolabel"
    ET.SubElement(task, "size").text = str(len(manifest_rows))
    ET.SubElement(task, "mode").text = "annotation"
    ET.SubElement(task, "overlap").text = "0"
    ET.SubElement(task, "bugtracker").text = ""
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    ET.SubElement(task, "created").text = now
    ET.SubElement(task, "updated").text = now
    labels = ET.SubElement(task, "labels")
    for name in (label_name, no_hand_label):
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "color").text = "#1f77b4" if name == label_name else "#d62728"
        ET.SubElement(label, "type").text = "any"
        ET.SubElement(label, "attributes")

    width = int(cfg["hand_roi"]["output_width"])
    height = int(cfg["hand_roi"]["output_height"])
    positives = 0
    negatives = 0
    for idx, manifest in enumerate(manifest_rows):
        crop_name = Path(str(manifest["crop_path"])).name
        image_el = ET.SubElement(annotations, "image", id=str(idx), name=crop_name, width=str(width), height=str(height))
        label = label_by_crop.get(str(manifest["crop_id"]))
        if label and bool((label.get("hand_presence") or {}).get("present", False)) and len(label.get("landmarks_crop_px") or []) == 21:
            ET.SubElement(
                image_el,
                "points",
                label=label_name,
                source="auto",
                occluded="0",
                points=_points_to_cvat(label["landmarks_crop_px"]),
                z_order="0",
            )
            positives += 1
        else:
            ET.SubElement(image_el, "tag", label=no_hand_label, source="auto")
            negatives += 1

    ET.indent(annotations, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(annotations).write(xml_path, encoding="utf-8", xml_declaration=True)
    return {"images": len(manifest_rows), "copied_images": copied, "positive_shapes": positives, "negative_tags": negatives, "xml_path": relpath(xml_path, root)}


def import_cvat_xml(
    xml_path: Path,
    manifest_rows: List[Mapping[str, Any]],
    draft_rows: List[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"CVAT reviewed XML not found: {xml_path}")
    label_name = str(cfg["cvat"].get("label_name", "hand_landmarks"))
    no_hand_label = str(cfg["cvat"].get("no_hand_label_name", "no_hand"))
    manifest_by_name = basename_index_by_path(manifest_rows, "crop_path")
    draft_by_crop = index_by(draft_rows, "crop_id")
    tree = ET.parse(xml_path)
    root_el = tree.getroot()
    rows: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    seen_names = set()
    width = int(cfg["hand_roi"]["output_width"])
    height = int(cfg["hand_roi"]["output_height"])

    for image_el in root_el.findall("image"):
        image_name = Path(str(image_el.attrib.get("name", ""))).name
        seen_names.add(image_name)
        manifest = manifest_by_name.get(image_name)
        if manifest is None:
            errors.append({"image": image_name, "errors": ["image_not_in_manifest"]})
            continue
        draft = draft_by_crop.get(str(manifest["crop_id"]), {})
        no_hand = any(tag.attrib.get("label") == no_hand_label for tag in image_el.findall("tag"))
        point_shapes = [p for p in image_el.findall("points") if p.attrib.get("label") == label_name]
        row_warnings: List[str] = []
        row_errors: List[str] = []
        if len(point_shapes) > 1:
            row_warnings.append(f"multiple_hand_shapes:{len(point_shapes)}")

        base = merge_label_with_manifest(dict(draft), manifest, cfg)
        if no_hand or not point_shapes:
            base.update(
                {
                    "hand_id": None,
                    "hand_presence": {"present": False},
                    "handedness": {"label": "unknown", "score": None},
                    "landmarks_crop_norm": [],
                    "landmarks_crop_px": [],
                    "landmarks_image_px": [],
                    "source": "cvat_reviewed",
                    "needs_review": bool(row_warnings or row_errors),
                }
            )
            rows.append(base)
            if row_warnings:
                warnings.append({"crop_id": base.get("crop_id"), "warnings": row_warnings})
            continue

        try:
            pts = _parse_cvat_points(point_shapes[0].attrib.get("points", ""))
        except Exception as exc:
            pts = []
            row_errors.append(f"invalid_points:{exc}")
        if len(pts) != 21:
            row_errors.append(f"point_count_not_21:{len(pts)}")
        crop_px = [{"id": idx, "x": float(x), "y": float(y), "visible": 1} for idx, (x, y) in enumerate(pts)]
        for p in crop_px:
            if p["x"] < 0.0 or p["y"] < 0.0 or p["x"] > width - 1 or p["y"] > height - 1:
                row_warnings.append(f"point_out_of_bounds:{p['id']}")
                break
        crop_norm = [
            {"id": p["id"], "x": p["x"] / float(max(1, width - 1)), "y": p["y"] / float(max(1, height - 1)), "visible": 1}
            for p in crop_px
        ]
        image_pts = project_px_points_to_image([(p["x"], p["y"]) for p in crop_px], manifest["roi_corners_px"], width, height)
        image_px = [{"id": idx, "x": x, "y": y, "visible": 1} for idx, (x, y) in enumerate(image_pts)]
        base.update(
            {
                "hand_id": base.get("hand_id") or f"{manifest['palm_det_id']}:hand0",
                "hand_presence": {"present": len(pts) == 21},
                "handedness": base.get("handedness") or {"label": "unknown", "score": None},
                "landmarks_crop_norm": crop_norm if len(pts) == 21 else [],
                "landmarks_crop_px": crop_px if len(pts) == 21 else [],
                "landmarks_image_px": image_px if len(pts) == 21 else [],
                "source": "cvat_reviewed",
                "needs_review": bool(row_warnings or row_errors),
            }
        )
        rows.append(base)
        if row_warnings:
            warnings.append({"crop_id": base.get("crop_id"), "warnings": row_warnings})
        if row_errors:
            errors.append({"crop_id": base.get("crop_id"), "errors": row_errors})

    for manifest in manifest_rows:
        crop_name = Path(str(manifest["crop_path"])).name
        if crop_name in seen_names:
            continue
        draft = draft_by_crop.get(str(manifest["crop_id"]), {})
        base = merge_label_with_manifest(dict(draft), manifest, cfg)
        base["needs_review"] = True
        base["source"] = "cvat_reviewed_missing_image"
        rows.append(base)
        warnings.append({"crop_id": base.get("crop_id"), "warnings": ["missing_from_cvat_xml"]})

    stats = {"rows": len(rows), "warnings": warnings, "errors": errors, "reviewed_xml": str(xml_path)}
    return rows, stats
