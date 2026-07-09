from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .formats import basename_index_by_path, index_by, make_hand_id, merge_label_with_manifest, relpath
from .projection import project_px_points_to_image


DEFAULT_HAND_SKELETON_POINT_LABELS = tuple(str(i) for i in range(1, 22))
HAND_SKELETON_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
HAND_SKELETON_SVG_POSITIONS = (
    (50, 92),
    (36, 76),
    (26, 60),
    (18, 46),
    (10, 34),
    (42, 58),
    (38, 40),
    (35, 26),
    (32, 12),
    (52, 56),
    (52, 36),
    (52, 20),
    (52, 6),
    (62, 60),
    (68, 42),
    (72, 28),
    (76, 14),
    (72, 68),
    (82, 54),
    (88, 42),
    (94, 30),
)


def _cvat_skeleton_point_labels(cfg: Mapping[str, Any]) -> List[str]:
    raw = cfg.get("cvat", {}).get("skeleton_point_labels")
    labels = [str(v) for v in (raw or DEFAULT_HAND_SKELETON_POINT_LABELS)]
    if len(labels) != 21:
        raise ValueError(f"cvat.skeleton_point_labels must contain 21 labels, got {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError("cvat.skeleton_point_labels must be unique")
    return labels


def _hand_skeleton_svg(point_labels: Sequence[str]) -> str:
    parts: List[str] = []
    for start, end in HAND_SKELETON_EDGES:
        x1, y1 = HAND_SKELETON_SVG_POSITIONS[start]
        x2, y2 = HAND_SKELETON_SVG_POSITIONS[end]
        line = ET.Element(
            "line",
            {
                "x1": str(x1),
                "y1": str(y1),
                "x2": str(x2),
                "y2": str(y2),
                "stroke": "black",
                "data-type": "edge",
                "data-node-from": str(start + 1),
                "data-node-to": str(end + 1),
                "stroke-width": "0.5",
            },
        )
        parts.append(ET.tostring(line, encoding="unicode"))
    for idx, (x, y) in enumerate(HAND_SKELETON_SVG_POSITIONS):
        circle = ET.Element(
            "circle",
            {
                "r": "1.5",
                "stroke": "black",
                "fill": "#b3b3b3",
                "cx": str(x),
                "cy": str(y),
                "stroke-width": "0.1",
                "data-type": "element node",
                "data-element-id": str(idx + 1),
                "data-node-id": str(idx + 1),
                "data-label-name": point_labels[idx],
            },
        )
        parts.append(ET.tostring(circle, encoding="unicode"))
    return "".join(parts)


def _add_cvat_label(labels_el: ET.Element, name: str, color: str, shape_type: str, parent: Optional[str] = None, svg: Optional[str] = None) -> None:
    label = ET.SubElement(labels_el, "label")
    ET.SubElement(label, "name").text = name
    ET.SubElement(label, "color").text = color
    ET.SubElement(label, "type").text = shape_type
    ET.SubElement(label, "attributes")
    if svg is not None:
        ET.SubElement(label, "svg").text = svg
    if parent is not None:
        ET.SubElement(label, "parent").text = parent


def _point_to_cvat(point: Mapping[str, Any]) -> str:
    return f"{float(point['x']):.3f},{float(point['y']):.3f}"


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


def export_cvat_xml(
    manifest_rows: List[Mapping[str, Any]],
    label_rows: List[Mapping[str, Any]],
    root: Path,
    xml_path: Path,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    label_by_crop = index_by(label_rows, "crop_id")
    label_name = str(cfg["cvat"].get("label_name", "hand_landmarks"))
    no_hand_label = str(cfg["cvat"].get("no_hand_label_name", "no_hand"))
    point_labels = _cvat_skeleton_point_labels(cfg)

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
    _add_cvat_label(labels, label_name, "#1f77b4", "skeleton", svg=_hand_skeleton_svg(point_labels))
    for point_label in point_labels:
        _add_cvat_label(labels, point_label, "#2ca02c", "points", parent=label_name)
    _add_cvat_label(labels, no_hand_label, "#d62728", "tag")

    width = int(cfg["hand_roi"]["output_width"])
    height = int(cfg["hand_roi"]["output_height"])
    positives = 0
    negatives = 0
    for idx, manifest in enumerate(manifest_rows):
        crop_name = Path(str(manifest["crop_path"])).name
        image_el = ET.SubElement(annotations, "image", id=str(idx), name=crop_name, width=str(width), height=str(height))
        label = label_by_crop.get(str(manifest["crop_id"]))
        if label and bool((label.get("hand_presence") or {}).get("present", False)) and len(label.get("landmarks_crop_px") or []) == 21:
            skeleton_el = ET.SubElement(image_el, "skeleton", label=label_name, source="auto", z_order="0")
            for point_label, point in zip(point_labels, label["landmarks_crop_px"]):
                ET.SubElement(
                    skeleton_el,
                    "points",
                    label=point_label,
                    source="auto",
                    occluded="0",
                    outside="0",
                    points=_point_to_cvat(point),
                )
            positives += 1
        else:
            ET.SubElement(image_el, "tag", label=no_hand_label, source="auto")
            negatives += 1

    ET.indent(annotations, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(annotations).write(xml_path, encoding="utf-8", xml_declaration=True)
    return {
        "images": len(manifest_rows),
        "copied_images": 0,
        "copy_policy": "disabled_use_roi_crops_images_directly",
        "upload_images_dir": str(cfg["paths"].get("roi_crops_dir", "data/02_roi_crops")).rstrip("/") + "/images",
        "positive_shape_type": "skeleton",
        "skeleton_point_labels": point_labels,
        "positive_shapes": positives,
        "negative_tags": negatives,
        "xml_path": relpath(xml_path, root),
    }


def _parse_cvat_skeleton(skeleton_el: ET.Element, point_labels: Sequence[str]) -> tuple[List[Tuple[float, float]], List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    expected = set(point_labels)
    children: Dict[str, ET.Element] = {}
    for point_el in skeleton_el.findall("points"):
        label = str(point_el.attrib.get("label", ""))
        if label not in expected:
            warnings.append(f"unexpected_skeleton_point_label:{label}")
            continue
        if label in children:
            errors.append(f"duplicate_skeleton_point:{label}")
            continue
        children[label] = point_el

    points: List[Tuple[float, float]] = []
    for label in point_labels:
        point_el = children.get(label)
        if point_el is None:
            errors.append(f"missing_skeleton_point:{label}")
            continue
        if str(point_el.attrib.get("outside", "0")) == "1":
            errors.append(f"skeleton_point_outside:{label}")
            continue
        try:
            parsed = _parse_cvat_points(point_el.attrib.get("points", ""))
        except Exception as exc:
            errors.append(f"invalid_skeleton_point:{label}:{exc}")
            continue
        if len(parsed) != 1:
            errors.append(f"skeleton_point_coordinate_count_not_1:{label}:{len(parsed)}")
            continue
        points.append(parsed[0])
    return points, warnings, errors


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
    point_labels = _cvat_skeleton_point_labels(cfg)
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
        skeleton_shapes = [s for s in image_el.findall("skeleton") if s.attrib.get("label") == label_name]
        legacy_point_shapes = [p for p in image_el.findall("points") if p.attrib.get("label") == label_name]
        row_warnings: List[str] = []
        row_errors: List[str] = []
        if legacy_point_shapes:
            row_errors.append(f"legacy_points_shape_not_supported:{len(legacy_point_shapes)}")
        if len(skeleton_shapes) > 1:
            row_warnings.append(f"multiple_hand_skeletons:{len(skeleton_shapes)}")
        if no_hand and skeleton_shapes:
            row_errors.append("conflicting_no_hand_and_skeleton")

        base = merge_label_with_manifest(dict(draft), manifest, cfg)
        if no_hand or not skeleton_shapes:
            if not no_hand and not skeleton_shapes:
                row_warnings.append("missing_no_hand_tag")
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
            if row_errors:
                errors.append({"crop_id": base.get("crop_id"), "errors": row_errors})
            continue

        pts, skeleton_warnings, skeleton_errors = _parse_cvat_skeleton(skeleton_shapes[0], point_labels)
        row_warnings.extend(skeleton_warnings)
        row_errors.extend(skeleton_errors)
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
        present = len(pts) == 21
        base.update(
            {
                "hand_id": (base.get("hand_id") or make_hand_id(str(manifest["crop_id"]))) if present else None,
                "hand_presence": {"present": present},
                "handedness": base.get("handedness") or {"label": "unknown", "score": None},
                "landmarks_crop_norm": crop_norm if present else [],
                "landmarks_crop_px": crop_px if present else [],
                "landmarks_image_px": image_px if present else [],
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
