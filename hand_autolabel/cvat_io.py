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


def _cvat_handedness_labels(cfg: Mapping[str, Any]) -> Dict[str, str]:
    cvat = cfg.get("cvat", {})
    return {
        "Left": str(cvat.get("left_label_name", "Left")),
        "Right": str(cvat.get("right_label_name", "Right")),
        "unknown": str(cvat.get("unknown_handedness_label_name", "unknown_handedness")),
    }


def _cvat_ignore_label(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("cvat", {}).get("ignore_for_training_label_name", "ignore_for_training"))


def _normalize_handedness_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label == "left":
        return "Left"
    if label == "right":
        return "Right"
    return "unknown"


def _cvat_handedness_tag_for_label(label: Any, handedness_labels: Mapping[str, str]) -> Optional[str]:
    normalized = _normalize_handedness_label(label)
    if normalized in {"Left", "Right", "unknown"}:
        return handedness_labels[normalized]
    return None


def _parse_cvat_handedness_tags(
    image_el: ET.Element, handedness_labels: Mapping[str, str]
) -> tuple[str, bool, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    tag_to_label = {tag_name: label for label, tag_name in handedness_labels.items()}
    present = []
    for tag_el in image_el.findall("tag"):
        tag_name = str(tag_el.attrib.get("label", ""))
        if tag_name in tag_to_label:
            present.append(tag_to_label[tag_name])
    unique = sorted(set(present))
    if len(present) > len(unique):
        warnings.append("duplicate_handedness_tag")
    if len(unique) > 1:
        errors.append("conflicting_handedness_tags")
        return "unknown", True, warnings, errors
    if not unique:
        return "unknown", False, warnings, errors
    return unique[0], True, warnings, errors


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


def _ordered_export_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[List[Mapping[str, Any]], Dict[str, Mapping[str, Any]], int]:
    """Bind labels one-to-one and order frames like a lexicographical CVAT upload."""
    manifest_by_crop: Dict[str, Mapping[str, Any]] = {}
    crop_names: set[str] = set()
    for row in manifest_rows:
        crop_id = str(row.get("crop_id") or "")
        crop_name = Path(str(row.get("crop_path") or "")).name
        if not crop_id or not crop_name:
            raise ValueError("CVAT export manifest row is missing crop_id or crop_path")
        if crop_id in manifest_by_crop:
            raise ValueError(f"CVAT export has duplicate manifest crop_id: {crop_id}")
        if crop_name in crop_names:
            raise ValueError(f"CVAT export has duplicate crop filename: {crop_name}")
        manifest_by_crop[crop_id] = row
        crop_names.add(crop_name)

    label_by_crop: Dict[str, Mapping[str, Any]] = {}
    for row in label_rows:
        crop_id = str(row.get("crop_id") or "")
        if not crop_id:
            raise ValueError("CVAT export label row is missing crop_id")
        if crop_id in label_by_crop:
            raise ValueError(f"CVAT export has duplicate label crop_id: {crop_id}")
        label_by_crop[crop_id] = row

    manifest_ids = set(manifest_by_crop)
    label_ids = set(label_by_crop)
    missing = sorted(manifest_ids - label_ids)
    unexpected = sorted(label_ids - manifest_ids)
    if missing or unexpected:
        raise ValueError(
            "CVAT export requires one label per manifest ROI: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    ordered = sorted(
        manifest_rows,
        key=lambda row: Path(str(row["crop_path"])).name,
    )
    input_names = [Path(str(row["crop_path"])).name for row in manifest_rows]
    ordered_names = [Path(str(row["crop_path"])).name for row in ordered]
    reordered = sum(left != right for left, right in zip(input_names, ordered_names))
    return ordered, label_by_crop, reordered


def _ordered_landmarks_for_cvat(
    label: Mapping[str, Any],
    crop_id: str,
) -> List[Mapping[str, Any]]:
    by_id: Dict[int, Mapping[str, Any]] = {}
    for position, point in enumerate(label.get("landmarks_crop_px") or []):
        try:
            landmark_id = int(point.get("id", position))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CVAT export has invalid landmark id for {crop_id}") from exc
        if landmark_id in by_id:
            raise ValueError(
                f"CVAT export has duplicate landmark id {landmark_id} for {crop_id}"
            )
        by_id[landmark_id] = point
    expected = set(range(21))
    if set(by_id) != expected:
        raise ValueError(
            f"CVAT export positive ROI must contain landmark ids 0..20: {crop_id}"
        )
    return [by_id[landmark_id] for landmark_id in range(21)]


def export_cvat_xml(
    manifest_rows: List[Mapping[str, Any]],
    label_rows: List[Mapping[str, Any]],
    root: Path,
    xml_path: Path,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    ordered_manifest, label_by_crop, reordered_images = _ordered_export_rows(
        manifest_rows,
        label_rows,
    )
    label_name = str(cfg["cvat"].get("label_name", "hand_landmarks"))
    no_hand_label = str(cfg["cvat"].get("no_hand_label_name", "no_hand"))
    ignore_label = _cvat_ignore_label(cfg)
    point_labels = _cvat_skeleton_point_labels(cfg)
    handedness_labels = _cvat_handedness_labels(cfg)
    review_cfg = cfg.get("review", {})
    strip_teacher_handedness = bool(review_cfg.get("strip_teacher_handedness", False))

    annotations = ET.Element("annotations")
    ET.SubElement(annotations, "version").text = str(cfg["cvat"].get("xml_version", "1.1"))
    meta = ET.SubElement(annotations, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "id").text = "0"
    ET.SubElement(task, "name").text = "hand_landmarker_autolabel"
    ET.SubElement(task, "size").text = str(len(ordered_manifest))
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
    _add_cvat_label(labels, handedness_labels["Left"], "#9467bd", "tag")
    _add_cvat_label(labels, handedness_labels["Right"], "#ff7f0e", "tag")
    if "unknown_handedness_label_name" in cfg.get("cvat", {}) or strip_teacher_handedness:
        _add_cvat_label(labels, handedness_labels["unknown"], "#17becf", "tag")
    _add_cvat_label(labels, ignore_label, "#7f7f7f", "tag")

    width = int(cfg["hand_roi"]["output_width"])
    height = int(cfg["hand_roi"]["output_height"])

    positives = 0
    negatives = 0
    handedness_tags = {"Left": 0, "Right": 0, "unknown": 0}
    for idx, manifest in enumerate(ordered_manifest):
        crop_name = Path(str(manifest["crop_path"])).name
        image_el = ET.SubElement(annotations, "image", id=str(idx), name=crop_name, width=str(width), height=str(height))
        crop_id = str(manifest["crop_id"])
        label = label_by_crop[crop_id]
        if bool((label.get("hand_presence") or {}).get("present", False)):
            landmarks = _ordered_landmarks_for_cvat(label, crop_id)
            skeleton_el = ET.SubElement(image_el, "skeleton", label=label_name, source="auto", z_order="0")
            for point_label, point in zip(point_labels, landmarks):
                ET.SubElement(
                    skeleton_el,
                    "points",
                    label=point_label,
                    source="auto",
                    occluded="0",
                    outside="0",
                    points=_point_to_cvat(point),
                )
            handedness_label = _normalize_handedness_label((label.get("handedness") or {}).get("label"))
            handedness_tag = _cvat_handedness_tag_for_label(handedness_label, handedness_labels)
            if not strip_teacher_handedness and handedness_tag is not None:
                ET.SubElement(image_el, "tag", label=handedness_tag, source="auto")
                handedness_tags[handedness_label] += 1
            else:
                handedness_tags["unknown"] += 1
            positives += 1
        else:
            ET.SubElement(image_el, "tag", label=no_hand_label, source="auto")
            negatives += 1

    ET.indent(annotations, space="  ")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(annotations).write(xml_path, encoding="utf-8", xml_declaration=True)
    return {
        "images": len(ordered_manifest),
        "copied_images": 0,
        "copy_policy": "disabled_use_roi_crops_images_directly",
        "image_order": "crop_filename_lexicographic",
        "reordered_from_manifest_input": reordered_images,
        "upload_images_dir": str(cfg["paths"].get("roi_crops_dir", "data/02_roi_crops")).rstrip("/") + "/images",
        "positive_shape_type": "skeleton",
        "skeleton_point_labels": point_labels,
        "handedness_tag_labels": handedness_labels,
        "teacher_handedness_stripped": strip_teacher_handedness,
        "ignore_for_training_label": ignore_label,
        "handedness_tags": handedness_tags,
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
    ignore_label = _cvat_ignore_label(cfg)
    point_labels = _cvat_skeleton_point_labels(cfg)
    handedness_labels = _cvat_handedness_labels(cfg)
    review_cfg = cfg.get("review", {})
    strict_presence = bool(review_cfg.get("require_explicit_presence_decision", False))
    strict_handedness = bool(review_cfg.get("require_explicit_handedness_decision", False))
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

    def append_reviewed_row(base: Dict[str, Any], row_warnings: List[str], row_errors: List[str]) -> None:
        present = bool((base.get("hand_presence") or {}).get("present", False))
        if row_errors:
            status = "import_conflict"
        elif bool(base.get("ignore_for_training", False)):
            status = "reviewed_ignored"
        elif present:
            status = "reviewed_positive"
        else:
            status = "reviewed_negative"
        base["cvat_image_seen"] = True
        base["cvat_review_status"] = status
        base["cvat_import_warnings"] = sorted(set(row_warnings))
        base["cvat_import_errors"] = sorted(set(row_errors))
        rows.append(base)
        if row_warnings:
            warnings.append({"crop_id": base.get("crop_id"), "warnings": sorted(set(row_warnings))})
        if row_errors:
            errors.append({"crop_id": base.get("crop_id"), "errors": sorted(set(row_errors))})

    for image_el in root_el.findall("image"):
        image_name = Path(str(image_el.attrib.get("name", ""))).name
        seen_names.add(image_name)
        manifest = manifest_by_name.get(image_name)
        if manifest is None:
            errors.append({"image": image_name, "errors": ["image_not_in_manifest"]})
            continue
        draft = draft_by_crop.get(str(manifest["crop_id"]), {})
        no_hand = any(tag.attrib.get("label") == no_hand_label for tag in image_el.findall("tag"))
        ignore_for_training = any(tag.attrib.get("label") == ignore_label for tag in image_el.findall("tag"))
        handedness_label, handedness_explicit, handedness_warnings, handedness_errors = _parse_cvat_handedness_tags(
            image_el, handedness_labels
        )
        skeleton_shapes = [s for s in image_el.findall("skeleton") if s.attrib.get("label") == label_name]
        legacy_point_shapes = [p for p in image_el.findall("points") if p.attrib.get("label") == label_name]
        row_warnings: List[str] = []
        row_errors: List[str] = []
        row_warnings.extend(handedness_warnings)
        row_errors.extend(handedness_errors)
        if legacy_point_shapes:
            row_errors.append(f"legacy_points_shape_not_supported:{len(legacy_point_shapes)}")
        if len(skeleton_shapes) > 1:
            (row_errors if strict_presence else row_warnings).append(
                f"multiple_hand_skeletons:{len(skeleton_shapes)}"
            )
        if no_hand and skeleton_shapes:
            row_errors.append("conflicting_no_hand_and_skeleton")
        if no_hand and handedness_explicit and not (strict_presence and ignore_for_training):
            row_errors.append("conflicting_no_hand_and_handedness_tag")
        if not skeleton_shapes and handedness_explicit and not (strict_presence and ignore_for_training):
            row_errors.append("handedness_tag_without_skeleton")
        if strict_presence and not ignore_for_training and (bool(no_hand) == bool(skeleton_shapes)):
            row_errors.append("missing_or_conflicting_explicit_presence_decision")
        if strict_handedness and not ignore_for_training and skeleton_shapes and not handedness_explicit:
            row_errors.append("missing_explicit_handedness_decision")

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
                    "ignore_for_training": bool(ignore_for_training),
                }
            )
            if strict_presence or strict_handedness:
                base["finetune_review"] = {
                    "presence_decision": "no_hand",
                    "handedness_decision": "unknown",
                }
            append_reviewed_row(base, row_warnings, row_errors)
            continue

        pts, skeleton_warnings, skeleton_errors = _parse_cvat_skeleton(skeleton_shapes[0], point_labels)
        row_warnings.extend(skeleton_warnings)
        row_errors.extend(skeleton_errors)
        if len(pts) != 21:
            row_errors.append(f"point_count_not_21:{len(pts)}")
        if len(pts) == 21 and handedness_label == "unknown" and not handedness_explicit:
            row_warnings.append("missing_handedness_tag")
        crop_px = [{"id": idx, "x": float(x), "y": float(y)} for idx, (x, y) in enumerate(pts)]
        for p in crop_px:
            if p["x"] < 0.0 or p["y"] < 0.0 or p["x"] > width - 1 or p["y"] > height - 1:
                row_warnings.append(f"point_out_of_bounds:{p['id']}")
                break
        crop_norm = [
            {"id": p["id"], "x": p["x"] / float(max(1, width - 1)), "y": p["y"] / float(max(1, height - 1))}
            for p in crop_px
        ]
        image_pts = project_px_points_to_image([(p["x"], p["y"]) for p in crop_px], manifest["roi_corners_px"], width, height)
        image_px = [{"id": idx, "x": x, "y": y} for idx, (x, y) in enumerate(image_pts)]
        present = len(pts) == 21
        base.update(
            {
                "hand_id": (base.get("hand_id") or make_hand_id(str(manifest["crop_id"]))) if present else None,
                "hand_presence": {"present": present},
                "handedness": {"label": handedness_label if present else "unknown", "score": None},
                "landmarks_crop_norm": crop_norm if present else [],
                "landmarks_crop_px": crop_px if present else [],
                "landmarks_image_px": image_px if present else [],
                "source": "cvat_reviewed",
                "needs_review": bool(row_warnings or row_errors),
                "ignore_for_training": bool(ignore_for_training),
            }
        )
        if strict_presence or strict_handedness:
            base["finetune_review"] = {
                "presence_decision": "hand" if present else "no_hand",
                "handedness_decision": handedness_label if present else "unknown",
            }
        append_reviewed_row(base, row_warnings, row_errors)

    for manifest in manifest_rows:
        crop_name = Path(str(manifest["crop_path"])).name
        if crop_name in seen_names:
            continue
        draft = draft_by_crop.get(str(manifest["crop_id"]), {})
        base = merge_label_with_manifest(dict(draft), manifest, cfg)
        base["needs_review"] = True
        base["source"] = "cvat_reviewed_missing_image"
        base["cvat_image_seen"] = False
        base["cvat_review_status"] = "missing_from_xml"
        base["cvat_import_warnings"] = ["missing_from_cvat_xml"]
        base["cvat_import_errors"] = []
        rows.append(base)
        warnings.append({"crop_id": base.get("crop_id"), "warnings": ["missing_from_cvat_xml"]})

    stats = {
        "rows": len(rows),
        "warnings": warnings,
        "errors": errors,
        "reviewed_xml": str(xml_path),
        "coverage": {
            "manifest_images": len(manifest_rows),
            "xml_images": len(seen_names),
            "reviewed_rows": len(rows),
            "seen_manifest_images": sum(bool(row.get("cvat_image_seen")) for row in rows),
            "missing_from_xml": sum(row.get("cvat_review_status") == "missing_from_xml" for row in rows),
        },
    }
    return rows, stats
