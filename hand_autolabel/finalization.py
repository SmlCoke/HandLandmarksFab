from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .formats import load_yaml_config, merge_label_with_manifest, read_jsonl, resolve_path
from .projection import project_px_points_to_image


MANIFEST_FIELDS = ("image", "crop_path", "palm_det_id", "palm_valid", "palm_score", "roi_rect", "roi_corners_px")
LANDMARK_FIELDS = ("landmarks_crop_norm", "landmarks_crop_px", "landmarks_image_px")


class FinalizationError(RuntimeError):
    """Raised when a finalizer refuses to publish an invalid dataset."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in materialized:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return len(materialized)


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_index(rows: Sequence[Mapping[str, Any]], key: str, scope: str) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None or str(value) == "":
            continue
        grouped[str(value)].append(row)
    duplicates = [{"scope": scope, "key": key, "value": value, "count": len(items)} for value, items in grouped.items() if len(items) > 1]
    return {value: dict(items[0]) for value, items in grouped.items() if len(items) == 1}, duplicates


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same(a: Any, b: Any, tolerance: float = 1e-5) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _finite(a) and _finite(b) and abs(float(a) - float(b)) <= tolerance
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same(a[k], b[k], tolerance) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tolerance) for x, y in zip(a, b))
    return a == b


def manifest_conflicts(label: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    conflicts = []
    for field in MANIFEST_FIELDS:
        if field in label and label.get(field) is not None and not _same(label.get(field), manifest.get(field)):
            conflicts.append(f"manifest_conflict:{field}")
    return conflicts


def validate_manifest_row(row: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for field in ("crop_id",) + MANIFEST_FIELDS:
        if row.get(field) is None:
            errors.append(f"manifest_missing:{field}")
    if not _finite(row.get("palm_score")):
        errors.append("manifest_palm_score_non_finite")
    corners = row.get("roi_corners_px") or []
    if len(corners) != 4 or any(not isinstance(p, (list, tuple)) or len(p) != 2 or not _finite(p[0]) or not _finite(p[1]) for p in corners):
        errors.append("manifest_roi_corners_invalid")
    rect = row.get("roi_rect") or {}
    for key in ("x_center", "y_center", "width", "height", "rotation_rad"):
        if not _finite(rect.get(key)):
            errors.append(f"manifest_roi_rect_invalid:{key}")
    if _finite(rect.get("width")) and float(rect["width"]) <= 0:
        errors.append("manifest_roi_width_not_positive")
    if _finite(rect.get("height")) and float(rect["height"]) <= 0:
        errors.append("manifest_roi_height_not_positive")
    return sorted(set(errors))


def _point_errors(points: Any, name: str, expected: int) -> List[str]:
    errors: List[str] = []
    if not isinstance(points, list):
        return [f"{name}_not_list"]
    if len(points) != expected:
        errors.append(f"{name}_count:{len(points)}!={expected}")
        return errors
    ids = []
    for point in points:
        if not isinstance(point, Mapping):
            errors.append(f"{name}_point_not_object")
            continue
        try:
            ids.append(int(point.get("id")))
        except (TypeError, ValueError):
            errors.append(f"{name}_invalid_id")
        if not _finite(point.get("x")) or not _finite(point.get("y")):
            errors.append(f"{name}_non_finite")
    if sorted(ids) != list(range(expected)):
        errors.append(f"{name}_ids_not_0_to_20")
    return sorted(set(errors))


def validate_label_schema(
    row: Mapping[str, Any], cfg: Mapping[str, Any], *, gold: bool, check_image: bool = True, source_root: Path | None = None
) -> tuple[List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    width = int(row.get("width", cfg["hand_roi"]["output_width"]))
    height = int(row.get("height", cfg["hand_roi"]["output_height"]))
    present = bool((row.get("hand_presence") or {}).get("present", False))
    if present:
        for name in LANDMARK_FIELDS:
            errors.extend(_point_errors(row.get(name), name, 21))
        handedness = str((row.get("handedness") or {}).get("label", "unknown")).lower()
        if gold and handedness not in {"left", "right"}:
            errors.append("positive_handedness_not_left_or_right")
        norm = row.get("landmarks_crop_norm") or []
        crop = row.get("landmarks_crop_px") or []
        image = row.get("landmarks_image_px") or []
        if len(norm) == len(crop) == len(image) == 21:
            for idx, (n, p) in enumerate(zip(norm, crop)):
                if abs(float(n["x"]) * (width - 1) - float(p["x"])) > 0.08 or abs(float(n["y"]) * (height - 1) - float(p["y"])) > 0.08:
                    errors.append(f"norm_crop_inconsistent:{idx}")
                    break
            corners = row.get("roi_corners_px") or []
            if len(corners) == 4:
                projected = project_px_points_to_image([(float(p["x"]), float(p["y"])) for p in crop], corners, width, height)
                for idx, ((x, y), target) in enumerate(zip(projected, image)):
                    if abs(x - float(target["x"])) > 0.15 or abs(y - float(target["y"])) > 0.15:
                        errors.append(f"crop_image_projection_inconsistent:{idx}")
                        break
            else:
                errors.append("roi_corners_not_4")
            outside = sum(not (0.0 <= float(n["x"]) <= 1.0 and 0.0 <= float(n["y"]) <= 1.0 and 0.0 <= float(p["x"]) <= width - 1 and 0.0 <= float(p["y"]) <= height - 1) for n, p in zip(norm, crop))
            if outside:
                (errors if gold else warnings).append(f"landmarks_out_of_crop:{outside}")
    else:
        for name in LANDMARK_FIELDS:
            if row.get(name):
                errors.append(f"negative_{name}_not_empty")
        handedness = row.get("handedness") or {}
        if str(handedness.get("label", "unknown")).lower() not in {"unknown", "", "none"}:
            errors.append("negative_handedness_not_unknown")
        if handedness.get("score") is not None:
            errors.append("negative_handedness_score_not_null")
        if row.get("hand_id") is not None:
            errors.append("negative_hand_id_not_null")
    if check_image:
        crop_path = row.get("crop_path")
        path = resolve_path(source_root or Path.cwd(), str(crop_path)) if crop_path else None
        if path is None or not path.is_file():
            errors.append("crop_image_missing")
        else:
            try:
                from .image_io import read_image
                image = read_image(path)
                if image is None:
                    errors.append("crop_image_unreadable")
                elif int(image.shape[1]) != width or int(image.shape[0]) != height:
                    errors.append(f"crop_image_size:{image.shape[1]}x{image.shape[0]}!={width}x{height}")
            except Exception as exc:
                errors.append(f"crop_image_check_failed:{type(exc).__name__}")
    return sorted(set(warnings)), sorted(set(errors))


def _sample_type(row: Mapping[str, Any]) -> str:
    present = bool((row.get("hand_presence") or {}).get("present", False))
    runtime = bool(row.get("palm_valid", False))
    if present:
        return "POS_RUNTIME" if runtime else "POS_LOW_PALM"
    return "NEG_RUNTIME_CANDIDATE" if runtime else "NEG_LOW_PALM_CANDIDATE"


def _training_weights(row: Dict[str, Any], stage_cfg: Mapping[str, Any]) -> None:
    present = bool((row.get("hand_presence") or {}).get("present", False))
    handedness = str((row.get("handedness") or {}).get("label", "unknown")).lower()
    row["hand_presence_loss_weight"] = 1.0
    row["landmark_loss_weight"] = 1.0 if present else 0.0
    row["handedness_loss_weight"] = 1.0 if present and handedness in {"left", "right"} else 0.0
    tier = row["supervision_tier"]
    row["supervision_loss_weight"] = float(stage_cfg.get("supervision_loss_weights", {}).get(tier, 1.0 if tier == "gold" else 0.7))
    quality_defaults = {"HIGH": 1.0, "MEDIUM": 0.7, "PRESENCE_ONLY": 0.4, "AMBIGUOUS": 0.0, "INVALID": 0.0}
    quality_weight = float(stage_cfg.get("quality_loss_weights", {}).get(row["quality_tier"], quality_defaults[row["quality_tier"]]))
    row["presence_quality_weight"] = quality_weight
    row["landmark_quality_weight"] = quality_weight if present else 0.0
    row["handedness_quality_weight"] = quality_weight if present and handedness in {"left", "right"} else 0.0
    sampling_defaults = {"POS_RUNTIME": 0.70, "POS_LOW_PALM": 0.70, "NEG_RUNTIME_CANDIDATE": 0.25, "NEG_LOW_PALM_CANDIDATE": 0.05}
    row["sampling_weight"] = float(stage_cfg.get("sampling_weights", {}).get(row["sample_type"], sampling_defaults[row["sample_type"]]))
    row["sampling_bucket"] = f"{tier}:{row['sample_type']}"


def _positive_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    pa = a.get("landmarks_image_px") or []
    pb = b.get("landmarks_image_px") or []
    if len(pa) != 21 or len(pb) != 21:
        return float("inf")
    def diag(points: Sequence[Mapping[str, Any]]) -> float:
        xs = [float(p["x"]) for p in points]
        ys = [float(p["y"]) for p in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    scale = max(1.0, (diag(pa) + diag(pb)) / 2.0)
    distances = sorted(math.hypot(float(x["x"]) - float(y["x"]), float(x["y"]) - float(y["y"])) for x, y in zip(pa, pb))
    return distances[len(distances) // 2] / scale


def _cluster_positives(rows: List[Dict[str, Any]], threshold: float) -> List[List[Dict[str, Any]]]:
    clusters: List[List[Dict[str, Any]]] = []
    for row in sorted(rows, key=lambda r: r["global_crop_id"]):
        placed = False
        for cluster in clusters:
            if all(_positive_distance(row, member) <= threshold for member in cluster):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])
    return clusters


def _polygon_area(poly: Sequence[Sequence[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    return abs(sum(float(poly[i][0]) * float(poly[(i + 1) % len(poly)][1]) - float(poly[(i + 1) % len(poly)][0]) * float(poly[i][1]) for i in range(len(poly)))) / 2.0


def _inside(p: Sequence[float], a: Sequence[float], b: Sequence[float], orientation: float) -> bool:
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return cross * orientation >= -1e-7


def _intersection(s: Sequence[float], e: Sequence[float], a: Sequence[float], b: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = *map(float, s), *map(float, e)
    x3, y3, x4, y4 = *map(float, a), *map(float, b)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return [x2, y2]
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)]


def polygon_iou(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    subject = [list(map(float, p)) for p in first]
    clip = [list(map(float, p)) for p in second]
    if len(subject) < 3 or len(clip) < 3:
        return 0.0
    signed = sum(clip[i][0] * clip[(i + 1) % len(clip)][1] - clip[(i + 1) % len(clip)][0] * clip[i][1] for i in range(len(clip)))
    orientation = 1.0 if signed >= 0 else -1.0
    output = subject
    for a, b in zip(clip, clip[1:] + clip[:1]):
        input_poly, output = output, []
        if not input_poly:
            break
        s = input_poly[-1]
        for e in input_poly:
            if _inside(e, a, b, orientation):
                if not _inside(s, a, b, orientation):
                    output.append(_intersection(s, e, a, b))
                output.append(e)
            elif _inside(s, a, b, orientation):
                output.append(_intersection(s, e, a, b))
            s = e
    inter = _polygon_area(output)
    union = _polygon_area(subject) + _polygon_area(clip) - inter
    return inter / union if union > 0 else 0.0


def _train_quality(row: Mapping[str, Any], provenance: str, cfg: Mapping[str, Any]) -> tuple[str, List[str]]:
    if provenance == "human_gold":
        return "HIGH", []
    flags: List[str] = []
    handedness = row.get("handedness") or {}
    present = bool((row.get("hand_presence") or {}).get("present", False))
    if row.get("needs_review"):
        flags.append("teacher_needs_review")
    if int(row.get("mediapipe_num_hands_detected", 0)) > 1:
        flags.append("teacher_multiple_hands")
    score = handedness.get("score")
    if present and score is not None and float(score) < float(cfg.get("quality", {}).get("handedness_review_threshold", 0.7)):
        flags.append("low_handedness_score")
    palm_score = row.get("palm_score")
    if not present and bool(row.get("palm_valid")) and palm_score is not None:
        flags.append("runtime_hard_negative")
    if not present and palm_score is not None and float(palm_score) >= float(cfg.get("quality", {}).get("high_palm_score_review_threshold", 0.8)):
        flags.append("negative_high_palm_score")
    if any(f in flags for f in ("teacher_needs_review", "teacher_multiple_hands", "negative_high_palm_score")):
        return "AMBIGUOUS", flags
    if flags:
        return "MEDIUM", flags
    return "HIGH", flags


def _dimension_pair(
    row: Mapping[str, Any], first: str, second: str, *, scope: str, crop_id: Any
) -> tuple[int, int] | None:
    first_value = row.get(first)
    second_value = row.get(second)
    if first_value is None and second_value is None:
        return None
    if first_value is None or second_value is None:
        raise FinalizationError(
            f"{scope}: crop_id={crop_id!r} must provide both {first} and {second}"
        )
    try:
        first_number = float(first_value)
        second_number = float(second_value)
        pair = (int(first_number), int(second_number))
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalizationError(
            f"{scope}: crop_id={crop_id!r} has invalid {first}/{second}: "
            f"{first_value!r}/{second_value!r}"
        ) from exc
    if (
        not math.isfinite(first_number)
        or not math.isfinite(second_number)
        or first_number != pair[0]
        or second_number != pair[1]
        or pair[0] <= 0
        or pair[1] <= 0
    ):
        raise FinalizationError(
            f"{scope}: crop_id={crop_id!r} has invalid {first}/{second}: "
            f"{first_value!r}/{second_value!r}"
        )
    return pair


def _infer_source_config(
    scope: str,
    manifests: Sequence[Mapping[str, Any]],
    label_groups: Sequence[Sequence[Mapping[str, Any]]],
    quality: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the small config view needed by finalization from published artifacts."""
    roi_from_labels: set[tuple[int, int]] = set()
    source_image_sizes: set[tuple[int, int]] = set()
    for rows in label_groups:
        for row in rows:
            crop_id = row.get("crop_id")
            roi_size = _dimension_pair(row, "width", "height", scope=scope, crop_id=crop_id)
            if roi_size is not None:
                roi_from_labels.add(roi_size)
            image_size = _dimension_pair(
                row,
                "source_image_width",
                "source_image_height",
                scope=scope,
                crop_id=crop_id,
            )
            if image_size is not None:
                source_image_sizes.add(image_size)

    roi_from_manifests: set[tuple[int, int]] = set()
    for row in manifests:
        output_size = row.get("output_size")
        if output_size is None:
            continue
        if not isinstance(output_size, (list, tuple)) or len(output_size) != 2:
            raise FinalizationError(
                f"{scope}: crop_id={row.get('crop_id')!r} has invalid manifest output_size: {output_size!r}"
            )
        manifest_size = _dimension_pair(
            {"width": output_size[0], "height": output_size[1]},
            "width",
            "height",
            scope=scope,
            crop_id=row.get("crop_id"),
        )
        if manifest_size is not None:
            roi_from_manifests.add(manifest_size)

    if len(roi_from_labels) > 1:
        raise FinalizationError(f"{scope}: inconsistent label width/height values: {sorted(roi_from_labels)}")
    if len(roi_from_manifests) > 1:
        raise FinalizationError(f"{scope}: inconsistent manifest output_size values: {sorted(roi_from_manifests)}")
    if roi_from_labels and roi_from_manifests and roi_from_labels != roi_from_manifests:
        raise FinalizationError(
            f"{scope}: label width/height {sorted(roi_from_labels)} conflicts with "
            f"manifest output_size {sorted(roi_from_manifests)}"
        )
    if len(source_image_sizes) > 1:
        raise FinalizationError(
            f"{scope}: inconsistent source_image_width/source_image_height values: {sorted(source_image_sizes)}"
        )
    roi_sizes = roi_from_labels or roi_from_manifests
    if not roi_sizes:
        raise FinalizationError(
            f"{scope}: cannot infer ROI dimensions; labels need width/height or manifests need output_size"
        )
    if not source_image_sizes:
        raise FinalizationError(
            f"{scope}: cannot infer source image dimensions; label JSONL needs "
            "source_image_width/source_image_height"
        )

    roi_width, roi_height = next(iter(roi_sizes))
    image_width, image_height = next(iter(source_image_sizes))
    return {
        "hand_roi": {"output_width": roi_width, "output_height": roi_height},
        "image": {"width": image_width, "height": image_height},
        "quality": dict(quality or {}),
    }


def finalize_training(config_path: Path, stage: str) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    root = config_path.parents[1]
    if stage not in {"pretrain", "finetune"}:
        raise ValueError("stage must be pretrain or finetune")
    stage_cfg = cfg.get("stages", {}).get(stage, {})
    try:
        output_cfg = cfg["outputs"][stage]
        labels_dir = resolve_path(root, output_cfg["labels_dir"])
        qc_dir = resolve_path(root, output_cfg["qc_dir"])
    except KeyError as exc:
        raise FinalizationError(f"Missing required outputs.{stage}.{exc.args[0]} in {config_path}") from exc
    check_images = bool(cfg.get("validation", {}).get("check_crop_images", True))
    fatal: List[Dict[str, Any]] = []
    catalog: List[Dict[str, Any]] = []
    global_ids: set[str] = set()
    source_stats: Dict[str, Any] = {}
    for source in cfg.get("sources", []):
        dataset_id = str(source["dataset_id"])
        source_root = resolve_path(root, source.get("root", "."))
        crop_images_dir = resolve_path(source_root, source["crop_images_dir"]) if source.get("crop_images_dir") else None
        if crop_images_dir is not None and not crop_images_dir.is_dir():
            fatal.append({"scope": dataset_id, "error": "crop_images_dir_missing", "path": str(crop_images_dir)})
        manifest_path = resolve_path(source_root, source["manifest"])
        pseudo_path = resolve_path(source_root, source["pseudo_labels"])
        missing_inputs = [
            (name, path)
            for name, path in (("manifest", manifest_path), ("pseudo_labels", pseudo_path))
            if not path.is_file()
        ]
        if missing_inputs:
            details = ", ".join(f"{name}={path}" for name, path in missing_inputs)
            raise FinalizationError(f"{dataset_id}: required input file missing: {details}")
        manifests = read_jsonl(manifest_path)
        pseudo_rows = read_jsonl(pseudo_path)
        manifest_idx, duplicates = unique_index(manifests, "crop_id", f"{dataset_id}:manifest")
        pseudo_idx, pseudo_dups = unique_index(pseudo_rows, "crop_id", f"{dataset_id}:pseudo")
        fatal.extend(duplicates + pseudo_dups)
        basename_counts = Counter(Path(str(r.get("crop_path", ""))).name for r in manifests)
        fatal.extend({"scope": dataset_id, "error": "duplicate_crop_basename", "value": k, "count": v} for k, v in basename_counts.items() if k and v > 1)
        missing_pseudo = sorted(set(manifest_idx) - set(pseudo_idx))
        orphan_pseudo = sorted(set(pseudo_idx) - set(manifest_idx))
        if missing_pseudo:
            fatal.append({"scope": dataset_id, "error": "manifest_without_pseudo", "crop_ids": missing_pseudo})
        if orphan_pseudo:
            fatal.append({"scope": dataset_id, "error": "pseudo_orphan", "crop_ids": orphan_pseudo})
        gold_idx: Dict[str, Dict[str, Any]] = {}
        gold_manifest_idx: Dict[str, Dict[str, Any]] = {}
        if source.get("gold_labels"):
            gold_rows = read_jsonl(resolve_path(source_root, source["gold_labels"]))
            gold_idx, gold_dups = unique_index(gold_rows, "crop_id", f"{dataset_id}:gold")
            fatal.extend(gold_dups)
            orphan_gold = sorted(set(gold_idx) - set(manifest_idx))
            if orphan_gold:
                fatal.append({"scope": dataset_id, "error": "gold_orphan", "crop_ids": orphan_gold})
            if source.get("gold_manifest"):
                gold_manifest_rows = read_jsonl(resolve_path(source_root, source["gold_manifest"]))
                gold_manifest_idx, gold_manifest_dups = unique_index(gold_manifest_rows, "crop_id", f"{dataset_id}:gold_manifest")
                fatal.extend(gold_manifest_dups)
                if set(gold_manifest_idx) != set(gold_idx):
                    fatal.append({"scope": dataset_id, "error": "gold_manifest_label_coverage_mismatch", "manifest_only": sorted(set(gold_manifest_idx) - set(gold_idx)), "label_only": sorted(set(gold_idx) - set(gold_manifest_idx))})
                for gold_id, gold_manifest in gold_manifest_idx.items():
                    full_manifest = manifest_idx.get(gold_id)
                    if full_manifest is None:
                        fatal.append({"scope": dataset_id, "error": "gold_manifest_orphan", "crop_id": gold_id})
                    else:
                        conflicts = manifest_conflicts(gold_manifest, full_manifest)
                        if conflicts:
                            fatal.append({"scope": dataset_id, "error": "gold_manifest_conflict", "crop_id": gold_id, "details": conflicts})
            if source.get("gold_import_report"):
                report_path = resolve_path(source_root, source["gold_import_report"])
                if not report_path.is_file():
                    fatal.append({"scope": dataset_id, "error": "gold_import_report_missing", "path": str(report_path)})
                else:
                    with report_path.open("r", encoding="utf-8") as f:
                        gold_report = json.load(f)
                    report_errors = gold_report.get("import_integrity", gold_report).get("errors", [])
                    ignored_gold = {key for key, value in gold_idx.items() if bool(value.get("ignore_for_training"))}
                    blocking = [item for item in report_errors if not (isinstance(item, Mapping) and str(item.get("crop_id")) in ignored_gold)]
                    if blocking:
                        fatal.append({"scope": dataset_id, "error": "gold_import_report_errors", "details": blocking})
        source_cfg = _infer_source_config(
            dataset_id,
            manifests,
            [pseudo_rows, list(gold_idx.values())],
            source.get("quality", cfg.get("quality", {})),
        )
        source_rows: List[Dict[str, Any]] = []
        for local_id, manifest in manifest_idx.items():
            global_id = f"{dataset_id}:{local_id}"
            if global_id in global_ids:
                fatal.append({"scope": dataset_id, "error": "duplicate_global_crop_id", "value": global_id})
            global_ids.add(global_id)
            pseudo = pseudo_idx.get(local_id, {})
            gold = gold_idx.get(local_id)
            effective = gold if gold is not None else pseudo
            provenance = "human_gold" if gold is not None else "mediapipe_pseudo"
            manifest_errors = validate_manifest_row(manifest)
            if manifest_errors:
                fatal.append({"scope": dataset_id, "crop_id": local_id, "errors": manifest_errors, "source": "manifest"})
            conflicts = manifest_conflicts(effective, manifest)
            if conflicts:
                fatal.append({"scope": dataset_id, "crop_id": local_id, "errors": conflicts, "source": provenance})
            row = merge_label_with_manifest(effective, manifest, source_cfg)
            if crop_images_dir is not None:
                row["source_crop_path"] = manifest.get("crop_path")
                row["crop_path"] = str(crop_images_dir / Path(str(manifest.get("crop_path", ""))).name)
            row.update({
                "schema_version": str(cfg.get("schema_version", "train_finalize_v1")),
                "dataset_id": dataset_id,
                "source_crop_id": local_id,
                "global_crop_id": global_id,
                "crop_id": global_id,
                "source_group_id": f"{dataset_id}:{manifest.get('image')}",
                "annotation_provenance": provenance,
                "supervision_tier": "gold" if gold is not None else "pseudo",
                "training_stage": stage,
            })
            if gold is not None:
                pseudo_probe = merge_label_with_manifest(pseudo, manifest, source_cfg)
                pseudo_warnings, pseudo_errors = validate_label_schema(pseudo_probe, source_cfg, gold=False, check_image=False, source_root=source_root)
                pseudo_conflicts = manifest_conflicts(pseudo, manifest)
                if pseudo_warnings or pseudo_errors or pseudo_conflicts:
                    row["overridden_pseudo_issues"] = sorted(set(pseudo_warnings + pseudo_errors + pseudo_conflicts))
                if gold.get("cvat_image_seen") is False or gold.get("cvat_review_status") == "missing_from_xml":
                    fatal.append({"scope": dataset_id, "crop_id": local_id, "errors": ["gold_not_seen_in_cvat"]})
                if gold.get("cvat_import_errors") and not bool(gold.get("ignore_for_training")):
                    fatal.append({"scope": dataset_id, "crop_id": local_id, "errors": gold.get("cvat_import_errors"), "source": "human_gold"})
            if row.get("ignore_for_training"):
                row.update({"sample_type": _sample_type(row), "quality_tier": "INVALID", "quality_flags": ["ignore_for_training"], "selection_action": "drop_ignore"})
                source_rows.append(row)
                continue
            warnings, errors = validate_label_schema(row, source_cfg, gold=gold is not None, check_image=check_images, source_root=source_root)
            quality, flags = _train_quality(row, provenance, source_cfg)
            row["quality_flags"] = sorted(set(flags + warnings))
            row["sample_type"] = _sample_type(row)
            if errors:
                row.update({"quality_tier": "INVALID", "selection_action": "drop_invalid", "structural_errors": errors})
                if gold is not None:
                    fatal.append({"scope": dataset_id, "crop_id": local_id, "errors": errors, "source": "human_gold"})
            else:
                row.update({"quality_tier": quality, "selection_action": "include" if quality != "AMBIGUOUS" else "hold"})
            _training_weights(row, stage_cfg)
            source_rows.append(row)
        catalog.extend(source_rows)
        source_stats[dataset_id] = {
            "manifest": len(manifests), "pseudo": len(pseudo_rows), "gold": len(gold_idx),
            "crop_images_dir": str(crop_images_dir) if crop_images_dir is not None else None,
        }
    if not cfg.get("sources"):
        fatal.append({"error": "no_sources_configured"})
    if stage == "finetune" and not any(row.get("annotation_provenance") == "human_gold" for row in catalog):
        fatal.append({"error": "finetune_requires_at_least_one_human_gold_row"})

    dedup = cfg.get("dedup", {})
    pos_threshold = float(dedup.get("positive_normalized_distance", 0.18))
    max_pos = int(dedup.get("max_positive_representatives_per_cluster", 1))
    neg_iou = float(dedup.get("negative_roi_iou", 0.85))
    max_neg = int(dedup.get("max_negatives_per_type_per_image", 1))
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in catalog:
        if row.get("selection_action") in {"include", "hold"}:
            by_group[row["source_group_id"]].append(row)
    cluster_serial = 0
    for group_rows in by_group.values():
        positives = [r for r in group_rows if bool((r.get("hand_presence") or {}).get("present", False)) and r["quality_tier"] != "INVALID"]
        for cluster in _cluster_positives(positives, pos_threshold):
            cluster_serial += 1
            cluster_id = f"pos-{cluster_serial:08d}"
            ranked = sorted(cluster, key=lambda r: (r["annotation_provenance"] != "human_gold", bool(r["quality_flags"]), not bool(r.get("palm_valid")), -float(r.get("palm_score") or 0.0), r["global_crop_id"]))
            for rank, row in enumerate(ranked, 1):
                row.update({"duplicate_cluster_id": cluster_id, "duplicate_cluster_size": len(cluster), "duplicate_rank": rank})
                if rank > max_pos and row["selection_action"] == "include":
                    row["selection_action"] = "drop_duplicate"
                    row["quality_flags"] = sorted(set(row["quality_flags"] + ["duplicate_positive_roi"]))
        negatives = [r for r in group_rows if not bool((r.get("hand_presence") or {}).get("present", False)) and r["selection_action"] == "include"]
        for negative in negatives:
            if any(p["annotation_provenance"] == "human_gold" and polygon_iou(negative.get("roi_corners_px") or [], p.get("roi_corners_px") or []) >= float(dedup.get("negative_positive_overlap_iou", 0.7)) for p in positives):
                negative["selection_action"] = "hold"
                negative["quality_flags"] = sorted(set(negative["quality_flags"] + ["negative_overlaps_gold_positive"]))
        for sample_type in ("NEG_RUNTIME_CANDIDATE", "NEG_LOW_PALM_CANDIDATE"):
            candidates = sorted([r for r in negatives if r["sample_type"] == sample_type and r["selection_action"] == "include"], key=lambda r: (-float(r.get("palm_score") or 0.0), r["global_crop_id"]))
            kept: List[Dict[str, Any]] = []
            for row in candidates:
                duplicate = any(polygon_iou(row.get("roi_corners_px") or [], old.get("roi_corners_px") or []) >= neg_iou for old in kept)
                if duplicate or len(kept) >= max_neg:
                    row["selection_action"] = "drop_duplicate"
                    row["quality_flags"] = sorted(set(row["quality_flags"] + ["duplicate_or_capped_negative_roi"]))
                else:
                    kept.append(row)

    catalog.sort(key=lambda r: r["global_crop_id"])
    included = [r for r in catalog if r.get("selection_action") == "include"]
    excluded = [r for r in catalog if r.get("selection_action") != "include"]
    for dataset_id in source_stats:
        source_rows = [r for r in catalog if r.get("dataset_id") == dataset_id]
        source_stats[dataset_id].update({
            "included": sum(r.get("selection_action") == "include" for r in source_rows),
            "excluded": sum(r.get("selection_action") != "include" for r in source_rows),
            "sample_types": dict(Counter(r.get("sample_type") for r in source_rows)),
        })
    paths = {
        "catalog": labels_dir / f"hand_train_catalog_{stage}.jsonl",
        "included": labels_dir / f"hand_training_labels_{stage}.jsonl",
        "excluded": labels_dir / f"hand_training_excluded_{stage}.jsonl",
        "report": qc_dir / f"finalize_train_{stage}_report.json",
    }
    report: Dict[str, Any] = {
        "schema_version": cfg.get("schema_version", "train_finalize_v1"), "stage": stage,
        "status": "failed" if fatal else "ok", "fatal_errors": fatal, "sources": source_stats,
        "counts": {
            "catalog": len(catalog), "included": len(included), "excluded": len(excluded),
            "positive": sum(bool((r.get("hand_presence") or {}).get("present")) for r in catalog),
            "negative": sum(not bool((r.get("hand_presence") or {}).get("present")) for r in catalog),
            "sample_types": dict(Counter(r.get("sample_type") for r in catalog)),
            "quality_tiers": dict(Counter(r.get("quality_tier") for r in catalog)),
            "provenance": dict(Counter(r.get("annotation_provenance") for r in catalog)),
            "actions": dict(Counter(r.get("selection_action") for r in catalog)),
            "quality_flags": dict(Counter(flag for r in catalog for flag in (r.get("quality_flags") or []))),
            "sampling_buckets_included": dict(Counter(r.get("sampling_bucket") for r in included)),
            "duplicate_cluster_sizes": dict(Counter(str(r.get("duplicate_cluster_size")) for r in catalog if r.get("duplicate_cluster_size") is not None)),
        },
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    if fatal:
        atomic_write_json(paths["report"], report)
        raise FinalizationError(f"07A refused to publish: {len(fatal)} fatal error(s); see {paths['report']}")
    atomic_write_jsonl(paths["catalog"], catalog)
    atomic_write_jsonl(paths["included"], included)
    atomic_write_jsonl(paths["excluded"], excluded)
    report["sha256"] = {k: sha256_file(v) for k, v in paths.items() if k != "report"}
    atomic_write_json(paths["report"], report)
    return report


def _load_review_context(path: Path) -> tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]]]:
    if not path.exists():
        return {}, []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    index, duplicates = unique_index(rows, "crop_id", "review_context")
    return {k: {str(a): str(b or "") for a, b in v.items()} for k, v in index.items()}, duplicates


def finalize_evaluation(config_path: Path, split: str) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    root = config_path.parents[1]
    configured_split = str(cfg.get("dataset", {}).get("split", split))
    if split not in {"val", "test"} or configured_split != split:
        raise FinalizationError(f"split mismatch: CLI={split}, config={configured_split}")
    evaluation_dataset_id = str(cfg.get("dataset", {}).get("id", f"{split}_unversioned"))
    evaluation_cfg = cfg.get("evaluation", {})
    outputs_cfg = cfg.get("outputs", {})
    labels_dir = resolve_path(root, outputs_cfg["labels_dir"])
    qc_dir = resolve_path(root, outputs_cfg["qc_dir"])
    sources = list(cfg.get("sources") or [])
    fatal: List[Dict[str, Any]] = []
    if not sources:
        fatal.append({"error": "no_evaluation_sources_configured"})

    combined: List[Dict[str, Any]] = []
    source_reports: Dict[str, Any] = {}
    input_records: List[Dict[str, Any]] = []
    physical_crop_paths: Dict[str, str] = {}
    seen_source_ids: set[str] = set()
    seen_dataset_ids: set[str] = set()

    for source in sources:
        source_id = str(source["source_id"])
        if source_id in seen_source_ids:
            fatal.append({"source_id": source_id, "error": "duplicate_evaluation_source_id"})
        seen_source_ids.add(source_id)
        source_dataset_id = str(source["dataset_id"])
        if source_dataset_id in seen_dataset_ids:
            fatal.append({"source_id": source_id, "dataset_id": source_dataset_id, "error": "duplicate_evaluation_dataset_id"})
        seen_dataset_ids.add(source_dataset_id)
        partition = str(source.get("partition", source_id))
        owner = str(source.get("owner", "unknown"))
        source_root = resolve_path(root, source["root"])
        crop_images_dir = resolve_path(source_root, source.get("crop_images_dir", "02_roi_crops/images"))
        source_split = str(source.get("split", split))
        if source_split != split:
            fatal.append({"source_id": source_id, "error": "source_split_mismatch", "configured": source_split, "expected": split})
        manifest_path = resolve_path(source_root, source.get("manifest", "02_roi_crops/hand_roi_crops_manifest.jsonl"))
        reviewed_path = resolve_path(source_root, source.get("reviewed", "03_reviewed/hand_landmarks_reviewed.jsonl"))
        import_report_path = resolve_path(source_root, source.get("import_report", "qc/cvat_import_stats.json"))
        context_path = resolve_path(source_root, source.get("review_context", "03_reviewed/review_context.csv"))
        required_files_missing = False
        if not manifest_path.is_file():
            fatal.append({"source_id": source_id, "error": "manifest_file_missing", "path": str(manifest_path)})
            required_files_missing = True
        if not reviewed_path.is_file():
            fatal.append({"source_id": source_id, "error": "reviewed_file_missing", "path": str(reviewed_path)})
            required_files_missing = True
        if not crop_images_dir.is_dir():
            fatal.append({"source_id": source_id, "error": "crop_images_dir_missing", "path": str(crop_images_dir)})
        if required_files_missing:
            continue
        manifests, reviewed = read_jsonl(manifest_path), read_jsonl(reviewed_path)
        source_cfg = _infer_source_config(source_id, manifests, [reviewed])
        manifest_idx, manifest_dups = unique_index(manifests, "crop_id", f"{source_id}:manifest")
        reviewed_idx, reviewed_dups = unique_index(reviewed, "crop_id", f"{source_id}:reviewed")
        fatal.extend(manifest_dups + reviewed_dups)
        basename_counts = Counter(Path(str(row.get("crop_path", ""))).name for row in manifests)
        fatal.extend({"source_id": source_id, "error": "duplicate_crop_basename_within_source", "basename": name, "count": count} for name, count in basename_counts.items() if name and count > 1)
        missing = sorted(set(manifest_idx) - set(reviewed_idx))
        orphans = sorted(set(reviewed_idx) - set(manifest_idx))
        if missing:
            fatal.append({"source_id": source_id, "error": "manifest_without_reviewed", "crop_ids": missing})
        if orphans:
            fatal.append({"source_id": source_id, "error": "reviewed_orphan", "crop_ids": orphans})
        for crop_id, manifest in manifest_idx.items():
            basename = Path(str(manifest.get("crop_path", ""))).name
            physical_path = str((crop_images_dir / basename).resolve())
            previous_source = physical_crop_paths.get(physical_path)
            if previous_source is not None and previous_source != source_id:
                fatal.append({"error": "cross_source_same_physical_crop_path", "path": physical_path, "sources": [previous_source, source_id]})
            else:
                physical_crop_paths[physical_path] = source_id

        if not import_report_path.is_file():
            fatal.append({"source_id": source_id, "error": "missing_cvat_import_report", "path": str(import_report_path)})
            import_report: Dict[str, Any] = {}
        else:
            with import_report_path.open("r", encoding="utf-8") as f:
                import_report = json.load(f)
        integrity = import_report.get("import_integrity", import_report)
        report_errors = []
        for item in integrity.get("errors", []):
            crop_id = str(item.get("crop_id")) if isinstance(item, Mapping) and item.get("crop_id") is not None else None
            reviewed_row = reviewed_idx.get(crop_id) if crop_id else None
            if reviewed_row is not None and bool(reviewed_row.get("ignore_for_training")):
                continue
            report_errors.append(item)
        if report_errors:
            fatal.append({"source_id": source_id, "error": "cvat_import_report_errors", "details": report_errors})
        context, context_dups = _load_review_context(context_path)
        fatal.extend({**item, "source_id": source_id} for item in context_dups)
        context_orphans = sorted(set(context) - set(manifest_idx))
        if context_orphans:
            fatal.append({"source_id": source_id, "error": "review_context_orphan", "crop_ids": context_orphans})
        for crop_id, ctx in context.items():
            expected = manifest_idx.get(crop_id, {}).get("palm_det_id")
            if ctx.get("palm_det_id") and str(ctx["palm_det_id"]) != str(expected):
                fatal.append({"source_id": source_id, "error": "review_context_palm_det_id_mismatch", "crop_id": crop_id})
        combined.extend({
            "source_id": source_id,
            "dataset_id": source_dataset_id,
            "partition": partition,
            "owner": owner,
            "crop_images_dir": crop_images_dir,
            "source_cfg": source_cfg,
            "manifest": manifest_idx[crop_id],
            "reviewed": reviewed_idx[crop_id],
            "context": context.get(crop_id),
            "context_required": context_path.exists(),
        } for crop_id in sorted(set(manifest_idx) & set(reviewed_idx)))
        source_reports[source_id] = {
            "dataset_id": source_dataset_id, "partition": partition, "owner": owner, "crop_images_dir": str(crop_images_dir),
            "manifest": len(manifests),
            "reviewed": len(reviewed),
            "missing": len(missing),
            "orphans": len(orphans),
        }
        input_records.append({
            "source_id": source_id,
            "dataset_id": source_dataset_id,
            "partition": partition,
            "owner": owner,
            "root": str(source_root),
            "crop_images_dir": str(crop_images_dir),
            "autolabel_config": (
                str(resolve_path(root, source["autolabel_config"]))
                if source.get("autolabel_config")
                else None
            ),
            "manifest": str(manifest_path),
            "reviewed": str(reviewed_path),
            "cvat_import_report": str(import_report_path),
            "review_context": str(context_path) if context_path.exists() else None,
        })

    included: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    check_images = bool(evaluation_cfg.get("check_crop_images", True))
    require_palm = bool(evaluation_cfg.get("require_palm_valid", True))
    for item in sorted(combined, key=lambda value: (value["dataset_id"], str(value["manifest"].get("crop_id")))):
        manifest, raw = item["manifest"], item["reviewed"]
        source_cfg = item["source_cfg"]
        source_crop_id = str(manifest.get("crop_id"))
        global_crop_id = f"{item['dataset_id']}:{source_crop_id}"
        source_palm_det_id = str(manifest.get("palm_det_id"))
        global_palm_det_id = f"{item['dataset_id']}:{source_palm_det_id}"
        source_image = str(manifest.get("image"))
        row_errors = validate_manifest_row(manifest) + manifest_conflicts(raw, manifest)
        seen = raw.get("cvat_image_seen")
        status = raw.get("cvat_review_status")
        if seen is not True or status == "missing_from_xml" or raw.get("source") == "cvat_reviewed_missing_image":
            row_errors.append("not_reviewed_in_cvat_xml")
        row = merge_label_with_manifest(raw, manifest, source_cfg)
        row["source_crop_path"] = manifest.get("crop_path")
        row["crop_path"] = str(item["crop_images_dir"] / Path(str(manifest.get("crop_path", ""))).name)
        source_hand_id = row.get("hand_id")
        present = bool((row.get("hand_presence") or {}).get("present", False))
        row.update({
            "schema_version": "evaluation_gold_v1", "dataset_id": item["dataset_id"],
            "evaluation_dataset_id": evaluation_dataset_id, "split": split,
            "source_crop_id": source_crop_id, "global_crop_id": global_crop_id, "crop_id": global_crop_id,
            "source_palm_det_id": source_palm_det_id, "global_palm_det_id": global_palm_det_id, "palm_det_id": global_palm_det_id,
            "source_hand_id": source_hand_id, "hand_id": f"{global_crop_id}:hand" if present else None,
            "source_image": source_image, "global_image_id": f"{item['dataset_id']}:{source_image}",
            "source_group_id": f"{item['dataset_id']}:{source_image}",
            "evaluation_source_id": item["source_id"], "evaluation_partition": item["partition"],
            "evaluation_owner": item["owner"],
            "annotation_provenance": "human_gold", "supervision_tier": "gold",
            "ground_truth_valid": not bool(row.get("ignore_for_training")),
        })
        if row.get("ignore_for_training"):
            ctx = item["context"]
            if item["context_required"] and (ctx is None or not (ctx.get("reason") or ctx.get("context"))):
                row_errors.append("ignored_missing_review_context")
            if ctx and ctx.get("palm_det_id") and ctx["palm_det_id"] != str(manifest.get("palm_det_id")):
                row_errors.append("review_context_palm_det_id_mismatch")
            row["review_context"] = ctx
            row["ground_truth_valid"] = False
            ignored.append(row)
        else:
            row_errors.extend(raw.get("cvat_import_errors") or [])
            if status not in {"reviewed_positive", "reviewed_negative"}:
                row_errors.append(f"invalid_cvat_review_status:{status}")
            if require_palm and not bool(manifest.get("palm_valid")):
                row_errors.append("evaluation_requires_palm_valid")
            _, schema_errors = validate_label_schema(row, source_cfg, gold=True, check_image=check_images, source_root=root)
            row_errors.extend(schema_errors)
            row["hand_presence_loss_weight"] = 1.0
            row["landmark_loss_weight"] = 1.0 if present else 0.0
            row["handedness_loss_weight"] = 1.0 if present else 0.0
            included.append(row)
        if row_errors:
            invalid_rows.append({"source_id": item["source_id"], "source_crop_id": source_crop_id, "global_crop_id": global_crop_id, "errors": sorted(set(row_errors))})
    if invalid_rows:
        fatal.append({"error": "invalid_reviewed_rows", "rows": invalid_rows})
    for source_id, source_report in source_reports.items():
        source_report["included"] = sum(row.get("evaluation_source_id") == source_id for row in included)
        source_report["ignored"] = sum(row.get("evaluation_source_id") == source_id for row in ignored)
    minimum = int(evaluation_cfg.get("minimum_included", 1))
    if len(included) < minimum:
        fatal.append({"error": "included_below_minimum", "included": len(included), "minimum": minimum})
    prefix = "validation" if split == "val" else "test"
    main_path = labels_dir / f"hand_{prefix}_labels.jsonl"
    ignored_path = labels_dir / f"hand_{split}_ignored.jsonl"
    report_path = qc_dir / f"finalize_{split}_report.json"
    report: Dict[str, Any] = {
        "schema_version": "evaluation_gold_v1", "evaluation_dataset_id": evaluation_dataset_id, "split": split,
        "status": "failed" if fatal else "ok", "fatal_errors": fatal,
        "coverage": {"manifest": sum(v["manifest"] for v in source_reports.values()), "reviewed": sum(v["reviewed"] for v in source_reports.values()), "included": len(included), "ignored": len(ignored), "missing": sum(v["missing"] for v in source_reports.values()), "orphans": sum(v["orphans"] for v in source_reports.values())},
        "sources": source_reports,
        "label_counts": {
            "positive": sum(bool((r.get("hand_presence") or {}).get("present")) for r in included),
            "negative": sum(not bool((r.get("hand_presence") or {}).get("present")) for r in included),
            "left": sum(str((r.get("handedness") or {}).get("label", "")).lower() == "left" for r in included),
            "right": sum(str((r.get("handedness") or {}).get("label", "")).lower() == "right" for r in included),
            "cvat_review_status": dict(Counter(item["reviewed"].get("cvat_review_status") for item in combined)),
            "ignored_reasons": dict(Counter(str((r.get("review_context") or {}).get("reason") or (r.get("review_context") or {}).get("context") or "unspecified") for r in ignored)),
            "partitions_included": dict(Counter(r.get("evaluation_partition") for r in included)),
            "sources_included": dict(Counter(r.get("evaluation_source_id") for r in included)),
            "datasets_included": dict(Counter(r.get("dataset_id") for r in included)),
            "owners_included": dict(Counter(r.get("evaluation_owner") for r in included)),
            "partition_included_ratio": {
                key: count / max(1, len(included))
                for key, count in Counter(r.get("evaluation_partition") for r in included).items()
            },
        },
        "ignored_ratio": len(ignored) / max(1, len(included) + len(ignored)),
        "warnings": (["ignored_ratio_above_20_percent"] if len(ignored) / max(1, len(included) + len(ignored)) >= 0.2 else []),
        "inputs": input_records,
        "outputs": {"included": str(main_path), "ignored": str(ignored_path), "report": str(report_path)},
    }
    if fatal:
        atomic_write_json(report_path, report)
        raise FinalizationError(f"07B refused to publish: {len(fatal)} fatal error(s); see {report_path}")
    atomic_write_jsonl(main_path, included)
    atomic_write_jsonl(ignored_path, ignored)
    report["sha256"] = {"included": sha256_file(main_path), "ignored": sha256_file(ignored_path)}
    atomic_write_json(report_path, report)
    return report
