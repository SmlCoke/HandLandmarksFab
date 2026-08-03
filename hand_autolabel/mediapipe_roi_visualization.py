from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2

from .image_io import ensure_bgr, read_image, to_uint8_gray, write_image
from .formats import load_yaml_config, read_jsonl, resolve_path
from .progress import track_progress


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

LANDMARK_COLORS = (
    (255, 255, 255),
    *((255, 0, 255),) * 4,
    *((0, 255, 0),) * 4,
    *((0, 255, 255),) * 4,
    *((0, 128, 255),) * 4,
    *((255, 128, 0),) * 4,
)


class TrainingRoiVisualizationError(RuntimeError):
    """Raised when finalized training ROI visualization would be incomplete."""


def _draw_text(
    image,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.4,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _landmark_coordinates(
    row: Mapping[str, Any],
    field: str = "landmarks_crop_px",
) -> Dict[int, tuple[int, int]]:
    coordinates: Dict[int, tuple[int, int]] = {}
    for position, point in enumerate(row.get(field) or []):
        try:
            landmark_id = int(point.get("id", position))
            if 0 <= landmark_id < 21:
                coordinates[landmark_id] = (
                    int(round(float(point["x"]))),
                    int(round(float(point["y"]))),
                )
        except (KeyError, TypeError, ValueError):
            continue
    return coordinates


def _draw_landmarks(image, coordinates: Mapping[int, tuple[int, int]]) -> None:
    for start, end in HAND_CONNECTIONS:
        if start in coordinates and end in coordinates:
            cv2.line(
                image,
                coordinates[start],
                coordinates[end],
                LANDMARK_COLORS[end],
                2,
                cv2.LINE_AA,
            )
    for landmark_id, (x, y) in sorted(coordinates.items()):
        color = LANDMARK_COLORS[landmark_id]
        cv2.circle(image, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 3, color, -1, cv2.LINE_AA)
        _draw_text(image, str(landmark_id), (x + 4, y - 4), color, scale=0.3)


def render_mediapipe_roi_draft_overlays(
    label_rows: Iterable[Mapping[str, Any]],
    roi_images_dir: Path,
    output_dir: Path,
    *,
    show_progress: bool = False,
) -> Dict[str, int]:
    """Render draft landmarks without reading original images, Palm data, or manifests."""
    roi_images_dir = Path(roi_images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "rows": 0,
        "saved": 0,
        "positive": 0,
        "negative": 0,
        "invalid_landmark_count": 0,
        "out_of_bounds_landmarks": 0,
        "missing_crop_path": 0,
        "missing_crop_image": 0,
        "write_failures": 0,
    }
    for row in track_progress(
        label_rows,
        enabled=show_progress,
        description="Visualize ROIs",
        unit="roi",
    ):
        stats["rows"] += 1
        recorded_crop_path = row.get("crop_path")
        crop_name = Path(str(recorded_crop_path)).name if recorded_crop_path else ""
        if not crop_name:
            stats["missing_crop_path"] += 1
            continue

        # Only the basename is retained from the JSONL path. This deliberately
        # relocates moved datasets against the configured 02_roi_crops/images/.
        crop_path = roi_images_dir / crop_name
        image = read_image(crop_path)
        if image is None:
            stats["missing_crop_image"] += 1
            continue

        overlay = ensure_bgr(image)
        present = bool((row.get("hand_presence") or {}).get("present", False))
        coordinates = _landmark_coordinates(row)
        height, width = overlay.shape[:2]
        out_of_bounds = sum(
            1 for x, y in coordinates.values()
            if x < 0 or x >= width or y < 0 or y >= height
        )
        stats["out_of_bounds_landmarks"] += out_of_bounds

        if present:
            stats["positive"] += 1
            if len(coordinates) == 21:
                _draw_landmarks(overlay, coordinates)
                status_color = (0, 255, 0) if out_of_bounds == 0 else (0, 165, 255)
            else:
                stats["invalid_landmark_count"] += 1
                status_color = (0, 0, 255)
        else:
            stats["negative"] += 1
            status_color = (0, 0, 255)

        handedness = row.get("handedness") or {}
        handedness_label = str(handedness.get("label", "unknown"))
        handedness_score = handedness.get("score")
        score_text = "n/a"
        if handedness_score is not None:
            try:
                score_text = f"{float(handedness_score):.3f}"
            except (TypeError, ValueError):
                score_text = str(handedness_score)
        _draw_text(
            overlay,
            f"present={int(present)} {handedness_label}={score_text}",
            (5, 15),
            status_color,
        )
        _draw_text(
            overlay,
            f"points={len(coordinates)} oob={out_of_bounds}",
            (5, 30),
            status_color,
        )

        output_path = output_dir / crop_name
        if write_image(output_path, overlay):
            stats["saved"] += 1
        else:
            stats["write_failures"] += 1

    return stats


def render_original_image_visualizations(
    label_rows: Sequence[Mapping[str, Any]],
    source_images_dir: Path,
    output_dir: Path,
    *,
    proposal_variant: str,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Render projected draft landmarks on every flat source image."""
    source_images_dir = Path(source_images_dir)
    output_dir = Path(output_dir)
    if not source_images_dir.is_dir():
        raise TrainingRoiVisualizationError(
            f"Source images directory not found: {source_images_dir}"
        )

    source_images = sorted(
        (
            path
            for path in source_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: path.name,
    )
    if not source_images:
        raise TrainingRoiVisualizationError(
            f"No source TIFF images found: {source_images_dir}"
        )

    output_name_by_source: Dict[str, str] = {}
    source_by_output_key: Dict[str, str] = {}
    for source_image in source_images:
        output_name = f"{source_image.stem}.png"
        output_key = output_name.casefold()
        previous_source = source_by_output_key.get(output_key)
        if previous_source is not None:
            raise TrainingRoiVisualizationError(
                "Source image stems collide after PNG conversion: "
                f"{previous_source}, {source_image.name} -> {output_name}"
            )
        source_by_output_key[output_key] = source_image.name
        output_name_by_source[source_image.name] = output_name

    source_names = {path.name for path in source_images}
    expected_output_names = set(output_name_by_source.values())
    rows_by_image: Dict[str, List[Mapping[str, Any]]] = {
        name: [] for name in source_names
    }
    missing_image_references = 0
    unknown_image_references: set[str] = set()
    for row in label_rows:
        recorded_image = str(row.get("image") or "").strip()
        if not recorded_image:
            missing_image_references += 1
            continue
        image_name = Path(recorded_image).name
        if image_name not in source_names:
            unknown_image_references.add(image_name)
            continue
        rows_by_image[image_name].append(row)

    if missing_image_references or unknown_image_references:
        unknown_preview = ", ".join(sorted(unknown_image_references)[:10])
        raise TrainingRoiVisualizationError(
            "Autolabel draft cannot be matched to source images: "
            f"missing_image_references={missing_image_references} "
            f"unknown_image_references={len(unknown_image_references)}"
            + (f" ({unknown_preview})" if unknown_preview else "")
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {
        "source_images": len(source_images),
        "draft_rows": len(label_rows),
        "output_format": "png",
        "png_compression": 3,
        "saved": 0,
        "images_with_hands": 0,
        "images_without_hands": 0,
        "positive_hands": 0,
        "teacher_abstain_rois": 0,
        "invalid_landmark_count": 0,
        "out_of_bounds_landmarks": 0,
        "read_failures": 0,
        "write_failures": 0,
    }
    for source_image in track_progress(
        source_images,
        enabled=show_progress,
        description="Visualize originals",
        unit="image",
    ):
        image = read_image(source_image)
        if image is None:
            stats["read_failures"] += 1
            continue
        overlay = ensure_bgr(to_uint8_gray(image))
        height, width = overlay.shape[:2]
        positive_rows = [
            row
            for row in rows_by_image[source_image.name]
            if bool((row.get("hand_presence") or {}).get("present", False))
        ]
        stats["teacher_abstain_rois"] += (
            len(rows_by_image[source_image.name]) - len(positive_rows)
        )

        rendered_hands = 0
        for row in positive_rows:
            coordinates = _landmark_coordinates(row, "landmarks_image_px")
            if len(coordinates) != 21:
                stats["invalid_landmark_count"] += 1
                continue
            out_of_bounds = sum(
                1
                for x, y in coordinates.values()
                if x < 0 or x >= width or y < 0 or y >= height
            )
            stats["out_of_bounds_landmarks"] += out_of_bounds
            _draw_landmarks(overlay, coordinates)
            rendered_hands += 1

            handedness = row.get("handedness") or {}
            label = str(handedness.get("label", "unknown"))
            score = handedness.get("score")
            try:
                score_text = f"{float(score):.3f}" if score is not None else "n/a"
            except (TypeError, ValueError):
                score_text = str(score)
            wrist_x, wrist_y = coordinates[0]
            text_y = max(16, min(height - 5, wrist_y - 8))
            _draw_text(
                overlay,
                f"hand={rendered_hands} {label}={score_text}",
                (max(5, min(width - 175, wrist_x + 8)), text_y),
                (0, 255, 0) if out_of_bounds == 0 else (0, 165, 255),
            )

        stats["positive_hands"] += rendered_hands
        if rendered_hands:
            stats["images_with_hands"] += 1
            status_color = (0, 255, 0)
        else:
            stats["images_without_hands"] += 1
            status_color = (0, 0, 255)
        _draw_text(
            overlay,
            f"variant={proposal_variant} hands={rendered_hands}",
            (5, 16),
            status_color,
            scale=0.45,
        )

        if write_image(
            output_dir / output_name_by_source[source_image.name],
            overlay,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        ):
            stats["saved"] += 1
        else:
            stats["write_failures"] += 1

    stale_removed = 0
    for path in output_dir.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in {".png", ".tif", ".tiff"}
            and path.name not in expected_output_names
        ):
            path.unlink()
            stale_removed += 1
    stats["stale_removed"] = stale_removed
    stats["output_dir"] = str(output_dir.resolve())

    if stats["saved"] != len(source_images):
        raise TrainingRoiVisualizationError(
            "Original-image visualization is incomplete: "
            f"saved={stats['saved']} expected={len(source_images)}"
        )
    return stats


def evenly_spaced_sample(
    rows: Sequence[Mapping[str, Any]],
    max_samples: int,
) -> List[Mapping[str, Any]]:
    """Select a deterministic sample spread uniformly across ordered ROI rows."""
    if max_samples < 1:
        raise ValueError("visualization.train_max_samples must be >= 1")
    count = len(rows)
    if count <= max_samples:
        return list(rows)
    if max_samples == 1:
        return [rows[count // 2]]
    indices = [
        round(index * (count - 1) / (max_samples - 1))
        for index in range(max_samples)
    ]
    return [rows[index] for index in indices]


def render_autolabel_roi_visualizations(
    label_rows: Sequence[Mapping[str, Any]],
    roi_images_dir: Path,
    output_dir: Path,
    *,
    split: str,
    train_max_samples: int,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Render sampled Train overlays or every Val/Test overlay."""
    if split == "train":
        selected = evenly_spaced_sample(label_rows, train_max_samples)
        selection = "evenly_spaced"
    elif split in {"val", "test"}:
        selected = list(label_rows)
        selection = "all"
    else:
        raise ValueError(f"unsupported visualization split: {split}")

    stats = render_mediapipe_roi_draft_overlays(
        selected,
        roi_images_dir,
        output_dir,
        show_progress=show_progress,
    )
    if stats["saved"] != len(selected):
        raise TrainingRoiVisualizationError(
            "Autolabel ROI visualization is incomplete: "
            f"saved={stats['saved']} expected={len(selected)}"
        )

    expected_names = {
        Path(str(row.get("crop_path", ""))).name
        for row in selected
        if row.get("crop_path")
    }
    stale_removed = 0
    for path in Path(output_dir).glob("*.png"):
        if path.name not in expected_names:
            path.unlink()
            stale_removed += 1

    return {
        **stats,
        "selection": selection,
        "available_rows": len(label_rows),
        "selected_rows": len(selected),
        "train_max_samples": int(train_max_samples) if split == "train" else None,
        "stale_removed": stale_removed,
        "output_dir": str(Path(output_dir).resolve()),
    }


def _safe_directory_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "source"


def render_finalized_training_overlays(
    config_path: Path,
    stage: str,
) -> Dict[str, Any]:
    """Render every canonical included training ROI, grouped by configured source."""
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    root = config_path.parents[1]
    if stage not in {"pretrain", "finetune"}:
        raise TrainingRoiVisualizationError(f"Unsupported training stage: {stage}")

    try:
        output_cfg = cfg["outputs"][stage]
        labels_dir = resolve_path(root, output_cfg["labels_dir"])
        qc_dir = resolve_path(root, output_cfg["qc_dir"])
    except KeyError as exc:
        raise TrainingRoiVisualizationError(
            f"Missing outputs.{stage}.{exc.args[0]} in {config_path}"
        ) from exc
    if labels_dir.parent.resolve() != qc_dir.parent.resolve():
        raise TrainingRoiVisualizationError(
            "Finalized training labels_dir and qc_dir must have the same parent "
            f"for visualization: labels_dir={labels_dir}, qc_dir={qc_dir}"
        )

    included_path = labels_dir / f"hand_training_labels_{stage}.jsonl"
    if not included_path.is_file():
        raise TrainingRoiVisualizationError(
            f"Finalized included-label JSONL not found: {included_path}"
        )
    included_rows = read_jsonl(included_path)

    source_dirs: Dict[str, Path] = {}
    output_names: Dict[str, str] = {}
    preflight_errors = []
    for source in cfg.get("sources") or []:
        dataset_id = str(source.get("dataset_id", "")).strip()
        if not dataset_id:
            preflight_errors.append("source_missing_dataset_id")
            continue
        if dataset_id in source_dirs:
            preflight_errors.append(f"duplicate_dataset_id:{dataset_id}")
            continue
        source_root = resolve_path(root, source.get("root", "."))
        crop_images_value = source.get("crop_images_dir")
        if not crop_images_value:
            preflight_errors.append(f"source_missing_crop_images_dir:{dataset_id}")
            continue
        crop_images_dir = resolve_path(source_root, crop_images_value)
        if not crop_images_dir.is_dir():
            preflight_errors.append(
                f"source_crop_images_dir_missing:{dataset_id}:{crop_images_dir}"
            )
        source_dirs[dataset_id] = crop_images_dir
        safe_name = _safe_directory_name(dataset_id)
        if safe_name in output_names.values():
            preflight_errors.append(f"source_output_name_collision:{dataset_id}:{safe_name}")
        output_names[dataset_id] = safe_name

    if not source_dirs:
        preflight_errors.append("no_training_sources_configured")

    rows_by_source: Dict[str, list[Mapping[str, Any]]] = {
        dataset_id: [] for dataset_id in source_dirs
    }
    for row in included_rows:
        dataset_id = str(row.get("dataset_id", ""))
        if dataset_id not in source_dirs:
            preflight_errors.append(
                f"included_row_source_not_configured:{dataset_id}:{row.get('global_crop_id')}"
            )
            continue
        if row.get("selection_action") != "include":
            preflight_errors.append(
                f"canonical_row_not_included:{dataset_id}:{row.get('global_crop_id')}"
            )
        crop_name = Path(str(row.get("crop_path", ""))).name
        if not crop_name:
            preflight_errors.append(
                f"included_row_missing_crop_path:{dataset_id}:{row.get('global_crop_id')}"
            )
            continue
        crop_path = source_dirs[dataset_id] / crop_name
        if not crop_path.is_file():
            preflight_errors.append(
                f"included_crop_image_missing:{dataset_id}:{crop_path}"
            )
        rows_by_source[dataset_id].append(row)

    if preflight_errors:
        preview = "; ".join(preflight_errors[:20])
        remainder = len(preflight_errors) - min(len(preflight_errors), 20)
        suffix = f"; ... and {remainder} more" if remainder else ""
        raise TrainingRoiVisualizationError(
            f"Finalized training ROI visualization preflight failed with "
            f"{len(preflight_errors)} error(s): {preview}{suffix}"
        )

    visualization_dir = labels_dir.parent / "hand_landmarks_roi_visualization"
    source_stats: Dict[str, Dict[str, int]] = {}
    total_saved = 0
    for dataset_id, source_dir in source_dirs.items():
        source_output_dir = visualization_dir / output_names[dataset_id]
        stats = render_mediapipe_roi_draft_overlays(
            rows_by_source[dataset_id],
            source_dir,
            source_output_dir,
        )
        if stats["saved"] != len(rows_by_source[dataset_id]):
            raise TrainingRoiVisualizationError(
                f"Failed to render all included ROIs for {dataset_id}: "
                f"saved={stats['saved']} expected={len(rows_by_source[dataset_id])}"
            )
        source_stats[dataset_id] = stats
        total_saved += stats["saved"]

    return {
        "stage": stage,
        "included_jsonl": str(included_path),
        "output_dir": str(visualization_dir),
        "sources": source_stats,
        "rows": len(included_rows),
        "saved": total_saved,
    }
