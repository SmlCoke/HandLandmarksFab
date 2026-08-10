from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.cvat_io import export_cvat_xml, import_cvat_xml
from hand_autolabel.dataset_v3 import (
    DatasetContractError,
    ROI_CONTRACT_VERSION,
    SCHEMA_VERSION,
    WarehouseRegistry,
    apply_label_provenance,
    clean_variant_visualizations,
    delete_source_variant,
    enrich_palm_rows,
    enrich_roi_rows,
    parse_capture_source_id,
    prepare_negative_review,
    prepare_selection_review,
    proposal_paths,
    publish_negative_review,
    require_safe_id,
    publish_selection_review,
    source_root,
    stable_id,
    validate_and_normalize_source,
)
from hand_autolabel.formats import load_yaml_config, read_jsonl, resolve_path, write_json, write_jsonl
from hand_autolabel.image_io import read_image, write_image
from hand_autolabel.hand_landmark_labeler import label_hand_landmark_manifest
from hand_autolabel.mediapipe_roi_visualization import (
    TrainingRoiVisualizationError,
    render_autolabel_roi_visualizations,
    render_original_image_visualizations,
)
from hand_autolabel.palm_mediapipe import run_mediapipe_palm_detector
from hand_autolabel.palm_onnx import run_onnx_palm_detector
from hand_autolabel.progress import track_progress
from hand_autolabel.quality_checks import label_issues, palm_record_issues, roi_manifest_issues, summarize_label_rows
from hand_autolabel.roi_geometry import build_roi_rect_from_palm, crop_image_by_roi
from tools.png_to_video import create_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HLMF 3.0 Hand ROI dataset warehouse")
    parser.add_argument("--autolabel-config", default="configs/autolabel.yaml")
    parser.add_argument("--review-config", default="configs/review.yaml")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    source_commands = (
        "validate-source",
        "palm",
        "build-roi",
        "mediapipe",
        "export-cvat",
        "import-cvat",
        "publish-source",
        "autolabel-train",
        "autolabel-eval",
        "autolabel-visualize-roi",
        "autolabel-visualize-original",
        "clean-autolabel-visualizations",
        "delete-source-variant",
    )
    for name in source_commands:
        command = sub.add_parser(name)
        command.add_argument("--dataset-root", required=True)
        command.add_argument("--scope", choices=("pretrain", "eval"), required=True)
        command.add_argument("--dataset-id", required=True)
        command.add_argument("--capture-source-id", required=True)
        command.add_argument("--proposal-variant", required=True)
        if name in {"mediapipe", "autolabel-train", "autolabel-eval"}:
            command.add_argument(
                "--hand-landmark-backend",
                choices=("mediapipe_tasks", "rtmpose_onnx"),
                default=None,
                help="Override hand_landmark.backend for this run.",
            )
        if name in {"autolabel-train", "autolabel-eval"}:
            command.add_argument(
                "--roi-visualization",
                choices=("true", "false"),
                default=None,
                help="Override visualization.roi_enabled for this autolabel run.",
            )
            command.add_argument(
                "--original-visualization",
                choices=("true", "false"),
                default=None,
                help="Override visualization.original_image_enabled for this autolabel run.",
            )
        if name == "autolabel-visualize-original":
            command.add_argument(
                "--original-video",
                choices=("true", "false"),
                default=None,
                help="Override visualization.original_video_enabled for this render.",
            )
        if name == "delete-source-variant":
            command.add_argument(
                "--confirm-delete",
                required=True,
                help="Must exactly match --proposal-variant.",
            )
    rebuild_manifest = sub.add_parser("rebuild-dataset-manifest")
    rebuild_manifest.add_argument("--dataset-root", required=True)
    rebuild_manifest.add_argument(
        "--scope", choices=("pretrain", "eval"), required=True
    )
    rebuild_manifest.add_argument("--dataset-id", required=True)
    prepare_negative = sub.add_parser("prepare-negative-review")
    prepare_negative.add_argument("--dataset-root", required=True)
    prepare_negative.add_argument("--negative-dataset-id", required=True)
    prepare_negative.add_argument("--candidate-labels", action="append", required=True)
    publish_negative = sub.add_parser("publish-negative-review")
    publish_negative.add_argument("--dataset-root", required=True)
    publish_negative.add_argument("--negative-dataset-id", required=True)
    prepare_selection = sub.add_parser("prepare-selection-review")
    prepare_selection.add_argument("--dataset-root", required=True)
    prepare_selection.add_argument("--selection-id", required=True)
    prepare_selection.add_argument("--request", required=True)
    publish_selection = sub.add_parser("publish-selection-review")
    publish_selection.add_argument("--dataset-root", required=True)
    publish_selection.add_argument("--selection-id", required=True)
    registry = sub.add_parser("registry-check")
    registry.add_argument("--dataset-root", required=True)
    return parser


def _merge_config(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value
    return base


def _load_public_configs(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_yaml_config(resolve_path(ROOT, args.autolabel_config))
    _merge_config(cfg, load_yaml_config(resolve_path(ROOT, args.review_config)))
    _merge_config(cfg, load_yaml_config(resolve_path(ROOT, args.datasets_config)))
    roi_visualization_override = getattr(args, "roi_visualization", None)
    if roi_visualization_override is not None:
        cfg.setdefault("visualization", {})["roi_enabled"] = (
            roi_visualization_override == "true"
        )
    original_visualization_override = getattr(args, "original_visualization", None)
    if original_visualization_override is not None:
        cfg.setdefault("visualization", {})["original_image_enabled"] = (
            original_visualization_override == "true"
        )
    original_video_override = getattr(args, "original_video", None)
    if original_video_override is not None:
        cfg.setdefault("visualization", {})["original_video_enabled"] = (
            original_video_override == "true"
        )
    hand_landmark_backend = getattr(args, "hand_landmark_backend", None)
    if hand_landmark_backend is not None:
        cfg.setdefault("hand_landmark", {})["backend"] = hand_landmark_backend
    return cfg


def _source_context(args: argparse.Namespace, cfg: Dict[str, Any]) -> tuple[Path, Dict[str, Path]]:
    dataset_root = Path(args.dataset_root).resolve()
    root = source_root(dataset_root, args.scope, args.dataset_id, args.capture_source_id)
    WarehouseRegistry(dataset_root).assert_variant_writable(
        args.capture_source_id, args.proposal_variant
    )
    paths = proposal_paths(root, args.proposal_variant)
    for path in paths.values():
        if path.name != "images":
            path.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("dataset", {})["role"] = parse_capture_source_id(args.capture_source_id)["split"]
    cfg.setdefault("paths", {}).update(
        {
            "images_dir": str(paths["images"]),
            "palm_outputs_dir": str(paths["palm"]),
            "roi_crops_dir": str(paths["roi"]),
            "reviewed_dir": str(paths["reviewed"]),
            "labels_dir": str(paths["labels"]),
            "qc_dir": str(paths["qc"]),
        }
    )
    if cfg["dataset"]["role"] in {"val", "test"}:
        cfg["palm"]["keep_low_score_candidates_for_negatives"] = False
    return root, paths


def _run_validate(
    args: argparse.Namespace,
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    return validate_and_normalize_source(
        Path(args.dataset_root),
        args.scope,
        args.dataset_id,
        args.capture_source_id,
        show_progress=show_progress,
    )


def _run_palm(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    source, paths = _source_context(args, cfg)
    raw_rows = read_jsonl(source / "raw_images.jsonl")
    if not raw_rows:
        raise DatasetContractError("validate-source must run before Palm detection")
    images = [paths["images"] / Path(str(row["relative_path"])).name for row in raw_rows]
    backend = str(cfg["palm"].get("backend", "aethersign_onnx"))
    if backend == "aethersign_onnx":
        model = resolve_path(ROOT, cfg["paths"]["palm_model_onnx"])
        rows = run_onnx_palm_detector(images, cfg, model, show_progress=show_progress)
        backend_mode = "onnx"
    elif backend == "mediapipe_official":
        rows, backend_mode = run_mediapipe_palm_detector(
            images,
            cfg,
            show_progress=show_progress,
        )
    else:
        raise DatasetContractError(f"unsupported Palm backend: {backend}")
    rows = enrich_palm_rows(raw_rows, rows, args.proposal_variant)
    output = paths["palm"] / "palm_detections.jsonl"
    write_jsonl(output, rows)
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for row in rows:
        row_warnings, row_errors = palm_record_issues(row, cfg)
        if row_warnings:
            warnings.append({"raw_image_id": row["raw_image_id"], "warnings": row_warnings})
        if row_errors:
            errors.append({"raw_image_id": row["raw_image_id"], "errors": row_errors})
    report = {
        "schema_version": SCHEMA_VERSION,
        "proposal_variant": args.proposal_variant,
        "backend": backend,
        "backend_mode": backend_mode,
        "images": len(rows),
        "detections": sum(len(row.get("detections") or []) for row in rows),
        "negative_candidates": sum(len(row.get("negative_candidates") or []) for row in rows),
        "palm_output_policy": "model_output_is_never_human_modified",
        "warnings": warnings,
        "errors": errors,
    }
    write_json(paths["qc"] / "palm_detection_report.json", report)
    if errors:
        raise DatasetContractError(f"Palm output contract failed; see {paths['qc'] / 'palm_detection_report.json'}")
    return report


def _run_build_roi(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    source, paths = _source_context(args, cfg)
    palm_rows = read_jsonl(paths["palm"] / "palm_detections.jsonl")
    if not palm_rows:
        raise DatasetContractError("Palm detections are missing")
    crop_images = paths["roi"] / "images"
    crop_images.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for parent in track_progress(
        palm_rows,
        enabled=show_progress,
        description="ROI crops",
        unit="image",
    ):
        image = read_image(paths["images"] / str(parent["image"]))
        if image is None:
            failures.append({"raw_image_id": parent["raw_image_id"], "error": "unreadable_source"})
            continue
        candidates = list(parent.get("detections") or [])
        if bool(cfg["palm"].get("keep_low_score_candidates_for_negatives", True)):
            candidates.extend(parent.get("negative_candidates") or [])
        for det in candidates:
            try:
                roi_id = stable_id(
                    "roi",
                    parent["raw_image_id"],
                    args.proposal_variant,
                    det["proposal_slot"],
                    ROI_CONTRACT_VERSION,
                )
                rect = build_roi_rect_from_palm(
                    det,
                    int(cfg["image"]["width"]),
                    int(cfg["image"]["height"]),
                    scale_x=float(cfg["hand_roi"]["scale_x"]),
                    scale_y=float(cfg["hand_roi"]["scale_y"]),
                    shift_x=float(cfg["hand_roi"]["shift_x"]),
                    shift_y=float(cfg["hand_roi"]["shift_y"]),
                )
                crop, corners = crop_image_by_roi(
                    image,
                    rect,
                    int(cfg["hand_roi"]["output_width"]),
                    int(cfg["hand_roi"]["output_height"]),
                )
                crop_path = crop_images / f"{roi_id}.png"
                if crop_path.exists():
                    existing = read_image(crop_path)
                    if existing is None or existing.shape != crop.shape or not (existing == crop).all():
                        raise DatasetContractError(f"refusing to overwrite changed ROI: {crop_path}")
                elif not write_image(crop_path, crop):
                    raise DatasetContractError(f"failed to write ROI: {crop_path}")
                manifest.append(
                    {
                        "crop_id": roi_id,
                        "image": parent["image"],
                        "palm_det_id": det["palm_det_id"],
                        "palm_valid": det["proposal_kind"] == "runtime",
                        "palm_score": float(det.get("score", 0.0)),
                        "crop_path": str(crop_path),
                        "roi_rect": rect,
                        "roi_corners_px": [[float(x), float(y)] for x, y in corners.tolist()],
                        "output_size": [
                            int(cfg["hand_roi"]["output_width"]),
                            int(cfg["hand_roi"]["output_height"]),
                        ],
                    }
                )
            except Exception as exc:
                failures.append({"raw_image_id": parent["raw_image_id"], "error": str(exc)})
    manifest = enrich_roi_rows(manifest, palm_rows, Path(args.dataset_root), args.proposal_variant)
    write_jsonl(paths["roi"] / "hand_roi_crops_manifest.jsonl", manifest)
    WarehouseRegistry(Path(args.dataset_root)).register_rois(manifest)
    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for row in manifest:
        row_warnings, row_errors = roi_manifest_issues(row, cfg)
        if row_warnings:
            warnings.append({"roi_id": row["roi_id"], "warnings": row_warnings})
        if row_errors:
            errors.append({"roi_id": row["roi_id"], "errors": row_errors})
    report = {
        "schema_version": SCHEMA_VERSION,
        "proposal_variant": args.proposal_variant,
        "rois": len(manifest),
        "failures": failures,
        "warnings": warnings,
        "errors": errors,
    }
    write_json(paths["qc"] / "roi_build_report.json", report)
    if failures or errors:
        raise DatasetContractError(f"ROI build failed; see {paths['qc'] / 'roi_build_report.json'}")
    return report


def _attach_manifest_fields(
    labels: Iterable[Mapping[str, Any]], manifest: Iterable[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_id = {str(row["roi_id"]): row for row in manifest}
    output: List[Dict[str, Any]] = []
    for raw in labels:
        row = dict(raw)
        roi_id = str(row.get("roi_id") or row.get("crop_id"))
        source = by_id.get(roi_id)
        if source is None:
            raise DatasetContractError(f"label references unknown ROI: {roi_id}")
        for key in (
            "schema_version",
            "dataset_id",
            "capture_source_id",
            "split",
            "raw_image_id",
            "roi_id",
            "proposal_variant",
            "proposal_slot",
            "proposal_kind",
            "roi_contract_version",
            "crop_relpath",
        ):
            row[key] = source[key]
        row["crop_id"] = roi_id
        row["crop_path"] = source["crop_relpath"]
        output.append(row)
    return output


def _run_mediapipe(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    root = source_root(
        dataset_root,
        args.scope,
        args.dataset_id,
        args.capture_source_id,
    )
    paths = proposal_paths(root, args.proposal_variant)
    manifest = read_jsonl(paths["roi"] / "hand_roi_crops_manifest.jsonl")
    if not manifest:
        raise DatasetContractError("ROI manifest is missing or empty")
    runtime_manifest = []
    for row in manifest:
        item = dict(row)
        item["crop_path"] = str(Path(args.dataset_root).resolve() / row["crop_relpath"])
        runtime_manifest.append(item)
    rows, backend_info = label_hand_landmark_manifest(
        runtime_manifest,
        cfg,
        ROOT,
        show_progress=show_progress,
    )
    rows = _attach_manifest_fields(rows, manifest)
    rows = apply_label_provenance(rows, human_reviewed=False)
    backend = str(backend_info["backend"])
    draft_path = paths["roi"] / "hand_landmarks_autolabel_draft.jsonl"
    write_jsonl(draft_path, rows)
    roi_visualization_report = _run_roi_visualization(
        args,
        cfg,
        rows,
        paths,
        hand_landmark_backend=backend,
        enabled=(cfg.get("visualization") or {}).get("roi_enabled", False),
        trigger="autolabel",
        show_progress=show_progress,
    )
    original_visualization_report = _run_original_image_visualization(
        args,
        rows,
        root,
        paths,
        enabled=(cfg.get("visualization") or {}).get("original_image_enabled", False),
        trigger="autolabel",
        show_progress=show_progress,
    )
    stats = summarize_label_rows(rows, cfg)
    report = {
        "schema_version": SCHEMA_VERSION,
        # Retained for consumers of the existing report path/schema.
        "mediapipe_mode": backend_info["mode"] if backend == "mediapipe_tasks" else None,
        "hand_landmark_backend": backend,
        "hand_landmark_mode": backend_info["mode"],
        "execution_provider": backend_info["provider"],
        "hand_classifier_provider": backend_info.get("hand_classifier_provider"),
        "hand_classifier_model_id": backend_info.get("hand_classifier_model_id"),
        "hand_classifier_runtime_rois_labeled": backend_info.get(
            "hand_classifier_runtime_rois_labeled", 0
        ),
        "runtime_rois_labeled": backend_info["runtime_rois_labeled"],
        "negative_candidates_skipped": backend_info["negative_candidates_skipped"],
        "total": stats["total"],
        "positive": stats["positive"],
        "teacher_abstain": stats["negative"],
        "label_origin": "mediapipe" if backend == "mediapipe_tasks" else "rtmpose",
        "roi_visualization": roi_visualization_report,
        "original_image_visualization": original_visualization_report,
    }
    write_json(paths["qc"] / "mediapipe_report.json", report)
    return report


def _run_roi_visualization(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    rows: List[Dict[str, Any]],
    paths: Dict[str, Path],
    *,
    hand_landmark_backend: str,
    enabled: Any,
    trigger: str,
    show_progress: bool,
) -> Dict[str, Any]:
    split = parse_capture_source_id(args.capture_source_id)["split"]
    visualization_cfg = cfg.get("visualization") or {}
    if not isinstance(enabled, bool):
        raise DatasetContractError("visualization.roi_enabled must be true or false")
    try:
        train_max_samples = int(visualization_cfg.get("train_max_samples", 200))
    except (TypeError, ValueError) as exc:
        raise DatasetContractError("visualization.train_max_samples must be an integer") from exc
    if train_max_samples < 1:
        raise DatasetContractError("visualization.train_max_samples must be >= 1")

    visualization_rows = rows
    excluded_non_runtime = 0
    if hand_landmark_backend == "rtmpose_onnx":
        visualization_rows = [
            row for row in rows if str(row.get("proposal_kind")) == "runtime"
        ]
        excluded_non_runtime = len(rows) - len(visualization_rows)

    roi_report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "enabled": enabled,
        "split": split,
        "trigger": trigger,
        "hand_landmark_backend": hand_landmark_backend,
        "input_rows": len(rows),
        "excluded_non_runtime": excluded_non_runtime,
        "output_relpath": str(
            (paths["roi"] / "hand_landmarks_roi_visualization")
            .relative_to(Path(args.dataset_root).resolve())
        ).replace("\\", "/"),
    }
    if enabled:
        try:
            roi_report.update(
                render_autolabel_roi_visualizations(
                    visualization_rows,
                    paths["roi"] / "images",
                    paths["roi"] / "hand_landmarks_roi_visualization",
                    split=split,
                    train_max_samples=train_max_samples,
                    show_progress=show_progress,
                )
            )
        except TrainingRoiVisualizationError as exc:
            raise DatasetContractError(str(exc)) from exc
    write_json(paths["qc"] / "roi_visualization_report.json", roi_report)
    return roi_report


def _run_existing_roi_visualization(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    _, paths = _source_context(args, cfg)
    draft_path = paths["roi"] / "hand_landmarks_autolabel_draft.jsonl"
    rows = read_jsonl(draft_path)
    if not rows:
        raise DatasetContractError(
            f"Hand landmark autolabel draft is missing or empty: {draft_path}"
        )
    backend = ""
    backend_report_path = paths["qc"] / "mediapipe_report.json"
    if backend_report_path.is_file():
        backend_report = json.loads(backend_report_path.read_text(encoding="utf-8"))
        backend = str(backend_report.get("hand_landmark_backend") or "")
    if not backend:
        backend = (
            "rtmpose_onnx"
            if any(str(row.get("source")) == "rtmpose_m_hand5_onnx" for row in rows)
            else "mediapipe_tasks"
        )
    return _run_roi_visualization(
        args,
        cfg,
        rows,
        paths,
        hand_landmark_backend=backend,
        enabled=True,
        trigger="standalone",
        show_progress=show_progress,
    )


def _run_original_image_visualization(
    args: argparse.Namespace,
    rows: List[Dict[str, Any]],
    source: Path,
    paths: Dict[str, Path],
    *,
    enabled: Any,
    trigger: str,
    show_progress: bool,
) -> Dict[str, Any]:
    if not isinstance(enabled, bool):
        raise DatasetContractError(
            "visualization.original_image_enabled must be true or false"
        )
    output_dir = source / "visualizations" / "original_image_landmarks" / args.proposal_variant
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "enabled": enabled,
        "trigger": trigger,
        "output_relpath": str(
            output_dir.relative_to(Path(args.dataset_root).resolve())
        ).replace("\\", "/"),
    }
    if enabled:
        try:
            report.update(
                render_original_image_visualizations(
                    rows,
                    source / "images",
                    output_dir,
                    proposal_variant=args.proposal_variant,
                    show_progress=show_progress,
                )
            )
        except TrainingRoiVisualizationError as exc:
            raise DatasetContractError(str(exc)) from exc
    write_json(paths["qc"] / "original_image_visualization_report.json", report)
    return report


def _run_existing_original_image_visualization(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    *,
    show_progress: bool = True,
) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    source = source_root(
        dataset_root,
        args.scope,
        args.dataset_id,
        args.capture_source_id,
    )
    paths = proposal_paths(source, args.proposal_variant)
    draft_path = paths["roi"] / "hand_landmarks_autolabel_draft.jsonl"
    rows = read_jsonl(draft_path)
    if not rows:
        raise DatasetContractError(
            f"Hand landmark autolabel draft is missing or empty: {draft_path}"
        )
    report = _run_original_image_visualization(
        args,
        rows,
        source,
        paths,
        enabled=True,
        trigger="standalone",
        show_progress=show_progress,
    )
    video_enabled = (cfg.get("visualization") or {}).get(
        "original_video_enabled", True
    )
    if not isinstance(video_enabled, bool):
        raise DatasetContractError(
            "visualization.original_video_enabled must be true or false"
        )
    video_report: Dict[str, Any] = {"enabled": video_enabled}
    if video_enabled:
        output_dir = source / "visualizations" / "original_image_landmarks" / args.proposal_variant
        video_report.update(create_video(output_dir, output_dir.parent))
        video_report["video_relpath"] = str(
            Path(video_report["video_path"]).relative_to(Path(args.dataset_root).resolve())
        ).replace("\\", "/")
    report["video"] = video_report
    write_json(paths["qc"] / "original_image_visualization_report.json", report)
    return report


def _run_export_cvat(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    _, paths = _source_context(args, cfg)
    if parse_capture_source_id(args.capture_source_id)["split"] == "train":
        raise DatasetContractError("routine CVAT review is limited to Val/Test Hand ROIs")
    manifest = read_jsonl(paths["roi"] / "hand_roi_crops_manifest.jsonl")
    draft = read_jsonl(paths["roi"] / "hand_landmarks_autolabel_draft.jsonl")
    xml_path = paths["reviewed"] / "cvat_autolabel.xml"
    stats = export_cvat_xml(manifest, draft, Path(args.dataset_root).resolve(), xml_path, cfg)
    stats["review_scope"] = "hand_roi_only"
    stats["manual_roi_editing"] = False
    write_json(paths["qc"] / "cvat_export_report.json", stats)
    return stats


def _run_import_cvat(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    _, paths = _source_context(args, cfg)
    reviewed_xml = paths["reviewed"] / "cvat_reviewed.xml"
    manifest = read_jsonl(paths["roi"] / "hand_roi_crops_manifest.jsonl")
    draft = read_jsonl(paths["roi"] / "hand_landmarks_autolabel_draft.jsonl")
    rows, stats = import_cvat_xml(reviewed_xml, manifest, draft, cfg)
    rows = _attach_manifest_fields(rows, manifest)
    draft_by_roi = {str(row["roi_id"]): row for row in draft}
    rows = apply_label_provenance(rows, draft_by_roi=draft_by_roi, human_reviewed=True)
    errors = list(stats.get("errors") or [])
    if errors:
        write_json(paths["qc"] / "cvat_import_report.json", stats)
        raise DatasetContractError(f"CVAT import has blocking errors; see {paths['qc'] / 'cvat_import_report.json'}")
    write_jsonl(paths["reviewed"] / "hand_landmarks_reviewed.jsonl", rows)
    write_json(paths["qc"] / "cvat_import_report.json", stats)
    return stats


def _dataset_manifest(
    dataset_root: Path,
    scope: str,
    dataset_id: str,
    *,
    pending_report: Mapping[str, Any] | None = None,
    write: bool = True,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    scope = str(scope).strip().lower()
    if scope not in {"pretrain", "eval"}:
        raise DatasetContractError("dataset scope must be pretrain or eval")
    dataset_id = require_safe_id(dataset_id, "dataset_id")
    bucket = "PretrainSource" if scope == "pretrain" else "EValSource"
    root = dataset_root / bucket / dataset_id
    sources = []
    for descriptor_path in sorted(root.glob("*/source.json")):
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        capture_root = descriptor_path.parent
        published = []
        for report in sorted((capture_root / "qc").glob("*/source_publish_report.json")):
            published.append(json.loads(report.read_text(encoding="utf-8")))
        if pending_report is not None and str(descriptor.get("capture_source_id")) == str(
            pending_report.get("capture_source_id")
        ):
            pending_variant = str(pending_report.get("proposal_variant"))
            published = [
                report
                for report in published
                if str(report.get("proposal_variant")) != pending_variant
            ]
            published.append(dict(pending_report))
            published.sort(key=lambda report: str(report.get("proposal_variant")))
        if published:
            descriptor["published_variants"] = published
            sources.append(descriptor)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "scope": scope,
        "capture_sources": sources,
        "content_sha256": "not_computed",
    }
    if write:
        write_json(root / "dataset_manifest.json", manifest)
    return manifest


def _run_clean_visualizations(args: argparse.Namespace) -> Dict[str, Any]:
    return clean_variant_visualizations(
        Path(args.dataset_root),
        args.scope,
        args.dataset_id,
        args.capture_source_id,
        args.proposal_variant,
    )


def _run_delete_source_variant(args: argparse.Namespace) -> Dict[str, Any]:
    result = delete_source_variant(
        Path(args.dataset_root),
        args.scope,
        args.dataset_id,
        args.capture_source_id,
        args.proposal_variant,
        args.confirm_delete,
    )
    _dataset_manifest(
        Path(args.dataset_root).resolve(), args.scope, args.dataset_id
    )
    result["dataset_manifest_updated"] = True
    return result


def _validate_evaluation_limits(dataset: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    for eval_split in ("val", "test"):
        selected = [
            source
            for source in dataset.get("capture_sources", [])
            if source.get("split") == eval_split
        ]
        raw_count = sum(int(source["raw_image_count"]) for source in selected)
        roi_count = sum(
            int(variant["rois"])
            for source in selected
            for variant in source.get("published_variants", [])
        )
        if raw_count > int(cfg["evaluation_limits"]["max_raw_images_per_split"]):
            raise DatasetContractError(f"{eval_split} exceeds raw-image limit: {raw_count}")
        if roi_count > int(cfg["evaluation_limits"]["max_rois_per_split"]):
            raise DatasetContractError(f"{eval_split} exceeds ROI limit: {roi_count}")


def _partition_labels(
    rows: Iterable[Mapping[str, Any]], split: str, cfg: Mapping[str, Any]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    positives: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        present = bool((row.get("hand_presence") or {}).get("present"))
        quality_warnings, quality_errors, quality_needs_review = label_issues(row, cfg)
        row["quality_gate"] = {
            "passed": not quality_errors and not quality_needs_review,
            "warnings": quality_warnings,
            "errors": quality_errors,
        }
        presence_gate_failed = any(
            str(error).startswith("rtmpose_hand_presence_score_")
            for error in quality_errors
        )
        connection_gate_failed = any(
            str(error).startswith("rtmpose_connection_length_")
            for error in quality_errors
        )
        if split == "train" and presence_gate_failed:
            row["train_eligible"] = False
            row["ignore_reason"] = "rtmpose_hand_presence_gate"
            ignored.append(row)
        elif split == "train" and present and (quality_errors or quality_needs_review):
            row["train_eligible"] = False
            if any(
                str(error).startswith("rtmpose_boundary_coordinate_values:")
                for error in quality_errors
            ):
                row["ignore_reason"] = "rtmpose_boundary_coordinate_gate"
            elif connection_gate_failed:
                row["ignore_reason"] = "rtmpose_connection_length_gate"
            else:
                row["ignore_reason"] = "automatic_positive_failed_quality_gate"
            ignored.append(row)
        elif bool(row.get("ignore_for_training")):
            row["train_eligible"] = False
            ignored.append(row)
        elif present:
            row["train_eligible"] = split == "train"
            positives.append(row)
        elif split == "train":
            row["train_eligible"] = False
            row["candidate_negative"] = True
            candidates.append(row)
        else:
            row["train_eligible"] = False
            positives.append(row)
    if split in {"val", "test"} and candidates:
        raise DatasetContractError("Val/Test must never publish negative candidates")
    return positives, candidates, ignored


def _run_publish_source(args: argparse.Namespace, cfg: Dict[str, Any]) -> Dict[str, Any]:
    source, paths = _source_context(args, cfg)
    split = parse_capture_source_id(args.capture_source_id)["split"]
    raw_rows = read_jsonl(source / "raw_images.jsonl")
    manifest = read_jsonl(paths["roi"] / "hand_roi_crops_manifest.jsonl")
    draft = read_jsonl(paths["roi"] / "hand_landmarks_autolabel_draft.jsonl")
    if split == "train":
        rows = draft
    else:
        rows = read_jsonl(paths["reviewed"] / "hand_landmarks_reviewed.jsonl")
        if not rows:
            raise DatasetContractError("Val/Test publication requires reviewed CVAT labels")
    positives, candidates, ignored = _partition_labels(rows, split, cfg)
    labels_file = paths["labels"] / ("hand_training_labels.jsonl" if split == "train" else "hand_evaluation_labels.jsonl")
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": args.dataset_id,
        "capture_source_id": args.capture_source_id,
        "split": split,
        "proposal_variant": args.proposal_variant,
        "raw_images": len(raw_rows),
        "rois": len(manifest),
        "published_labels": len(positives),
        "candidate_negatives": len(candidates),
        "ignored": len(ignored),
        "labels_relpath": str(labels_file.relative_to(Path(args.dataset_root).resolve())).replace("\\", "/"),
        "palm_output_human_modified": False,
        "evaluation_scope": "fixed_hand_roi_only" if split in {"val", "test"} else None,
    }
    if args.scope == "eval":
        prospective_dataset = _dataset_manifest(
            Path(args.dataset_root).resolve(),
            args.scope,
            args.dataset_id,
            pending_report=report,
            write=False,
        )
        _validate_evaluation_limits(prospective_dataset, cfg)
    write_jsonl(labels_file, positives)
    write_jsonl(paths["labels"] / "candidate_negatives.jsonl", candidates)
    write_jsonl(paths["labels"] / "ignored.jsonl", ignored)
    write_json(paths["qc"] / "source_publish_report.json", report)
    _dataset_manifest(Path(args.dataset_root).resolve(), args.scope, args.dataset_id)
    return report


def _run_source_pipeline(args: argparse.Namespace, cfg: Dict[str, Any], evaluation: bool) -> Dict[str, Any]:
    split = parse_capture_source_id(args.capture_source_id)["split"]
    if evaluation != (split in {"val", "test"}):
        raise DatasetContractError("autolabel-train/eval does not match capture source split")
    print("[1/4] Source check", file=sys.stderr, flush=True)
    _run_validate(args, show_progress=True)
    print("[2/4] Palm inference", file=sys.stderr, flush=True)
    _run_palm(args, cfg, show_progress=True)
    print("[3/4] ROI crops", file=sys.stderr, flush=True)
    _run_build_roi(args, cfg, show_progress=True)
    print("[4/4] Hand landmark autolabel", file=sys.stderr, flush=True)
    result = _run_mediapipe(args, cfg, show_progress=True)
    if evaluation:
        result["next_step"] = "export-cvat, then place cvat_reviewed.xml and run import-cvat + publish-source"
    else:
        _run_publish_source(args, cfg)
        result["next_step"] = "optional negative review from candidate_negatives.jsonl"
    return result


def main() -> None:
    args = _parser().parse_args()
    cfg = _load_public_configs(args)
    try:
        if args.command == "validate-source":
            result = _run_validate(args)
        elif args.command == "palm":
            result = _run_palm(args, cfg)
        elif args.command == "build-roi":
            result = _run_build_roi(args, cfg)
        elif args.command == "mediapipe":
            result = _run_mediapipe(args, cfg)
        elif args.command == "export-cvat":
            result = _run_export_cvat(args, cfg)
        elif args.command == "import-cvat":
            result = _run_import_cvat(args, cfg)
        elif args.command == "publish-source":
            result = _run_publish_source(args, cfg)
        elif args.command == "autolabel-train":
            result = _run_source_pipeline(args, cfg, evaluation=False)
        elif args.command == "autolabel-eval":
            result = _run_source_pipeline(args, cfg, evaluation=True)
        elif args.command == "autolabel-visualize-roi":
            result = _run_existing_roi_visualization(args, cfg)
        elif args.command == "autolabel-visualize-original":
            result = _run_existing_original_image_visualization(args, cfg)
        elif args.command == "clean-autolabel-visualizations":
            result = _run_clean_visualizations(args)
        elif args.command == "delete-source-variant":
            result = _run_delete_source_variant(args)
        elif args.command == "rebuild-dataset-manifest":
            result = _dataset_manifest(
                Path(args.dataset_root).resolve(), args.scope, args.dataset_id
            )
        elif args.command == "prepare-negative-review":
            rows = [row for path in args.candidate_labels for row in read_jsonl(Path(path))]
            result = prepare_negative_review(Path(args.dataset_root), args.negative_dataset_id, rows)
        elif args.command == "publish-negative-review":
            result = publish_negative_review(Path(args.dataset_root), args.negative_dataset_id)
        elif args.command == "prepare-selection-review":
            result = prepare_selection_review(
                Path(args.dataset_root), args.selection_id, read_jsonl(Path(args.request))
            )
        elif args.command == "publish-selection-review":
            result = publish_selection_review(Path(args.dataset_root), args.selection_id)
        else:
            result = WarehouseRegistry(Path(args.dataset_root)).report()
    except (DatasetContractError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
