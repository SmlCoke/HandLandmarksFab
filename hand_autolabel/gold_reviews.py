"""Reusable CVAT review and publication for mined and recorded Gold data."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .cvat_io import export_cvat_xml, import_cvat_xml
from .dataset_v3 import (
    SCHEMA_VERSION,
    DatasetContractError,
    WarehouseRegistry,
    apply_label_provenance,
    require_safe_id,
)
from .formats import read_jsonl, write_json, write_jsonl


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DatasetContractError(f"refusing to overwrite published image: {destination}")
    shutil.copy2(source, destination)


def _hard_root(dataset_root: Path, hard_dataset_id: str) -> Path:
    require_safe_id(hard_dataset_id, "hard_dataset_id")
    return Path(dataset_root).resolve() / "GoldSource" / "HardSamples" / hard_dataset_id


def prepare_hard_review(
    dataset_root: Path,
    hard_dataset_id: str,
    request_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Materialize one reusable hard dataset and its CVAT 1.1 draft."""

    dataset_root = Path(dataset_root).resolve()
    root = _hard_root(dataset_root, hard_dataset_id)
    review_root = root / "review"
    if review_root.exists() or (root / "published").exists():
        raise DatasetContractError(f"hard dataset workspace already exists: {root}")
    if not request_rows:
        raise DatasetContractError("hard review request is empty")

    registry = WarehouseRegistry(dataset_root)
    registry.reserve_hard_dataset(hard_dataset_id)
    materialized: List[Dict[str, Any]] = []
    cvat_manifest: List[Dict[str, Any]] = []
    draft_rows: List[Dict[str, Any]] = []
    seen_roi_ids: set[str] = set()
    try:
        for raw in request_rows:
            row = dict(raw)
            if str(row.get("split")) != "train":
                raise DatasetContractError("hard dataset review accepts Train requests only")
            if not bool((row.get("hand_presence") or {}).get("present", False)):
                raise DatasetContractError("hard mining requests must start from positive labels")
            registry.assert_roi_reference(row)
            roi_id = str(row.get("roi_id") or row.get("crop_id") or "")
            if not roi_id or roi_id in seen_roi_ids:
                raise DatasetContractError(f"invalid or duplicate hard-review roi_id: {roi_id!r}")
            seen_roi_ids.add(roi_id)
            source_relpath = str(row.get("crop_relpath") or row.get("crop_path") or "")
            source = dataset_root / source_relpath
            if not source.is_file():
                raise DatasetContractError(f"requested ROI does not exist: {source}")
            destination = review_root / "images" / f"{roi_id}{source.suffix.lower()}"
            _copy_image(source, destination)
            review_relpath = str(destination.relative_to(dataset_root)).replace("\\", "/")
            row.update(
                {
                    "roi_id": roi_id,
                    "crop_id": roi_id,
                    "source_crop_relpath": source_relpath,
                    "review_relpath": review_relpath,
                }
            )
            materialized.append(row)
            cvat_row = dict(row)
            cvat_row["crop_relpath"] = review_relpath
            cvat_row["crop_path"] = review_relpath
            cvat_manifest.append(cvat_row)
            draft_rows.append(dict(cvat_row))

        write_jsonl(review_root / "request_manifest.jsonl", materialized)
        write_jsonl(review_root / "cvat_manifest.jsonl", cvat_manifest)
        write_jsonl(review_root / "teacher_draft.jsonl", draft_rows)
        cvat_cfg = dict(cfg)
        cvat_cfg["paths"] = dict(cfg.get("paths") or {})
        cvat_cfg["paths"]["roi_crops_dir"] = str(review_root)
        stats = export_cvat_xml(
            cvat_manifest,
            draft_rows,
            dataset_root,
            review_root / "cvat_autolabel.xml",
            cvat_cfg,
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "hard_dataset_id": hard_dataset_id,
            "candidate_count": len(materialized),
            "review_root": str(review_root),
            "images_dir": str(review_root / "images"),
            "cvat_xml": str(review_root / "cvat_autolabel.xml"),
            "cvat": stats,
            "instruction": (
                "Upload review/images with cvat_autolabel.xml, precisely review all "
                "landmarks/presence/handedness, then save CVAT 1.1 as cvat_reviewed.xml."
            ),
        }
        write_json(review_root / "prepare_report.json", report)
        return report
    except Exception:
        # The registry reservation deliberately remains as a tombstone so a
        # partially used public dataset ID cannot silently acquire new meaning.
        raise


def import_hard_review(
    dataset_root: Path,
    hard_dataset_id: str,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    review_root = _hard_root(dataset_root, hard_dataset_id) / "review"
    manifest = read_jsonl(review_root / "cvat_manifest.jsonl")
    draft = read_jsonl(review_root / "teacher_draft.jsonl")
    if not manifest or not draft:
        raise DatasetContractError("prepare-hard-review must run before import")
    rows, stats = import_cvat_xml(
        review_root / "cvat_reviewed.xml", manifest, draft, cfg
    )
    errors = list(stats.get("errors") or [])
    write_json(review_root / "cvat_import_report.json", stats)
    if errors:
        raise DatasetContractError(
            f"hard CVAT import has blocking errors; see {review_root / 'cvat_import_report.json'}"
        )
    draft_by_roi = {str(row["roi_id"]): row for row in draft}
    rows = apply_label_provenance(
        rows, draft_by_roi=draft_by_roi, human_reviewed=True
    )
    write_jsonl(review_root / "hand_landmarks_reviewed.jsonl", rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "hard_dataset_id": hard_dataset_id,
        "reviewed_rows": len(rows),
        "warnings": len(stats.get("warnings") or []),
        "errors": 0,
    }


def publish_hard_review(dataset_root: Path, hard_dataset_id: str) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    root = _hard_root(dataset_root, hard_dataset_id)
    review_root = root / "review"
    published_root = root / "published"
    if published_root.exists():
        raise DatasetContractError(f"hard dataset is already published: {hard_dataset_id}")
    requests = read_jsonl(review_root / "request_manifest.jsonl")
    reviewed = read_jsonl(review_root / "hand_landmarks_reviewed.jsonl")
    if not requests or not reviewed:
        raise DatasetContractError("import-hard-review must complete before publish")
    request_by_id = {str(row["roi_id"]): row for row in requests}
    if len(request_by_id) != len(requests):
        raise DatasetContractError("hard request contains duplicate ROI IDs")
    if {str(row.get("roi_id")) for row in reviewed} != set(request_by_id):
        raise DatasetContractError("hard reviewed rows do not exactly cover the request")

    published: List[Dict[str, Any]] = []
    ignored = 0
    for label in reviewed:
        roi_id = str(label.get("roi_id"))
        request = request_by_id[roi_id]
        if not bool(label.get("cvat_image_seen")):
            raise DatasetContractError(f"hard reviewed ROI is missing from CVAT XML: {roi_id}")
        if bool(label.get("ignore_for_training")):
            ignored += 1
            continue
        review_image = dataset_root / str(request["review_relpath"])
        destination = (
            published_root
            / "images"
            / str(request["capture_source_id"])
            / review_image.name
        )
        _copy_image(review_image, destination)
        item = dict(label)
        for key in (
            "dataset_id",
            "capture_source_id",
            "split",
            "raw_image_id",
            "roi_id",
            "proposal_variant",
            "proposal_slot",
            "proposal_kind",
            "roi_contract_version",
        ):
            if key in request:
                item[key] = request[key]
        source_relpath = str(request["source_crop_relpath"])
        item.update(
            {
                "crop_id": roi_id,
                "crop_relpath": source_relpath,
                "crop_path": source_relpath,
                "source_crop_relpath": source_relpath,
                "published_relpath": str(destination.relative_to(dataset_root)).replace("\\", "/"),
                "hard_dataset_id": hard_dataset_id,
            }
        )
        published.append(item)
    if not published:
        raise DatasetContractError("cannot publish an empty hard dataset")
    write_jsonl(published_root / "hard_labels.jsonl", published)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "hard_dataset_id": hard_dataset_id,
        "records": len(published),
        "positive": sum(
            bool((row.get("hand_presence") or {}).get("present")) for row in published
        ),
        "negative": sum(
            not bool((row.get("hand_presence") or {}).get("present")) for row in published
        ),
        "ignored": ignored,
        "labels": "hard_labels.jsonl",
        "review_contract": "cvat_xml_1.1_precise_hand_roi_review",
        "image_policy": "copied_review_and_published_images",
    }
    write_json(published_root / "manifest.json", manifest)
    WarehouseRegistry(dataset_root).publish_hard_dataset(hard_dataset_id)
    shutil.rmtree(review_root)
    return manifest
