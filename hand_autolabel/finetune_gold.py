from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from .cvat_io import export_cvat_xml, import_cvat_xml
from .finalization import (
    atomic_write_json,
    atomic_write_jsonl,
    manifest_conflicts,
    sha256_file,
    validate_label_schema,
    validate_manifest_row,
)
from .formats import load_yaml_config, read_jsonl, resolve_path
from .image_io import read_image, to_uint8_gray, write_image
from .roi_geometry import build_roi_rect_from_palm, crop_image_by_roi


class GoldPipelineError(RuntimeError):
    """Raised when a Gold source cannot be authenticated or published safely."""


ALLOWED_SOURCE_KINDS = {
    "external_gold",
    "reviewed_hard_gold",
    "disagreement_gold",
    "new_recorded_gold",
}
ROLE_PRECEDENCE = {
    "external_gold": 0,
    "reviewed_hard_gold": 1,
    "disagreement_gold": 2,
    "new_recorded_gold": 3,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _repo_root(config_path: Path) -> Path:
    return Path(config_path).resolve().parents[1]


def _workspace(cfg: Mapping[str, Any], config_path: Path) -> Path:
    value = cfg.get("workspace_root")
    if not value:
        raise GoldPipelineError("workspace_root is required")
    return resolve_path(_repo_root(config_path), str(value))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _json_sha(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_version(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _unique(rows: Iterable[Mapping[str, Any]], key: str, scope: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise GoldPipelineError(f"{scope}: missing {key}")
        if value in result:
            raise GoldPipelineError(f"{scope}: duplicate {key}={value}")
        result[value] = dict(row)
    return result


def _first_symlink(path: Path) -> Path | None:
    """Return the first symlink in an absolute path, including ancestors."""

    lexical = Path(os.path.abspath(os.fspath(path)))
    return next(
        (candidate for candidate in (lexical, *lexical.parents) if candidate.is_symlink()),
        None,
    )


def _assert_regular_file(path: Path, scope: str) -> Path:
    raw = Path(path)
    symlink = _first_symlink(raw)
    if symlink is not None:
        raise GoldPipelineError(f"{scope}: symlink is not allowed: {symlink}")
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise GoldPipelineError(f"{scope}: regular file required: {raw}") from exc
    if not resolved.is_file():
        raise GoldPipelineError(f"{scope}: regular file required: {resolved}")
    return resolved


def _assert_directory(path: Path, scope: str) -> Path:
    raw = Path(path)
    symlink = _first_symlink(raw)
    if symlink is not None:
        raise GoldPipelineError(f"{scope}: symlink is not allowed: {symlink}")
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise GoldPipelineError(f"{scope}: directory required: {raw}") from exc
    if not resolved.is_dir():
        raise GoldPipelineError(f"{scope}: directory required: {resolved}")
    return resolved


def _safe_relative_file(root: Path, relative: Any, scope: str) -> Path:
    text = str(relative or "")
    value = Path(text)
    if not text or value.is_absolute() or ".." in value.parts:
        raise GoldPipelineError(f"{scope}: safe relative file path required: {text!r}")
    trusted_root = _assert_directory(root, f"{scope} root")
    resolved = _assert_regular_file(trusted_root / value, scope)
    try:
        resolved.relative_to(trusted_root)
    except ValueError as exc:
        raise GoldPipelineError(f"{scope}: path escapes controlled root: {text}") from exc
    return resolved


def _safe_relative_directory(root: Path, relative: Any, scope: str) -> Path:
    text = str(relative or "")
    value = Path(text)
    if not text or value.is_absolute() or ".." in value.parts:
        raise GoldPipelineError(f"{scope}: safe relative directory path required: {text!r}")
    trusted_root = _assert_directory(root, f"{scope} root")
    resolved = _assert_directory(trusted_root / value, scope)
    try:
        resolved.relative_to(trusted_root)
    except ValueError as exc:
        raise GoldPipelineError(f"{scope}: path escapes controlled root: {text}") from exc
    return resolved


def _copy_or_link(source: Path, target: Path) -> None:
    source = _assert_regular_file(source, "copy source")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    if sha256_file(source) != sha256_file(target):
        raise GoldPipelineError(f"copy SHA mismatch: {source} -> {target}")


def _artifact(path: Path, root: Path, *, count: int | None = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "path": str(path.resolve().relative_to(root.resolve())).replace("\\", "/"),
        "sha256": sha256_file(path),
    }
    if count is not None:
        data["count"] = int(count)
    return data


def _write_hash_manifest(files: Iterable[Path], base: Path, output: Path) -> tuple[int, str]:
    rows = [
        {
            "path": str(path.resolve().relative_to(base.resolve())).replace("\\", "/"),
            "sha256": sha256_file(path),
        }
        for path in sorted({Path(value).resolve() for value in files}, key=lambda p: str(p).lower())
    ]
    atomic_write_jsonl(output, rows)
    aggregate = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        aggregate.update(f"{row['path']}:{row['sha256']}\n".encode("utf-8"))
    return len(rows), aggregate.hexdigest()


def _normalized_pixel_sha(path: Path) -> str:
    image = read_image(path)
    if image is None:
        raise GoldPipelineError(f"unreadable crop image: {path}")
    gray = to_uint8_gray(image)
    digest = hashlib.sha256()
    digest.update(f"{gray.shape[1]}x{gray.shape[0]}:uint8-gray\0".encode("ascii"))
    digest.update(gray.tobytes(order="C"))
    return digest.hexdigest()


def _dynamic_cvat_config(cfg: Mapping[str, Any], task_root: Path) -> Dict[str, Any]:
    dynamic = copy.deepcopy(dict(cfg))
    dynamic.setdefault("image", {"width": 1280, "height": 720, "channels": 1, "orientation": "upright"})
    dynamic.setdefault(
        "hand_roi",
        {"output_width": 256, "output_height": 256, "scale_x": 1.8, "scale_y": 1.8, "shift_x": 0.0, "shift_y": -0.1},
    )
    dynamic.setdefault("cvat", {})
    dynamic["cvat"].setdefault("label_name", "hand_landmarks")
    dynamic["cvat"].setdefault("no_hand_label_name", "no_hand")
    dynamic["cvat"].setdefault("left_label_name", "Left")
    dynamic["cvat"].setdefault("right_label_name", "Right")
    dynamic["cvat"].setdefault("unknown_handedness_label_name", "unknown_handedness")
    dynamic["cvat"].setdefault("ignore_for_training_label_name", "ignore_for_training")
    dynamic["cvat"].setdefault("xml_version", "1.1")
    dynamic.setdefault("review", {})
    dynamic["review"].update(
        {
            "strip_teacher_handedness": True,
            "require_explicit_presence_decision": True,
            "require_explicit_handedness_decision": True,
            "handedness_policy": "optional_per_row",
        }
    )
    dynamic["paths"] = {
        "roi_crops_dir": str(task_root / "02_roi_crops"),
        "reviewed_dir": str(task_root / "03_reviewed"),
        "qc_dir": str(task_root / "qc"),
    }
    return dynamic


def build_pretrain_source_registry(config_path: Path) -> Dict[str, Any]:
    """Publish the authenticated manifest/draft lookup used by HLML b/c selection."""
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    root = _repo_root(config_path)
    try:
        qc_dir = resolve_path(root, cfg["outputs"]["pretrain"]["qc_dir"])
    except KeyError as exc:
        raise GoldPipelineError(f"pretrain registry requires outputs.pretrain.qc_dir: {exc}") from exc
    rows: List[Dict[str, Any]] = []
    sources: Dict[str, Any] = {}
    seen_global: set[str] = set()
    for source in cfg.get("sources") or []:
        dataset_id = str(source.get("dataset_id", ""))
        if not dataset_id:
            raise GoldPipelineError("pretrain source registry: dataset_id is required")
        if "pretrain" not in [str(value) for value in source.get("enabled_stages", ["pretrain", "finetune"])]:
            sources[dataset_id] = {"status": "disabled_for_stage"}
            continue
        if str(source.get("source_mode", "pseudo_with_optional_gold")) == "gold_only":
            sources[dataset_id] = {"status": "disabled_for_stage"}
            continue
        raw_source_root = Path(str(source.get("root", ".")))
        source_root = _assert_directory(
            raw_source_root if raw_source_root.is_absolute() else root / raw_source_root,
            f"{dataset_id} source root",
        )
        manifest_path = _safe_relative_file(source_root, source["manifest"], f"{dataset_id} manifest")
        draft_path = _safe_relative_file(source_root, source["pseudo_labels"], f"{dataset_id} draft")
        crop_dir = _safe_relative_directory(source_root, source["crop_images_dir"], f"{dataset_id} crop_images_dir")
        manifests = _unique(read_jsonl(manifest_path), "crop_id", f"{dataset_id}:manifest")
        drafts = _unique(read_jsonl(draft_path), "crop_id", f"{dataset_id}:draft")
        if set(manifests) != set(drafts):
            raise GoldPipelineError(f"{dataset_id}: manifest/draft coverage mismatch")
        manifest_sha = sha256_file(manifest_path)
        draft_sha = sha256_file(draft_path)
        for source_crop_id in sorted(manifests):
            global_crop_id = f"{dataset_id}:{source_crop_id}"
            if global_crop_id in seen_global:
                raise GoldPipelineError(f"duplicate registry global_crop_id: {global_crop_id}")
            seen_global.add(global_crop_id)
            manifest = manifests[source_crop_id]
            crop_path = _assert_regular_file(
                crop_dir / Path(str(manifest.get("crop_path", ""))).name,
                f"{global_crop_id} crop",
            )
            rows.append(
                {
                    "schema_version": "pretrain_source_registry_v1",
                    "dataset_id": dataset_id,
                    "source_crop_id": source_crop_id,
                    "global_crop_id": global_crop_id,
                    "parent_manifest_path": str(manifest_path.resolve()),
                    "parent_manifest_sha256": manifest_sha,
                    "parent_draft_path": str(draft_path.resolve()),
                    "parent_draft_sha256": draft_sha,
                    "parent_crop_path": str(crop_path.resolve()),
                    "image_sha256": sha256_file(crop_path),
                }
            )
        sources[dataset_id] = {
            "status": "ok",
            "rows": len(manifests),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha,
            "draft_path": str(draft_path.resolve()),
            "draft_sha256": draft_sha,
        }
    if not rows:
        raise GoldPipelineError("pretrain source registry has no rows")
    registry_path = qc_dir / "pretrain_source_registry.jsonl"
    report_path = qc_dir / "pretrain_source_registry_report.json"
    atomic_write_jsonl(registry_path, sorted(rows, key=lambda row: row["global_crop_id"]))
    report = {
        "schema_version": "pretrain_source_registry_v1",
        "status": "ok",
        "rows": len(rows),
        "sources": sources,
        "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path), "count": len(rows)},
    }
    atomic_write_json(report_path, report)
    return report


def _resolve_request_path(value: Any, base: Path, field: str) -> Path:
    if not value:
        raise GoldPipelineError(f"selection row requires {field}")
    path = Path(str(value))
    return _assert_regular_file(path, field) if path.is_absolute() else _safe_relative_file(base, path, field)


def _materialize_rows_from_selection(
    request_path: Path, source_id: str, dataset_id: str, task_root: Path
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    requests = read_jsonl(request_path)
    if not requests:
        raise GoldPipelineError(f"empty selection request: {request_path}")
    cache: Dict[Path, Dict[str, Dict[str, Any]]] = {}
    digest_cache: Dict[Path, str] = {}
    manifests: List[Dict[str, Any]] = []
    drafts: List[Dict[str, Any]] = []
    image_dir = task_root / "02_roi_crops" / "images"
    seen_parent_ids: set[str] = set()
    source_kinds = {str(row.get("source_kind", "")) for row in requests if row.get("source_kind")}
    if len(source_kinds) > 1:
        raise GoldPipelineError(f"selection request mixes source_kind values: {sorted(source_kinds)}")

    for request in requests:
        parent_dataset = str(request.get("parent_dataset_id") or request.get("dataset_id") or "")
        parent_local = str(request.get("parent_source_crop_id") or request.get("source_crop_id") or "")
        parent_global = str(request.get("parent_global_crop_id") or request.get("global_crop_id") or "")
        if not parent_dataset or not parent_local or not parent_global:
            raise GoldPipelineError("selection row requires parent_dataset_id, parent_source_crop_id and parent_global_crop_id")
        if parent_global in seen_parent_ids:
            raise GoldPipelineError(f"selection request contains duplicate parent_global_crop_id: {parent_global}")
        seen_parent_ids.add(parent_global)
        manifest_path = _resolve_request_path(
            request.get("parent_manifest_path") or request.get("source_manifest_path"), request_path.parent, "parent_manifest_path"
        )
        draft_path = _resolve_request_path(
            request.get("parent_draft_path") or request.get("source_draft_path"), request_path.parent, "parent_draft_path"
        )
        expected_manifest_sha = str(
            request.get("parent_manifest_sha256") or request.get("source_manifest_sha256") or ""
        )
        expected_draft_sha = str(
            request.get("parent_draft_sha256") or request.get("source_draft_sha256") or ""
        )
        manifest_sha = digest_cache.get(manifest_path)
        if manifest_sha is None:
            manifest_sha = sha256_file(manifest_path)
            digest_cache[manifest_path] = manifest_sha
        draft_sha = digest_cache.get(draft_path)
        if draft_sha is None:
            draft_sha = sha256_file(draft_path)
            digest_cache[draft_path] = draft_sha
        if not expected_manifest_sha or manifest_sha.lower() != expected_manifest_sha.lower():
            raise GoldPipelineError(f"selection manifest SHA missing/mismatch for {parent_global}")
        if not expected_draft_sha or draft_sha.lower() != expected_draft_sha.lower():
            raise GoldPipelineError(f"selection draft SHA missing/mismatch for {parent_global}")
        if manifest_path not in cache:
            cache[manifest_path] = _unique(read_jsonl(manifest_path), "crop_id", str(manifest_path))
        if draft_path not in cache:
            cache[draft_path] = _unique(read_jsonl(draft_path), "crop_id", str(draft_path))
        manifest = cache[manifest_path].get(parent_local)
        draft = cache[draft_path].get(parent_local)
        if manifest is None or draft is None:
            raise GoldPipelineError(f"parent crop {parent_local} not covered by manifest and draft")
        conflicts = manifest_conflicts(draft, manifest)
        if conflicts:
            raise GoldPipelineError(f"parent manifest/draft conflict for {parent_global}: {conflicts}")
        parent_crop = _resolve_request_path(
            request.get("parent_crop_path") or request.get("source_crop_path") or request.get("crop_path"),
            request_path.parent,
            "parent_crop_path",
        )
        actual_sha = sha256_file(parent_crop)
        expected_sha = str(request.get("image_sha256") or request.get("parent_image_sha256") or "")
        if not expected_sha:
            raise GoldPipelineError(f"selection row {parent_global} requires image_sha256")
        if actual_sha.lower() != expected_sha.lower():
            raise GoldPipelineError(f"selection image SHA mismatch for {parent_global}")
        token = hashlib.sha256(parent_global.encode("utf-8")).hexdigest()[:20]
        palm_id = f"{source_id}:palm:{token}"
        crop_id = f"{palm_id}:crop"
        filename = f"{_safe_name(source_id)}_{token}.png"
        target = image_dir / filename
        _copy_or_link(parent_crop, target)

        new_manifest = copy.deepcopy(manifest)
        new_manifest.update(
            {
                "dataset_id": dataset_id,
                "source_crop_id": crop_id,
                "crop_id": crop_id,
                "palm_det_id": palm_id,
                "crop_path": f"02_roi_crops/images/{filename}",
                "image_sha256": actual_sha,
                "parent_dataset_id": parent_dataset,
                "parent_source_crop_id": parent_local,
                "parent_global_crop_id": parent_global,
                "parent_source_sequence_id": request.get("source_sequence_id"),
                "parent_source_frame_index": request.get("source_frame_index"),
            }
        )
        new_draft = copy.deepcopy(draft)
        teacher_handedness = copy.deepcopy(new_draft.get("handedness"))
        new_draft.update(
            {
                "dataset_id": dataset_id,
                "source_crop_id": crop_id,
                "crop_id": crop_id,
                "palm_det_id": palm_id,
                "hand_id": f"{crop_id}:hand" if bool((new_draft.get("hand_presence") or {}).get("present")) else None,
                "crop_path": f"02_roi_crops/images/{filename}",
                "image_sha256": actual_sha,
                "parent_dataset_id": parent_dataset,
                "parent_source_crop_id": parent_local,
                "parent_global_crop_id": parent_global,
                "teacher_handedness_audit": teacher_handedness,
                "handedness": {"label": "unknown", "score": None},
            }
        )
        manifests.append(new_manifest)
        drafts.append(new_draft)

    return manifests, drafts, {
        "selection_request": str(request_path),
        "selection_request_sha256": sha256_file(request_path),
        "source_kind": next(iter(source_kinds), None),
        "rows": len(requests),
    }


def _materialize_rows_from_native(
    raw_root: Path, source_id: str, dataset_id: str, task_root: Path
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    raw_root = _assert_directory(Path(raw_root), "native source root")
    manifest_path = _safe_relative_file(
        raw_root, "02_roi_crops/hand_roi_crops_manifest.jsonl", "native manifest"
    )
    draft_path = _safe_relative_file(
        raw_root, "02_roi_crops/hand_landmarks_autolabel_draft.jsonl", "native draft"
    )
    native_images = _safe_relative_directory(raw_root, "02_roi_crops/images", "native crop images")
    manifest_idx = _unique(read_jsonl(manifest_path), "crop_id", "native manifest")
    draft_idx = _unique(read_jsonl(draft_path), "crop_id", "native draft")
    if set(manifest_idx) != set(draft_idx):
        raise GoldPipelineError("native manifest/draft coverage mismatch")
    manifests: List[Dict[str, Any]] = []
    drafts: List[Dict[str, Any]] = []
    image_dir = task_root / "02_roi_crops" / "images"
    seen_names: set[str] = set()
    for index, parent_local in enumerate(sorted(manifest_idx)):
        manifest = copy.deepcopy(manifest_idx[parent_local])
        draft = copy.deepcopy(draft_idx[parent_local])
        original = Path(str(manifest.get("crop_path", "")))
        candidates: List[Path] = []
        if not original.is_absolute() and ".." not in original.parts:
            candidates.append(raw_root / original)
        candidates.append(native_images / original.name)
        parent_crop = next(
            (
                _assert_regular_file(value, f"native crop {parent_local}")
                for value in candidates
                if value.is_file()
            ),
            None,
        )
        if parent_crop is None:
            raise GoldPipelineError(f"native crop image missing for {parent_local}")
        filename = original.name or f"{_safe_name(source_id)}_{index:08d}.png"
        if filename in seen_names:
            raise GoldPipelineError(f"duplicate native crop basename: {filename}")
        seen_names.add(filename)
        target = image_dir / filename
        _copy_or_link(parent_crop, target)
        image_sha = sha256_file(target)
        token = hashlib.sha256(f"{source_id}\0{parent_local}".encode("utf-8")).hexdigest()[:20]
        palm_id = f"{source_id}:palm:{token}"
        crop_id = f"{palm_id}:crop"
        common = {
            "dataset_id": dataset_id,
            "source_crop_id": crop_id,
            "crop_id": crop_id,
            "palm_det_id": palm_id,
            "crop_path": f"02_roi_crops/images/{filename}",
            "image_sha256": image_sha,
            "parent_dataset_id": None,
            "parent_source_crop_id": None,
            "parent_global_crop_id": None,
            "native_source_root": str(raw_root),
            "native_source_crop_id": parent_local,
        }
        manifest.update(common)
        teacher_handedness = copy.deepcopy(draft.get("handedness"))
        draft.update(common)
        draft["hand_id"] = f"{crop_id}:hand" if bool((draft.get("hand_presence") or {}).get("present")) else None
        draft["teacher_handedness_audit"] = teacher_handedness
        draft["handedness"] = {"label": "unknown", "score": None}
        manifests.append(manifest)
        drafts.append(draft)
    return manifests, drafts, {
        "raw_source_root": str(raw_root),
        "manifest_sha256": sha256_file(manifest_path),
        "draft_sha256": sha256_file(draft_path),
        "rows": len(manifests),
    }


def export_finetune_gold(
    config_path: Path,
    *,
    source_id: str,
    source_mode: str,
    raw_source_root: Path | None = None,
    selection_request: Path | None = None,
    source_kind: str | None = None,
) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    workspace = _workspace(cfg, config_path)
    if source_mode not in {"selection_subset", "native_existing"}:
        raise GoldPipelineError("source_mode must be selection_subset or native_existing")
    if not source_id or source_id in {".", ".."} or Path(source_id).name != source_id:
        raise GoldPipelineError(f"invalid source_id: {source_id!r}")
    dataset_id = str(cfg.get("dataset_id") or source_id)
    task_root = workspace / "cvat" / source_id
    published_root = workspace / "sources" / "gold" / source_id
    if task_root.exists() or published_root.exists():
        raise GoldPipelineError(f"source/task already exists: {source_id}")
    task_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{_safe_name(source_id)}.", dir=task_root.parent))
    try:
        if source_mode == "selection_subset":
            request = selection_request or workspace / "mining" / source_id / "selection_request.jsonl"
            request = _assert_regular_file(request, "selection request")
            manifests, drafts, provenance = _materialize_rows_from_selection(request, source_id, dataset_id, temp_root)
            resolved_kind = source_kind or provenance.get("source_kind")
            if resolved_kind not in {"reviewed_hard_gold", "disagreement_gold"}:
                raise GoldPipelineError("selection_subset requires source_kind reviewed_hard_gold or disagreement_gold")
        else:
            if raw_source_root is None:
                raise GoldPipelineError("native_existing requires raw_source_root")
            manifests, drafts, provenance = _materialize_rows_from_native(
                raw_source_root, source_id, dataset_id, temp_root
            )
            resolved_kind = source_kind or "new_recorded_gold"
            if resolved_kind != "new_recorded_gold":
                raise GoldPipelineError("native_existing requires source_kind new_recorded_gold")
        if resolved_kind not in ALLOWED_SOURCE_KINDS:
            raise GoldPipelineError(f"unsupported source_kind: {resolved_kind}")
        manifest_path = temp_root / "02_roi_crops" / "hand_roi_crops_manifest.jsonl"
        draft_path = temp_root / "02_roi_crops" / "hand_landmarks_autolabel_draft.jsonl"
        atomic_write_jsonl(manifest_path, manifests)
        atomic_write_jsonl(draft_path, drafts)
        crop_hash_path = temp_root / "qc" / "crop_images_sha256.jsonl"
        _write_hash_manifest(
            list((temp_root / "02_roi_crops" / "images").iterdir()), temp_root, crop_hash_path
        )
        dynamic = _dynamic_cvat_config(cfg, temp_root)
        xml_path = temp_root / "cvat_autolabel.xml"
        export_stats = export_cvat_xml(manifests, drafts, _repo_root(config_path), xml_path, dynamic)
        export_stats["source_id"] = source_id
        export_stats["source_mode"] = source_mode
        export_stats["source_kind"] = resolved_kind
        atomic_write_json(temp_root / "qc" / "cvat_export_stats.json", export_stats)
        task_descriptor = {
            "schema_version": "finetune_gold_task_v1",
            "status": "awaiting_human_review",
            "source_id": source_id,
            "dataset_id": dataset_id,
            "source_kind": resolved_kind,
            "source_mode": source_mode,
            "created_at": _utc_now(),
            "producer": "hlmf_finetune_gold",
            "producer_version": _git_version(_repo_root(config_path)),
            "artifacts": {
                "manifest": _artifact(manifest_path, temp_root, count=len(manifests)),
                "draft": _artifact(draft_path, temp_root, count=len(drafts)),
                "crop_images_sha256": _artifact(crop_hash_path, temp_root, count=len(manifests)),
                "cvat_xml": _artifact(xml_path, temp_root, count=len(manifests)),
                "export_report": _artifact(temp_root / "qc" / "cvat_export_stats.json", temp_root),
            },
            "reviewed_xml": "reviewed.xml",
            "provenance": provenance,
        }
        atomic_write_json(temp_root / "task_descriptor.json", task_descriptor)
        os.replace(temp_root, task_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return task_descriptor


def _validate_task_artifacts(task_root: Path, descriptor: Mapping[str, Any]) -> None:
    for name, artifact in (descriptor.get("artifacts") or {}).items():
        path = _safe_relative_file(task_root, artifact.get("path"), f"task artifact {name}")
        if sha256_file(path).lower() != str(artifact.get("sha256", "")).lower():
            raise GoldPipelineError(f"task artifact SHA mismatch: {name}")
        if path.suffix == ".jsonl" and artifact.get("count") is not None:
            if len(read_jsonl(path)) != int(artifact["count"]):
                raise GoldPipelineError(f"task artifact count mismatch: {name}")
    crop_hash_artifact = (descriptor.get("artifacts") or {}).get("crop_images_sha256")
    if not isinstance(crop_hash_artifact, Mapping):
        raise GoldPipelineError("task descriptor requires crop_images_sha256 artifact")
    crop_hash_path = _safe_relative_file(
        task_root, crop_hash_artifact.get("path"), "task crop image hash manifest"
    )
    for row in read_jsonl(crop_hash_path):
        crop_path = _safe_relative_file(task_root, row.get("path"), "task crop image")
        if sha256_file(crop_path).lower() != str(row.get("sha256", "")).lower():
            raise GoldPipelineError(f"task crop image SHA mismatch: {row.get('path')}")


def _source_descriptor(
    source_root: Path,
    *,
    source_id: str,
    dataset_id: str,
    source_kind: str,
    source_mode: str,
    handedness_policy: str,
    producer: str,
    producer_version: str,
    parent_pretrain_id: str | None,
    input_sha256: Mapping[str, Any],
    manifest_count: int,
    gold_count: int,
    ignored_count: int,
    source_image_count: int | None = None,
    source_images_root: Path | None = None,
) -> Dict[str, Any]:
    manifest = source_root / "02_roi_crops" / "hand_roi_crops_manifest.jsonl"
    gold = source_root / "03_reviewed" / "hand_landmarks_reviewed.jsonl"
    ignored = source_root / "03_reviewed" / "ignored.jsonl"
    crop_hashes = source_root / "qc" / "crop_images_sha256.jsonl"
    report = source_root / "qc" / "gold_source_report.json"
    crop_rows = read_jsonl(crop_hashes)
    aggregate_hash = hashlib.sha256()
    for row in sorted(crop_rows, key=lambda item: item["path"]):
        aggregate_hash.update(f"{row['path']}:{row['sha256']}\n".encode("utf-8"))
    artifacts: Dict[str, Any] = {
        "manifest": _artifact(manifest, source_root, count=manifest_count),
        "crop_images": {
            "root": "02_roi_crops/images",
            "sha256_manifest": "qc/crop_images_sha256.jsonl",
            "manifest_sha256": sha256_file(crop_hashes),
            "aggregate_sha256": aggregate_hash.hexdigest(),
            "count": len(crop_rows),
        },
        "gold_labels": _artifact(gold, source_root, count=gold_count),
        "ignored_sidecar": _artifact(ignored, source_root, count=ignored_count),
        "qc_report": _artifact(report, source_root),
    }
    source_hashes = source_root / "qc" / "source_images_sha256.jsonl"
    if source_hashes.is_file():
        source_rows = read_jsonl(source_hashes)
        source_aggregate = hashlib.sha256()
        for row in sorted(source_rows, key=lambda item: item["path"]):
            source_aggregate.update(f"{row['path']}:{row['sha256']}\n".encode("utf-8"))
        artifacts["source_images"] = {
            "root": str(source_images_root).replace("\\", "/") if source_images_root is not None else None,
            "read_only": True,
            "sha256_manifest": "qc/source_images_sha256.jsonl",
            "manifest_sha256": sha256_file(source_hashes),
            "aggregate_sha256": source_aggregate.hexdigest(),
            "count": source_image_count if source_image_count is not None else len(source_rows),
        }
    task_descriptor = source_root / "audit" / "task_descriptor.json"
    if task_descriptor.is_file():
        artifacts["task_descriptor"] = _artifact(task_descriptor, source_root)
    selection = source_root / "audit" / "selection_request.jsonl"
    if selection.is_file():
        artifacts["selection_audit"] = _artifact(selection, source_root, count=len(read_jsonl(selection)))
    return {
        "schema_version": "finetune_source_v1",
        "source_id": source_id,
        "dataset_id": dataset_id,
        "source_kind": source_kind,
        "source_mode": source_mode,
        "producer": producer,
        "producer_version": producer_version,
        "created_at": _utc_now(),
        "parent_pretrain_id": parent_pretrain_id,
        "enabled_stages": ["finetune"],
        "supervision_tier": "gold",
        "handedness_policy": handedness_policy,
        "input_sha256": dict(input_sha256),
        "artifacts": artifacts,
        "counts": {
            "manifest": manifest_count,
            "gold_labels": gold_count,
            "included": gold_count - ignored_count,
            "ignored": ignored_count,
        },
    }


def import_finetune_gold(config_path: Path, *, source_id: str, dry_run: bool = False) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    workspace = _workspace(cfg, config_path)
    task_root = workspace / "cvat" / source_id
    descriptor_path = _safe_relative_file(task_root, "task_descriptor.json", "task descriptor")
    with descriptor_path.open("r", encoding="utf-8") as handle:
        task = json.load(handle)
    if task.get("schema_version") != "finetune_gold_task_v1" or task.get("source_id") != source_id:
        raise GoldPipelineError("task descriptor identity/schema mismatch")
    _validate_task_artifacts(task_root, task)
    reviewed_xml = _safe_relative_file(
        task_root, task.get("reviewed_xml", "reviewed.xml"), "reviewed XML"
    )
    manifests = read_jsonl(task_root / task["artifacts"]["manifest"]["path"])
    drafts = read_jsonl(task_root / task["artifacts"]["draft"]["path"])
    dynamic = _dynamic_cvat_config(cfg, task_root)
    rows, stats = import_cvat_xml(reviewed_xml, manifests, drafts, dynamic)
    row_by_id = {str(row.get("crop_id")): row for row in rows}
    blocking = []
    for item in stats.get("errors", []):
        crop_id = str(item.get("crop_id", "")) if isinstance(item, Mapping) else ""
        if crop_id and bool(row_by_id.get(crop_id, {}).get("ignore_for_training")):
            continue
        blocking.append(item)
    coverage = stats.get("coverage", {})
    stats["strict_gate"] = {
        "blocking_errors": blocking,
        "blocking_error_count": len(blocking),
        "status": "failed"
        if blocking or coverage.get("missing_from_xml") or len(rows) != len(manifests)
        else "ok",
    }
    atomic_write_json(task_root / "qc" / "cvat_import_stats.json", stats)
    if blocking or coverage.get("missing_from_xml") or len(rows) != len(manifests):
        raise GoldPipelineError(
            f"strict CVAT import failed: blocking_errors={len(blocking)} missing={coverage.get('missing_from_xml', 0)}"
        )
    task_sha = sha256_file(descriptor_path)
    for row in rows:
        row["dataset_id"] = task["dataset_id"]
        row["source_crop_id"] = row["crop_id"]
        row["annotation_provenance"] = "human_gold"
        row["supervision_tier"] = "gold"
        row["training_stage"] = "finetune"
        row.setdefault("finetune_review", {})
        row["finetune_review"].update(
            {"task_descriptor_sha256": task_sha, "source_descriptor_sha256": None}
        )

    if dry_run:
        return {
            "schema_version": "finetune_gold_import_preflight_v1",
            "status": "ok",
            "source_id": source_id,
            "rows": len(rows),
            "ignored": sum(bool(row.get("ignore_for_training")) for row in rows),
            "task_descriptor_sha256": task_sha,
            "reviewed_xml_sha256": sha256_file(reviewed_xml),
        }

    source_root = workspace / "sources" / "gold" / source_id
    if source_root.exists():
        raise GoldPipelineError(f"published source already exists: {source_root}")
    source_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{_safe_name(source_id)}.", dir=source_root.parent))
    try:
        (staging / "02_roi_crops" / "images").mkdir(parents=True, exist_ok=True)
        for manifest in manifests:
            filename = Path(str(manifest["crop_path"])).name
            _copy_or_link(task_root / "02_roi_crops" / "images" / filename, staging / "02_roi_crops" / "images" / filename)
        atomic_write_jsonl(staging / "02_roi_crops" / "hand_roi_crops_manifest.jsonl", manifests)
        atomic_write_jsonl(staging / "02_roi_crops" / "hand_landmarks_autolabel_draft.jsonl", drafts)
        atomic_write_jsonl(staging / "03_reviewed" / "hand_landmarks_reviewed.jsonl", rows)
        ignored = [row for row in rows if bool(row.get("ignore_for_training"))]
        atomic_write_jsonl(staging / "03_reviewed" / "ignored.jsonl", ignored)
        _copy_or_link(descriptor_path, staging / "audit" / "task_descriptor.json")
        provenance = task.get("provenance") or {}
        selection_path = provenance.get("selection_request")
        if selection_path:
            _copy_or_link(Path(selection_path), staging / "audit" / "selection_request.jsonl")
        crop_files = list((staging / "02_roi_crops" / "images").iterdir())
        _write_hash_manifest(crop_files, staging, staging / "qc" / "crop_images_sha256.jsonl")
        report = {
            "schema_version": "finetune_gold_source_report_v1",
            "status": "ok",
            "source_id": source_id,
            "source_mode": task["source_mode"],
            "source_kind": task["source_kind"],
            "counts": {
                "manifest": len(manifests),
                "reviewed": len(rows),
                "included": len(rows) - len(ignored),
                "ignored": len(ignored),
                "positive": sum(bool((row.get("hand_presence") or {}).get("present")) for row in rows if not row.get("ignore_for_training")),
                "negative": sum(not bool((row.get("hand_presence") or {}).get("present")) for row in rows if not row.get("ignore_for_training")),
            },
            "cvat_import": stats,
            "reviewed_xml_sha256": sha256_file(reviewed_xml),
        }
        atomic_write_json(staging / "qc" / "gold_source_report.json", report)
        source_descriptor = _source_descriptor(
            staging,
            source_id=source_id,
            dataset_id=str(task["dataset_id"]),
            source_kind=str(task["source_kind"]),
            source_mode="reviewed_gold",
            handedness_policy="optional_per_row",
            producer="hlmf_strict_cvat",
            producer_version=_git_version(_repo_root(config_path)),
            parent_pretrain_id=cfg.get("parent_pretrain_id"),
            input_sha256={"task_descriptor": task_sha, "reviewed_xml": sha256_file(reviewed_xml)},
            manifest_count=len(manifests),
            gold_count=len(rows),
            ignored_count=len(ignored),
        )
        atomic_write_json(staging / "finetune_source.json", source_descriptor)
        os.replace(staging, source_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return source_descriptor


def import_all_finetune_gold(config_path: Path) -> Dict[str, Any]:
    """Preflight every returned task, then publish each source and one batch report."""
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    workspace = _workspace(cfg, config_path)
    task_descriptors = sorted((workspace / "cvat").glob("*/task_descriptor.json"))
    if not task_descriptors:
        raise GoldPipelineError(f"no finetune Gold tasks found under {workspace / 'cvat'}")
    pending: List[str] = []
    already_published: List[str] = []
    for path in task_descriptors:
        source_id = path.parent.name
        if (workspace / "sources" / "gold" / source_id / "finetune_source.json").is_file():
            already_published.append(source_id)
        else:
            pending.append(source_id)
    if not pending:
        raise GoldPipelineError("all discovered finetune Gold tasks are already published")
    preflight = [import_finetune_gold(config_path, source_id=source_id, dry_run=True) for source_id in pending]
    published = [import_finetune_gold(config_path, source_id=source_id) for source_id in pending]
    report = {
        "schema_version": "finetune_gold_batch_import_v1",
        "status": "ok",
        "created_at": _utc_now(),
        "preflight": preflight,
        "published": [
            {
                "source_id": item["source_id"],
                "source_kind": item["source_kind"],
                "counts": item["counts"],
                "descriptor_sha256": sha256_file(
                    workspace / "sources" / "gold" / item["source_id"] / "finetune_source.json"
                ),
            }
            for item in published
        ],
        "already_published": already_published,
    }
    atomic_write_json(workspace / "cvat" / "batch_import_report.json", report)
    return report


def parse_dragon_hand_annotations(path: Path) -> Dict[str, List[List[float]]]:
    records: Dict[str, List[List[float]]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, 1):
            tokens = raw.split()
            if len(tokens) < 2:
                raise GoldPipelineError(f"invalid Dragon hand row {line_no}")
            name, count_text = tokens[0], tokens[1]
            if name in records:
                raise GoldPipelineError(f"duplicate Dragon hand key: {name}")
            try:
                count = int(count_text)
                values = [float(value) for value in tokens[2:]]
            except ValueError as exc:
                raise GoldPipelineError(f"invalid Dragon hand numeric value at line {line_no}") from exc
            if count not in {0, 1, 2} or len(values) != count * 42:
                raise GoldPipelineError(f"Dragon hand row {line_no}: count/value mismatch")
            records[name] = [values[index * 42 : (index + 1) * 42] for index in range(count)]
    return records


def parse_dragon_palm_annotations(path: Path) -> Dict[str, List[List[float]]]:
    records: Dict[str, List[List[float]]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, 1):
            tokens = raw.split()
            if not tokens:
                continue
            if tokens[0].rstrip(":").lower() == "frame":
                tokens = tokens[1:]
            if not tokens:
                raise GoldPipelineError(f"invalid Dragon palm row {line_no}")
            name = tokens[0]
            if name in records:
                raise GoldPipelineError(f"duplicate Dragon palm key: {name}")
            try:
                values = [float(value) for value in tokens[1:]]
            except ValueError as exc:
                raise GoldPipelineError(f"invalid Dragon palm numeric value at line {line_no}") from exc
            if len(values) % 8 or len(values) // 8 not in {0, 1, 2}:
                raise GoldPipelineError(f"Dragon palm row {line_no}: value count must be 0, 8 or 16")
            records[name] = [values[index : index + 8] for index in range(0, len(values), 8)]
    return records


def match_dragon_hands_to_palms(hands: Sequence[Sequence[float]], palms: Sequence[Sequence[float]]) -> List[int] | None:
    if len(hands) not in {1, 2} or len(hands) > len(palms):
        return None
    matches: List[int] = []
    for hand in hands:
        xs = [float(hand[index]) for index in range(0, 42, 2)]
        ys = [float(hand[index]) for index in range(1, 42, 2)]
        center = (sum(xs) / 21.0, sum(ys) / 21.0)
        containing = [
            index
            for index, palm in enumerate(palms)
            if min(palm[0], palm[2]) <= center[0] <= max(palm[0], palm[2])
            and min(palm[1], palm[3]) <= center[1] <= max(palm[1], palm[3])
        ]
        if len(containing) != 1:
            return None
        matches.append(containing[0])
    if len(set(matches)) != len(matches):
        return None
    return matches


def _load_dragon_image(path: Path, *, expected_orientation: int, logical_size: tuple[int, int]) -> tuple[np.ndarray, int]:
    with Image.open(path) as image:
        orientation = int(image.getexif().get(274, 1))
        if orientation != expected_orientation:
            raise GoldPipelineError(f"Dragon image {path.name}: EXIF orientation {orientation}, expected {expected_orientation}")
        logical = ImageOps.exif_transpose(image).convert("RGB")
        if logical.size != logical_size:
            raise GoldPipelineError(f"Dragon image {path.name}: logical size {logical.size}, expected {logical_size}")
        array = np.asarray(logical)
    if array.ndim != 3 or array.shape[2] != 3:
        raise GoldPipelineError(f"Dragon image {path.name}: RGB image required")
    if not (np.array_equal(array[:, :, 0], array[:, :, 1]) and np.array_equal(array[:, :, 0], array[:, :, 2])):
        raise GoldPipelineError(f"Dragon image {path.name}: RGB channels are not identical grayscale")
    return array[:, :, 0], orientation


def _project_image_landmarks_to_crop(
    points: Sequence[tuple[float, float]], corners: Sequence[Sequence[float]]
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    corner_array = np.asarray(corners, dtype=np.float64)
    basis = np.column_stack((corner_array[1] - corner_array[0], corner_array[3] - corner_array[0]))
    if abs(float(np.linalg.det(basis))) < 1e-12:
        raise GoldPipelineError("degenerate ROI transform")
    crop_norm: List[Dict[str, Any]] = []
    crop_px: List[Dict[str, Any]] = []
    outside = False
    for index, point in enumerate(points):
        uv = np.linalg.solve(basis, np.asarray(point, dtype=np.float64) - corner_array[0])
        u, v = float(uv[0]), float(uv[1])
        outside = outside or not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0)
        crop_norm.append({"id": index, "x": u, "y": v})
        crop_px.append({"id": index, "x": u * 255.0, "y": v * 255.0})
    return crop_norm, crop_px, outside


def _draw_dragon_overlay(path: Path, crop: np.ndarray, points: Sequence[Mapping[str, Any]]) -> None:
    canvas = cv2.cvtColor(to_uint8_gray(crop), cv2.COLOR_GRAY2BGR)
    edges = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))
    coords = [(int(round(float(point["x"]))), int(round(float(point["y"])))) for point in points]
    for first, second in edges:
        cv2.line(canvas, coords[first], coords[second], (0, 220, 0), 1, cv2.LINE_AA)
    for index, coord in enumerate(coords):
        cv2.circle(canvas, coord, 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, str(index), coord, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1, cv2.LINE_AA)
    if not write_image(path, canvas):
        raise GoldPipelineError(f"failed to write overlay: {path}")


def prepare_dragon_gold(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    workspace = _workspace(cfg, config_path)
    dragon = cfg.get("dragon") or {}
    raw_value = Path(str(dragon["raw_root"]))
    raw_root = _assert_directory(
        raw_value if raw_value.is_absolute() else _repo_root(config_path) / raw_value,
        "Dragon raw root",
    )
    source_id = str(dragon["source_id"])
    dataset_id = str(dragon.get("dataset_id", source_id))
    source_root = workspace / "sources" / "gold" / source_id
    if source_root.exists():
        raise GoldPipelineError(f"Dragon source already exists: {source_root}")
    hand_path = _safe_relative_file(
        raw_root, dragon.get("hand_annotations", "annotations_hand.txt"), "Dragon hand annotations"
    )
    palm_path = _safe_relative_file(
        raw_root, dragon.get("palm_annotations", "annotations_palm.txt"), "Dragon palm annotations"
    )
    readme_path = _safe_relative_file(raw_root, dragon.get("readme", "README.md"), "Dragon README")
    expected_sha = dragon.get("expected_sha256") or {}
    actual_sha = {"annotations_hand": sha256_file(hand_path), "annotations_palm": sha256_file(palm_path), "readme": sha256_file(readme_path)}
    for name, expected in expected_sha.items():
        if name in actual_sha and str(expected).lower() != actual_sha[name].lower():
            raise GoldPipelineError(f"Dragon input SHA mismatch: {name}")
    hands = parse_dragon_hand_annotations(hand_path)
    palms = parse_dragon_palm_annotations(palm_path)
    if set(hands) != set(palms):
        raise GoldPipelineError("Dragon hand/palm annotation key sets differ")
    images_dir = _safe_relative_directory(raw_root, dragon.get("images_dir", "images"), "Dragon images")
    image_files = sorted(
        _assert_regular_file(path, "Dragon source image")
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_by_name = {path.name: path for path in image_files}
    if len(image_by_name) != len(image_files):
        raise GoldPipelineError("Dragon image basenames are not unique")

    source_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{_safe_name(source_id)}.", dir=source_root.parent))
    manifests: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    source_image_paths: Dict[str, Path] = {}
    overlay_candidates: List[tuple[str, Path, List[Dict[str, Any]]]] = []
    logical_width, logical_height = [int(value) for value in dragon.get("logical_size", [1280, 720])]
    hand_roi = cfg.get("hand_roi") or {}
    try:
        for image_name in hands:
            image_hands, image_palms = hands[image_name], palms[image_name]
            if not image_hands:
                rejects.append({"image": image_name, "reason": "HAND_COUNT_ZERO"})
                continue
            image_path = image_by_name.get(image_name)
            if image_path is None:
                rejects.append({"image": image_name, "reason": "SOURCE_IMAGE_MISSING"})
                continue
            matches = match_dragon_hands_to_palms(image_hands, image_palms)
            if matches is None:
                reason = "HAND_COUNT_EXCEEDS_PALM_COUNT" if len(image_hands) > len(image_palms) else "HAND_PALM_MATCH_AMBIGUOUS"
                rejects.append({"image": image_name, "reason": reason, "hands": len(image_hands), "palms": len(image_palms)})
                continue
            gray, orientation = _load_dragon_image(
                image_path,
                expected_orientation=int(dragon.get("expected_exif_orientation", 6)),
                logical_size=(logical_width, logical_height),
            )
            source_image_paths[image_name] = image_path
            for hand_index, palm_index in enumerate(matches):
                palm = image_palms[palm_index]
                suffix = "A" if palm_index == 0 else "B"
                palm_id = f"{Path(image_name).stem}:palm{suffix}"
                crop_id = f"{palm_id}:crop"
                detection = {
                    "bbox_px": [palm[0] * logical_width, palm[1] * logical_height, palm[2] * logical_width, palm[3] * logical_height],
                    "keypoints_px": {
                        "p0": [palm[4] * logical_width, palm[5] * logical_height],
                        "p9": [palm[6] * logical_width, palm[7] * logical_height],
                    },
                }
                rect = build_roi_rect_from_palm(
                    detection,
                    logical_width,
                    logical_height,
                    scale_x=float(hand_roi.get("scale_x", 1.8)),
                    scale_y=float(hand_roi.get("scale_y", 1.8)),
                    shift_x=float(hand_roi.get("shift_x", 0.0)),
                    shift_y=float(hand_roi.get("shift_y", -0.1)),
                )
                crop, corners = crop_image_by_roi(gray, rect, 256, 256)
                filename = f"{_safe_name(palm_id)}_crop.png"
                crop_path = staging / "02_roi_crops" / "images" / filename
                if not write_image(crop_path, crop):
                    raise GoldPipelineError(f"failed to write Dragon crop: {crop_path}")
                crop_sha = sha256_file(crop_path)
                hand = image_hands[hand_index]
                image_points = [(hand[index] * logical_width, hand[index + 1] * logical_height) for index in range(0, 42, 2)]
                crop_norm, crop_px, outside = _project_image_landmarks_to_crop(image_points, corners.tolist())
                manifest = {
                    "dataset_id": dataset_id,
                    "source_crop_id": crop_id,
                    "crop_id": crop_id,
                    "image": image_name,
                    "palm_det_id": palm_id,
                    "palm_valid": True,
                    "palm_score": 0.5,
                    "palm_score_observed": False,
                    "palm_score_source": "legacy_export_missing",
                    "crop_path": f"02_roi_crops/images/{filename}",
                    "image_sha256": crop_sha,
                    "source_image_sha256": sha256_file(image_path),
                    "roi_rect": rect,
                    "roi_corners_px": [[float(x), float(y)] for x, y in corners.tolist()],
                    "output_size": [256, 256],
                    "parent_dataset_id": None,
                    "parent_source_crop_id": None,
                    "parent_global_crop_id": None,
                    "dragon_hand_index": hand_index,
                    "dragon_palm_index": palm_index,
                    "source_exif_orientation": orientation,
                }
                label = copy.deepcopy(manifest)
                label.update(
                    {
                        "hand_id": f"{crop_id}:hand",
                        "hand_presence": {"present": True},
                        "handedness": {"label": "unknown", "score": None},
                        "landmarks_crop_norm": crop_norm,
                        "landmarks_crop_px": crop_px,
                        "landmarks_image_px": [
                            {"id": index, "x": float(point[0]), "y": float(point[1])}
                            for index, point in enumerate(image_points)
                        ],
                        "width": 256,
                        "height": 256,
                        "source_image_width": logical_width,
                        "source_image_height": logical_height,
                        "source": "dragon_human_gold",
                        "annotation_provenance": "human_gold",
                        "supervision_tier": "gold",
                        "training_stage": "finetune",
                        "ignore_for_training": outside,
                        "ignore_reason": "LANDMARK_OUTSIDE_ROI" if outside else None,
                        "finetune_review": {
                            "presence_decision": "hand",
                            "handedness_decision": "unknown",
                            "task_descriptor_sha256": None,
                            "source_descriptor_sha256": None,
                        },
                    }
                )
                manifests.append(manifest)
                labels.append(label)
                if not outside:
                    overlay_candidates.append((crop_id, crop_path, crop_px))

        expected_counts = dragon.get("expected_counts") or {}
        observed = {
            "images": len(image_files),
            "annotation_rows": len(hands),
            "unannotated_images": len(image_files) - len(hands),
            "p0_images": sum(not value for value in hands.values()),
            "raw_hands": sum(len(value) for value in hands.values()),
            "matched_images": len(source_image_paths),
            "matched_rois": len(manifests),
            "included": sum(not row["ignore_for_training"] for row in labels),
            "ignored": sum(bool(row["ignore_for_training"]) for row in labels),
        }
        for key, expected in expected_counts.items():
            if key in observed and int(expected) != int(observed[key]):
                raise GoldPipelineError(f"Dragon count mismatch {key}: {observed[key]} != {expected}")
        manifests.sort(key=lambda row: row["crop_id"])
        labels.sort(key=lambda row: row["crop_id"])
        ignored = [row for row in labels if row["ignore_for_training"]]
        atomic_write_jsonl(staging / "02_roi_crops" / "hand_roi_crops_manifest.jsonl", manifests)
        atomic_write_jsonl(staging / "03_reviewed" / "hand_landmarks_reviewed.jsonl", labels)
        atomic_write_jsonl(staging / "03_reviewed" / "ignored.jsonl", ignored)
        atomic_write_jsonl(staging / "audit" / "dragon_rejected.jsonl", rejects)
        source_hash_rows = []
        for image_name, path in sorted(source_image_paths.items()):
            packaged_source = staging / "source_images" / image_name
            _copy_or_link(path, packaged_source)
            source_hash_rows.append({"path": image_name, "sha256": sha256_file(packaged_source)})
        atomic_write_jsonl(staging / "qc" / "source_images_sha256.jsonl", source_hash_rows)
        crop_files = list((staging / "02_roi_crops" / "images").iterdir())
        _write_hash_manifest(crop_files, staging, staging / "qc" / "crop_images_sha256.jsonl")
        overlay_count = int(dragon.get("overlay_count", 64))
        selected_overlays = sorted(
            overlay_candidates,
            key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
        )[:overlay_count]
        for crop_id, crop_path, points in selected_overlays:
            _draw_dragon_overlay(staging / "qc" / "overlays" / f"{_safe_name(crop_id)}.png", read_image(crop_path), points)
        report = {
            "schema_version": "dragon_gold_import_v1",
            "status": "ok",
            "source_id": source_id,
            "counts": observed,
            "reject_reasons": dict(Counter(row["reason"] for row in rejects)),
            "input_sha256": actual_sha,
            "roi": {
                "scale_x": float(hand_roi.get("scale_x", 1.8)),
                "scale_y": float(hand_roi.get("scale_y", 1.8)),
                "shift_x": float(hand_roi.get("shift_x", 0.0)),
                "shift_y": float(hand_roi.get("shift_y", -0.1)),
                "output_size": [256, 256],
            },
            "palm_score": {"value": 0.5, "observed": False, "source": "legacy_export_missing"},
            "overlays": len(selected_overlays),
        }
        atomic_write_json(staging / "qc" / "gold_source_report.json", report)
        descriptor = _source_descriptor(
            staging,
            source_id=source_id,
            dataset_id=dataset_id,
            source_kind="external_gold",
            source_mode="gold_only",
            handedness_policy="unavailable",
            producer="hlmf_dragon_adapter",
            producer_version=_git_version(_repo_root(config_path)),
            parent_pretrain_id=None,
            input_sha256=actual_sha,
            manifest_count=len(manifests),
            gold_count=len(labels),
            ignored_count=len(ignored),
            source_image_count=len(source_image_paths),
            source_images_root=Path("source_images"),
        )
        atomic_write_json(staging / "finetune_source.json", descriptor)
        os.replace(staging, source_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return descriptor


def _validate_source_descriptor(source_root: Path, descriptor: Mapping[str, Any]) -> None:
    if descriptor.get("schema_version") != "finetune_source_v1":
        raise GoldPipelineError(f"invalid source descriptor schema: {source_root}")
    if str(descriptor.get("source_id", "")) != source_root.name:
        raise GoldPipelineError(f"source_id does not match source directory: {source_root}")
    if descriptor.get("source_kind") not in ALLOWED_SOURCE_KINDS:
        raise GoldPipelineError(f"invalid source_kind: {descriptor.get('source_kind')}")
    if descriptor.get("supervision_tier") != "gold" or descriptor.get("enabled_stages") != ["finetune"]:
        raise GoldPipelineError(f"source is not finetune-only Gold: {source_root}")
    if descriptor.get("handedness_policy") not in {"unavailable", "optional_per_row", "required"}:
        raise GoldPipelineError(f"invalid handedness_policy: {source_root}")
    for name in ("manifest", "gold_labels", "ignored_sidecar", "qc_report"):
        artifact = (descriptor.get("artifacts") or {}).get(name)
        if not isinstance(artifact, Mapping):
            raise GoldPipelineError(f"source descriptor missing artifact: {name}")
        path = _safe_relative_file(source_root, artifact.get("path"), f"source artifact {name}")
        if sha256_file(path).lower() != str(artifact.get("sha256", "")).lower():
            raise GoldPipelineError(f"source artifact SHA mismatch: {name}")
        if artifact.get("count") is not None and path.suffix == ".jsonl":
            if len(read_jsonl(path)) != int(artifact["count"]):
                raise GoldPipelineError(f"source artifact count mismatch: {name}")
    crop_artifact = (descriptor.get("artifacts") or {}).get("crop_images") or {}
    crop_root = _safe_relative_directory(
        source_root, crop_artifact.get("root"), "authenticated crop image root"
    )
    hash_path = _safe_relative_file(
        source_root, crop_artifact.get("sha256_manifest"), "crop image hash manifest"
    )
    if sha256_file(hash_path).lower() != str(crop_artifact.get("manifest_sha256", "")).lower():
        raise GoldPipelineError("crop image hash manifest SHA mismatch")
    crop_hash_rows = read_jsonl(hash_path)
    if len(crop_hash_rows) != int(crop_artifact.get("count", -1)):
        raise GoldPipelineError("crop image hash manifest count mismatch")
    crop_aggregate = hashlib.sha256()
    for row in sorted(crop_hash_rows, key=lambda item: item["path"]):
        crop_aggregate.update(f"{row['path']}:{row['sha256']}\n".encode("utf-8"))
        image = _safe_relative_file(source_root, row.get("path"), "authenticated crop image")
        try:
            image.relative_to(crop_root)
        except ValueError as exc:
            raise GoldPipelineError(f"crop image is outside declared crop root: {row.get('path')}") from exc
        if sha256_file(image).lower() != str(row["sha256"]).lower():
            raise GoldPipelineError(f"crop image SHA mismatch: {row['path']}")
    if crop_aggregate.hexdigest().lower() != str(crop_artifact.get("aggregate_sha256", "")).lower():
        raise GoldPipelineError("crop image aggregate SHA mismatch")
    source_artifact = (descriptor.get("artifacts") or {}).get("source_images")
    if source_artifact:
        root_reference = Path(str(source_artifact.get("root", "")))
        if (
            root_reference.is_absolute()
            or not str(root_reference)
            or ".." in root_reference.parts
            or source_artifact.get("read_only") is not True
        ):
            raise GoldPipelineError("source_images requires a safe package-relative read-only-declared root")
        raw_physical_root = source_root / root_reference
        if raw_physical_root.is_symlink() or not raw_physical_root.is_dir():
            raise GoldPipelineError("source_images package root is missing or a symlink")
        physical_root = raw_physical_root.resolve()
        try:
            physical_root.relative_to(source_root.resolve())
        except ValueError as exc:
            raise GoldPipelineError("source_images package root escapes source root") from exc
        source_hash_path = _safe_relative_file(
            source_root, source_artifact.get("sha256_manifest"), "source image hash manifest"
        )
        if sha256_file(source_hash_path).lower() != str(source_artifact.get("manifest_sha256", "")).lower():
            raise GoldPipelineError("source image hash manifest SHA mismatch")
        source_rows = read_jsonl(source_hash_path)
        if len(source_rows) != int(source_artifact.get("count", -1)):
            raise GoldPipelineError("source image hash manifest count mismatch")
        source_aggregate = hashlib.sha256()
        for row in sorted(source_rows, key=lambda item: item["path"]):
            source_aggregate.update(f"{row['path']}:{row['sha256']}\n".encode("utf-8"))
            physical = _safe_relative_file(physical_root, row.get("path"), "authenticated source image")
            if sha256_file(physical).lower() != str(row["sha256"]).lower():
                raise GoldPipelineError(f"source image SHA mismatch: {row['path']}")
        if source_aggregate.hexdigest().lower() != str(source_artifact.get("aggregate_sha256", "")).lower():
            raise GoldPipelineError("source image aggregate SHA mismatch")


def _round_landmarks(points: Any) -> Any:
    return [
        {"id": int(point["id"]), "x": round(float(point["x"]), 8), "y": round(float(point["y"]), 8)}
        for point in (points or [])
    ]


def _label_signature(row: Mapping[str, Any], handedness_policy: str) -> str:
    present = bool((row.get("hand_presence") or {}).get("present"))
    handedness = str((row.get("handedness") or {}).get("label", "unknown"))
    return _json_sha(
        {
            "ignore": bool(row.get("ignore_for_training")),
            "present": present,
            "landmarks": _round_landmarks(row.get("landmarks_crop_norm")) if present else [],
            "handedness": handedness if handedness_policy != "unavailable" else "unavailable",
        }
    )


def _sample_type(row: Mapping[str, Any]) -> str:
    present = bool((row.get("hand_presence") or {}).get("present"))
    runtime = bool(row.get("palm_valid"))
    if present:
        return "POS_RUNTIME" if runtime else "POS_LOW_PALM"
    return "NEG_RUNTIME_CANDIDATE" if runtime else "NEG_LOW_PALM_CANDIDATE"


class _UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def finalize_gold_aggregate(config_path: Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml_config(config_path)
    workspace = _workspace(cfg, config_path)
    discovery = cfg.get("source_discovery") or {}
    source_value = Path(str(discovery.get("root", workspace / "sources" / "gold")))
    source_root = _assert_directory(
        source_value if source_value.is_absolute() else _repo_root(config_path) / source_value,
        "Gold source discovery root",
    )
    descriptor_name = str(discovery.get("descriptor_name", "finetune_source.json"))
    descriptor_paths: List[Path] = []
    if source_root.is_dir():
        for candidate in sorted(source_root.iterdir(), key=lambda path: path.name):
            if candidate.name.startswith("."):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise GoldPipelineError(f"invalid entry in Gold source discovery root: {candidate}")
            descriptor = _safe_relative_file(candidate, descriptor_name, "Gold source descriptor")
            descriptor_paths.append(descriptor)
    if not descriptor_paths:
        raise GoldPipelineError(f"no Gold source descriptors found under {source_root}")
    identity_cfg = cfg.get("cross_source_identity") or {}
    identity_keys = [
        str(value)
        for value in identity_cfg.get(
            "keys", ["parent_global_crop_id", "global_crop_id", "image_sha256", "normalized_pixel_sha256"]
        )
    ]
    allowed_identity_keys = {"parent_global_crop_id", "global_crop_id", "image_sha256", "normalized_pixel_sha256"}
    if not identity_keys or any(value not in allowed_identity_keys for value in identity_keys):
        raise GoldPipelineError(f"invalid cross_source_identity.keys: {identity_keys}")
    if identity_cfg.get("conflicting_label", "fail") != "fail":
        raise GoldPipelineError("cross_source_identity.conflicting_label must be fail")
    if identity_cfg.get("identical_label", "keep_by_role_then_source_id") != "keep_by_role_then_source_id":
        raise GoldPipelineError("cross_source_identity.identical_label must be keep_by_role_then_source_id")
    role_order = [str(value) for value in identity_cfg.get("role_precedence", list(ROLE_PRECEDENCE))]
    if len(role_order) != len(ALLOWED_SOURCE_KINDS) or set(role_order) != ALLOWED_SOURCE_KINDS:
        raise GoldPipelineError(f"role_precedence must list every Gold source kind once: {role_order}")
    role_precedence = {value: index for index, value in enumerate(role_order)}
    records: List[Dict[str, Any]] = []
    source_descriptors: List[Dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_dataset_ids: set[str] = set()
    for descriptor_path in descriptor_paths:
        with descriptor_path.open("r", encoding="utf-8") as handle:
            descriptor = json.load(handle)
        root = descriptor_path.parent
        _validate_source_descriptor(root, descriptor)
        source_id = str(descriptor["source_id"])
        dataset_id = str(descriptor["dataset_id"])
        if source_id in seen_source_ids or dataset_id in seen_dataset_ids:
            raise GoldPipelineError(f"duplicate source_id/dataset_id: {source_id}/{dataset_id}")
        seen_source_ids.add(source_id)
        seen_dataset_ids.add(dataset_id)
        descriptor_sha = sha256_file(descriptor_path)
        try:
            relative_descriptor = descriptor_path.resolve().relative_to(workspace.resolve())
        except ValueError as exc:
            raise GoldPipelineError(f"descriptor outside finetune workspace: {descriptor_path}") from exc
        source_descriptors.append(
            {"source_id": source_id, "path": str(relative_descriptor).replace("\\", "/"), "sha256": descriptor_sha}
        )
        manifest_path = _safe_relative_file(
            root, descriptor["artifacts"]["manifest"]["path"], f"{source_id} manifest"
        )
        labels_path = _safe_relative_file(
            root, descriptor["artifacts"]["gold_labels"]["path"], f"{source_id} Gold labels"
        )
        manifest_idx = _unique(read_jsonl(manifest_path), "crop_id", f"{source_id}:manifest")
        label_idx = _unique(read_jsonl(labels_path), "crop_id", f"{source_id}:gold")
        if set(manifest_idx) != set(label_idx):
            raise GoldPipelineError(f"{source_id}: manifest/Gold full coverage mismatch")
        policy = str(descriptor["handedness_policy"])
        source_cfg = {
            "hand_roi": {"output_width": 256, "output_height": 256},
            "image": {"width": 1280, "height": 720},
        }
        for local_id in sorted(manifest_idx):
            manifest, raw = manifest_idx[local_id], copy.deepcopy(label_idx[local_id])
            crop_path = _safe_relative_file(root, manifest.get("crop_path"), f"{source_id}:{local_id} crop")
            image_sha = sha256_file(crop_path)
            if manifest.get("image_sha256") and str(manifest["image_sha256"]).lower() != image_sha.lower():
                raise GoldPipelineError(f"{source_id}:{local_id}: manifest image SHA mismatch")
            raw.update(
                {
                    "schema_version": "train_finalize_v1",
                    "dataset_id": dataset_id,
                    "source_id": source_id,
                    "source_kind": descriptor["source_kind"],
                    "source_crop_id": local_id,
                    "global_crop_id": f"{dataset_id}:{local_id}",
                    "crop_id": f"{dataset_id}:{local_id}",
                    "crop_path": str(crop_path.resolve()),
                    "image_sha256": image_sha,
                    "normalized_pixel_sha256": _normalized_pixel_sha(crop_path),
                    "annotation_provenance": "human_gold",
                    "supervision_tier": "gold",
                    "training_stage": "finetune",
                    "source_group_id": f"{dataset_id}:{manifest.get('image')}",
                    "sample_type": _sample_type(raw),
                    "quality_tier": "HIGH" if not raw.get("ignore_for_training") else "INVALID",
                    "quality_flags": ["ignore_for_training"] if raw.get("ignore_for_training") else [],
                    "selection_action": "drop_ignore" if raw.get("ignore_for_training") else "include",
                    "hand_presence_loss_weight": 1.0,
                    "landmark_loss_weight": 1.0 if bool((raw.get("hand_presence") or {}).get("present")) else 0.0,
                    "handedness_loss_weight": 1.0
                    if bool((raw.get("hand_presence") or {}).get("present"))
                    and str((raw.get("handedness") or {}).get("label", "")).lower() in {"left", "right"}
                    else 0.0,
                    "supervision_loss_weight": 1.0,
                    "presence_quality_weight": 1.0,
                    "landmark_quality_weight": 1.0 if bool((raw.get("hand_presence") or {}).get("present")) else 0.0,
                    "handedness_quality_weight": 1.0
                    if bool((raw.get("hand_presence") or {}).get("present"))
                    and str((raw.get("handedness") or {}).get("label", "")).lower() in {"left", "right"}
                    else 0.0,
                    "sampling_weight": 1.0,
                }
            )
            raw["sampling_bucket"] = f"gold:{raw['sample_type']}"
            raw.setdefault("finetune_review", {})
            raw["finetune_review"]["source_descriptor_sha256"] = descriptor_sha
            if not raw.get("ignore_for_training"):
                manifest_errors = validate_manifest_row(manifest)
                _, label_errors = validate_label_schema(
                    raw,
                    source_cfg,
                    gold=True,
                    check_image=True,
                    source_root=root,
                    handedness_policy=policy,
                )
                if manifest_errors or label_errors:
                    raise GoldPipelineError(f"{source_id}:{local_id}: invalid Gold row: {manifest_errors + label_errors}")
            records.append(
                {
                    "row": raw,
                    "source_id": source_id,
                    "source_kind": descriptor["source_kind"],
                    "handedness_policy": policy,
                    "signature": _label_signature(raw, policy),
                }
            )

    union = _UnionFind(len(records))
    identity_index: Dict[str, int] = {}
    for index, item in enumerate(records):
        row = item["row"]
        identities = []
        for name in identity_keys:
            value = row.get(name)
            if value:
                identities.append(f"{name}:{value}")
        for identity in identities:
            previous = identity_index.get(identity)
            if previous is not None:
                union.union(index, previous)
            else:
                identity_index[identity] = index
    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)
    duplicate_count = 0
    conflicts: List[Dict[str, Any]] = []
    for indices in groups.values():
        if len(indices) == 1:
            continue
        signatures = {records[index]["signature"] for index in indices}
        if len(signatures) > 1:
            conflicts.append(
                {
                    "rows": [
                        {
                            "source_id": records[index]["source_id"],
                            "global_crop_id": records[index]["row"]["global_crop_id"],
                            "signature": records[index]["signature"],
                        }
                        for index in indices
                    ]
                }
            )
            continue
        ordered = sorted(
            indices,
            key=lambda index: (
                role_precedence.get(str(records[index]["source_kind"]), 99),
                records[index]["source_id"],
                records[index]["row"]["global_crop_id"],
            ),
        )
        owner = ordered[0]
        for duplicate in ordered[1:]:
            row = records[duplicate]["row"]
            row["selection_action"] = "drop_duplicate"
            row["quality_flags"] = sorted(set((row.get("quality_flags") or []) + ["DUPLICATE_GOLD_SAME_LABEL"]))
            row["duplicate_gold_owner"] = records[owner]["row"]["global_crop_id"]
            duplicate_count += 1
    if conflicts:
        raise GoldPipelineError(f"cross-source Gold label conflicts: {json.dumps(conflicts[:10], ensure_ascii=False)}")

    catalog = sorted((item["row"] for item in records), key=lambda row: row["global_crop_id"])
    included = [row for row in catalog if row["selection_action"] == "include"]
    excluded = [row for row in catalog if row["selection_action"] != "include"]
    output_cfg = cfg.get("outputs") or {}
    output_root = resolve_path(_repo_root(config_path), output_cfg.get("root", workspace / "hmlf_gold_merged"))
    if output_root.exists():
        raise GoldPipelineError(f"Gold aggregate output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hmlf_gold_merged.", dir=output_root.parent))
    try:
        catalog_path = staging / "05_labels" / "hand_train_catalog_finetune.jsonl"
        included_path = staging / "05_labels" / "hand_training_labels_finetune.jsonl"
        excluded_path = staging / "05_labels" / "hand_training_excluded_finetune.jsonl"
        report_path = staging / "qc" / "finalize_train_finetune_report.json"
        atomic_write_jsonl(catalog_path, catalog)
        atomic_write_jsonl(included_path, included)
        atomic_write_jsonl(excluded_path, excluded)
        report = {
            "schema_version": "hmlf_gold_finalize_report_v1",
            "status": "ok",
            "finetune_id": os.environ.get("HAND_FINETUNE_ID"),
            "counts": {
                "catalog": len(catalog),
                "included": len(included),
                "excluded": len(excluded),
                "duplicates": duplicate_count,
                "conflicts": 0,
                "sources": len(source_descriptors),
            },
            "identity_algorithm": {
                "version": "gold_identity_v1",
                "keys": identity_keys,
                "matching": "union_on_any_authenticated_identity",
            },
            "source_descriptors": source_descriptors,
        }
        atomic_write_json(report_path, report)
        artifacts = {
            "catalog": _artifact(catalog_path, staging, count=len(catalog)),
            "included": _artifact(included_path, staging, count=len(included)),
            "excluded": _artifact(excluded_path, staging, count=len(excluded)),
            "report": _artifact(report_path, staging, count=1),
        }
        aggregate = {
            "schema_version": "hmlf_gold_aggregate_v1",
            "finetune_id": os.environ.get("HAND_FINETUNE_ID"),
            "created_at": _utc_now(),
            "source_descriptors": source_descriptors,
            "artifacts": artifacts,
            "counts": {"catalog": len(catalog), "included": len(included), "excluded": len(excluded)},
            "identity_algorithm_version": "gold_identity_v1",
            "duplicate_count": duplicate_count,
            "conflict_count": 0,
        }
        atomic_write_json(staging / "hmlf_gold_aggregate.json", aggregate)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return aggregate
