"""HLMF 3.0 long-lived dataset warehouse contracts.

The warehouse is intentionally identified by logical IDs and cached lightweight
image fingerprints.  Source images are never authenticated by repeatedly
hashing their complete byte streams.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import cv2
import numpy as np

from .formats import read_jsonl, resolve_path, write_json, write_jsonl
from .handedness_classifier import HandClassifierONNX
from .image_io import read_image
from .onnx_runtime import onnx_provider_for, onnx_runtime_settings
from .progress import track_progress


SCHEMA_VERSION = "hlmf_dataset_v1"
ROI_CONTRACT_VERSION = "aethersign_roi_v1"
CAPTURE_SOURCE_RE = re.compile(
    r"^(?P<background>[a-z0-9_]+)-(?P<distance>[a-z0-9_]+)-"
    r"(?P<lighting>[a-z0-9_]+)-(?P<condition>[a-z0-9_]+)-"
    r"(?P<split>train|val|test)-(?P<session>s[0-9]+)-"
    r"(?P<performer>[a-z0-9_]+)$"
)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DatasetContractError(RuntimeError):
    """Raised when an HLMF 3.0 warehouse invariant is violated."""


def stable_id(prefix: str, *parts: object) -> str:
    canonical = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}_{hashlib.blake2s(canonical, digest_size=12).hexdigest()}"


def require_safe_id(value: str, field: str) -> str:
    value = str(value).strip()
    if not SAFE_ID_RE.fullmatch(value):
        raise DatasetContractError(
            f"{field} must use letters, digits, '.', '_' or '-': {value!r}"
        )
    return value


def parse_capture_source_id(value: str) -> Dict[str, str]:
    match = CAPTURE_SOURCE_RE.fullmatch(str(value).strip())
    if match is None:
        raise DatasetContractError(
            "capture_source_id must be "
            "<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>; "
            "condition must not contain '-': {!r}".format(value)
        )
    return dict(match.groupdict())


def source_root(
    dataset_root: Path,
    scope: str,
    dataset_id: str,
    capture_source_id: str,
) -> Path:
    scope = str(scope).strip().lower()
    if scope not in {"pretrain", "eval"}:
        raise DatasetContractError("dataset scope must be pretrain or eval")
    require_safe_id(dataset_id, "dataset_id")
    parsed = parse_capture_source_id(capture_source_id)
    if scope == "pretrain" and parsed["split"] != "train":
        raise DatasetContractError("PretrainSource accepts only capture sources whose split field is train")
    if scope == "eval" and parsed["split"] not in {"val", "test"}:
        raise DatasetContractError("EValSource accepts only capture sources whose split field is val or test")
    bucket = "PretrainSource" if scope == "pretrain" else "EValSource"
    return Path(dataset_root).resolve() / bucket / dataset_id / capture_source_id


def proposal_paths(root: Path, proposal_variant: str) -> Dict[str, Path]:
    variant = require_safe_id(proposal_variant, "proposal_variant")
    root = Path(root).resolve()
    return {
        "images": root / "images",
        "palm": root / "01_palm" / variant,
        "roi": root / "02_roi_crops" / variant,
        "reviewed": root / "03_reviewed" / variant,
        "labels": root / "05_labels" / variant,
        "qc": root / "qc" / variant,
    }


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        raise DatasetContractError("source TIFF must be single-channel grayscale")
    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)
    if gray.dtype == np.uint16:
        return np.ascontiguousarray(gray)
    raise DatasetContractError(f"unsupported TIFF dtype: {gray.dtype}")


def _dhash64(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] >= small[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def lightweight_fingerprint(gray: np.ndarray, file_size: int) -> Dict[str, Any]:
    pixels = np.ascontiguousarray(gray).tobytes()
    return {
        "byte_size": int(file_size),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "pixel_crc32": f"{zlib.crc32(pixels) & 0xFFFFFFFF:08x}",
        "dhash64": _dhash64(gray),
    }


def _fingerprint_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(value.get("byte_size", -1)),
        int(value.get("width", -1)),
        int(value.get("height", -1)),
        str(value.get("pixel_crc32", "")),
        str(value.get("dhash64", "")),
    )


def _write_tiff_atomic(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.hlmf-normalizing.tiff")
    try:
        ok = cv2.imwrite(
            str(temporary),
            image,
            [int(cv2.IMWRITE_TIFF_COMPRESSION), 1],
        )
        if not ok:
            raise DatasetContractError(f"failed to write normalized TIFF: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_existing_raw_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for row in rows:
        raw_id = str(row.get("raw_image_id", ""))
        if not raw_id or raw_id in seen:
            raise DatasetContractError(f"invalid or duplicate raw_image_id in {path}: {raw_id!r}")
        seen.add(raw_id)
    return rows


def validate_and_normalize_source(
    dataset_root: Path,
    scope: str,
    dataset_id: str,
    capture_source_id: str,
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Normalize TIFF orientation and freeze stable raw image identities."""

    root = source_root(dataset_root, scope, dataset_id, capture_source_id)
    images_dir = root / "images"
    if not images_dir.is_dir():
        raise DatasetContractError(f"source images directory does not exist: {images_dir}")
    nested = [path for path in images_dir.iterdir() if path.is_dir()]
    if nested:
        raise DatasetContractError("source images/ must be flat; nested directories are forbidden")
    image_paths = sorted(
        [path for path in images_dir.iterdir() if path.is_file()],
        key=lambda path: path.name,
    )
    if not image_paths:
        raise DatasetContractError(f"no TIFF images found in {images_dir}")
    non_tiff = [path.name for path in image_paths if path.suffix.lower() not in {".tif", ".tiff"}]
    if non_tiff:
        report = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "capture_source_id": capture_source_id,
            "scope": scope,
            "split": parse_capture_source_id(capture_source_id)["split"],
            "total": len(image_paths),
            "valid": 0,
            "failed": len(non_tiff),
            "rotated_clockwise": 0,
            "errors": [
                {"image": name, "error": "non_tiff_input"} for name in non_tiff
            ],
            "validation_policy": {
                "accepted_extensions": [".tif", ".tiff"],
                "content_sha256": "not_computed",
            },
        }
        (root / "qc").mkdir(parents=True, exist_ok=True)
        write_json(root / "qc" / "image_validation_report.json", report)
        raise DatasetContractError(
            f"source images/ accepts TIFF only; see {root / 'qc' / 'image_validation_report.json'}"
        )

    parsed = parse_capture_source_id(capture_source_id)
    manifest_path = root / "raw_images.jsonl"
    previous = _load_existing_raw_manifest(manifest_path)
    previous_by_path = {str(row.get("relative_path")): row for row in previous}
    previous_by_fp: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in previous:
        previous_by_fp.setdefault(_fingerprint_key(row.get("fingerprint") or {}), []).append(row)

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    rotations = 0
    reused_after_rename = 0
    assigned_ids: set[str] = set()
    for path in track_progress(
        image_paths,
        enabled=show_progress,
        description="Source check",
        unit="image",
    ):
        try:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise DatasetContractError("unreadable TIFF")
            gray = _gray_uint8(image)
            height, width = gray.shape
            rotated = False
            if (width, height) == (720, 1280):
                gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                _write_tiff_atomic(path, gray)
                rotations += 1
                rotated = True
            elif (width, height) != (1280, 720):
                raise DatasetContractError(f"unexpected_size:{width}x{height}")
            fingerprint = lightweight_fingerprint(gray, path.stat().st_size)
            relative = f"images/{path.name}"
            existing = previous_by_path.get(relative)
            if existing is None:
                candidates = [
                    row
                    for row in previous_by_fp.get(_fingerprint_key(fingerprint), [])
                    if str(row.get("raw_image_id")) not in assigned_ids
                ]
                if len(candidates) > 1:
                    raise DatasetContractError(
                        "ambiguous renamed image fingerprint; restore the original name or resolve manually"
                    )
                existing = candidates[0] if candidates else None
                if existing is not None:
                    reused_after_rename += 1
            raw_id = (
                str(existing["raw_image_id"])
                if existing is not None
                else stable_id("raw", dataset_id, capture_source_id, path.name)
            )
            if raw_id in assigned_ids:
                raise DatasetContractError(f"duplicate raw identity in source: {raw_id}")
            assigned_ids.add(raw_id)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "capture_source_id": capture_source_id,
                    "split": parsed["split"],
                    "raw_image_id": raw_id,
                    "relative_path": relative,
                    "fingerprint": fingerprint,
                    "normalized_rotation_applied": rotated,
                }
            )
        except Exception as exc:
            errors.append({"image": path.name, "error": str(exc)})

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "capture_source_id": capture_source_id,
        "scope": scope,
        "split": parsed["split"],
        "total": len(image_paths),
        "valid": len(records),
        "failed": len(errors),
        "rotated_clockwise": rotations,
        "raw_ids_reused_after_rename": reused_after_rename,
        "errors": errors,
        "validation_policy": {
            "accepted_input_sizes": ["720x1280", "1280x720"],
            "normalized_size": "1280x720",
            "channels": 1,
            "content_sha256": "not_computed",
        },
    }
    (root / "qc").mkdir(parents=True, exist_ok=True)
    write_json(root / "qc" / "image_validation_report.json", report)
    if errors:
        raise DatasetContractError(
            f"image validation failed for {len(errors)} file(s); see {root / 'qc' / 'image_validation_report.json'}"
        )
    write_jsonl(manifest_path, records)
    source_descriptor = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "dataset_id": dataset_id,
        "capture_source_id": capture_source_id,
        **parsed,
        "raw_image_count": len(records),
        "raw_manifest": "raw_images.jsonl",
    }
    write_json(root / "source.json", source_descriptor)
    registry = WarehouseRegistry(Path(dataset_root))
    registry.register_source(source_descriptor, records)
    return report


class WarehouseRegistry:
    """SQLite-backed global identity registry with manifest-relative paths."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root).resolve()
        self.path = self.dataset_root / "Registry" / "registry.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL CHECK(scope IN ('pretrain','eval'))
                );
                CREATE TABLE IF NOT EXISTS capture_sources (
                    capture_source_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
                    split TEXT NOT NULL CHECK(split IN ('train','val','test')),
                    performer TEXT NOT NULL,
                    source_relpath TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS raw_images (
                    raw_image_id TEXT PRIMARY KEY,
                    capture_source_id TEXT NOT NULL REFERENCES capture_sources(capture_source_id),
                    relative_path TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    pixel_crc32 TEXT NOT NULL,
                    dhash64 TEXT NOT NULL,
                    UNIQUE(capture_source_id, relative_path)
                );
                CREATE TABLE IF NOT EXISTS proposal_variants (
                    capture_source_id TEXT NOT NULL REFERENCES capture_sources(capture_source_id),
                    proposal_variant TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active','retired')),
                    retired_at TEXT,
                    PRIMARY KEY(capture_source_id, proposal_variant)
                );
                CREATE TABLE IF NOT EXISTS rois (
                    roi_id TEXT PRIMARY KEY,
                    raw_image_id TEXT NOT NULL REFERENCES raw_images(raw_image_id),
                    capture_source_id TEXT NOT NULL,
                    proposal_variant TEXT NOT NULL,
                    proposal_slot INTEGER NOT NULL,
                    crop_relpath TEXT NOT NULL,
                    UNIQUE(raw_image_id, proposal_variant, proposal_slot)
                );
                CREATE TABLE IF NOT EXISTS negative_datasets (
                    negative_dataset_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('reserved','published','retired'))
                );
                CREATE TABLE IF NOT EXISTS published_negatives (
                    roi_id TEXT PRIMARY KEY REFERENCES rois(roi_id),
                    negative_dataset_id TEXT NOT NULL REFERENCES negative_datasets(negative_dataset_id),
                    published_relpath TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS selections (
                    selection_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('reserved','published','retired'))
                );
                """
            )
            db.execute(
                """INSERT OR IGNORE INTO proposal_variants
                   (capture_source_id,proposal_variant,status,retired_at)
                   SELECT DISTINCT capture_source_id,proposal_variant,'active',NULL FROM rois"""
            )

    def register_source(
        self,
        descriptor: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        dataset_id = str(descriptor["dataset_id"])
        capture_id = str(descriptor["capture_source_id"])
        scope = str(descriptor["scope"])
        bucket = "PretrainSource" if scope == "pretrain" else "EValSource"
        source_relpath = f"{bucket}/{dataset_id}/{capture_id}"
        with self.connect() as db:
            existing = db.execute(
                "SELECT scope FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            if existing is not None and existing["scope"] != scope:
                raise DatasetContractError(f"dataset_id reused across scopes: {dataset_id}")
            db.execute(
                "INSERT OR IGNORE INTO datasets(dataset_id,scope) VALUES(?,?)",
                (dataset_id, scope),
            )
            existing_source = db.execute(
                "SELECT dataset_id,split FROM capture_sources WHERE capture_source_id=?",
                (capture_id,),
            ).fetchone()
            if existing_source is not None and (
                existing_source["dataset_id"] != dataset_id
                or existing_source["split"] != descriptor["split"]
            ):
                raise DatasetContractError(f"capture_source_id reused with different ownership: {capture_id}")
            db.execute(
                """INSERT OR IGNORE INTO capture_sources
                   (capture_source_id,dataset_id,split,performer,source_relpath)
                   VALUES(?,?,?,?,?)""",
                (
                    capture_id,
                    dataset_id,
                    descriptor["split"],
                    descriptor["performer"],
                    source_relpath,
                ),
            )
            for row in rows:
                fp = row["fingerprint"]
                current = db.execute(
                    "SELECT capture_source_id FROM raw_images WHERE raw_image_id=?",
                    (row["raw_image_id"],),
                ).fetchone()
                if current is not None and current["capture_source_id"] != capture_id:
                    raise DatasetContractError(
                        f"raw_image_id reused across capture sources: {row['raw_image_id']}"
                    )
                db.execute(
                    """INSERT INTO raw_images
                       (raw_image_id,capture_source_id,relative_path,byte_size,width,height,pixel_crc32,dhash64)
                       VALUES(?,?,?,?,?,?,?,?)
                       ON CONFLICT(raw_image_id) DO UPDATE SET
                         relative_path=excluded.relative_path,
                         byte_size=excluded.byte_size,
                         width=excluded.width,
                         height=excluded.height,
                         pixel_crc32=excluded.pixel_crc32,
                         dhash64=excluded.dhash64""",
                    (
                        row["raw_image_id"],
                        capture_id,
                        row["relative_path"],
                        fp["byte_size"],
                        fp["width"],
                        fp["height"],
                        fp["pixel_crc32"],
                        fp["dhash64"],
                    ),
                )

    def variant_status(
        self, capture_source_id: str, proposal_variant: str
    ) -> str | None:
        capture_source_id = require_safe_id(capture_source_id, "capture_source_id")
        proposal_variant = require_safe_id(proposal_variant, "proposal_variant")
        with self.connect() as db:
            stored = db.execute(
                "SELECT status FROM proposal_variants "
                "WHERE capture_source_id=? AND proposal_variant=?",
                (capture_source_id, proposal_variant),
            ).fetchone()
        return str(stored["status"]) if stored is not None else None

    def assert_variant_writable(
        self, capture_source_id: str, proposal_variant: str
    ) -> None:
        capture_source_id = require_safe_id(capture_source_id, "capture_source_id")
        proposal_variant = require_safe_id(proposal_variant, "proposal_variant")
        with self.connect() as db:
            stored = db.execute(
                "SELECT status FROM proposal_variants "
                "WHERE capture_source_id=? AND proposal_variant=?",
                (capture_source_id, proposal_variant),
            ).fetchone()
        if stored is not None and stored["status"] == "retired":
            raise DatasetContractError(
                "proposal variant is retired and cannot be reused: "
                f"{capture_source_id}/{proposal_variant}"
            )

    def retire_variant(
        self, capture_source_id: str, proposal_variant: str
    ) -> str:
        capture_source_id = require_safe_id(capture_source_id, "capture_source_id")
        proposal_variant = require_safe_id(proposal_variant, "proposal_variant")
        retired_at = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            source = db.execute(
                "SELECT 1 FROM capture_sources WHERE capture_source_id=?",
                (capture_source_id,),
            ).fetchone()
            if source is None:
                raise DatasetContractError(
                    f"capture source is not registered: {capture_source_id}"
                )
            stored = db.execute(
                "SELECT status FROM proposal_variants "
                "WHERE capture_source_id=? AND proposal_variant=?",
                (capture_source_id, proposal_variant),
            ).fetchone()
            previous = str(stored["status"]) if stored is not None else "unregistered"
            if stored is None:
                db.execute(
                    "INSERT INTO proposal_variants "
                    "(capture_source_id,proposal_variant,status,retired_at) "
                    "VALUES(?,?,'retired',?)",
                    (capture_source_id, proposal_variant, retired_at),
                )
            elif stored["status"] != "retired":
                db.execute(
                    "UPDATE proposal_variants SET status='retired',retired_at=? "
                    "WHERE capture_source_id=? AND proposal_variant=?",
                    (retired_at, capture_source_id, proposal_variant),
                )
        return previous

    def register_rois(self, rows: Sequence[Mapping[str, Any]]) -> None:
        pairs = {
            (str(row["capture_source_id"]), str(row["proposal_variant"]))
            for row in rows
        }
        with self.connect() as db:
            for capture_source_id, proposal_variant in sorted(pairs):
                stored = db.execute(
                    "SELECT status FROM proposal_variants "
                    "WHERE capture_source_id=? AND proposal_variant=?",
                    (capture_source_id, proposal_variant),
                ).fetchone()
                if stored is not None and stored["status"] == "retired":
                    raise DatasetContractError(
                        "proposal variant is retired and cannot be reused: "
                        f"{capture_source_id}/{proposal_variant}"
                    )
                db.execute(
                    "INSERT OR IGNORE INTO proposal_variants "
                    "(capture_source_id,proposal_variant,status,retired_at) "
                    "VALUES(?,?,'active',NULL)",
                    (capture_source_id, proposal_variant),
                )
            for row in rows:
                current = db.execute(
                    "SELECT raw_image_id,proposal_variant,proposal_slot FROM rois WHERE roi_id=?",
                    (row["roi_id"],),
                ).fetchone()
                expected = (
                    str(row["raw_image_id"]),
                    str(row["proposal_variant"]),
                    int(row["proposal_slot"]),
                )
                if current is not None and tuple(current) != expected:
                    raise DatasetContractError(f"roi_id collision: {row['roi_id']}")
                db.execute(
                    """INSERT OR IGNORE INTO rois
                       (roi_id,raw_image_id,capture_source_id,proposal_variant,proposal_slot,crop_relpath)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        row["roi_id"],
                        row["raw_image_id"],
                        row["capture_source_id"],
                        row["proposal_variant"],
                        int(row["proposal_slot"]),
                        row["crop_relpath"],
                    ),
                )

    def reserve_negative_dataset(self, negative_dataset_id: str) -> None:
        require_safe_id(negative_dataset_id, "negative_dataset_id")
        with self.connect() as db:
            existing = db.execute(
                "SELECT status FROM negative_datasets WHERE negative_dataset_id=?",
                (negative_dataset_id,),
            ).fetchone()
            if existing is not None:
                raise DatasetContractError(
                    f"negative_dataset_id has already been used ({existing['status']}): {negative_dataset_id}"
                )
            db.execute(
                "INSERT INTO negative_datasets(negative_dataset_id,status) VALUES(?,'reserved')",
                (negative_dataset_id,),
            )

    def assert_roi_reference(self, row: Mapping[str, Any]) -> None:
        """Reject review requests that do not exactly reference a registered ROI."""

        roi_id = str(row.get("roi_id") or row.get("crop_id") or "")
        with self.connect() as db:
            stored = db.execute(
                "SELECT r.raw_image_id,r.capture_source_id,r.proposal_variant,r.crop_relpath, "
                "v.status AS variant_status FROM rois AS r "
                "JOIN proposal_variants AS v "
                "ON v.capture_source_id=r.capture_source_id "
                "AND v.proposal_variant=r.proposal_variant "
                "WHERE r.roi_id=?",
                (roi_id,),
            ).fetchone()
        if stored is None:
            raise DatasetContractError(f"review request references unregistered roi_id: {roi_id}")
        if stored["variant_status"] != "active":
            raise DatasetContractError(f"review request references retired roi_id: {roi_id}")
        expected = (
            str(row.get("raw_image_id")),
            str(row.get("capture_source_id")),
            str(row.get("proposal_variant")),
            str(row.get("crop_relpath") or row.get("crop_path")),
        )
        actual = tuple(stored[key] for key in (
            "raw_image_id", "capture_source_id", "proposal_variant", "crop_relpath"
        ))
        if actual != expected:
            raise DatasetContractError(
                f"review request disagrees with registry for {roi_id}: {expected} != {actual}"
            )

    def publish_negatives(
        self,
        negative_dataset_id: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        with self.connect() as db:
            status = db.execute(
                "SELECT status FROM negative_datasets WHERE negative_dataset_id=?",
                (negative_dataset_id,),
            ).fetchone()
            if status is None or status["status"] != "reserved":
                raise DatasetContractError(
                    f"negative dataset must be uniquely reserved before publish: {negative_dataset_id}"
                )
            for row in rows:
                db.execute(
                    "INSERT INTO published_negatives(roi_id,negative_dataset_id,published_relpath) VALUES(?,?,?)",
                    (row["roi_id"], negative_dataset_id, row["published_relpath"]),
                )
            db.execute(
                "UPDATE negative_datasets SET status='published' WHERE negative_dataset_id=?",
                (negative_dataset_id,),
            )

    def reserve_selection(self, selection_id: str) -> None:
        require_safe_id(selection_id, "selection_id")
        with self.connect() as db:
            if db.execute("SELECT 1 FROM selections WHERE selection_id=?", (selection_id,)).fetchone():
                raise DatasetContractError(f"selection_id has already been used: {selection_id}")
            db.execute("INSERT INTO selections(selection_id,status) VALUES(?,'reserved')", (selection_id,))

    def publish_selection(self, selection_id: str) -> None:
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE selections SET status='published' WHERE selection_id=? AND status='reserved'",
                (selection_id,),
            )
            if cursor.rowcount != 1:
                raise DatasetContractError(f"selection is not reserved: {selection_id}")

    def report(self) -> Dict[str, Any]:
        with self.connect() as db:
            counts = {}
            for table in (
                "datasets",
                "capture_sources",
                "raw_images",
                "proposal_variants",
                "rois",
                "negative_datasets",
                "published_negatives",
                "selections",
            ):
                counts[table] = int(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            duplicate_variants = [
                dict(row)
                for row in db.execute(
                    """SELECT r.capture_source_id,r.proposal_variant,v.status,COUNT(*) AS count
                       FROM rois AS r JOIN proposal_variants AS v
                       ON v.capture_source_id=r.capture_source_id
                       AND v.proposal_variant=r.proposal_variant
                       GROUP BY r.capture_source_id,r.proposal_variant,v.status"""
                )
            ]
            variant_status_counts = {
                str(row["status"]): int(row["count"])
                for row in db.execute(
                    "SELECT status,COUNT(*) AS count FROM proposal_variants GROUP BY status"
                )
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "registry": str(self.path),
            "counts": counts,
            "roi_counts_by_source_variant": duplicate_variants,
            "proposal_variant_status_counts": variant_status_counts,
            "content_sha256": "not_computed",
        }


def enrich_palm_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    palm_rows: Sequence[Mapping[str, Any]],
    proposal_variant: str,
) -> List[Dict[str, Any]]:
    """Attach stable raw/proposal identities to Palm backend output."""

    raw_by_name = {Path(str(row["relative_path"])).name: row for row in raw_rows}
    output: List[Dict[str, Any]] = []
    for row in palm_rows:
        raw = raw_by_name.get(str(row.get("image")))
        if raw is None:
            raise DatasetContractError(f"Palm output references unknown image: {row.get('image')}")
        item = dict(row)
        item.update(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": raw["dataset_id"],
                "capture_source_id": raw["capture_source_id"],
                "split": raw["split"],
                "raw_image_id": raw["raw_image_id"],
                "proposal_variant": proposal_variant,
            }
        )
        for key in ("detections", "negative_candidates"):
            candidates = list(item.get(key) or [])
            for det in candidates:
                det["proposal_kind"] = "runtime" if key == "detections" else "negative_candidate"
            item[key] = candidates
        combined = list(item["detections"]) + list(item["negative_candidates"])
        combined.sort(
            key=lambda det: (
                round((float(det["bbox_norm"][0]) + float(det["bbox_norm"][2])) / 2.0, 8),
                round((float(det["bbox_norm"][1]) + float(det["bbox_norm"][3])) / 2.0, 8),
                -round(float(det.get("score", 0.0)), 8),
                str(det["proposal_kind"]),
            )
        )
        for slot, det in enumerate(combined):
            det["proposal_slot"] = slot
            det["palm_det_id"] = stable_id(
                "proposal", raw["raw_image_id"], proposal_variant, slot
            )
        output.append(item)
    return output


def enrich_roi_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    palm_rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    proposal_variant: str,
) -> List[Dict[str, Any]]:
    palm_by_det: Dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for parent in palm_rows:
        for det in list(parent.get("detections") or []) + list(parent.get("negative_candidates") or []):
            palm_by_det[str(det["palm_det_id"])] = (parent, det)
    enriched: List[Dict[str, Any]] = []
    for row in manifest_rows:
        parent_det = palm_by_det.get(str(row.get("palm_det_id")))
        if parent_det is None:
            raise DatasetContractError(f"ROI references unknown proposal: {row.get('palm_det_id')}")
        parent, det = parent_det
        slot = int(det["proposal_slot"])
        roi_id = stable_id(
            "roi",
            parent["raw_image_id"],
            proposal_variant,
            slot,
            ROI_CONTRACT_VERSION,
        )
        crop_path = Path(str(row["crop_path"]))
        absolute = crop_path if crop_path.is_absolute() else Path(dataset_root).resolve() / crop_path
        try:
            crop_relpath = str(absolute.resolve().relative_to(Path(dataset_root).resolve())).replace("\\", "/")
        except ValueError as exc:
            raise DatasetContractError(f"ROI crop escapes HAND_DATASET_ROOT: {absolute}") from exc
        item = dict(row)
        item.update(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": parent["dataset_id"],
                "capture_source_id": parent["capture_source_id"],
                "split": parent["split"],
                "raw_image_id": parent["raw_image_id"],
                "roi_id": roi_id,
                "crop_id": roi_id,
                "proposal_variant": proposal_variant,
                "proposal_slot": slot,
                "proposal_kind": det["proposal_kind"],
                "roi_contract_version": ROI_CONTRACT_VERSION,
                "crop_relpath": crop_relpath,
                "crop_path": crop_relpath,
            }
        )
        enriched.append(item)
    return enriched


def apply_label_provenance(
    rows: Sequence[Mapping[str, Any]],
    draft_by_roi: Mapping[str, Mapping[str, Any]] | None = None,
    human_reviewed: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize label origin/style and record the exact human changes."""

    def teacher_identity(source_row: Mapping[str, Any]) -> tuple[str, str, str | None]:
        source = str(source_row.get("source") or "")
        if source == "rtmpose_m_hand5_onnx":
            return "rtmpose", "rtmpose_m_hand5_v1", "rtmpose-m_hand5_256x256_onnx"
        if source == "mediapipe_hand_landmarker_full_tflite_rtmpose_rescue":
            return (
                "mediapipe",
                "mediapipe_tflite_rescue_v1",
                "mediapipe-hand-landmark-full-tflite",
            )
        if source == "eos_negative_candidate_unassessed":
            return "unresolved", "unlabeled_v1", None
        return "mediapipe", "mediapipe_v1", "mediapipe_hand_landmarker_task"

    drafts = draft_by_roi or {}
    output: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        roi_id = str(row.get("roi_id") or row.get("crop_id"))
        row["roi_id"] = roi_id
        row["crop_id"] = roi_id
        draft = drafts.get(roi_id)
        present = bool((row.get("hand_presence") or {}).get("present"))
        teacher_present = bool((draft.get("hand_presence") or {}).get("present")) if draft else False
        teacher_origin, teacher_style, teacher_model_id = teacher_identity(draft or row)
        handedness_teacher_model_id = (draft or row).get("handedness_teacher_model_id")
        hand_presence_teacher_model_id = (draft or row).get(
            "hand_presence_teacher_model_id"
        )
        if teacher_origin == "mediapipe":
            if handedness_teacher_model_id is None:
                handedness_teacher_model_id = "mediapipe_hand_landmarker_task"
            if hand_presence_teacher_model_id is None:
                hand_presence_teacher_model_id = "mediapipe_hand_landmarker_task"
        teacher_handedness = str(
            ((draft or row).get("handedness") or {}).get("label", "unknown")
        ).lower()
        current_handedness = str(
            (row.get("handedness") or {}).get("label", "unknown")
        ).lower()
        modified_handedness = bool(
            human_reviewed and current_handedness != teacher_handedness
        )
        modified_presence = bool(human_reviewed and present != teacher_present)
        modified: List[int] = []
        if human_reviewed and present and teacher_present:
            old = {int(point["id"]): point for point in draft.get("landmarks_crop_norm") or []}
            for point in row.get("landmarks_crop_norm") or []:
                before = old.get(int(point["id"]))
                if before is None or abs(float(point["x"]) - float(before["x"])) > 1e-6 or abs(
                    float(point["y"]) - float(before["y"])
                ) > 1e-6:
                    modified.append(int(point["id"]))
        if not human_reviewed:
            origin = teacher_origin
            style = teacher_style
        elif present and teacher_present and not modified:
            origin = teacher_origin
            style = teacher_style
        elif present and teacher_present:
            origin = f"{teacher_origin}_human_corrected"
            style = "project_consensus_v1"
        else:
            origin = "human"
            style = "project_consensus_v1"
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "label_origin": origin,
                "annotation_style": style,
                "teacher_model_id": teacher_model_id,
                "teacher_detected": teacher_present if human_reviewed else present,
                "human_reviewed": bool(human_reviewed),
                "human_modified_landmark_ids": sorted(modified),
                "handedness_teacher_model_id": handedness_teacher_model_id,
                "hand_presence_teacher_model_id": hand_presence_teacher_model_id,
                "human_modified_handedness": modified_handedness,
                "human_modified_presence": modified_presence,
            }
        )
        output.append(row)
    return output


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise DatasetContractError(f"refusing to overwrite review/published image: {destination}")
    try:
        shutil.copy2(source, destination)
    except OSError as exc:
        raise DatasetContractError(
            f"image copy failed: {source} -> {destination}"
        ) from exc


def _remove_variant_target(
    target: Path, source: Path, dataset_root: Path
) -> str | None:
    target = Path(target)
    source = Path(source).resolve()
    dataset_root = Path(dataset_root).resolve()
    if target.is_symlink():
        raise DatasetContractError(f"refusing to delete symlinked variant artifact: {target}")
    try:
        target.resolve(strict=False).relative_to(source)
    except ValueError as exc:
        raise DatasetContractError(f"variant artifact escapes capture source: {target}") from exc
    if not target.exists():
        return None
    relpath = str(target.resolve().relative_to(dataset_root)).replace("\\", "/")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return relpath


def clean_variant_visualizations(
    dataset_root: Path,
    scope: str,
    dataset_id: str,
    capture_source_id: str,
    proposal_variant: str,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    source = source_root(dataset_root, scope, dataset_id, capture_source_id)
    if not source.is_dir():
        raise DatasetContractError(f"capture source does not exist: {source}")
    paths = proposal_paths(source, proposal_variant)
    original_root = source / "visualizations" / "original_image_landmarks"
    targets = [
        paths["roi"] / "hand_landmarks_roi_visualization",
        original_root / proposal_variant,
        original_root / f"{proposal_variant}.mp4",
        paths["qc"] / "roi_visualization_report.json",
        paths["qc"] / "original_image_visualization_report.json",
    ]
    removed = []
    for target in targets:
        relpath = _remove_variant_target(target, source, dataset_root)
        if relpath is not None:
            removed.append(relpath)
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_source_id": capture_source_id,
        "proposal_variant": proposal_variant,
        "removed": removed,
        "removed_count": len(removed),
    }


def delete_source_variant(
    dataset_root: Path,
    scope: str,
    dataset_id: str,
    capture_source_id: str,
    proposal_variant: str,
    confirmation: str,
) -> Dict[str, Any]:
    proposal_variant = require_safe_id(proposal_variant, "proposal_variant")
    if str(confirmation) != proposal_variant:
        raise DatasetContractError(
            "deletion confirmation must exactly match proposal_variant"
        )
    dataset_root = Path(dataset_root).resolve()
    source = source_root(dataset_root, scope, dataset_id, capture_source_id)
    if not source.is_dir() or not (source / "source.json").is_file():
        raise DatasetContractError(f"registered capture source does not exist: {source}")
    paths = proposal_paths(source, proposal_variant)
    original_root = source / "visualizations" / "original_image_landmarks"
    targets = [
        paths["palm"],
        paths["roi"],
        paths["reviewed"],
        paths["labels"],
        paths["qc"],
        original_root / proposal_variant,
        original_root / f"{proposal_variant}.mp4",
    ]
    existing_targets = [target for target in targets if target.exists()]
    registry = WarehouseRegistry(dataset_root)
    status = registry.variant_status(capture_source_id, proposal_variant)
    if status is None and not existing_targets:
        raise DatasetContractError(
            f"proposal variant has no registry entry or artifacts: {capture_source_id}/{proposal_variant}"
        )
    previous_status = registry.retire_variant(capture_source_id, proposal_variant)
    removed = []
    for target in targets:
        relpath = _remove_variant_target(target, source, dataset_root)
        if relpath is not None:
            removed.append(relpath)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "capture_source_id": capture_source_id,
        "proposal_variant": proposal_variant,
        "previous_registry_status": previous_status,
        "registry_status": "retired",
        "removed": removed,
        "removed_count": len(removed),
        "raw_source_preserved": True,
    }


def prepare_negative_review(
    dataset_root: Path,
    negative_dataset_id: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    model_root: Path,
    *,
    show_progress: bool = False,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    negative_dataset_id = require_safe_id(negative_dataset_id, "negative_dataset_id")
    batch_root = dataset_root / "GoldSource" / "NegativeSamples" / negative_dataset_id
    review_root = batch_root / "review"
    if review_root.exists() or (batch_root / "published").exists():
        raise DatasetContractError(f"negative dataset workspace already exists: {batch_root}")
    review_cfg = cfg.get("negative_review") or {}
    try:
        threshold = float(review_cfg.get("hand_presence_threshold", 0.5))
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(
            "negative_review.hand_presence_threshold must be a finite number in [0, 1]"
        ) from exc
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise DatasetContractError(
            "negative_review.hand_presence_threshold must be a finite number in [0, 1]"
        )
    _default_provider, batch_size = onnx_runtime_settings(cfg)
    provider_preference = onnx_provider_for(cfg, "hand_classifier")
    classifier_cfg = cfg.get("hand_classifier") or {}
    model_path = resolve_path(model_root, classifier_cfg.get("model_onnx_path", ""))
    registry = WarehouseRegistry(dataset_root)
    validated: List[tuple[Dict[str, Any], Path]] = []
    for raw in candidate_rows:
        row = dict(raw)
        if str(row.get("split")) != "train":
            raise DatasetContractError("negative review accepts Train candidates only")
        registry.assert_roi_reference(row)
        source = dataset_root / str(row["crop_relpath"])
        if not source.is_file():
            raise DatasetContractError(f"candidate ROI does not exist: {source}")
        validated.append((row, source))
    if not validated:
        raise DatasetContractError("negative review requires at least one candidate")

    classifier = HandClassifierONNX(model_path, provider_preference)
    selected: List[tuple[Dict[str, Any], Path]] = []
    excluded: List[Dict[str, Any]] = []
    for offset in track_progress(
        range(0, len(validated), batch_size),
        enabled=show_progress,
        description="Negative pre-review HCF",
        unit="batch",
    ):
        chunk = validated[offset : offset + batch_size]
        images = []
        for _, source in chunk:
            image = read_image(source)
            if image is None:
                raise DatasetContractError(f"candidate ROI is unreadable: {source}")
            images.append(image)
        predictions = classifier.classify_batch(images)
        if len(predictions) != len(chunk):
            raise DatasetContractError(
                "Hand classifier output count does not match negative candidates"
            )
        for (row, source), prediction in zip(chunk, predictions):
            score = float(prediction["hand_presence"]["score"])
            precheck = {
                "hand_presence_score": score,
                "threshold": threshold,
                "selected_for_human_review": score < threshold,
                "model_id": classifier.model_id,
            }
            item = dict(row)
            item["negative_review_precheck"] = precheck
            if score < threshold:
                selected.append((item, source))
            else:
                excluded.append(item)
    if not selected:
        raise DatasetContractError(
            "negative pre-review selected no candidates below the hand-presence threshold"
        )

    registry.reserve_negative_dataset(negative_dataset_id)
    materialized: List[Dict[str, Any]] = []
    for row, source in selected:
        capture_id = str(row["capture_source_id"])
        destination = review_root / "images" / capture_id / source.name
        _copy_image(source, destination)
        item = dict(row)
        item["review_relpath"] = str(destination.relative_to(batch_root)).replace("\\", "/")
        materialized.append(item)
    write_jsonl(review_root / "candidate_manifest.jsonl", materialized)
    write_jsonl(review_root / "precheck_excluded.jsonl", excluded)
    write_json(
        review_root / "README.json",
        {
            "negative_dataset_id": negative_dataset_id,
            "instruction": "Delete every image containing a hand or any uncertain content; keep only true background negatives.",
            "input_candidate_count": len(validated),
            "candidate_count": len(materialized),
            "precheck_excluded_count": len(excluded),
            "hand_presence_threshold": threshold,
            "selection_rule": "P(has_hand) < threshold",
            "hand_classifier_model_id": classifier.model_id,
            "onnx_provider": classifier.provider,
            "onnx_provider_fallback_reason": classifier.fallback_reason,
            "onnx_batch_size": batch_size,
        },
    )
    return {
        "negative_dataset_id": negative_dataset_id,
        "input_candidate_count": len(validated),
        "candidate_count": len(materialized),
        "precheck_excluded_count": len(excluded),
        "hand_presence_threshold": threshold,
        "hand_classifier_model_id": classifier.model_id,
        "onnx_provider": classifier.provider,
        "onnx_batch_size": batch_size,
        "review_root": str(review_root),
    }


def publish_negative_review(dataset_root: Path, negative_dataset_id: str) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    batch_root = dataset_root / "GoldSource" / "NegativeSamples" / negative_dataset_id
    review_root = batch_root / "review"
    published_root = batch_root / "published"
    if published_root.exists():
        raise DatasetContractError(f"negative dataset is already published: {negative_dataset_id}")
    candidates = read_jsonl(review_root / "candidate_manifest.jsonl")
    retained_by_path = {
        str(path.relative_to(batch_root)).replace("\\", "/"): path
        for path in (review_root / "images").rglob("*")
        if path.is_file()
    }
    retained: List[Dict[str, Any]] = []
    removed: List[str] = []
    for row in candidates:
        review_relpath = str(row["review_relpath"])
        source = retained_by_path.pop(review_relpath, None)
        if source is None:
            removed.append(str(row["roi_id"]))
            continue
        destination = published_root / "images" / str(row["capture_source_id"]) / source.name
        _copy_image(source, destination)
        item = dict(row)
        item.update(
            {
                "negative_dataset_id": negative_dataset_id,
                "label_origin": "human",
                "annotation_style": "project_consensus_v1",
                "human_reviewed": True,
                "hand_presence": {"present": False},
                "landmarks_crop_norm": [],
                "published_relpath": str(destination.relative_to(dataset_root)).replace("\\", "/"),
            }
        )
        retained.append(item)
    if retained_by_path:
        raise DatasetContractError(
            f"review tree contains unmanifested files: {sorted(retained_by_path)[:8]}"
        )
    if not retained:
        raise DatasetContractError("cannot publish an empty negative dataset")
    write_jsonl(published_root / "negative_labels.jsonl", retained)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "negative_dataset_id": negative_dataset_id,
        "records": len(retained),
        "capture_sources": sorted({str(row["capture_source_id"]) for row in retained}),
        "labels": "negative_labels.jsonl",
        "image_policy": "copied_review_and_published_images",
        "content_sha256": "not_computed",
    }
    write_json(published_root / "manifest.json", manifest)
    write_json(
        published_root / "review_report.json",
        {
            "candidates": len(candidates),
            "retained_true_negatives": len(retained),
            "removed_false_or_uncertain_negatives": len(removed),
            "removed_roi_ids": removed,
        },
    )
    WarehouseRegistry(dataset_root).publish_negatives(negative_dataset_id, retained)
    shutil.rmtree(review_root)
    return manifest


def prepare_selection_review(
    dataset_root: Path,
    selection_id: str,
    request_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    registry = WarehouseRegistry(dataset_root)
    registry.reserve_selection(selection_id)
    root = dataset_root / "Selections" / selection_id
    review_root = root / "review"
    if review_root.exists() or (root / "published").exists():
        raise DatasetContractError(f"selection workspace already exists: {root}")
    materialized: List[Dict[str, Any]] = []
    for row in request_rows:
        if str(row.get("split")) != "train":
            raise DatasetContractError("hard-positive selection accepts Train requests only")
        registry.assert_roi_reference(row)
        source = dataset_root / str(row["crop_relpath"])
        if not source.is_file():
            raise DatasetContractError(f"requested ROI does not exist: {source}")
        destination = review_root / "images" / str(row["capture_source_id"]) / source.name
        _copy_image(source, destination)
        item = dict(row)
        item["review_relpath"] = str(destination.relative_to(root)).replace("\\", "/")
        materialized.append(item)
    write_jsonl(review_root / "request_manifest.jsonl", materialized)
    write_json(
        review_root / "README.json",
        {
            "selection_id": selection_id,
            "instruction": "Delete only ROIs whose teacher landmarks are clearly wrong; do not relabel points.",
            "candidate_count": len(materialized),
        },
    )
    return {"selection_id": selection_id, "candidate_count": len(materialized), "review_root": str(review_root)}


def publish_selection_review(dataset_root: Path, selection_id: str) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    root = dataset_root / "Selections" / selection_id
    review_root = root / "review"
    published_root = root / "published"
    if published_root.exists():
        raise DatasetContractError(f"selection is already published: {selection_id}")
    rows = read_jsonl(review_root / "request_manifest.jsonl")
    retained_by_path = {
        str(path.relative_to(root)).replace("\\", "/"): path
        for path in (review_root / "images").rglob("*")
        if path.is_file()
    }
    known = {str(row["review_relpath"]) for row in rows}
    unknown = sorted(set(retained_by_path) - known)
    if unknown:
        raise DatasetContractError(f"selection review contains unmanifested files: {unknown[:8]}")
    retained: List[Dict[str, Any]] = []
    for row in rows:
        review_relpath = str(row["review_relpath"])
        source = retained_by_path.get(review_relpath)
        if source is None:
            continue
        destination = (
            published_root / "images" / str(row["capture_source_id"]) / source.name
        )
        _copy_image(source, destination)
        item = dict(row)
        item.pop("review_relpath", None)
        item["selection_id"] = selection_id
        item["source_crop_relpath"] = str(item.get("crop_relpath") or item.get("crop_path"))
        item["published_relpath"] = str(
            destination.relative_to(dataset_root)
        ).replace("\\", "/")
        retained.append(item)
    if not retained:
        raise DatasetContractError("cannot publish an empty hard-positive selection")
    write_jsonl(published_root / "selection.jsonl", retained)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": selection_id,
        "records": len(retained),
        "removed_teacher_errors": len(rows) - len(retained),
        "selection": "selection.jsonl",
        "image_policy": "copied_review_and_published_images",
    }
    write_json(published_root / "manifest.json", manifest)
    WarehouseRegistry(dataset_root).publish_selection(selection_id)
    shutil.rmtree(review_root)
    return manifest
