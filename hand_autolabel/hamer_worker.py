from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .formats import read_jsonl, write_jsonl


HAMER_SOURCE = "hamer_official_cvpr24"
HAMER_TFLITE_RESCUE_SOURCE = (
    "mediapipe_hand_landmarker_full_tflite_hamer_rescue"
)


def _resolve_python_executable(value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("hamer.python_executable is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(
                f"HaMeR Python executable does not exist: {candidate}"
            )
        return candidate
    resolved = shutil.which(raw)
    if resolved is None:
        raise FileNotFoundError(
            f"HaMeR Python executable is not available: {raw}"
        )
    return Path(resolved)


def _validate_keypoints(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 21:
        raise ValueError("HaMeR response keypoints_2d must contain 21 points")
    output: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("HaMeR response keypoint must contain x and y")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("HaMeR response keypoint is invalid") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("HaMeR response keypoints must be finite")
        output.append([x, y])
    return output


class HaMeRWorkerClient:
    def __init__(self, cfg: Mapping[str, Any], repo_root: Path) -> None:
        hamer_cfg = cfg.get("hamer") or {}
        if not isinstance(hamer_cfg, Mapping):
            raise ValueError("hamer must be a mapping")
        self.python_executable = _resolve_python_executable(
            hamer_cfg.get("python_executable")
        )
        repository_value = str(hamer_cfg.get("repository_path") or "").strip()
        if not repository_value:
            raise ValueError("hamer.repository_path is required")
        self.repository_path = Path(repository_value).expanduser().resolve()
        if not (self.repository_path / "predict_hand_keypoints.py").is_file():
            raise FileNotFoundError(
                "HaMeR repository must contain predict_hand_keypoints.py: "
                f"{self.repository_path}"
            )
        expected_commit = str(
            hamer_cfg.get("repository_commit") or ""
        ).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
            raise ValueError("hamer.repository_commit must be a full Git commit ID")
        git_result = subprocess.run(
            ["git", "-C", str(self.repository_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if git_result.returncode != 0:
            raise RuntimeError(
                "cannot resolve HaMeR repository HEAD: "
                f"{(git_result.stderr or git_result.stdout).strip()}"
            )
        self.repository_commit = git_result.stdout.strip().lower()
        if self.repository_commit != expected_commit:
            raise RuntimeError(
                "HaMeR repository HEAD does not match hamer.repository_commit: "
                f"actual={self.repository_commit}, expected={expected_commit}"
            )
        checkpoint_value = str(hamer_cfg.get("checkpoint_path") or "").strip()
        if not checkpoint_value:
            raise ValueError("hamer.checkpoint_path is required")
        checkpoint = Path(checkpoint_value).expanduser()
        self.checkpoint_path = (
            checkpoint.resolve()
            if checkpoint.is_absolute()
            else (self.repository_path / checkpoint).resolve()
        )
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"HaMeR checkpoint does not exist: {self.checkpoint_path}"
            )
        self.model_id = str(hamer_cfg.get("model_id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.model_id):
            raise ValueError("hamer.model_id must be a non-empty safe identifier")
        try:
            self.rescale = float(hamer_cfg.get("rescale", 0.75))
        except (TypeError, ValueError) as exc:
            raise ValueError("hamer.rescale must be a positive finite number") from exc
        if not math.isfinite(self.rescale) or self.rescale <= 0.0:
            raise ValueError("hamer.rescale must be a positive finite number")
        self.device = str(hamer_cfg.get("device", "cuda")).strip().lower()
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("hamer.device must be cuda or cpu")
        self.worker_path = Path(repo_root).resolve() / "tools" / "hamer_worker.py"
        if not self.worker_path.is_file():
            raise FileNotFoundError(
                f"HaMeR worker does not exist: {self.worker_path}"
            )

    def predict(
        self, requests: Iterable[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        request_rows = []
        for row in requests:
            handedness = str(row.get("handedness") or "").strip().lower()
            request_rows.append(
                {
                    "crop_id": str(row.get("crop_id") or ""),
                    "crop_path": str(row.get("crop_path") or ""),
                    "handedness": handedness,
                }
            )
        if not request_rows:
            return {}
        expected_ids = [row["crop_id"] for row in request_rows]
        if any(not crop_id for crop_id in expected_ids):
            raise ValueError("HaMeR request requires crop_id")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("HaMeR request contains duplicate crop_id")
        if any(not row["crop_path"] for row in request_rows):
            raise ValueError("HaMeR request requires crop_path")
        invalid_handedness = sorted(
            {
                row["handedness"]
                for row in request_rows
                if row["handedness"] not in {"left", "right"}
            }
        )
        if invalid_handedness:
            raise ValueError(
                "HaMeR request handedness must be Left or Right; "
                f"got {invalid_handedness}"
            )

        with tempfile.TemporaryDirectory(prefix="hlmf-hamer-") as temp_dir:
            temp_root = Path(temp_dir)
            request_path = temp_root / "request.jsonl"
            output_path = temp_root / "output.jsonl"
            write_jsonl(request_path, request_rows)
            environment = os.environ.copy()
            environment.setdefault("PYOPENGL_PLATFORM", "egl")
            result = subprocess.run(
                [
                    str(self.python_executable),
                    "-B",
                    str(self.worker_path),
                    "--repository",
                    str(self.repository_path),
                    "--checkpoint",
                    str(self.checkpoint_path),
                    "--device",
                    self.device,
                    "--rescale",
                    str(self.rescale),
                    "--request",
                    str(request_path),
                    "--output",
                    str(output_path),
                ],
                cwd=str(self.repository_path),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "worker failed").strip()
                raise RuntimeError(
                    f"HaMeR worker failed ({result.returncode}): {detail}"
                )
            response_rows = read_jsonl(output_path)

        by_id: Dict[str, Dict[str, Any]] = {}
        for row in response_rows:
            crop_id = str(row.get("crop_id") or "")
            if not crop_id or crop_id in by_id:
                raise ValueError(
                    "HaMeR response has a missing or duplicate crop_id"
                )
            value = dict(row)
            value["keypoints_2d"] = _validate_keypoints(
                value.get("keypoints_2d")
            )
            by_id[crop_id] = value
        if set(by_id) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(by_id))
            extra = sorted(set(by_id) - set(expected_ids))
            raise ValueError(
                f"HaMeR response mismatch: missing={missing}, extra={extra}"
            )
        return by_id
