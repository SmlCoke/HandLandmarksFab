from __future__ import annotations

import shutil
import subprocess
import tempfile
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .formats import read_jsonl, resolve_path, write_jsonl


MEDIAPIPE_TFLITE_MODEL_ID = "mediapipe-hand-landmark-full-tflite"
MEDIAPIPE_TFLITE_RESCUE_SOURCE = (
    "mediapipe_hand_landmarker_full_tflite_rtmpose_rescue"
)


def _validate_landmark_rows(value: Any, field: str) -> None:
    if not isinstance(value, list) or len(value) != 21:
        raise ValueError(f"MediaPipe TFLite response {field} must contain 21 points")
    ids: set[int] = set()
    for point in value:
        if not isinstance(point, Mapping):
            raise ValueError(f"MediaPipe TFLite response {field} point must be an object")
        raw_id = point.get("id")
        if isinstance(raw_id, bool):
            raise ValueError(f"MediaPipe TFLite response {field} id must be an integer")
        try:
            point_id = int(raw_id)
            if float(raw_id) != float(point_id):
                raise ValueError
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"MediaPipe TFLite response {field} point is invalid"
            ) from exc
        if point_id < 0 or point_id > 20 or point_id in ids:
            raise ValueError(
                f"MediaPipe TFLite response {field} ids must be unique 0..20"
            )
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"MediaPipe TFLite response {field} coordinates must be finite"
            )
        ids.add(point_id)
    if ids != set(range(21)):
        raise ValueError(f"MediaPipe TFLite response {field} ids must be exactly 0..20")


def mediapipe_tflite_rescue_enabled(cfg: Mapping[str, Any]) -> bool:
    raw = (cfg.get("quality") or {}).get(
        "rtmpose_train_mediapipe_tflite_rescue_enabled", True
    )
    if not isinstance(raw, bool):
        raise ValueError(
            "quality.rtmpose_train_mediapipe_tflite_rescue_enabled must be a boolean"
        )
    return raw


def _resolve_python_executable(value: Any) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("mediapipe_tflite.python_executable is required")
    candidate = Path(raw)
    if candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(
                f"MediaPipe TFLite Python executable does not exist: {candidate}"
            )
        return candidate
    resolved = shutil.which(raw)
    if resolved is None:
        raise FileNotFoundError(
            f"MediaPipe TFLite Python executable is not available: {raw}"
        )
    return Path(resolved)


class MediaPipeTFLiteRescueClient:
    def __init__(self, cfg: Mapping[str, Any], repo_root: Path) -> None:
        rescue_cfg = cfg.get("mediapipe_tflite") or {}
        if not isinstance(rescue_cfg, Mapping):
            raise ValueError("mediapipe_tflite must be a mapping")
        self.python_executable = _resolve_python_executable(
            rescue_cfg.get("python_executable")
        )
        self.model_path = resolve_path(
            repo_root, str(rescue_cfg.get("model_asset_path") or "")
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe TFLite model does not exist: {self.model_path}"
            )
        self.worker_path = Path(repo_root).resolve() / "tools" / "mediapipe_tflite_worker.py"
        if not self.worker_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe TFLite worker does not exist: {self.worker_path}"
            )

    def predict(
        self, requests: Iterable[Mapping[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        request_rows = [
            {
                "crop_id": str(row.get("crop_id") or ""),
                "crop_path": str(row.get("crop_path") or ""),
            }
            for row in requests
        ]
        if not request_rows:
            return {}
        expected_ids = [row["crop_id"] for row in request_rows]
        if any(not crop_id for crop_id in expected_ids):
            raise ValueError("MediaPipe TFLite rescue request requires crop_id")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("MediaPipe TFLite rescue request contains duplicate crop_id")
        if any(not row["crop_path"] for row in request_rows):
            raise ValueError("MediaPipe TFLite rescue request requires crop_path")

        with tempfile.TemporaryDirectory(prefix="hlmf-mp-tflite-") as temp_dir:
            temp_root = Path(temp_dir)
            request_path = temp_root / "request.jsonl"
            output_path = temp_root / "output.jsonl"
            write_jsonl(request_path, request_rows)
            result = subprocess.run(
                [
                    str(self.python_executable),
                    "-B",
                    str(self.worker_path),
                    "--model",
                    str(self.model_path),
                    "--request",
                    str(request_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "worker failed").strip()
                raise RuntimeError(
                    f"MediaPipe TFLite rescue worker failed ({result.returncode}): {detail}"
                )
            response_rows = read_jsonl(output_path)

        by_id: Dict[str, Dict[str, Any]] = {}
        for row in response_rows:
            crop_id = str(row.get("crop_id") or "")
            if not crop_id or crop_id in by_id:
                raise ValueError(
                    "MediaPipe TFLite rescue response has a missing or duplicate crop_id"
                )
            _validate_landmark_rows(
                row.get("landmarks_crop_px"), "landmarks_crop_px"
            )
            _validate_landmark_rows(
                row.get("landmarks_crop_norm"), "landmarks_crop_norm"
            )
            by_id[crop_id] = dict(row)
        if set(by_id) != set(expected_ids):
            missing = sorted(set(expected_ids) - set(by_id))
            extra = sorted(set(by_id) - set(expected_ids))
            raise ValueError(
                f"MediaPipe TFLite rescue response mismatch: missing={missing}, extra={extra}"
            )
        return by_id
