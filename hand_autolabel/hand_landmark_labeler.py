from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .hamer_hand_labeler import label_roi_manifest_hamer
from .mediapipe_roi_labeler import label_roi_manifest as label_roi_manifest_mediapipe
from .rtmpose_hand_labeler import label_roi_manifest_rtmpose


SUPPORTED_HAND_LANDMARK_BACKENDS = ("mediapipe_tasks", "rtmpose_onnx", "hamer")


def label_hand_landmark_manifest(
    manifest_rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    root: Path,
    *,
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    backend = str((cfg.get("hand_landmark") or {}).get("backend", "mediapipe_tasks"))
    if backend == "mediapipe_tasks":
        rows, mode = label_roi_manifest_mediapipe(
            manifest_rows,
            cfg,
            root,
            show_progress=show_progress,
        )
        return rows, {
            "backend": backend,
            "mode": mode,
            "provider": None,
            "runtime_rois_labeled": sum(
                bool((row.get("hand_presence") or {}).get("present")) for row in rows
            ),
            "negative_candidates_skipped": 0,
        }
    if backend == "rtmpose_onnx":
        return label_roi_manifest_rtmpose(
            manifest_rows,
            cfg,
            root,
            show_progress=show_progress,
        )
    if backend == "hamer":
        return label_roi_manifest_hamer(
            manifest_rows,
            cfg,
            root,
            show_progress=show_progress,
        )
    supported = ", ".join(SUPPORTED_HAND_LANDMARK_BACKENDS)
    raise ValueError(f"unsupported Hand landmark backend {backend!r}; choose one of: {supported}")
