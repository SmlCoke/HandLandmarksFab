from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


ONNX_PROVIDER_AUTO = "auto"
ONNX_PROVIDER_CUDA = "cuda"
ONNX_PROVIDER_CPU = "cpu"
ONNX_PROVIDER_CHOICES = {
    ONNX_PROVIDER_AUTO,
    ONNX_PROVIDER_CUDA,
    ONNX_PROVIDER_CPU,
}


def onnx_runtime_settings(cfg: Mapping[str, Any]) -> tuple[str, int]:
    runtime_cfg = cfg.get("onnx_runtime") or {}
    if not isinstance(runtime_cfg, Mapping):
        raise ValueError("onnx_runtime must be a mapping")
    provider = str(runtime_cfg.get("provider", ONNX_PROVIDER_AUTO)).strip().lower()
    if provider not in ONNX_PROVIDER_CHOICES:
        raise ValueError("onnx_runtime.provider must be one of: auto, cuda, cpu")
    batch_size = runtime_cfg.get("batch_size", 32)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("onnx_runtime.batch_size must be a positive integer")
    return provider, batch_size


def onnx_provider_for(cfg: Mapping[str, Any], model_name: str) -> str:
    default_provider, _batch_size = onnx_runtime_settings(cfg)
    runtime_cfg = cfg.get("onnx_runtime") or {}
    overrides = runtime_cfg.get("model_providers") or {}
    if not isinstance(overrides, Mapping):
        raise ValueError("onnx_runtime.model_providers must be a mapping")
    provider = str(overrides.get(model_name, default_provider)).strip().lower()
    if provider not in ONNX_PROVIDER_CHOICES:
        raise ValueError(
            f"onnx_runtime.model_providers.{model_name} must be one of: auto, cuda, cpu"
        )
    return provider


def create_onnx_session(
    model_path: Path,
    provider_preference: str = ONNX_PROVIDER_AUTO,
) -> tuple[Any, str, str | None]:
    """Create a CUDA-first session, with CPU fallback only in auto mode."""

    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("onnxruntime is required for ONNX model inference") from exc

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    preference = str(provider_preference).strip().lower()
    if preference not in ONNX_PROVIDER_CHOICES:
        raise ValueError("ONNX provider preference must be one of: auto, cuda, cpu")

    available = set(ort.get_available_providers())
    fallback_reason: str | None = None
    if preference == ONNX_PROVIDER_CPU:
        requested = ["CPUExecutionProvider"]
    elif "CUDAExecutionProvider" in available:
        requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif preference == ONNX_PROVIDER_CUDA:
        raise RuntimeError(
            "onnx_runtime.provider=cuda but CUDAExecutionProvider is unavailable"
        )
    else:
        requested = ["CPUExecutionProvider"]
        fallback_reason = "CUDAExecutionProvider unavailable"

    try:
        session = ort.InferenceSession(str(model_path), providers=requested)
    except Exception as exc:
        if preference != ONNX_PROVIDER_AUTO or requested == ["CPUExecutionProvider"]:
            raise
        session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        fallback_reason = f"CUDA session initialization failed: {type(exc).__name__}"

    active = list(session.get_providers())
    if not active:
        raise RuntimeError("ONNX Runtime did not activate an execution provider")
    provider = str(active[0])
    if preference == ONNX_PROVIDER_CUDA and provider != "CUDAExecutionProvider":
        raise RuntimeError(
            "onnx_runtime.provider=cuda but CUDAExecutionProvider did not activate"
        )
    if preference == ONNX_PROVIDER_AUTO and provider != "CUDAExecutionProvider":
        fallback_reason = fallback_reason or "CUDAExecutionProvider did not activate"
    return session, provider, fallback_reason
