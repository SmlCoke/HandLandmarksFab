from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hand_autolabel.onnx_runtime import (
    create_onnx_session,
    onnx_provider_for,
    onnx_runtime_settings,
)


class _Session:
    def __init__(self, _path: str, providers: list[str]) -> None:
        self.providers = providers

    def get_providers(self) -> list[str]:
        return self.providers


class ONNXRuntimeTests(unittest.TestCase):
    def test_settings_validate_provider_and_batch_size(self) -> None:
        self.assertEqual(("auto", 32), onnx_runtime_settings({}))
        self.assertEqual(
            ("cpu", 8),
            onnx_runtime_settings(
                {"onnx_runtime": {"provider": "CPU", "batch_size": 8}}
            ),
        )
        self.assertEqual(
            "cuda",
            onnx_provider_for(
                {
                    "onnx_runtime": {
                        "provider": "cpu",
                        "model_providers": {"rtmpose": "cuda"},
                    }
                },
                "rtmpose",
            ),
        )
        with self.assertRaisesRegex(ValueError, "provider"):
            onnx_runtime_settings({"onnx_runtime": {"provider": "tensorrt"}})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            onnx_runtime_settings({"onnx_runtime": {"batch_size": 0}})

    def test_auto_falls_back_to_cpu_and_explicit_cuda_is_strict(self) -> None:
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=_Session,
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.onnx"
            model.write_bytes(b"model")
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                session, provider, reason = create_onnx_session(model, "auto")
                self.assertEqual("CPUExecutionProvider", provider)
                self.assertIn("unavailable", str(reason))
                self.assertEqual(["CPUExecutionProvider"], session.providers)
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    create_onnx_session(model, "cuda")


if __name__ == "__main__":
    unittest.main()
