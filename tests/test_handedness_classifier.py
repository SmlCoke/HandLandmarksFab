from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hand_autolabel.handedness_classifier import (
    HAND_CLASSIFIER_MEAN,
    HAND_CLASSIFIER_STD,
    HandednessONNXClassifier,
    decode_handedness_logits,
    preprocess_handedness_image,
)


class _FakeSession:
    def __init__(self, _path: str, providers: list[str]) -> None:
        self.requested_providers = providers

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input", shape=["batch", 1, "height", "width"])]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="output", shape=["batch", 2])]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if names != ["output"] or feeds["input"].shape != (1, 1, 256, 256):
            raise AssertionError((names, feeds))
        return [np.asarray([[-1.0, 2.0]], dtype=np.float32)]


class HandednessClassifierTests(unittest.TestCase):
    def test_preprocess_matches_training_normalization(self) -> None:
        tensor = preprocess_handedness_image(
            np.full((128, 128), 100, dtype=np.uint8)
        )
        self.assertEqual((1, 1, 256, 256), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        expected = (100.0 / 255.0 - HAND_CLASSIFIER_MEAN) / HAND_CLASSIFIER_STD
        np.testing.assert_allclose(tensor[0, 0, 0, 0], expected, rtol=1e-6)
        self.assertTrue(np.isfinite(tensor).all())

    def test_decode_uses_argmax_and_softmax_probability(self) -> None:
        result = decode_handedness_logits(
            np.asarray([[0.0, 2.0]], dtype=np.float32)
        )
        self.assertEqual("Right", result["label"])
        self.assertAlmostEqual(0.880797, result["score"], places=6)
        with self.assertRaisesRegex(ValueError, "output shape"):
            decode_handedness_logits(np.zeros((1, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            decode_handedness_logits(
                np.asarray([[np.nan, 1.0]], dtype=np.float32)
            )

    def test_session_prefers_cuda_and_validates_runtime_output(self) -> None:
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=_FakeSession,
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.onnx"
            model.write_bytes(b"test")
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                classifier = HandednessONNXClassifier(model)
                result = classifier.classify(
                    np.zeros((256, 256), dtype=np.uint8)
                )
        self.assertEqual("CUDAExecutionProvider", classifier.provider)
        self.assertEqual("Right", result["label"])
        self.assertGreater(result["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
