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
    HandClassifierONNX,
    decode_hand_classifier_logits,
    hand_classifier_model_id_from_path,
    preprocess_hand_classifier_image,
)


def _node(name: str, shape: list[object], node_type: str = "tensor(float)") -> SimpleNamespace:
    return SimpleNamespace(name=name, shape=shape, type=node_type)


class _FakeSession:
    def __init__(self, _path: str, providers: list[str]) -> None:
        self.requested_providers = providers

    def get_inputs(self) -> list[SimpleNamespace]:
        return [_node("input", ["batch", 1, 256, 256])]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [
            _node("handedness", ["batch", 2]),
            _node("hand_presence", ["batch", 2]),
        ]

    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if names != ["handedness", "hand_presence"] or feeds["input"].shape != (
            1,
            1,
            256,
            256,
        ):
            raise AssertionError((names, feeds))
        return [
            np.asarray([[-1.0, 2.0]], dtype=np.float32),
            np.asarray([[1.0, 3.0]], dtype=np.float32),
        ]


class HandClassifierTests(unittest.TestCase):
    def test_preprocess_matches_training_normalization(self) -> None:
        tensor = preprocess_hand_classifier_image(
            np.full((128, 128, 3), 100, dtype=np.uint8)
        )
        self.assertEqual((1, 1, 256, 256), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        expected = (100.0 / 255.0 - HAND_CLASSIFIER_MEAN) / HAND_CLASSIFIER_STD
        np.testing.assert_allclose(tensor[0, 0, 0, 0], expected, rtol=1e-6)
        self.assertTrue(np.isfinite(tensor).all())

    def test_decode_returns_handedness_and_probability_of_has_hand(self) -> None:
        result = decode_hand_classifier_logits(
            np.asarray([[0.0, 2.0]], dtype=np.float32),
            np.asarray([[2.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual("Right", result["handedness"]["label"])
        self.assertAlmostEqual(0.880797, result["handedness"]["score"], places=6)
        self.assertFalse(result["hand_presence"]["present"])
        self.assertAlmostEqual(0.119203, result["hand_presence"]["score"], places=6)
        with self.assertRaisesRegex(ValueError, "hand_presence output shape"):
            decode_hand_classifier_logits(
                np.zeros((1, 2), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "handedness output contains non-finite"):
            decode_hand_classifier_logits(
                np.asarray([[np.nan, 1.0]], dtype=np.float32),
                np.zeros((1, 2), dtype=np.float32),
            )

    def test_session_prefers_cuda_and_runs_both_heads(self) -> None:
        fake_ort = SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=_FakeSession,
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "handedness-handpresence-0809" / "model.onnx"
            model.parent.mkdir()
            model.write_bytes(b"test")
            with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                classifier = HandClassifierONNX(model)
                result = classifier.classify(np.zeros((256, 256), dtype=np.uint8))
        self.assertEqual("CUDAExecutionProvider", classifier.provider)
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0809", classifier.model_id
        )
        self.assertEqual("Right", result["handedness"]["label"])
        self.assertTrue(result["hand_presence"]["present"])
        self.assertAlmostEqual(0.880797, result["hand_presence"]["score"], places=6)

    def test_model_id_is_derived_from_version_directory(self) -> None:
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0809",
            hand_classifier_model_id_from_path(
                Path("models/hand_classifier/handedness-handpresence-0809/model.onnx")
            ),
        )
        with self.assertRaisesRegex(ValueError, "safely named"):
            hand_classifier_model_id_from_path(
                Path("models/hand_classifier/bad version/model.onnx")
            )

    def test_interface_rejects_static_batch_and_wrong_outputs(self) -> None:
        session = _FakeSession("unused", ["CPUExecutionProvider"])
        session.get_inputs = lambda: [_node("input", [1, 1, 256, 256])]
        classifier = HandClassifierONNX.__new__(HandClassifierONNX)
        classifier.session = session
        with self.assertRaisesRegex(ValueError, "input shape"):
            classifier._validate_model_interface()

        session.get_inputs = lambda: [_node("input", ["batch", 1, 256, 256])]
        session.get_outputs = lambda: [_node("output", ["batch", 2])]
        with self.assertRaisesRegex(ValueError, "outputs must be exactly"):
            classifier._validate_model_interface()


if __name__ == "__main__":
    unittest.main()
