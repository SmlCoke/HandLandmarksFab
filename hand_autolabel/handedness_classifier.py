from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from .image_io import to_uint8_gray


HAND_CLASSIFIER_INPUT_NAME = "input"
HAND_CLASSIFIER_OUTPUT_NAME = "output"
HAND_CLASSIFIER_INPUT_SIZE = (256, 256)
HAND_CLASSIFIER_LABELS = ("Left", "Right")
HAND_CLASSIFIER_MEAN = 0.485
HAND_CLASSIFIER_STD = 0.229
HAND_CLASSIFIER_MODEL_ID = "hand-classifier-mobilenetv3-small-v1"


def preprocess_handedness_image(image: np.ndarray) -> np.ndarray:
    """Build the grayscale tensor used to train the handedness classifier."""

    gray = to_uint8_gray(image)
    width, height = HAND_CLASSIFIER_INPUT_SIZE
    if gray.shape != (height, width):
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    tensor = gray.astype(np.float32) / 255.0
    tensor = (tensor - HAND_CLASSIFIER_MEAN) / HAND_CLASSIFIER_STD
    tensor = tensor[None, None, :, :]
    if tensor.shape != (1, 1, height, width) or not np.isfinite(tensor).all():
        raise ValueError("Hand classifier preprocessing produced an invalid input tensor")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def decode_handedness_logits(logits: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(logits, dtype=np.float32)
    if values.shape != (1, 2):
        raise ValueError(f"unexpected Hand classifier output shape: {values.shape}; expected (1, 2)")
    if not np.isfinite(values).all():
        raise ValueError("Hand classifier output contains non-finite logits")
    shifted = values[0] - float(np.max(values[0]))
    exponentials = np.exp(shifted)
    probabilities = exponentials / float(np.sum(exponentials))
    class_id = int(np.argmax(values[0]))
    return {
        "label": HAND_CLASSIFIER_LABELS[class_id],
        "score": float(probabilities[class_id]),
    }


def _shape_matches(actual: Sequence[Any], expected: Sequence[int | None]) -> bool:
    if len(actual) != len(expected):
        return False
    for value, wanted in zip(actual, expected):
        if wanted is None:
            continue
        if isinstance(value, int) and value != wanted:
            return False
    return True


class HandednessONNXClassifier:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("onnxruntime is required for the Hand classifier") from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Hand classifier ONNX model does not exist: {model_path}")
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self._validate_model_interface()
        active = self.session.get_providers()
        if not active:
            raise RuntimeError("ONNX Runtime did not activate a Hand classifier execution provider")
        self.provider = str(active[0])
        self.model_id = HAND_CLASSIFIER_MODEL_ID

    def _validate_model_interface(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != HAND_CLASSIFIER_INPUT_NAME:
            raise ValueError("Hand classifier ONNX must expose exactly one input named 'input'")
        if not _shape_matches(inputs[0].shape, (None, 1, 256, 256)):
            raise ValueError(f"unexpected Hand classifier input shape: {inputs[0].shape}")
        if len(outputs) != 1 or outputs[0].name != HAND_CLASSIFIER_OUTPUT_NAME:
            raise ValueError("Hand classifier ONNX must expose exactly one output named 'output'")
        if not _shape_matches(outputs[0].shape, (None, 2)):
            raise ValueError(f"unexpected Hand classifier output shape: {outputs[0].shape}")

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        tensor = preprocess_handedness_image(image)
        outputs = self.session.run(
            [HAND_CLASSIFIER_OUTPUT_NAME],
            {HAND_CLASSIFIER_INPUT_NAME: tensor},
        )
        if len(outputs) != 1:
            raise ValueError(f"Hand classifier ONNX returned {len(outputs)} outputs, expected 1")
        return decode_handedness_logits(outputs[0])
