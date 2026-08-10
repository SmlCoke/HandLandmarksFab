from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from .image_io import to_uint8_gray
from .onnx_runtime import create_onnx_session


HAND_CLASSIFIER_INPUT_NAME = "input"
HAND_CLASSIFIER_OUTPUT_NAMES = ("handedness", "hand_presence")
HAND_CLASSIFIER_INPUT_SIZE = (256, 256)
HANDEDNESS_LABELS = ("Left", "Right")
HAND_CLASSIFIER_MEAN = 0.485
HAND_CLASSIFIER_STD = 0.229


def hand_classifier_model_id_from_path(model_path: Path) -> str:
    version = Path(model_path).parent.name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError(
            "Hand classifier model must be stored in a safely named version directory"
        )
    return f"hand-classifier-{version}"


def preprocess_hand_classifier_image(image: np.ndarray) -> np.ndarray:
    """Build the normalized grayscale tensor used to train the classifier."""

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


def preprocess_hand_classifier_images(images: Sequence[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("Hand classifier batch must contain at least one image")
    return np.concatenate(
        [preprocess_hand_classifier_image(image) for image in images], axis=0
    )


def _softmax_probabilities(logits: np.ndarray, *, output_name: str) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.shape != (1, 2):
        raise ValueError(
            f"unexpected Hand classifier {output_name} output shape: {values.shape}; "
            "expected (1, 2)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Hand classifier {output_name} output contains non-finite logits")
    shifted = values[0] - float(np.max(values[0]))
    exponentials = np.exp(shifted)
    probabilities = exponentials / float(np.sum(exponentials))
    if probabilities.shape != (2,) or not np.isfinite(probabilities).all():
        raise ValueError(f"Hand classifier {output_name} softmax is invalid")
    return probabilities


def decode_hand_classifier_logits(
    handedness_logits: np.ndarray,
    hand_presence_logits: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    handedness_probabilities = _softmax_probabilities(
        handedness_logits, output_name="handedness"
    )
    presence_probabilities = _softmax_probabilities(
        hand_presence_logits, output_name="hand_presence"
    )
    handedness_class_id = int(np.argmax(np.asarray(handedness_logits)[0]))
    presence_class_id = int(np.argmax(np.asarray(hand_presence_logits)[0]))
    return {
        "handedness": {
            "label": HANDEDNESS_LABELS[handedness_class_id],
            "score": float(handedness_probabilities[handedness_class_id]),
        },
        "hand_presence": {
            "present": presence_class_id == 1,
            # This is always P(has_hand), not the winning-class probability.
            "score": float(presence_probabilities[1]),
        },
    }


def decode_hand_classifier_batch(
    handedness_logits: np.ndarray,
    hand_presence_logits: np.ndarray,
) -> list[Dict[str, Dict[str, Any]]]:
    handedness = np.asarray(handedness_logits)
    presence = np.asarray(hand_presence_logits)
    if (
        handedness.ndim != 2
        or presence.ndim != 2
        or handedness.shape[1:] != (2,)
        or presence.shape != handedness.shape
        or handedness.shape[0] < 1
    ):
        raise ValueError(
            "unexpected Hand classifier batch output shapes: "
            f"handedness={handedness.shape}, hand_presence={presence.shape}"
        )
    return [
        decode_hand_classifier_logits(
            handedness[index : index + 1], presence[index : index + 1]
        )
        for index in range(handedness.shape[0])
    ]


def _shape_matches(actual: Sequence[Any], expected: Sequence[int | None]) -> bool:
    if len(actual) != len(expected):
        return False
    for value, wanted in zip(actual, expected):
        if wanted is None:
            if isinstance(value, int):
                return False
            continue
        if not isinstance(value, int) or value != wanted:
            return False
    return True


class HandClassifierONNX:
    def __init__(self, model_path: Path, provider_preference: str = "auto") -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Hand classifier ONNX model does not exist: {model_path}")
        self.model_id = hand_classifier_model_id_from_path(model_path)
        self.session, self.provider, self.fallback_reason = create_onnx_session(
            model_path, provider_preference
        )
        self._validate_model_interface()

    @staticmethod
    def _validate_tensor_type(node: Any, *, description: str) -> None:
        node_type = getattr(node, "type", None)
        if node_type is not None and node_type != "tensor(float)":
            raise ValueError(f"{description} must use tensor(float), got {node_type!r}")

    def _validate_model_interface(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != HAND_CLASSIFIER_INPUT_NAME:
            raise ValueError("Hand classifier ONNX must expose exactly one input named 'input'")
        if not _shape_matches(inputs[0].shape, (None, 1, 256, 256)):
            raise ValueError(f"unexpected Hand classifier input shape: {inputs[0].shape}")
        self._validate_tensor_type(inputs[0], description="Hand classifier input")
        by_name = {output.name: output for output in outputs}
        if set(by_name) != set(HAND_CLASSIFIER_OUTPUT_NAMES):
            raise ValueError(
                "Hand classifier ONNX outputs must be exactly 'handedness' and 'hand_presence'"
            )
        for name in HAND_CLASSIFIER_OUTPUT_NAMES:
            if not _shape_matches(by_name[name].shape, (None, 2)):
                raise ValueError(
                    f"unexpected Hand classifier output shape for {name}: {by_name[name].shape}"
                )
            self._validate_tensor_type(
                by_name[name], description=f"Hand classifier {name} output"
            )

    def classify(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        return self.classify_batch([image])[0]

    def classify_batch(
        self, images: Sequence[np.ndarray]
    ) -> list[Dict[str, Dict[str, Any]]]:
        tensor = preprocess_hand_classifier_images(images)
        outputs = self.session.run(
            list(HAND_CLASSIFIER_OUTPUT_NAMES),
            {HAND_CLASSIFIER_INPUT_NAME: tensor},
        )
        if len(outputs) != 2:
            raise ValueError(f"Hand classifier ONNX returned {len(outputs)} outputs, expected 2")
        return decode_hand_classifier_batch(outputs[0], outputs[1])
