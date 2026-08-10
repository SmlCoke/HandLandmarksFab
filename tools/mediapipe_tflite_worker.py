from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import cv2
import numpy as np


MODEL_INPUT_SIZE = 224
CROP_SIZE = 256
EXPECTED_OUTPUT_SHAPES = ((1, 63), (1, 1), (1, 1), (1, 63))


def preprocess_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unreadable Hand ROI: {path}")
    resized = cv2.resize(
        image,
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = np.repeat(resized[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    tensor = rgb[None, ...]
    if tensor.shape != (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
        raise ValueError(f"unexpected preprocessed tensor shape: {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("MediaPipe TFLite preprocessing produced non-finite values")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def decode_landmarks(raw: np.ndarray) -> tuple[list[Dict[str, float]], list[Dict[str, float]]]:
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (1, 63):
        raise ValueError(f"unexpected MediaPipe landmark output shape: {values.shape}")
    points = values.reshape(21, 3)
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("MediaPipe TFLite landmarks contain non-finite coordinates")
    crop_px: list[Dict[str, float]] = []
    crop_norm: list[Dict[str, float]] = []
    for index, point in enumerate(points):
        x_norm = float(point[0]) / float(MODEL_INPUT_SIZE)
        y_norm = float(point[1]) / float(MODEL_INPUT_SIZE)
        crop_norm.append({"id": index, "x": x_norm, "y": y_norm})
        crop_px.append(
            {
                "id": index,
                "x": x_norm * float(CROP_SIZE),
                "y": y_norm * float(CROP_SIZE),
            }
        )
    return crop_px, crop_norm


class MediaPipeTFLiteInterpreter:
    def __init__(self, model_path: Path) -> None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except Exception as exc:  # pragma: no cover - independent runtime only.
            raise RuntimeError(
                "tflite-runtime is required in the MediaPipe TFLite environment"
            ) from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MediaPipe TFLite model does not exist: {model_path}")
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1:
            raise ValueError("MediaPipe TFLite model must expose exactly one input")
        input_shape = tuple(int(value) for value in inputs[0]["shape"])
        if input_shape != (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3):
            raise ValueError(f"unexpected MediaPipe TFLite input shape: {input_shape}")
        if np.dtype(inputs[0]["dtype"]) != np.dtype(np.float32):
            raise ValueError(f"unexpected MediaPipe TFLite input dtype: {inputs[0]['dtype']}")
        if len(outputs) != len(EXPECTED_OUTPUT_SHAPES):
            raise ValueError(
                f"MediaPipe TFLite model returned {len(outputs)} outputs, expected 4"
            )
        for index, (detail, expected_shape) in enumerate(
            zip(outputs, EXPECTED_OUTPUT_SHAPES)
        ):
            actual_shape = tuple(int(value) for value in detail["shape"])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"unexpected MediaPipe TFLite output {index} shape: {actual_shape}; "
                    f"expected {expected_shape}"
                )
            if np.dtype(detail["dtype"]) != np.dtype(np.float32):
                raise ValueError(
                    f"unexpected MediaPipe TFLite output {index} dtype: {detail['dtype']}"
                )
        self.input_index = int(inputs[0]["index"])
        # Output order is fixed by hand_landmark_full.tflite: landmarks,
        # handflag, handedness, world landmarks. Only landmarks are consumed.
        self.landmark_output_index = int(outputs[0]["index"])

    def predict(self, image_path: Path) -> tuple[list[Dict[str, float]], list[Dict[str, float]]]:
        tensor = preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_index, tensor)
        self.interpreter.invoke()
        raw_landmarks = self.interpreter.get_tensor(self.landmark_output_index)
        return decode_landmarks(raw_landmarks)


def _read_requests(path: Path) -> list[Dict[str, str]]:
    requests: list[Dict[str, str]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid request JSONL at line {line_number}") from exc
            if not isinstance(item, Mapping):
                raise ValueError(f"request line {line_number} must be an object")
            crop_id = str(item.get("crop_id") or "")
            crop_path = str(item.get("crop_path") or "")
            if not crop_id or not crop_path:
                raise ValueError(f"request line {line_number} requires crop_id and crop_path")
            if crop_id in seen:
                raise ValueError(f"duplicate crop_id in request: {crop_id}")
            seen.add(crop_id)
            requests.append({"crop_id": crop_id, "crop_path": crop_path})
    if not requests:
        raise ValueError("MediaPipe TFLite request is empty")
    return requests


def _write_results(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch MediaPipe Hand Landmarker TFLite worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        requests = _read_requests(Path(args.request))
        detector = MediaPipeTFLiteInterpreter(Path(args.model))
        output = []
        for request in requests:
            crop_px, crop_norm = detector.predict(Path(request["crop_path"]))
            output.append(
                {
                    "crop_id": request["crop_id"],
                    "landmarks_crop_px": crop_px,
                    "landmarks_crop_norm": crop_norm,
                }
            )
        _write_results(Path(args.output), output)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
