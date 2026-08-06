from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np

from .handedness_classifier import (
    HAND_CLASSIFIER_MODEL_ID,
    HandednessONNXClassifier,
)
from .formats import make_hand_id, merge_label_with_manifest, resolve_path
from .image_io import read_image, to_uint8_gray
from .progress import track_progress
from .projection import landmark_dicts_from_norm


RTMPOSE_INPUT_NAME = "input"
RTMPOSE_OUTPUT_NAMES = ("simcc_x", "simcc_y")
RTMPOSE_INPUT_SIZE = (256, 256)
RTMPOSE_KEYPOINTS = 21
RTMPOSE_SIMCC_BINS = 512
RTMPOSE_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
RTMPOSE_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_rtmpose_image(image: np.ndarray) -> np.ndarray:
    """Create the official MMPose RGB/mean/std NCHW input tensor."""

    gray = to_uint8_gray(image)
    width, height = RTMPOSE_INPUT_SIZE
    if gray.shape != (height, width):
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    tensor = (rgb - RTMPOSE_MEAN) / RTMPOSE_STD
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    if tensor.shape != (1, 3, height, width) or not np.isfinite(tensor).all():
        raise ValueError("RTMPose preprocessing produced an invalid input tensor")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    *,
    split_ratio: float,
    input_size: Sequence[int] = RTMPOSE_INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw SimCC logits as MMPose does: argmax, then divide by ratio."""

    x = np.asarray(simcc_x)
    y = np.asarray(simcc_y)
    expected_x = (1, RTMPOSE_KEYPOINTS, RTMPOSE_SIMCC_BINS)
    expected_y = (1, RTMPOSE_KEYPOINTS, RTMPOSE_SIMCC_BINS)
    if x.shape != expected_x or y.shape != expected_y:
        raise ValueError(
            f"unexpected RTMPose output shapes: simcc_x={x.shape}, simcc_y={y.shape}; "
            f"expected {expected_x} and {expected_y}"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("RTMPose output contains non-finite SimCC logits")
    if not np.isfinite(split_ratio) or split_ratio <= 0:
        raise ValueError("rtmpose.simcc_split_ratio must be a positive finite number")

    x_indices = np.argmax(x[0], axis=1).astype(np.float32)
    y_indices = np.argmax(y[0], axis=1).astype(np.float32)
    scores = np.minimum(np.max(x[0], axis=1), np.max(y[0], axis=1)).astype(np.float32)

    coordinates = np.stack((x_indices, y_indices), axis=1) / float(split_ratio)
    width, height = (int(input_size[0]), int(input_size[1]))
    if width < 2 or height < 2:
        raise ValueError("RTMPose input dimensions must be at least 2 pixels")
    # The last SimCC bin decodes to 255.5 at ratio 2.0. Clamp only this
    # half-pixel border overshoot before normalization to the ROI contract.
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0.0, float(width - 1))
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0.0, float(height - 1))
    if coordinates.shape != (RTMPOSE_KEYPOINTS, 2) or not np.isfinite(coordinates).all():
        raise ValueError("RTMPose decoding produced invalid keypoint coordinates")
    return coordinates, scores


def _shape_matches(actual: Sequence[Any], expected: Sequence[int | None]) -> bool:
    if len(actual) != len(expected):
        return False
    for value, wanted in zip(actual, expected):
        if wanted is None:
            continue
        # ONNX exports may retain a symbolic dimension even when every runtime
        # output is fixed. Reject contradictory static dimensions here; the
        # decoder validates the concrete output tensor exactly on every call.
        if isinstance(value, int) and value != wanted:
            return False
    return True


class RTMPoseONNXHandLabeler:
    def __init__(self, model_path: Path, split_ratio: float) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError("onnxruntime is required for the RTMPose ONNX backend") from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"RTMPose ONNX model does not exist: {model_path}")
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.split_ratio = float(split_ratio)
        self._validate_model_interface()
        active = self.session.get_providers()
        if not active:
            raise RuntimeError("ONNX Runtime did not activate an execution provider")
        self.provider = str(active[0])

    def _validate_model_interface(self) -> None:
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != RTMPOSE_INPUT_NAME:
            raise ValueError("RTMPose ONNX must expose exactly one input named 'input'")
        if not _shape_matches(inputs[0].shape, (None, 3, 256, 256)):
            raise ValueError(f"unexpected RTMPose input shape: {inputs[0].shape}")
        by_name = {output.name: output for output in outputs}
        if set(by_name) != set(RTMPOSE_OUTPUT_NAMES):
            raise ValueError("RTMPose ONNX outputs must be exactly 'simcc_x' and 'simcc_y'")
        for name in RTMPOSE_OUTPUT_NAMES:
            if not _shape_matches(by_name[name].shape, (None, 21, 512)):
                raise ValueError(f"unexpected RTMPose output shape for {name}: {by_name[name].shape}")

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = preprocess_rtmpose_image(image)
        outputs = self.session.run(list(RTMPOSE_OUTPUT_NAMES), {RTMPOSE_INPUT_NAME: tensor})
        if len(outputs) != 2:
            raise ValueError(f"RTMPose ONNX returned {len(outputs)} outputs, expected 2")
        return decode_simcc(
            outputs[0],
            outputs[1],
            split_ratio=self.split_ratio,
            input_size=RTMPOSE_INPUT_SIZE,
        )


def _is_negative_candidate(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("palm_valid") is False or manifest.get("proposal_kind") == "negative_candidate"


def _unassessed_candidate_label(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "crop_id": manifest["crop_id"],
        "image": manifest["image"],
        "palm_det_id": manifest["palm_det_id"],
        "hand_id": None,
        "hand_presence": {"present": False},
        "handedness": {"label": "unknown", "score": None},
        "handedness_teacher_model_id": None,
        "human_modified_handedness": False,
        "landmarks_crop_norm": [],
        "landmarks_crop_px": [],
        "landmarks_image_px": [],
        "source": "eos_negative_candidate_unassessed",
    }


def label_one_roi_rtmpose(
    manifest: Mapping[str, Any],
    image: np.ndarray,
    detector: Any,
    handedness_classifier: Any,
    cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    if _is_negative_candidate(manifest):
        return merge_label_with_manifest(_unassessed_candidate_label(manifest), manifest, cfg)

    output_size = manifest.get("output_size") or [
        cfg["hand_roi"]["output_width"],
        cfg["hand_roi"]["output_height"],
    ]
    crop_width, crop_height = int(output_size[0]), int(output_size[1])
    if (crop_width, crop_height) != RTMPOSE_INPUT_SIZE:
        raise ValueError(
            f"RTMPose requires a 256x256 Hand ROI, got {crop_width}x{crop_height} "
            f"for {manifest.get('crop_id')}"
        )
    coordinates, _scores = detector.detect(image)
    if coordinates.shape != (RTMPOSE_KEYPOINTS, 2):
        raise ValueError(f"RTMPose must return 21 keypoints, got {coordinates.shape}")
    handedness = handedness_classifier.classify(image)
    crop_px = [
        {"id": idx, "x": float(point[0]), "y": float(point[1])}
        for idx, point in enumerate(coordinates)
    ]
    crop_norm = [
        {
            "id": point["id"],
            "x": point["x"] / float(crop_width - 1),
            "y": point["y"] / float(crop_height - 1),
        }
        for point in crop_px
    ]
    image_px = landmark_dicts_from_norm(crop_norm, manifest["roi_corners_px"])
    row = {
        "crop_id": manifest["crop_id"],
        "image": manifest["image"],
        "palm_det_id": manifest["palm_det_id"],
        "hand_id": make_hand_id(str(manifest["crop_id"])),
        # Routing sentinel only. It is not a supervised presence label.
        "hand_presence": {"present": True},
        "handedness": handedness,
        "handedness_teacher_model_id": HAND_CLASSIFIER_MODEL_ID,
        "human_modified_handedness": False,
        "landmarks_crop_norm": crop_norm,
        "landmarks_crop_px": crop_px,
        "landmarks_image_px": image_px,
        "source": "rtmpose_m_hand5_onnx",
        # Legacy draft/release schema field. For this backend the value means
        # one runtime ROI was labeled, while provenance remains RTMPose.
        "mediapipe_num_hands_detected": 1,
    }
    return merge_label_with_manifest(row, manifest, cfg)


def label_roi_manifest_rtmpose(
    manifest_rows: Iterable[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    root: Path,
    *,
    show_progress: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rtmpose_cfg = cfg.get("rtmpose") or {}
    split_ratio = float(rtmpose_cfg.get("simcc_split_ratio", 2.0))
    if split_ratio != 2.0:
        raise ValueError("rtmpose.simcc_split_ratio is fixed at 2.0 for this model")
    model_path = resolve_path(root, rtmpose_cfg.get("model_onnx_path", ""))
    classifier_cfg = cfg.get("hand_classifier") or {}
    classifier_model_path = resolve_path(root, classifier_cfg.get("model_onnx_path", ""))
    detector: RTMPoseONNXHandLabeler | None = None
    handedness_classifier: HandednessONNXClassifier | None = None
    rows: List[Dict[str, Any]] = []
    runtime_labeled = 0
    candidates_skipped = 0
    for manifest in track_progress(
        manifest_rows,
        enabled=show_progress,
        description="RTMPose hand landmarks",
        unit="roi",
    ):
        if _is_negative_candidate(manifest):
            rows.append(
                merge_label_with_manifest(_unassessed_candidate_label(manifest), manifest, cfg)
            )
            candidates_skipped += 1
            continue
        if detector is None:
            detector = RTMPoseONNXHandLabeler(model_path, split_ratio)
            handedness_classifier = HandednessONNXClassifier(classifier_model_path)
        crop_path = resolve_path(root, manifest["crop_path"])
        image = read_image(crop_path)
        if image is None:
            raise RuntimeError(f"unreadable runtime Hand ROI for RTMPose: {crop_path}")
        if handedness_classifier is None:
            raise RuntimeError("Hand classifier was not initialized for a runtime ROI")
        rows.append(
            label_one_roi_rtmpose(manifest, image, detector, handedness_classifier, cfg)
        )
        runtime_labeled += 1
    return rows, {
        "backend": "rtmpose_onnx",
        "mode": "rtmpose_onnx",
        "provider": detector.provider if detector is not None else None,
        "handedness_classifier_provider": (
            handedness_classifier.provider if handedness_classifier is not None else None
        ),
        "handedness_classifier_model_id": (
            handedness_classifier.model_id if handedness_classifier is not None else None
        ),
        "handedness_runtime_rois_labeled": runtime_labeled,
        "runtime_rois_labeled": runtime_labeled,
        "negative_candidates_skipped": candidates_skipped,
    }
