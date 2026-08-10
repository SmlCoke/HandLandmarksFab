from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hand_autolabel.dataset_v3 import apply_label_provenance
from hand_autolabel.hand_landmark_labeler import label_hand_landmark_manifest
from hand_autolabel.mediapipe_roi_labeler import label_one_roi as label_one_roi_mediapipe
from hand_autolabel.rtmpose_hand_labeler import (
    RTMPOSE_MEAN,
    RTMPOSE_STD,
    decode_simcc,
    decode_simcc_batch,
    label_one_roi_rtmpose,
    label_roi_manifest_rtmpose,
    preprocess_rtmpose_image,
    preprocess_rtmpose_images,
)
from scripts.hlmf import _partition_labels


class _FakeDetector:
    provider = "FakeExecutionProvider"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, _image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        coordinates = np.asarray(
            [[20.0 + index, 30.0 + index] for index in range(21)], dtype=np.float32
        )
        return coordinates, np.ones(21, dtype=np.float32)


class _FakeHandClassifier:
    provider = "FakeExecutionProvider"
    model_id = "hand-classifier-handedness-handpresence-0809"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, _image: np.ndarray) -> dict:
        self.calls += 1
        return {
            "handedness": {"label": "Right", "score": 0.93},
            "hand_presence": {"present": True, "score": 0.98},
        }


class _FakeMediaPipeDetector:
    def detect(self, _image: np.ndarray) -> tuple[list, list]:
        landmarks = [
            SimpleNamespace(x=(20.0 + index) / 255.0, y=(30.0 + index) / 255.0)
            for index in range(21)
        ]
        return [landmarks], []


def _config() -> dict:
    return {
        "image": {"width": 1280, "height": 720},
        "hand_roi": {"output_width": 256, "output_height": 256},
        "hand_landmark": {"backend": "rtmpose_onnx"},
        "rtmpose": {"model_onnx_path": "unused.onnx", "simcc_split_ratio": 2.0},
        "hand_classifier": {"model_onnx_path": "unused-classifier.onnx"},
        "quality": {
            "handedness_review_threshold": 0.7,
            "rtmpose_train_hand_presence_threshold": 0.5,
            "rtmpose_train_boundary_coordinate_reject_threshold": 3,
            "rtmpose_train_connection_length_gate_enabled": False,
        },
    }


def _manifest(*, candidate: bool = False) -> dict:
    return {
        "crop_id": "roi_candidate" if candidate else "roi_runtime",
        "image": "frame.tiff",
        "palm_det_id": "palm_1",
        "palm_valid": not candidate,
        "proposal_kind": "negative_candidate" if candidate else "runtime",
        "palm_score": 0.2 if candidate else 0.9,
        "output_size": [256, 256],
        "roi_corners_px": [[100.0, 100.0], [355.0, 100.0], [355.0, 355.0], [100.0, 355.0]],
    }


class RTMPoseHandLabelerTests(unittest.TestCase):
    def test_grayscale_preprocess_replicates_rgb_and_uses_official_normalization(self) -> None:
        tensor = preprocess_rtmpose_image(np.full((256, 256), 100, dtype=np.uint8))
        self.assertEqual((1, 3, 256, 256), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        expected = (100.0 - RTMPOSE_MEAN) / RTMPOSE_STD
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6)
        for channel in range(3):
            self.assertTrue(np.all(tensor[0, channel] == tensor[0, channel, 0, 0]))

    def test_simcc_decode_uses_raw_argmax_and_minimum_peak(self) -> None:
        x = np.full((1, 21, 512), -4.0, dtype=np.float32)
        y = np.full((1, 21, 512), -5.0, dtype=np.float32)
        for index in range(21):
            x[0, index, 2 * index] = 7.0 + index
            y[0, index, 2 * index + 2] = 3.0 + index
        coordinates, scores = decode_simcc(x, y, split_ratio=2.0)
        np.testing.assert_allclose(coordinates[:, 0], np.arange(21, dtype=np.float32))
        np.testing.assert_allclose(coordinates[:, 1], np.arange(1, 22, dtype=np.float32))
        np.testing.assert_allclose(scores, 3.0 + np.arange(21, dtype=np.float32))

        # Negative logits are still decoded; scores are diagnostics, not a gate.
        negative_x = x - 100.0
        negative_y = y - 100.0
        negative_coordinates, negative_scores = decode_simcc(
            negative_x, negative_y, split_ratio=2.0
        )
        np.testing.assert_array_equal(negative_coordinates, coordinates)
        self.assertTrue(np.all(negative_scores < 0.0))

    def test_simcc_decode_validates_shape_finite_values_and_clamps_border_bin(self) -> None:
        x = np.zeros((1, 21, 512), dtype=np.float32)
        y = np.zeros((1, 21, 512), dtype=np.float32)
        x[:, :, 511] = 1.0
        y[:, :, 511] = 2.0
        coordinates, _ = decode_simcc(x, y, split_ratio=2.0)
        self.assertTrue(np.all(coordinates == 255.0))
        with self.assertRaisesRegex(ValueError, "output shapes"):
            decode_simcc(x[:, :20], y, split_ratio=2.0)
        x[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            decode_simcc(x, y, split_ratio=2.0)

    def test_batch_preprocess_and_decode_preserve_each_roi(self) -> None:
        tensor = preprocess_rtmpose_images(
            [
                np.zeros((256, 256), dtype=np.uint8),
                np.full((256, 256), 255, dtype=np.uint8),
            ]
        )
        self.assertEqual((2, 3, 256, 256), tensor.shape)
        x = np.zeros((2, 21, 512), dtype=np.float32)
        y = np.zeros((2, 21, 512), dtype=np.float32)
        x[0, :, 20] = 1.0
        y[0, :, 40] = 1.0
        x[1, :, 60] = 1.0
        y[1, :, 80] = 1.0
        decoded = decode_simcc_batch(x, y, split_ratio=2.0)
        self.assertEqual(2, len(decoded))
        self.assertTrue(np.all(decoded[0][0] == [10.0, 20.0]))
        self.assertTrue(np.all(decoded[1][0] == [30.0, 40.0]))

    def test_runtime_roi_gets_21_points_and_handedness_classification(self) -> None:
        detector = _FakeDetector()
        classifier = _FakeHandClassifier()
        row = label_one_roi_rtmpose(
            _manifest(),
            np.zeros((256, 256), dtype=np.uint8),
            detector,
            classifier,
            _config(),
        )
        self.assertEqual(1, detector.calls)
        self.assertEqual(1, classifier.calls)
        self.assertEqual({"present": True, "score": 0.98}, row["hand_presence"])
        self.assertEqual({"label": "Right", "score": 0.93}, row["handedness"])
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0809",
            row["handedness_teacher_model_id"],
        )
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0809",
            row["hand_presence_teacher_model_id"],
        )
        self.assertEqual(21, len(row["landmarks_crop_norm"]))
        self.assertEqual(21, len(row["landmarks_image_px"]))
        self.assertTrue(
            np.isfinite(
                [[point["x"], point["y"]] for point in row["landmarks_crop_norm"]]
            ).all()
        )
        mediapipe_row = label_one_roi_mediapipe(
            _manifest(),
            np.zeros((256, 256), dtype=np.uint8),
            _FakeMediaPipeDetector(),
            _config(),
        )
        rtmpose_published = apply_label_provenance([row], human_reviewed=False)[0]
        mediapipe_published = apply_label_provenance(
            [mediapipe_row], human_reviewed=False
        )[0]
        self.assertEqual(set(mediapipe_published), set(rtmpose_published))

    def test_low_presence_runtime_keeps_points_and_is_ignored_for_train(self) -> None:
        detector = _FakeDetector()
        classifier = _FakeHandClassifier()
        classifier.classify = lambda _image: {
            "handedness": {"label": "Left", "score": 0.99},
            "hand_presence": {"present": False, "score": 0.01},
        }
        row = label_one_roi_rtmpose(
            _manifest(),
            np.zeros((256, 256), dtype=np.uint8),
            detector,
            classifier,
            _config(),
        )
        row["split"] = "train"
        row["proposal_kind"] = "runtime"
        row = apply_label_provenance([row], human_reviewed=False)[0]
        self.assertEqual(21, len(row["landmarks_crop_norm"]))
        self.assertFalse(row["hand_presence"]["present"])
        positives, candidates, ignored = _partition_labels([row], "train", _config())
        self.assertEqual([], positives)
        self.assertEqual([], candidates)
        self.assertEqual(["roi_runtime"], [item["crop_id"] for item in ignored])
        self.assertEqual("rtmpose_hand_presence_gate", ignored[0]["ignore_reason"])

    def test_negative_candidate_is_not_sent_to_rtmpose(self) -> None:
        detector = _FakeDetector()
        classifier = _FakeHandClassifier()
        row = label_one_roi_rtmpose(
            _manifest(candidate=True),
            np.zeros((256, 256), dtype=np.uint8),
            detector,
            classifier,
            _config(),
        )
        self.assertEqual(0, detector.calls)
        self.assertEqual(0, classifier.calls)
        self.assertFalse(row["hand_presence"]["present"])
        self.assertEqual([], row["landmarks_crop_norm"])
        self.assertEqual("eos_negative_candidate_unassessed", row["source"])
        row = apply_label_provenance([row], human_reviewed=False)[0]
        positives, candidates, ignored = _partition_labels([row], "train", _config())
        self.assertEqual([], positives)
        self.assertEqual(["roi_candidate"], [item["crop_id"] for item in candidates])
        self.assertEqual([], ignored)
        self.assertEqual("unresolved", candidates[0]["label_origin"])

    def test_candidate_only_manifest_does_not_load_model_or_read_crop(self) -> None:
        candidate = _manifest(candidate=True)
        candidate["crop_path"] = "does-not-exist.png"
        with patch(
            "hand_autolabel.rtmpose_hand_labeler.RTMPoseONNXHandLabeler"
        ) as pose_constructor, patch(
            "hand_autolabel.rtmpose_hand_labeler.HandClassifierONNX"
        ) as hand_classifier_constructor:
            rows, info = label_roi_manifest_rtmpose(
                [candidate], _config(), Path("."), show_progress=False
            )
        pose_constructor.assert_not_called()
        hand_classifier_constructor.assert_not_called()
        self.assertEqual(1, len(rows))
        self.assertEqual(1, info["negative_candidates_skipped"])
        self.assertEqual(0, info["runtime_rois_labeled"])
        self.assertIsNone(info["provider"])
        self.assertIsNone(info["hand_classifier_provider"])

    def test_unknown_backend_fails_immediately(self) -> None:
        cfg = _config()
        cfg["hand_landmark"]["backend"] = "typo_backend"
        with self.assertRaisesRegex(ValueError, "unsupported Hand landmark backend"):
            label_hand_landmark_manifest([], cfg, Path("."))


if __name__ == "__main__":
    unittest.main()
