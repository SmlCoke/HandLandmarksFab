from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from hand_autolabel.dataset_v3 import apply_label_provenance
from hand_autolabel.mediapipe_tflite_rescue import (
    MediaPipeTFLiteRescueClient,
    mediapipe_tflite_rescue_enabled,
)
from hand_autolabel.quality_checks import RTMPOSE_CONNECTION_PAIRS
from hand_autolabel.rtmpose_hand_labeler import label_roi_manifest_rtmpose
from scripts.hlmf import _partition_labels
from tools.mediapipe_tflite_worker import decode_landmarks, preprocess_image


class _Detector:
    provider = "FakeExecutionProvider"

    def __init__(self, coordinates: np.ndarray) -> None:
        self.coordinates = np.asarray(coordinates, dtype=np.float32)

    def detect(self, _image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.coordinates.copy(), np.ones(21, dtype=np.float32)


class _Classifier:
    provider = "FakeExecutionProvider"
    model_id = "hand-classifier-handedness-handpresence-0807"

    def __init__(self, *, presence_score: float = 0.98) -> None:
        self.presence_score = presence_score

    def classify(self, _image: np.ndarray) -> dict:
        return {
            "handedness": {"label": "Right", "score": 0.93},
            "hand_presence": {
                "present": self.presence_score >= 0.5,
                "score": self.presence_score,
            },
        }


def _points(value: float = 100.0, step: float = 0.1) -> list[dict]:
    return [
        {"id": index, "x": value + index * step, "y": value + index * step}
        for index in range(21)
    ]


def _prediction(points: list[dict]) -> dict:
    return {
        "landmarks_crop_px": points,
        "landmarks_crop_norm": [
            {
                "id": point["id"],
                "x": float(point["x"]) / 256.0,
                "y": float(point["y"]) / 256.0,
            }
            for point in points
        ],
    }


def _thresholds(default: float = 1000.0) -> dict:
    return {
        distance: {
            f"{start}-{end}": default for start, end in RTMPOSE_CONNECTION_PAIRS
        }
        for distance in ("near", "mid", "far")
    }


def _config(*, rescue_enabled: bool = True, connection_enabled: bool = False) -> dict:
    return {
        "image": {"width": 1280, "height": 720},
        "hand_roi": {"output_width": 256, "output_height": 256},
        "rtmpose": {"model_onnx_path": "unused.onnx", "simcc_split_ratio": 2.0},
        "hand_classifier": {"model_onnx_path": "unused-classifier.onnx"},
        "quality": {
            "handedness_review_threshold": 0.7,
            "rtmpose_train_hand_presence_threshold": 0.5,
            "rtmpose_train_boundary_coordinate_reject_threshold": 2,
            "rtmpose_train_connection_length_gate_enabled": connection_enabled,
            "rtmpose_train_connection_length_thresholds_px": _thresholds(),
            "rtmpose_train_mediapipe_tflite_rescue_enabled": rescue_enabled,
        },
    }


def _manifest(crop_path: Path, *, split: str = "train") -> dict:
    return {
        "crop_id": "roi_runtime",
        "image": "frame.tiff",
        "crop_path": str(crop_path),
        "palm_det_id": "palm_1",
        "palm_valid": True,
        "proposal_kind": "runtime",
        "palm_score": 0.9,
        "capture_source_id": f"complex-mid-bright-random-{split}-s01-peak",
        "split": split,
        "output_size": [256, 256],
        "roi_corners_px": [
            [100.0, 100.0],
            [355.0, 100.0],
            [355.0, 355.0],
            [100.0, 355.0],
        ],
    }


class _RescueClient:
    prediction: dict = _prediction(_points())
    calls = 0

    def __init__(self, _cfg: dict, _root: Path) -> None:
        pass

    def predict(self, requests) -> dict:
        type(self).calls += 1
        rows = list(requests)
        return {str(row["crop_id"]): dict(type(self).prediction) for row in rows}


class MediaPipeTFLiteWorkerTests(unittest.TestCase):
    def test_preprocess_and_decode_match_evaluated_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "roi.png"
            image = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            tensor = preprocess_image(image_path)
        self.assertEqual((1, 224, 224, 3), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        np.testing.assert_array_equal(tensor[0, :, :, 0], tensor[0, :, :, 1])
        np.testing.assert_array_equal(tensor[0, :, :, 1], tensor[0, :, :, 2])
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

        raw = np.zeros((1, 63), dtype=np.float32)
        raw[0, 0] = 112.0
        raw[0, 1] = 56.0
        crop_px, crop_norm = decode_landmarks(raw)
        self.assertEqual({"id": 0, "x": 128.0, "y": 64.0}, crop_px[0])
        self.assertEqual({"id": 0, "x": 0.5, "y": 0.25}, crop_norm[0])
        with self.assertRaisesRegex(ValueError, "output shape"):
            decode_landmarks(np.zeros((1, 62), dtype=np.float32))

    def test_rescue_switch_requires_a_boolean(self) -> None:
        self.assertTrue(mediapipe_tflite_rescue_enabled({}))
        self.assertFalse(
            mediapipe_tflite_rescue_enabled(
                {"quality": {"rtmpose_train_mediapipe_tflite_rescue_enabled": False}}
            )
        )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            mediapipe_tflite_rescue_enabled(
                {"quality": {"rtmpose_train_mediapipe_tflite_rescue_enabled": "false"}}
            )

    def test_client_rejects_missing_asset_and_incomplete_worker_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = {
                "mediapipe_tflite": {
                    "python_executable": sys.executable,
                    "model_asset_path": "model.tflite",
                }
            }
            with self.assertRaisesRegex(FileNotFoundError, "model does not exist"):
                MediaPipeTFLiteRescueClient(cfg, root)

            (root / "model.tflite").write_bytes(b"placeholder")
            worker = root / "tools" / "mediapipe_tflite_worker.py"
            worker.parent.mkdir(parents=True)
            worker.write_text("# placeholder\n", encoding="utf-8")
            client = MediaPipeTFLiteRescueClient(cfg, root)

            def empty_response(command, **_kwargs):
                output_path = Path(command[command.index("--output") + 1])
                output_path.write_text("", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch(
                "hand_autolabel.mediapipe_tflite_rescue.subprocess.run",
                side_effect=empty_response,
            ), self.assertRaisesRegex(ValueError, "response mismatch"):
                client.predict([{"crop_id": "roi_1", "crop_path": "roi.png"}])


class RTMPoseTFLiteRescueTests(unittest.TestCase):
    def setUp(self) -> None:
        _RescueClient.calls = 0
        _RescueClient.prediction = _prediction(_points())
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.crop_path = self.root / "roi.png"
        self.assertTrue(
            cv2.imwrite(str(self.crop_path), np.full((256, 256), 100, dtype=np.uint8))
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(
        self,
        coordinates: np.ndarray,
        cfg: dict,
        *, split: str = "train",
        presence_score: float = 0.98,
    ) -> tuple[list[dict], dict]:
        detector = _Detector(coordinates)
        classifier = _Classifier(presence_score=presence_score)
        with patch(
            "hand_autolabel.rtmpose_hand_labeler.RTMPoseONNXHandLabeler",
            return_value=detector,
        ), patch(
            "hand_autolabel.rtmpose_hand_labeler.HandClassifierONNX",
            return_value=classifier,
        ), patch(
            "hand_autolabel.rtmpose_hand_labeler.MediaPipeTFLiteRescueClient",
            _RescueClient,
        ):
            return label_roi_manifest_rtmpose(
                [_manifest(self.crop_path, split=split)], cfg, self.root
            )

    def test_boundary_failure_is_rescued_and_hcf_outputs_remain_authoritative(self) -> None:
        coordinates = np.asarray(
            [[40.0 + index, 60.0 + index] for index in range(21)], dtype=np.float32
        )
        coordinates[0, 0] = 0.0
        coordinates[1, 1] = 255.0
        cfg = _config()
        rows, info = self._run(coordinates, cfg)
        row = rows[0]
        self.assertEqual(1, _RescueClient.calls)
        self.assertEqual(
            "mediapipe_hand_landmarker_full_tflite_rtmpose_rescue", row["source"]
        )
        self.assertEqual({"present": True, "score": 0.98}, row["hand_presence"])
        self.assertEqual({"label": "Right", "score": 0.93}, row["handedness"])
        self.assertTrue(row["rtmpose_geometry_rescue"]["accepted"])
        self.assertEqual(1, info["mediapipe_tflite_rescue_attempted"])
        self.assertEqual(1, info["mediapipe_tflite_rescue_accepted"])

        published = apply_label_provenance(rows, human_reviewed=False)[0]
        self.assertEqual("mediapipe", published["label_origin"])
        self.assertEqual(
            "mediapipe-hand-landmark-full-tflite", published["teacher_model_id"]
        )
        self.assertEqual(
            "hand-classifier-handedness-handpresence-0807",
            published["hand_presence_teacher_model_id"],
        )
        positives, candidates, ignored = _partition_labels([published], "train", cfg)
        self.assertEqual(1, len(positives))
        self.assertEqual([], candidates)
        self.assertEqual([], ignored)

    def test_failed_rescue_keeps_original_rtmpose_landmarks(self) -> None:
        coordinates = np.asarray(
            [[40.0 + index, 60.0 + index] for index in range(21)], dtype=np.float32
        )
        coordinates[0, 0] = 0.0
        coordinates[1, 1] = 255.0
        failed = _points()
        failed[0]["x"] = 0.0
        failed[1]["y"] = 255.0
        _RescueClient.prediction = _prediction(failed)
        cfg = _config()
        rows, info = self._run(coordinates, cfg)
        row = rows[0]
        self.assertEqual("rtmpose_m_hand5_onnx", row["source"])
        self.assertEqual(0.0, row["landmarks_crop_px"][0]["x"])
        self.assertFalse(row["rtmpose_geometry_rescue"]["accepted"])
        self.assertEqual(1, info["mediapipe_tflite_rescue_rejected"])
        published = apply_label_provenance(rows, human_reviewed=False)[0]
        _, _, ignored = _partition_labels([published], "train", cfg)
        self.assertEqual("rtmpose_boundary_coordinate_gate", ignored[0]["ignore_reason"])

    def test_connection_failure_triggers_rescue_and_disabled_switch_skips_it(self) -> None:
        coordinates = np.asarray(
            [[100.0 + index, 100.0 + index] for index in range(21)], dtype=np.float32
        )
        cfg = _config(connection_enabled=True)
        for distance in ("near", "mid", "far"):
            cfg["quality"]["rtmpose_train_connection_length_thresholds_px"][distance][
                "0-1"
            ] = 1.0
        rows, _ = self._run(coordinates, cfg)
        self.assertEqual(1, _RescueClient.calls)
        self.assertTrue(rows[0]["rtmpose_geometry_rescue"]["accepted"])

        disabled = _config(rescue_enabled=False, connection_enabled=True)
        for distance in ("near", "mid", "far"):
            disabled["quality"]["rtmpose_train_connection_length_thresholds_px"][distance][
                "0-1"
            ] = 1.0
        _RescueClient.calls = 0
        rows, info = self._run(coordinates, disabled)
        self.assertEqual(0, _RescueClient.calls)
        self.assertEqual("rtmpose_m_hand5_onnx", rows[0]["source"])
        self.assertFalse(info["mediapipe_tflite_rescue_enabled"])

    def test_non_train_and_geometry_pass_do_not_start_worker(self) -> None:
        coordinates = np.asarray(
            [[100.0 + index * 0.1, 100.0 + index * 0.1] for index in range(21)],
            dtype=np.float32,
        )
        rows, _ = self._run(coordinates, _config())
        self.assertEqual(0, _RescueClient.calls)
        self.assertNotIn("rtmpose_geometry_rescue", rows[0])

        bad_geometry = coordinates.copy()
        bad_geometry[0, 0] = 0.0
        bad_geometry[1, 1] = 255.0
        self._run(bad_geometry, _config(), split="val")
        self.assertEqual(0, _RescueClient.calls)

    def test_presence_gate_still_wins_after_successful_geometry_rescue(self) -> None:
        coordinates = np.asarray(
            [[40.0 + index, 60.0 + index] for index in range(21)], dtype=np.float32
        )
        coordinates[0, 0] = 0.0
        coordinates[1, 1] = 255.0
        cfg = _config()
        rows, _ = self._run(coordinates, cfg, presence_score=0.1)
        published = apply_label_provenance(rows, human_reviewed=False)[0]
        _, _, ignored = _partition_labels([published], "train", cfg)
        self.assertEqual("rtmpose_hand_presence_gate", ignored[0]["ignore_reason"])


if __name__ == "__main__":
    unittest.main()
