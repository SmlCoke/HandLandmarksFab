from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hand_autolabel.dataset_v3 import apply_label_provenance
from hand_autolabel.hamer_hand_labeler import label_roi_manifest_hamer
from hand_autolabel.quality_checks import RTMPOSE_CONNECTION_PAIRS
from scripts.hlmf import _partition_labels, _quality_gate_rejection_counts


def _thresholds(default: float = 1000.0) -> dict:
    return {
        distance: {
            f"{start}-{end}": default for start, end in RTMPOSE_CONNECTION_PAIRS
        }
        for distance in ("near", "mid", "far")
    }


def _config(*, rescue_enabled: bool = False) -> dict:
    return {
        "image": {"width": 1280, "height": 720},
        "hand_roi": {"output_width": 256, "output_height": 256},
        "hand_landmark": {"backend": "hamer"},
        "onnx_runtime": {
            "provider": "auto",
            "model_providers": {"hand_classifier": "auto"},
            "batch_size": 4,
        },
        "hamer": {"hand_classifier_model_onnx_path": "unused.onnx"},
        "quality": {
            "handedness_review_threshold": 0.8,
            "rtmpose_train_hand_presence_threshold": 0.5,
            "rtmpose_train_boundary_coordinate_reject_threshold": 2,
            "rtmpose_train_connection_length_gate_enabled": True,
            "rtmpose_train_connection_length_thresholds_px": _thresholds(),
            "rtmpose_train_mediapipe_tflite_rescue_enabled": rescue_enabled,
        },
    }


def _manifest(crop_path: Path, *, candidate: bool = False) -> dict:
    return {
        "crop_id": "roi_candidate" if candidate else "roi_runtime",
        "image": "frame.tiff",
        "crop_path": str(crop_path),
        "palm_det_id": "palm_1",
        "palm_valid": not candidate,
        "proposal_kind": "negative_candidate" if candidate else "runtime",
        "palm_score": 0.2 if candidate else 0.9,
        "capture_source_id": "complex-mid-bright-random-train-s01-peak",
        "split": "train",
        "output_size": [256, 256],
        "roi_corners_px": [
            [100.0, 100.0],
            [355.0, 100.0],
            [355.0, 355.0],
            [100.0, 355.0],
        ],
    }


class _Classifier:
    provider = "FakeExecutionProvider"
    fallback_reason = None
    model_id = "hand-classifier-v1-mobilenet_v3_large"
    result = {
        "handedness": {"label": "Right", "score": 0.93},
        "hand_presence": {"present": True, "score": 0.98},
    }

    def __init__(self, _path: Path, _provider: str) -> None:
        pass

    def classify_batch(self, images) -> list[dict]:
        return [dict(type(self).result) for _ in images]


class _HaMeRClient:
    model_id = "hamer-cvpr24-official-test"
    device = "cuda"
    rescale = 0.75
    repository_path = Path("/hamer")
    repository_commit = "b29f1b397ed5ef36eba8f9498dd719949615fe09"
    checkpoint_path = Path("/hamer/hamer.ckpt")
    points = [[60.0 + index, 70.0 + index] for index in range(21)]
    calls = 0

    def __init__(self, _cfg: dict, _root: Path) -> None:
        pass

    def predict(self, requests) -> dict:
        type(self).calls += 1
        rows = list(requests)
        return {
            str(row["crop_id"]): {
                "crop_id": row["crop_id"],
                "keypoints_2d": [list(point) for point in type(self).points],
                "flipped": str(row["handedness"]).lower() == "left",
                "bbox_size": 256.0,
            }
            for row in rows
        }


class _RescueClient:
    def __init__(self, _cfg: dict, _root: Path) -> None:
        pass

    def predict(self, requests) -> dict:
        return {
            str(row["crop_id"]): {
                "landmarks_crop_px": [
                    {"id": index, "x": 100.0 + index, "y": 110.0 + index}
                    for index in range(21)
                ],
                "landmarks_crop_norm": [
                    {
                        "id": index,
                        "x": (100.0 + index) / 256.0,
                        "y": (110.0 + index) / 256.0,
                    }
                    for index in range(21)
                ],
            }
            for row in list(requests)
        }


class HaMeRHandLabelerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.crop_path = self.root / "roi.png"
        self.assertTrue(
            cv2.imwrite(
                str(self.crop_path), np.full((256, 256), 100, dtype=np.uint8)
            )
        )
        _HaMeRClient.calls = 0
        _HaMeRClient.points = [
            [60.0 + index, 70.0 + index] for index in range(21)
        ]
        _Classifier.result = {
            "handedness": {"label": "Right", "score": 0.93},
            "hand_presence": {"present": True, "score": 0.98},
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(self, cfg: dict | None = None) -> tuple[list[dict], dict]:
        with patch(
            "hand_autolabel.hamer_hand_labeler.HandClassifierONNX", _Classifier
        ), patch(
            "hand_autolabel.hamer_hand_labeler.HaMeRWorkerClient", _HaMeRClient
        ), patch(
            "hand_autolabel.hamer_hand_labeler.MediaPipeTFLiteRescueClient",
            _RescueClient,
        ):
            return label_roi_manifest_hamer(
                [_manifest(self.crop_path)], cfg or _config(), self.root
            )

    def test_runtime_uses_hcf_handedness_and_publishes_hamer_provenance(self) -> None:
        rows, info = self._run()
        row = apply_label_provenance(rows, human_reviewed=False)[0]
        self.assertEqual(21, len(row["landmarks_crop_px"]))
        self.assertEqual({"label": "Right", "score": 0.93}, row["handedness"])
        self.assertEqual({"present": True, "score": 0.98}, row["hand_presence"])
        self.assertEqual("hamer", row["label_origin"])
        self.assertEqual("hamer_openpose21_v1", row["annotation_style"])
        self.assertEqual(_HaMeRClient.model_id, row["teacher_model_id"])
        self.assertEqual(_Classifier.model_id, row["handedness_teacher_model_id"])
        self.assertEqual("hand_classifier", row["hamer_inference"]["handedness_source"])
        self.assertEqual("hamer", info["backend"])
        self.assertEqual("cuda", info["provider"])
        self.assertEqual(1, info["hand_classifier_runtime_rois_labeled"])

    def test_candidate_does_not_start_hamer_or_hcf(self) -> None:
        candidate = _manifest(self.crop_path, candidate=True)
        with patch(
            "hand_autolabel.hamer_hand_labeler.HandClassifierONNX"
        ) as classifier_constructor, patch(
            "hand_autolabel.hamer_hand_labeler.HaMeRWorkerClient"
        ) as hamer_constructor:
            rows, info = label_roi_manifest_hamer(
                [candidate], _config(), self.root
            )
        classifier_constructor.assert_not_called()
        hamer_constructor.assert_not_called()
        self.assertEqual(1, info["negative_candidates_skipped"])
        self.assertEqual("eos_negative_candidate_unassessed", rows[0]["source"])

    def test_four_quality_gates_route_hamer_rows_with_exclusive_counts(self) -> None:
        cases = []

        _Classifier.result = {
            "handedness": {"label": "Right", "score": 0.99},
            "hand_presence": {"present": False, "score": 0.01},
        }
        cases.append(
            (self._run()[0][0], "hamer_hand_presence_gate", _config())
        )

        _Classifier.result = {
            "handedness": {"label": "Right", "score": 0.99},
            "hand_presence": {"present": True, "score": 0.98},
        }
        _HaMeRClient.points = [
            [-3.0, 60.0],
            [80.0, 300.0],
            *[[90.0 + index, 90.0 + index] for index in range(19)],
        ]
        cases.append(
            (self._run()[0][0], "hamer_boundary_coordinate_gate", _config())
        )

        _HaMeRClient.points = [
            [20.0, 20.0],
            [200.0, 200.0],
            *[[100.0 + index, 100.0 + index] for index in range(19)],
        ]
        connection_cfg = _config()
        connection_cfg["quality"]["rtmpose_train_connection_length_thresholds_px"][
            "mid"
        ]["0-1"] = 1.0
        cases.append(
            (
                self._run(connection_cfg)[0][0],
                "hamer_connection_length_gate",
                connection_cfg,
            )
        )

        _HaMeRClient.points = [
            [60.0 + index, 70.0 + index] for index in range(21)
        ]
        _Classifier.result = {
            "handedness": {"label": "Right", "score": 0.7},
            "hand_presence": {"present": True, "score": 0.98},
        }
        cases.append(
            (
                self._run()[0][0],
                "automatic_positive_failed_quality_gate",
                _config(),
            )
        )

        ignored = []
        for row, expected_reason, case_cfg in cases:
            published = apply_label_provenance([row], human_reviewed=False)[0]
            positives, candidates, rejected = _partition_labels(
                [published], "train", case_cfg
            )
            self.assertEqual([], positives)
            self.assertEqual([], candidates)
            self.assertEqual(expected_reason, rejected[0]["ignore_reason"])
            ignored.extend(rejected)
        self.assertEqual(
            {
                "hand_presence": 1,
                "boundary_coordinate": 1,
                "connection_length": 1,
                "handedness": 1,
            },
            _quality_gate_rejection_counts(ignored),
        )

    def test_geometry_failure_can_use_tflite_rescue_without_replacing_hcf(self) -> None:
        _HaMeRClient.points = [
            [-3.0, 60.0],
            [80.0, 300.0],
            *[[90.0 + index, 90.0 + index] for index in range(19)],
        ]
        rows, info = self._run(_config(rescue_enabled=True))
        row = apply_label_provenance(rows, human_reviewed=False)[0]
        self.assertEqual(
            "mediapipe_hand_landmarker_full_tflite_hamer_rescue", row["source"]
        )
        self.assertTrue(row["hamer_geometry_rescue"]["accepted"])
        self.assertEqual(_Classifier.model_id, row["hand_presence_teacher_model_id"])
        self.assertEqual("mediapipe", row["label_origin"])
        self.assertEqual(1, info["mediapipe_tflite_rescue_accepted"])


if __name__ == "__main__":
    unittest.main()
