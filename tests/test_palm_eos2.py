from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_autolabel.palm_decode import (
    decode_onnx_outputs,
    feature_level_anchor_count,
    generate_anchors,
    normalize_feature_levels,
)
from hand_autolabel.palm_onnx import palm_model_contract, preprocess_for_onnx


def palm_config() -> dict:
    return {
        "paths": {"palm_model_onnx": "models/palm_detector/eos-2.0/model_384x224_opt.onnx"},
        "palm": {
            "model_id": "eos-2.0",
            "input_width": 384,
            "input_height": 224,
            "onnx_output_layout": "nchw",
            "feature_levels": [
                {
                    "name": "14x24",
                    "height": 14,
                    "width": 24,
                    "anchor_sizes": [
                        [0.1188828125, 0.2170138889],
                        [0.129875, 0.34175],
                    ],
                },
                {
                    "name": "7x12",
                    "height": 7,
                    "width": 12,
                    "anchor_sizes": [
                        [0.171640625, 0.2737222222],
                        [0.193296875, 0.4076527778],
                    ],
                },
            ],
            "score_threshold": 0.25,
            "nms_iou_threshold": 0.10,
            "max_detections": 2,
            "negative_candidate_threshold": 0.15,
        },
    }


class _Meta:
    def __init__(self, name: str, shape: list[int], tensor_type: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _Session:
    def __init__(self, input_shape: list[int] | None = None) -> None:
        self._inputs = [_Meta("inputs", input_shape or [1, 1, 224, 384])]
        self._outputs = [
            _Meta("reg14", [1, 16, 14, 24]),
            _Meta("cls14", [1, 2, 14, 24]),
            _Meta("reg7", [1, 16, 7, 12]),
            _Meta("cls7", [1, 2, 7, 12]),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


class Eos2PalmTest(unittest.TestCase):
    def test_rectangular_preprocess_matches_area_resize(self) -> None:
        image = np.arange(720 * 1280, dtype=np.uint32).reshape(720, 1280) % 256
        image = image.astype(np.uint8)
        actual = preprocess_for_onnx(image, 384, 224)
        expected = cv2.resize(image, (384, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self.assertEqual((1, 1, 224, 384), actual.shape)
        self.assertEqual(np.float32, actual.dtype)
        np.testing.assert_array_equal(expected, actual[0, 0])

    def test_rectangular_anchors_have_expected_count_and_centers(self) -> None:
        levels = normalize_feature_levels(palm_config()["palm"])
        self.assertEqual(840, feature_level_anchor_count(levels))
        anchors = generate_anchors(14, 24, levels[0]["anchor_sizes"])
        self.assertEqual((672, 4), anchors.shape)
        np.testing.assert_allclose(
            [1.0 / 48.0, 1.0 / 28.0], anchors[0, :2], rtol=0.0, atol=1e-7
        )
        np.testing.assert_allclose(
            [23.5 / 24.0, 13.5 / 14.0], anchors[-1, :2], rtol=0.0, atol=1e-7
        )

    def test_decode_finds_shuffled_outputs_and_applies_global_nms(self) -> None:
        config = palm_config()["palm"]
        levels = normalize_feature_levels(config)
        reg14 = np.zeros((1, 16, 14, 24), dtype=np.float32)
        cls14 = np.zeros((1, 2, 14, 24), dtype=np.float32)
        reg7 = np.zeros((1, 16, 7, 12), dtype=np.float32)
        cls7 = np.zeros((1, 2, 7, 12), dtype=np.float32)
        cls14[0, 0, 0, 0] = 0.90
        cls14[0, 1, 0, 3] = 0.20
        cls7[0, 0, 0, 0] = 0.80
        detections, negatives = decode_onnx_outputs(
            [cls7, reg14, cls14, reg7],
            levels,
            score_threshold=0.25,
            nms_iou_threshold=0.10,
            max_detections=2,
            negative_candidate_threshold=0.15,
            output_layout="nchw",
        )
        self.assertEqual(1, len(detections))
        self.assertEqual("14x24", detections[0]["head"])
        self.assertAlmostEqual(0.90, detections[0]["score"], places=6)
        self.assertEqual(1, len(negatives))
        self.assertAlmostEqual(0.20, negatives[0]["score"], places=6)

    def test_score_threshold_equality_passes(self) -> None:
        config = palm_config()["palm"]
        levels = normalize_feature_levels(config)
        outputs = [
            np.zeros((1, 16, 14, 24), dtype=np.float32),
            np.zeros((1, 2, 14, 24), dtype=np.float32),
            np.zeros((1, 16, 7, 12), dtype=np.float32),
            np.zeros((1, 2, 7, 12), dtype=np.float32),
        ]
        outputs[1][0, 0, 2, 3] = 0.25
        detections, _ = decode_onnx_outputs(
            outputs,
            levels,
            score_threshold=0.25,
            nms_iou_threshold=0.10,
            max_detections=2,
            negative_candidate_threshold=0.15,
            output_layout="nchw",
        )
        self.assertEqual(1, len(detections))

    def test_invalid_feature_level_config_is_rejected(self) -> None:
        config = palm_config()["palm"]
        config["feature_levels"][1]["height"] = 14
        config["feature_levels"][1]["width"] = 24
        with self.assertRaisesRegex(ValueError, "shapes must be unique"):
            normalize_feature_levels(config)
        config = palm_config()["palm"]
        config["feature_levels"][0]["anchor_sizes"] = [[0.1, 0.2]]
        with self.assertRaisesRegex(ValueError, "exactly two"):
            normalize_feature_levels(config)

    def test_model_contract_reports_eos2_geometry(self) -> None:
        contract = palm_model_contract(
            _Session(),
            palm_config(),
            Path("models/palm_detector/eos-2.0/model_384x224_opt.onnx"),
        )
        self.assertEqual("eos-2.0", contract["model_id"])
        self.assertEqual([1, 1, 224, 384], contract["input_shape"])
        self.assertEqual(840, contract["anchor_count"])
        self.assertEqual(0.25, contract["score_threshold"])
        self.assertEqual(0.10, contract["nms_iou_threshold"])
        with self.assertRaisesRegex(ValueError, "does not match configured"):
            palm_model_contract(
                _Session([1, 1, 224, 224]),
                palm_config(),
                Path("bad.onnx"),
            )
        config = palm_config()
        config["palm"]["input_width"] = True
        with self.assertRaisesRegex(ValueError, "positive integers"):
            palm_model_contract(_Session(), config, Path("bad.onnx"))
        config = palm_config()
        config["palm"]["max_detections"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            palm_model_contract(_Session(), config, Path("bad.onnx"))


if __name__ == "__main__":
    unittest.main()
