from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hand_autolabel.dataset_v3 import (
    DatasetContractError,
    WarehouseRegistry,
    enrich_roi_rows,
)
from hand_autolabel.formats import basename_index_by_path
from hand_autolabel.image_io import read_image, write_image
from hand_autolabel.roi_geometry import crop_image_by_roi


CAPTURE_SOURCE_ID = "complex-mid-bright-random-val-s01-peak"


class RoiPixelContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_lossless_png_and_tiff_decode_to_identical_uint8_pixels(self) -> None:
        y, x = np.indices((256, 256), dtype=np.uint16)
        roi = ((x * 17 + y * 31) % 256).astype(np.uint8)
        png_path = self.root / "roi.png"
        tiff_path = self.root / "roi.tiff"

        self.assertTrue(write_image(png_path, roi))
        self.assertTrue(
            write_image(
                tiff_path,
                roi,
                [int(cv2.IMWRITE_TIFF_COMPRESSION), 1],
            )
        )

        decoded_png = read_image(png_path)
        decoded_tiff = read_image(tiff_path)
        self.assertIsNotNone(decoded_png)
        self.assertIsNotNone(decoded_tiff)
        self.assertEqual(np.uint8, decoded_png.dtype)
        self.assertEqual(np.uint8, decoded_tiff.dtype)
        np.testing.assert_array_equal(roi, decoded_png)
        np.testing.assert_array_equal(decoded_png, decoded_tiff)

    def test_hand_roi_crop_is_single_channel_uint8_256_square(self) -> None:
        y, x = np.indices((720, 1280), dtype=np.uint16)
        raw = ((x + y * 3) % 256).astype(np.uint8)
        roi_rect = {
            "x_center": 640.0,
            "y_center": 360.0,
            "width": 320.0,
            "height": 320.0,
            "rotation_rad": 0.17,
        }

        crop, corners = crop_image_by_roi(raw, roi_rect, 256, 256)

        self.assertEqual((256, 256), crop.shape)
        self.assertEqual(np.uint8, crop.dtype)
        self.assertEqual(2, crop.ndim)
        self.assertEqual((4, 2), corners.shape)

    def test_roi_id_ignores_suffix_but_cvat_and_registry_paths_do_not(self) -> None:
        palm_rows = [
            {
                "schema_version": "hlmf_dataset_v1",
                "dataset_id": "eval-r1",
                "capture_source_id": CAPTURE_SOURCE_ID,
                "split": "val",
                "raw_image_id": "raw_contract001",
                "detections": [
                    {
                        "palm_det_id": "proposal_contract001",
                        "proposal_slot": 0,
                        "proposal_kind": "runtime",
                    }
                ],
                "negative_candidates": [],
            }
        ]
        common = {
            "crop_id": "temporary",
            "image": "frame001.tiff",
            "palm_det_id": "proposal_contract001",
        }
        png_row = enrich_roi_rows(
            [dict(common, crop_path=self.root / "derived" / "roi.png")],
            palm_rows,
            self.root,
            "eos-1.0",
        )[0]
        tiff_row = enrich_roi_rows(
            [dict(common, crop_path=self.root / "derived" / "roi.tiff")],
            palm_rows,
            self.root,
            "eos-1.0",
        )[0]

        self.assertEqual(png_row["roi_id"], tiff_row["roi_id"])
        self.assertNotEqual(png_row["crop_relpath"], tiff_row["crop_relpath"])
        cvat_index = basename_index_by_path([png_row], "crop_path")
        self.assertIn("roi.png", cvat_index)
        self.assertNotIn("roi.tiff", cvat_index)

        registry = WarehouseRegistry(self.root)
        registry.register_source(
            {
                "dataset_id": "eval-r1",
                "capture_source_id": CAPTURE_SOURCE_ID,
                "scope": "eval",
                "split": "val",
                "performer": "peak",
            },
            [
                {
                    "raw_image_id": "raw_contract001",
                    "relative_path": "images/frame001.tiff",
                    "fingerprint": {
                        "byte_size": 1,
                        "width": 1280,
                        "height": 720,
                        "pixel_crc32": "00000000",
                        "dhash64": "0000000000000000",
                    },
                }
            ],
        )
        registry.register_rois([png_row])
        registry.assert_roi_reference(png_row)
        with self.assertRaisesRegex(DatasetContractError, "disagrees with registry"):
            registry.assert_roi_reference(tiff_row)


if __name__ == "__main__":
    unittest.main()
