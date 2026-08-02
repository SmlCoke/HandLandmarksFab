from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from hand_autolabel.cvat_io import export_cvat_xml


def cvat_config() -> dict:
    return {
        "cvat": {
            "xml_version": "1.1",
            "label_name": "hand_landmarks",
            "no_hand_label_name": "no_hand",
            "left_label_name": "Left",
            "right_label_name": "Right",
            "unknown_handedness_label_name": "unknown_handedness",
            "ignore_for_training_label_name": "ignore_for_training",
            "skeleton_point_labels": [str(index) for index in range(1, 22)],
        },
        "review": {"strip_teacher_handedness": False},
        "hand_roi": {"output_width": 256, "output_height": 256},
        "paths": {"roi_crops_dir": "roi"},
    }


def positive_label(crop_id: str, offset: float, handedness: str = "Right") -> dict:
    return {
        "crop_id": crop_id,
        "hand_presence": {"present": True},
        "handedness": {"label": handedness, "score": 0.9},
        # Deliberately reversed: export must bind skeleton sublabels by point id.
        "landmarks_crop_px": [
            {"id": landmark_id, "x": offset + landmark_id, "y": 20.0 + landmark_id}
            for landmark_id in reversed(range(21))
        ],
    }


class CvatExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_export_frame_ids_follow_lexicographic_roi_filename_order(self) -> None:
        manifest = [
            {"crop_id": "z", "crop_path": "images/roi_f0.png"},
            {"crop_id": "a", "crop_path": "images/roi_01.png"},
            {"crop_id": "m", "crop_path": "images/roi_a2.png"},
        ]
        labels = [
            positive_label("z", 100.0),
            {
                "crop_id": "a",
                "hand_presence": {"present": False},
                "landmarks_crop_px": [],
            },
            positive_label("m", 50.0, handedness="unknown"),
        ]
        xml_path = self.root / "cvat.xml"
        report = export_cvat_xml(manifest, labels, self.root, xml_path, cvat_config())

        images = sorted(
            ET.parse(xml_path).getroot().findall("image"),
            key=lambda element: int(element.attrib["id"]),
        )
        self.assertEqual(["0", "1", "2"], [image.attrib["id"] for image in images])
        self.assertEqual(
            ["roi_01.png", "roi_a2.png", "roi_f0.png"],
            [image.attrib["name"] for image in images],
        )
        self.assertEqual("no_hand", images[0].find("tag").attrib["label"])
        self.assertIsNotNone(images[1].find("skeleton"))
        self.assertEqual("unknown_handedness", images[1].find("tag").attrib["label"])
        z_points = images[2].find("skeleton").findall("points")
        self.assertEqual([str(index) for index in range(1, 22)], [point.attrib["label"] for point in z_points])
        self.assertEqual("100.000,20.000", z_points[0].attrib["points"])
        self.assertEqual("crop_filename_lexicographic", report["image_order"])
        self.assertEqual(3, report["reordered_from_manifest_input"])

    def test_export_rejects_missing_or_malformed_positive_labels(self) -> None:
        manifest = [{"crop_id": "a", "crop_path": "images/roi_a.png"}]
        with self.assertRaisesRegex(ValueError, "one label per manifest ROI"):
            export_cvat_xml(manifest, [], self.root, self.root / "missing.xml", cvat_config())

        malformed = positive_label("a", 0.0)
        malformed["landmarks_crop_px"] = malformed["landmarks_crop_px"][:-1]
        with self.assertRaisesRegex(ValueError, "landmark ids 0..20"):
            export_cvat_xml(
                manifest,
                [malformed],
                self.root,
                self.root / "malformed.xml",
                cvat_config(),
            )


if __name__ == "__main__":
    unittest.main()
