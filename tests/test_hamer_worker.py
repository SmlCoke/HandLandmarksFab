from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.hamer_worker import _read_requests, _validate_keypoints


class HaMeRWorkerProtocolTests(unittest.TestCase):
    def test_request_protocol_requires_unique_ids_paths_and_handedness(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "request.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "crop_id": "roi-1",
                        "crop_path": "/tmp/roi.png",
                        "handedness": "Left",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _read_requests(path)
            self.assertEqual("left", rows[0]["handedness"])
            path.write_text(
                json.dumps(
                    {
                        "crop_id": "roi-1",
                        "crop_path": "/tmp/roi.png",
                        "handedness": "unknown",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "left or right"):
                _read_requests(path)

    def test_response_validation_requires_21_finite_xy_points(self) -> None:
        points = [[float(index), float(index + 1)] for index in range(21)]
        self.assertEqual(points, _validate_keypoints(points))
        with self.assertRaisesRegex(ValueError, "21"):
            _validate_keypoints(points[:20])


if __name__ == "__main__":
    unittest.main()
