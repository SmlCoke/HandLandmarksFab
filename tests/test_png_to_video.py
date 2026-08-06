from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tools.png_to_video import create_video


class _FakeWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.frames: list[int] = []

    def isOpened(self) -> bool:
        return True

    def write(self, image: np.ndarray) -> None:
        self.frames.append(int(round(float(image.mean()))))

    def release(self) -> None:
        self.path.write_bytes(b"video")


class PNGToVideoTests(unittest.TestCase):
    def test_video_uses_lexicographic_png_order_and_parent_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "eos-2.0"
            images.mkdir()
            for name, value in (("z.png", 220), ("a.png", 20), ("m.png", 120)):
                self.assertTrue(
                    cv2.imwrite(
                        str(images / name),
                        np.full((16, 24, 3), value, dtype=np.uint8),
                    )
                )
            holder: dict[str, _FakeWriter] = {}

            def factory(path: str, *_args: object) -> _FakeWriter:
                writer = _FakeWriter(path)
                holder["writer"] = writer
                return writer

            with patch("tools.png_to_video.cv2.VideoWriter", side_effect=factory), patch(
                "tools.png_to_video.cv2.VideoWriter_fourcc", return_value=0
            ):
                report = create_video(images, root)

            self.assertEqual([20, 120, 220], holder["writer"].frames)
            self.assertEqual(3, report["frames"])
            self.assertEqual(root / "eos-2.0.mp4", Path(report["video_path"]))
            self.assertTrue((root / "eos-2.0.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
