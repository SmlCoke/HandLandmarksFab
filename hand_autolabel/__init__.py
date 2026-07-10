"""Semi-automatic Hand Landmarker annotation helpers."""

import os


# Suppress noisy TIFF metadata warnings while preserving OpenCV errors.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

__all__ = [
    "cvat_io",
    "formats",
    "image_io",
    "mediapipe_roi_labeler",
    "nms",
    "palm_decode",
    "palm_mediapipe",
    "palm_onnx",
    "projection",
    "quality_checks",
    "roi_geometry",
    "visualization",
]
