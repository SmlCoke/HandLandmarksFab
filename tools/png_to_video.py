#!/usr/bin/env python3
"""Combine lexicographically ordered PNG files into one MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import cv2


def get_image_paths(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder!s} is not a valid directory")
    images = sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise RuntimeError(f"No .png images found in {folder}")
    return images


def create_video(
    input_folder: Path,
    output_folder: Path,
    *,
    fps: float = 30.0,
) -> Dict[str, Any]:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    if fps <= 0:
        raise ValueError("video fps must be greater than zero")
    image_paths = get_image_paths(input_folder)
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read first PNG: {image_paths[0]}")
    height, width = first.shape[:2]
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / f"{input_folder.name}.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    written = 0
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Failed to read PNG: {image_path}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.ndim != 3 or image.shape[2] != 3:
                raise RuntimeError(f"Unexpected PNG channel layout: {image_path}")
            writer.write(image)
            written += 1
    finally:
        writer.release()
    if written != len(image_paths) or not output_path.is_file():
        raise RuntimeError(
            f"Video creation is incomplete: written={written} expected={len(image_paths)}"
        )
    return {
        "video_path": str(output_path.resolve()),
        "frames": written,
        "fps": float(fps),
        "codec": "mp4v",
        "frame_width": width,
        "frame_height": height,
        "ordering": "filename_lexicographic_case_insensitive",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    report = create_video(Path(args.input_folder), Path(args.output_folder), fps=args.fps)
    print(f"Video successfully created: {report['video_path']}")


if __name__ == "__main__":
    main()
