#!/usr/bin/env python3
"""Local-only TIFF frame sampler used before uploading an HLMF source."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


TIFF_EXTENSIONS = {".tif", ".tiff"}


def tiff_files(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {directory}")
    nested = [path for path in directory.iterdir() if path.is_dir()]
    if nested:
        raise ValueError("input must be one flat camera-frame directory")
    unsupported = [
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() not in TIFF_EXTENSIONS
    ]
    if unsupported:
        raise ValueError(f"only lossless TIFF input is accepted: {unsupported[:8]}")
    return sorted(
        [path for path in directory.iterdir() if path.is_file()],
        key=lambda path: path.name,
    )


def downsample(input_dir: Path, interval: int, output_dir: Path) -> dict[str, int | str]:
    if interval < 1:
        raise ValueError("interval must be >= 1")
    source_files = tiff_files(input_dir)
    if not source_files:
        raise ValueError("no TIFF camera frames found")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = source_files[::interval]
    for source in selected:
        destination = output_dir / source.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite: {destination}")
        shutil.copy2(source, destination)
    return {
        "input_frames": len(source_files),
        "interval": interval,
        "retained_frames": len(selected),
        "output": str(output_dir.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain every Nth lossless TIFF locally before server upload."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("interval", type=int)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        report = downsample(args.input_dir, args.interval, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(
        "input_frames={input_frames} interval={interval} retained_frames={retained_frames} output={output}".format(
            **report
        )
    )


if __name__ == "__main__":
    main()
