#!/usr/bin/env python3
"""
PNG to Video Converter

Usage:
    python png_to_video.py <input_folder> <output_folder>

Arguments:
    input_folder   Path to the folder containing .png images.
    output_folder  Path to the folder where the video will be saved.

The video will be named after the input folder and saved as <folder_name>.mp4.
Default frame rate is 30 fps, codec is MP4V (H.264 compatible).
"""

import os
import sys
import cv2

SUPPORTED_EXT = ('.png',)

def get_image_paths(folder):
    """Return a sorted list of full paths to all .png images in folder."""
    if not os.path.isdir(folder):
        raise ValueError(f"'{folder}' is not a valid directory.")

    files = []
    for f in os.listdir(folder):
        if f.lower().endswith(SUPPORTED_EXT):
            full_path = os.path.join(folder, f)
            if os.path.isfile(full_path):
                files.append(full_path)

    if not files:
        raise RuntimeError(f"No .png images found in '{folder}'.")

    # Sort by filename (lexicographic order, case-insensitive for better sorting)
    files.sort(key=lambda x: os.path.basename(x).lower())
    return files

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    os.makedirs(output_folder, exist_ok=True)

    try:
        image_paths = get_image_paths(input_folder)
    except Exception as e:
        print(f"Error reading images: {e}")
        sys.exit(1)

    # Read first image to determine size and color mode
    first_img = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        print(f"Failed to read first image: {image_paths[0]}")
        sys.exit(1)

    height, width = first_img.shape[:2]
    # Determine if image is grayscale or color (BGR)
    if len(first_img.shape) == 2:
        is_gray = True
    elif len(first_img.shape) == 3 and first_img.shape[2] == 3:
        is_gray = False
    elif len(first_img.shape) == 3 and first_img.shape[2] == 4:
        # PNG with alpha – we'll convert to BGR later
        is_gray = False
    else:
        is_gray = False

    # Prepare video writer
    output_video_name = os.path.basename(os.path.normpath(input_folder)) + '.mp4'
    output_path = os.path.join(output_folder, output_video_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0

    # Create writer; if first image is grayscale, we'll convert to BGR for consistency
    # So writer always receives BGR frames
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Failed to open video writer for {output_path}")
        sys.exit(1)

    print(f"Writing video to: {output_path}")
    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: skipping unreadable image: {img_path}")
            continue

        # Ensure same dimensions
        h, w = img.shape[:2]
        if h != height or w != width:
            print(f"Warning: image {img_path} has size {w}x{h}, expected {width}x{height}. Resizing.")
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Convert to BGR if needed
        if len(img.shape) == 2:
            # Grayscale -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA -> BGR (discard alpha)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # Already BGR (imread with default reads as BGR)
            pass
        else:
            print(f"Warning: unexpected channel count in {img_path}, skipping.")
            continue

        writer.write(img)

        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total}")

    writer.release()
    print(f"Video successfully created: {output_path}")

if __name__ == "__main__":
    main()