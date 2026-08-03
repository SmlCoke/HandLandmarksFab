from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    if img is not None:
        return img
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im)
    except Exception:
        return None


def write_image(
    path: Path,
    image: np.ndarray,
    encode_params: Optional[Sequence[int]] = None,
) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    if encode_params is None:
        ok, encoded = cv2.imencode(ext, image)
    else:
        ok, encoded = cv2.imencode(ext, image, list(encode_params))
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    gray = ensure_gray(image)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    gray = ensure_gray(image)
    if gray.dtype == np.uint8:
        return gray
    if np.issubdtype(gray.dtype, np.integer):
        max_value = float(np.iinfo(gray.dtype).max)
        out = gray.astype(np.float32) * (255.0 / max_value)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    out = gray.astype(np.float32)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    max_value = float(np.max(finite))
    if max_value <= 1.0:
        out = out * 255.0
    else:
        out = out * (255.0 / max_value)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def image_shape_info(image: np.ndarray) -> dict:
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return {"width": int(width), "height": int(height), "channels": channels, "dtype": str(image.dtype)}
