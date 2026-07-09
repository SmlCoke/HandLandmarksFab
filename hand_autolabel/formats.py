from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional


VALID_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - exercised by runtime users.
        raise RuntimeError("PyYAML is required. Install dependencies from requirements.txt.") from exc

    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def repo_root_from_config(config_path: Path) -> Path:
    return Path(config_path).resolve().parents[1]


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (Path(root) / p).resolve()


def cfg_path(cfg: Mapping[str, Any], root: Path, key: str) -> Path:
    try:
        value = cfg["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing paths.{key} in config") from exc
    return resolve_path(root, value)


def cfg_path_any(cfg: Mapping[str, Any], root: Path, keys: Iterable[str]) -> Path:
    for key in keys:
        value = cfg.get("paths", {}).get(key)
        if value is not None:
            return resolve_path(root, value)
    joined = ", ".join(keys)
    raise KeyError(f"Missing all configured paths: {joined}")


def relpath(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_no}")
            yield item


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not Path(path).exists():
        return []
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def image_files(images_dir: Path) -> List[Path]:
    files = [p for p in Path(images_dir).iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def image_stem(image_name: str) -> str:
    return Path(image_name).stem


def make_palm_det_id(image_name: str, index: int, prefix: str = "palm") -> str:
    return f"{image_stem(image_name)}:{prefix}{index}"


def make_crop_id(palm_det_id: str) -> str:
    return f"{palm_det_id}:crop"


def make_hand_id(crop_id: str) -> str:
    return f"{crop_id}:hand"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def clamp01(v: float) -> float:
    return clamp(v, 0.0, 1.0)


def bbox_norm_to_px(bbox: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def bbox_px_to_norm(bbox: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def keypoints_norm_to_px(keypoints: Mapping[str, Iterable[float]], width: int, height: int) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, pt in keypoints.items():
        x, y = [float(v) for v in pt]
        out[name] = [x * width, y * height]
    return out


def keypoints_px_to_norm(keypoints: Mapping[str, Iterable[float]], width: int, height: int) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, pt in keypoints.items():
        x, y = [float(v) for v in pt]
        out[name] = [x / width, y / height]
    return out


def normalize_detection_schema(det: MutableMapping[str, Any], image_name: str, index: int, width: int, height: int) -> Dict[str, Any]:
    bbox_norm = [clamp01(v) for v in det["bbox_norm"]]
    x1, y1, x2, y2 = bbox_norm
    bbox_norm = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    keypoints_norm = {
        "p0": [clamp01(det["keypoints_norm"]["p0"][0]), clamp01(det["keypoints_norm"]["p0"][1])],
        "p9": [clamp01(det["keypoints_norm"]["p9"][0]), clamp01(det["keypoints_norm"]["p9"][1])],
    }
    result = dict(det)
    result["palm_det_id"] = det.get("palm_det_id") or make_palm_det_id(image_name, index)
    result["valid"] = bool(det.get("valid", True))
    result["score"] = float(det.get("score", 0.0))
    result["bbox_norm"] = bbox_norm
    result["bbox_px"] = bbox_norm_to_px(bbox_norm, width, height)
    result["keypoints_norm"] = keypoints_norm
    result["keypoints_px"] = keypoints_norm_to_px(keypoints_norm, width, height)
    return result


def index_by(items: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if value is not None:
            out[str(value)] = dict(item)
    return out


def basename_index_by_path(items: Iterable[Mapping[str, Any]], path_key: str = "crop_path") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in items:
        value = item.get(path_key)
        if value:
            out[Path(str(value)).name] = dict(item)
    return out


def merge_label_with_manifest(label: Mapping[str, Any], manifest: Mapping[str, Any], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    width = int(cfg["hand_roi"]["output_width"])
    height = int(cfg["hand_roi"]["output_height"])
    image_width = int(cfg["image"]["width"])
    image_height = int(cfg["image"]["height"])
    row = dict(label)
    row.setdefault("crop_id", manifest.get("crop_id"))
    row.setdefault("image", manifest.get("image"))
    row.setdefault("crop_path", manifest.get("crop_path"))
    row.setdefault("palm_det_id", manifest.get("palm_det_id"))
    row.setdefault("palm_valid", manifest.get("palm_valid"))
    row.setdefault("palm_score", manifest.get("palm_score"))
    row.setdefault("width", width)
    row.setdefault("height", height)
    row.setdefault("source_image_width", image_width)
    row.setdefault("source_image_height", image_height)
    row.setdefault("roi_rect", manifest.get("roi_rect"))
    row.setdefault("roi_corners_px", manifest.get("roi_corners_px"))
    row.setdefault("hand_presence", {"present": False})
    row.setdefault("handedness", {"label": "unknown", "score": None})
    row.setdefault("landmarks_crop_norm", [])
    row.setdefault("landmarks_crop_px", [])
    row.setdefault("landmarks_image_px", [])
    normalize_landmark_fields(row)
    return row


def normalize_landmark_fields(row: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in ("landmarks_crop_norm", "landmarks_crop_px", "landmarks_image_px"):
        normalized = []
        for idx, point in enumerate(row.get(key) or []):
            normalized.append(
                {
                    "id": int(point.get("id", idx)),
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                }
            )
        row[key] = normalized
    return row


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
