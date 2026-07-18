from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import cfg_path, iter_jsonl, load_yaml_config, make_crop_id, relpath, repo_root_from_config, resolve_path, write_json, write_jsonl
from hand_autolabel.image_io import read_image, write_image
from hand_autolabel.quality_checks import roi_manifest_issues
from hand_autolabel.roi_geometry import build_roi_rect_from_palm, crop_image_by_roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 256x256 hand ROI crops from palm detections.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--palm-jsonl", default=None)
    parser.add_argument("--output-manifest", default=None)
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    images_dir = cfg_path(cfg, root, "images_dir")
    palm_dir = cfg_path(cfg, root, "palm_outputs_dir")
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    palm_jsonl = resolve_path(root, args.palm_jsonl) if args.palm_jsonl else palm_dir / "palm_detections.jsonl"
    manifest_path = resolve_path(root, args.output_manifest) if args.output_manifest else roi_dir / "hand_roi_crops_manifest.jsonl"
    crop_images_dir = roi_dir / "images"
    crop_images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for record in iter_jsonl(palm_jsonl):
        image_name = record["image"]
        img = read_image(images_dir / image_name)
        if img is None:
            failures.append({"image": image_name, "error": "unreadable_source_image"})
            continue
        candidates = list(record.get("detections") or [])
        if cfg["palm"].get("keep_low_score_candidates_for_negatives", True):
            candidates.extend(record.get("negative_candidates") or [])
        for det in candidates:
            try:
                rect = build_roi_rect_from_palm(
                    det,
                    int(cfg["image"]["width"]),
                    int(cfg["image"]["height"]),
                    scale_x=float(cfg["hand_roi"]["scale_x"]),
                    scale_y=float(cfg["hand_roi"]["scale_y"]),
                    shift_x=float(cfg["hand_roi"]["shift_x"]),
                    shift_y=float(cfg["hand_roi"]["shift_y"]),
                )
                crop, corners = crop_image_by_roi(
                    img,
                    rect,
                    int(cfg["hand_roi"]["output_width"]),
                    int(cfg["hand_roi"]["output_height"]),
                )
                crop_id = make_crop_id(det["palm_det_id"])
                crop_path = crop_images_dir / f"{_safe_name(crop_id)}.png"
                if not write_image(crop_path, crop):
                    raise RuntimeError(f"failed_to_write_crop:{crop_path}")
                rows.append(
                    {
                        "crop_id": crop_id,
                        "image": image_name,
                        "palm_det_id": det["palm_det_id"],
                        "palm_valid": bool(det.get("valid", True)),
                        "palm_score": det.get("score"),
                        "crop_path": relpath(crop_path, root),
                        "roi_rect": rect,
                        "roi_corners_px": [[float(x), float(y)] for x, y in corners.tolist()],
                        "output_size": [int(cfg["hand_roi"]["output_width"]), int(cfg["hand_roi"]["output_height"])],
                    }
                )
            except Exception as exc:
                failures.append({"image": image_name, "palm_det_id": det.get("palm_det_id"), "error": str(exc)})

    write_jsonl(manifest_path, rows)
    warnings = []
    errors = []
    for row in rows:
        w, e = roi_manifest_issues(row, cfg)
        if w:
            warnings.append({"crop_id": row.get("crop_id"), "warnings": w})
        if e:
            errors.append({"crop_id": row.get("crop_id"), "errors": e})
    stats = {
        "crops": len(rows),
        "failures": failures,
        "warnings": warnings,
        "errors": errors,
        "manifest": str(manifest_path),
        "autolabel_runtime": cfg.get("_autolabel_runtime"),
    }
    write_json(qc_dir / "roi_crop_stats.json", stats)
    print(f"crops={len(rows)} failures={len(failures)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
