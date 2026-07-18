from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import cfg_path, image_files, load_yaml_config, repo_root_from_config, resolve_path, write_json, write_jsonl
from hand_autolabel.palm_mediapipe import run_mediapipe_palm_detector
from hand_autolabel.palm_onnx import run_onnx_palm_detector
from hand_autolabel.quality_checks import palm_record_issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export palm or palm-compatible detections.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--backend", choices=["mediapipe_official", "aethersign_onnx"], default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    if args.backend:
        cfg.setdefault("palm", {})["backend"] = args.backend
    backend = str(cfg["palm"].get("backend", "mediapipe_official"))
    images = image_files(cfg_path(cfg, root, "images_dir"))
    palm_dir = cfg_path(cfg, root, "palm_outputs_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    output_jsonl = resolve_path(root, args.output_jsonl) if args.output_jsonl else palm_dir / "palm_detections.jsonl"
    image_progress = tqdm(images, desc=f"Palm detection ({backend})", unit="image", dynamic_ncols=True)

    backend_mode = None
    if backend == "mediapipe_official":
        rows, backend_mode = run_mediapipe_palm_detector(image_progress, cfg)
    elif backend == "aethersign_onnx":
        rows = run_onnx_palm_detector(image_progress, cfg, cfg_path(cfg, root, "palm_model_onnx"))
    else:
        raise ValueError(f"Unsupported palm.backend: {backend}")

    write_jsonl(output_jsonl, rows)
    warnings = []
    errors = []
    for row in rows:
        w, e = palm_record_issues(row, cfg)
        if w:
            warnings.append({"image": row.get("image"), "warnings": w})
        if e:
            errors.append({"image": row.get("image"), "errors": e})
    stats = {
        "backend": backend,
        "backend_mode": backend_mode,
        "images": len(rows),
        "detections": sum(len(r.get("detections") or []) for r in rows),
        "negative_candidates": sum(len(r.get("negative_candidates") or []) for r in rows),
        "autolabel_runtime": cfg.get("_autolabel_runtime"),
        "warnings": warnings,
        "errors": errors,
        "output_jsonl": str(output_jsonl),
    }
    write_json(qc_dir / "palm_detection_stats.json", stats)
    print(f"backend={backend} images={stats['images']} detections={stats['detections']} output={output_jsonl}")


if __name__ == "__main__":
    main()
