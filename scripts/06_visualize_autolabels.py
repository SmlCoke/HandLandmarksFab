from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import cfg_path, load_yaml_config, read_jsonl, repo_root_from_config, resolve_path, write_json
from hand_autolabel.visualization import render_overlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render palm/ROI/landmark overlays on source images.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--labels-jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    images_dir = cfg_path(cfg, root, "images_dir")
    palm_dir = cfg_path(cfg, root, "palm_outputs_dir")
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    labels_dir = cfg_path(cfg, root, "labels_dir")
    review_dir = cfg_path(cfg, root, "review_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    labels_path = resolve_path(root, args.labels_jsonl) if args.labels_jsonl else labels_dir / "hand_landmarks_reviewed.jsonl"
    stats = render_overlays(
        images_dir,
        read_jsonl(palm_dir / "palm_detections.jsonl"),
        read_jsonl(roi_dir / "hand_roi_crops_manifest.jsonl"),
        read_jsonl(labels_path),
        root,
        review_dir / "overlay_images",
        review_dir / "review_index.csv",
        cfg,
    )
    write_json(qc_dir / "visualization_stats.json", stats)
    print(f"overlays={stats['overlay_images']} review_index={review_dir / 'review_index.csv'}")


if __name__ == "__main__":
    main()
