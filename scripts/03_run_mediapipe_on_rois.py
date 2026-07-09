from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import cfg_path, read_jsonl, load_yaml_config, repo_root_from_config, resolve_path, write_json, write_jsonl
from hand_autolabel.mediapipe_roi_labeler import label_roi_manifest
from hand_autolabel.quality_checks import summarize_label_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official MediaPipe hand landmarker on ROI crops.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    labels_dir = cfg_path(cfg, root, "labels_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    manifest_path = resolve_path(root, args.manifest) if args.manifest else roi_dir / "hand_roi_crops_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    rows, mode = label_roi_manifest(manifest_rows, cfg, root)
    raw_path = labels_dir / "hand_landmarks_mediapipe_raw.jsonl"
    draft_path = labels_dir / "hand_landmarks_autolabel_draft.jsonl"
    write_jsonl(raw_path, rows)
    write_jsonl(draft_path, rows)
    stats = summarize_label_rows(rows, cfg)
    stats.update(
        {
            "mediapipe_mode": mode,
            "raw_jsonl": str(raw_path),
            "draft_jsonl": str(draft_path),
            "hand_presence_score_policy": "omitted: current MediaPipe Python API does not expose hand presence score",
        }
    )
    write_json(qc_dir / "mediapipe_roi_stats.json", stats)
    print(f"rois={len(rows)} positive={stats['positive']} negative={stats['negative']} mode={mode}")


if __name__ == "__main__":
    main()
