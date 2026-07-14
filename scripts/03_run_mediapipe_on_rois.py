from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import (
    cfg_path,
    load_yaml_config,
    parse_bool,
    read_jsonl,
    repo_root_from_config,
    resolve_path,
    write_json,
    write_jsonl,
)
from hand_autolabel.mediapipe_roi_labeler import label_roi_manifest
from hand_autolabel.mediapipe_roi_visualization import (
    render_mediapipe_roi_draft_overlays,
)
from hand_autolabel.quality_checks import summarize_label_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official MediaPipe hand landmarker on ROI crops.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--visualize-rois",
        default="0",
        help="Render draft landmarks on 02_roi_crops/images (1/true/yes/on enables it).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    manifest_path = resolve_path(root, args.manifest) if args.manifest else roi_dir / "hand_roi_crops_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    roi_progress = tqdm(manifest_rows, desc="MediaPipe ROI labeling", unit="ROI", dynamic_ncols=True)
    rows, mode = label_roi_manifest(roi_progress, cfg, root)
    draft_path = roi_dir / "hand_landmarks_autolabel_draft.jsonl"
    write_jsonl(draft_path, rows)
    stats = summarize_label_rows(rows, cfg)
    stats.update(
        {
            "mediapipe_mode": mode,
            "draft_jsonl": str(draft_path),
            "raw_jsonl": None,
            "raw_output_policy": "omitted_to_avoid_duplicate_of_autolabel_draft",
            "hand_presence_score_policy": "omitted: current MediaPipe Python API does not expose hand presence score",
        }
    )
    write_json(qc_dir / "mediapipe_roi_stats.json", stats)
    print(f"rois={len(rows)} positive={stats['positive']} negative={stats['negative']} mode={mode}")

    if parse_bool(args.visualize_rois):
        roi_images_dir = roi_dir / "images"
        visualization_dir = roi_dir / "hand_landmarks_visualization"
        visualization_rows = read_jsonl(draft_path)
        visualization_progress = tqdm(
            visualization_rows,
            desc="Visualize MediaPipe ROI landmarks",
            unit="ROI",
            dynamic_ncols=True,
        )
        visualization_stats = render_mediapipe_roi_draft_overlays(
            visualization_progress,
            roi_images_dir,
            visualization_dir,
        )
        print(
            f"roi_landmark_overlays={visualization_stats['saved']}/{visualization_stats['rows']} "
            f"positive={visualization_stats['positive']} negative={visualization_stats['negative']} "
            f"invalid_points={visualization_stats['invalid_landmark_count']} "
            f"missing_images={visualization_stats['missing_crop_image']} "
            f"output={visualization_dir}"
        )


if __name__ == "__main__":
    main()
