from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.finalization import FinalizationError, finalize_training
from hand_autolabel.formats import parse_bool, resolve_path
from hand_autolabel.mediapipe_roi_visualization import (
    TrainingRoiVisualizationError,
    render_finalized_training_overlays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize noisy pseudo labels and optional Gold overrides for training.")
    parser.add_argument("--config", default="configs/finalize_train.yaml")
    parser.add_argument("--stage", choices=("pretrain", "finetune"), required=True)
    parser.add_argument(
        "--visualize-rois",
        default="0",
        help="Render every canonical included training ROI (1/true/yes/on enables it).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    try:
        report = finalize_training(config_path, args.stage)
        visualization_stats = None
        if parse_bool(args.visualize_rois):
            visualization_stats = render_finalized_training_overlays(
                config_path,
                args.stage,
            )
    except (FinalizationError, TrainingRoiVisualizationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    counts = report["counts"]
    print(f"07A {args.stage}: included={counts['included']} excluded={counts['excluded']} catalog={counts['catalog']}")
    print(f"report={report['outputs']['report']}")
    if visualization_stats is not None:
        print(
            f"finalized_roi_overlays={visualization_stats['saved']}/"
            f"{visualization_stats['rows']} sources={len(visualization_stats['sources'])} "
            f"output={visualization_stats['output_dir']}"
        )


if __name__ == "__main__":
    main()
