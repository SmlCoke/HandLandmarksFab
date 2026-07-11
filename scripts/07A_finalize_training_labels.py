from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.finalization import FinalizationError, finalize_training
from hand_autolabel.formats import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize noisy pseudo labels and optional Gold overrides for training.")
    parser.add_argument("--config", default="configs/finalize_train.yaml")
    parser.add_argument("--stage", choices=("pretrain", "finetune"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = finalize_training(resolve_path(ROOT, args.config), args.stage)
    except FinalizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    counts = report["counts"]
    print(f"07A {args.stage}: included={counts['included']} excluded={counts['excluded']} catalog={counts['catalog']}")
    print(f"report={report['outputs']['report']}")


if __name__ == "__main__":
    main()
