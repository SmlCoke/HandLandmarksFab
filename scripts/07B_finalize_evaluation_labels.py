from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.finalization import FinalizationError, finalize_evaluation
from hand_autolabel.formats import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly finalize fully reviewed validation/test Gold labels.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = finalize_evaluation(resolve_path(ROOT, args.config), args.split)
    except FinalizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    coverage = report["coverage"]
    print(f"07B {args.split}: included={coverage['included']} ignored={coverage['ignored']} reviewed={coverage['reviewed']}")
    print(f"report={report['outputs']['report']}")


if __name__ == "__main__":
    main()
