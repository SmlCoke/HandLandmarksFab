from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.finetune_gold import (
    GoldPipelineError,
    build_pretrain_source_registry,
    export_finetune_gold,
    finalize_gold_aggregate,
    import_all_finetune_gold,
    import_finetune_gold,
    prepare_dragon_gold,
    seed_finetune_gold,
)
from hand_autolabel.formats import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, review and aggregate finetune-only HLMF Gold sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dragon = subparsers.add_parser("prepare-dragon", help="Import the legacy Dragon external-Gold dataset.")
    dragon.add_argument("--config", default="configs/dragon_gold.yaml")

    registry = subparsers.add_parser("source-registry", help="Build the authenticated pretrain source lookup for HLML.")
    registry.add_argument("--config", default="configs/finalize_train.yaml")

    export = subparsers.add_parser("export", help="Materialize a source and export its strict CVAT task.")
    export.add_argument("--config", default="configs/finetune_gold.yaml")
    export.add_argument("--source-id", required=True)
    export.add_argument("--source-mode", choices=("selection_subset", "native_existing"), required=True)
    export.add_argument("--raw-source-root")
    export.add_argument("--selection-request")
    export.add_argument("--max-items", type=int)

    seed = subparsers.add_parser("seed", help="Seed a new workspace from authenticated historical Gold.")
    seed.add_argument("--config", default="configs/finetune_gold.yaml")
    seed.add_argument("--base-finetune-id", required=True)
    seed.add_argument("--finetune-id", required=True)

    imported = subparsers.add_parser("import", help="Strictly import one or all returned CVAT Gold tasks.")
    imported.add_argument("--config", default="configs/finetune_gold.yaml")
    imported_group = imported.add_mutually_exclusive_group(required=True)
    imported_group.add_argument("--source-id")
    imported_group.add_argument("--all", action="store_true", help="Preflight and publish every returned task.")

    finalize = subparsers.add_parser("finalize", help="Authenticate and aggregate all published HLMF Gold sources.")
    finalize.add_argument("--config", default="configs/finalize_finetune.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    try:
        if args.command == "prepare-dragon":
            result = prepare_dragon_gold(config_path)
        elif args.command == "source-registry":
            result = build_pretrain_source_registry(config_path)
        elif args.command == "export":
            result = export_finetune_gold(
                config_path,
                source_id=args.source_id,
                source_mode=args.source_mode,
                raw_source_root=Path(args.raw_source_root).resolve() if args.raw_source_root else None,
                selection_request=Path(args.selection_request).resolve() if args.selection_request else None,
                max_items=args.max_items,
            )
        elif args.command == "seed":
            result = seed_finetune_gold(
                config_path,
                base_finetune_id=args.base_finetune_id,
                finetune_id=args.finetune_id,
            )
        elif args.command == "import":
            result = (
                import_all_finetune_gold(config_path)
                if args.all
                else import_finetune_gold(config_path, source_id=args.source_id)
            )
        else:
            result = finalize_gold_aggregate(config_path)
    except (GoldPipelineError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
