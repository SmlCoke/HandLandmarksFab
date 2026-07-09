from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import cfg_path, image_files, load_yaml_config, repo_root_from_config, resolve_path, write_json
from hand_autolabel.quality_checks import validate_image_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate upright 1280x720 grayscale TIFF images.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--output-report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    images_dir = resolve_path(root, args.images_dir) if args.images_dir else cfg_path(cfg, root, "images_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    report_path = resolve_path(root, args.output_report) if args.output_report else qc_dir / "image_validation_report.json"

    rows = [validate_image_file(p, int(cfg["image"]["width"]), int(cfg["image"]["height"])) for p in image_files(images_dir)]
    report = {
        "images_dir": str(images_dir),
        "total": len(rows),
        "ok": sum(1 for r in rows if r["ok"]),
        "failed": sum(1 for r in rows if not r["ok"]),
        "records": rows,
        "note": "Images are validated as-is; this script never rotates source images.",
    }
    write_json(report_path, report)
    print(f"validated={report['total']} ok={report['ok']} failed={report['failed']} report={report_path}")


if __name__ == "__main__":
    main()
