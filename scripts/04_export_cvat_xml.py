from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.cvat_io import export_cvat_xml
from hand_autolabel.formats import cfg_path, load_yaml_config, read_jsonl, repo_root_from_config, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export autolabel draft to CVAT for images 1.1 XML.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--draft-jsonl", default=None)
    parser.add_argument("--output-xml", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    manifest_path = resolve_path(root, args.manifest) if args.manifest else roi_dir / "hand_roi_crops_manifest.jsonl"
    draft_path = resolve_path(root, args.draft_jsonl) if args.draft_jsonl else roi_dir / "hand_landmarks_autolabel_draft.jsonl"
    xml_path = resolve_path(root, args.output_xml) if args.output_xml else roi_dir / "cvat_autolabel.xml"
    stats = export_cvat_xml(read_jsonl(manifest_path), read_jsonl(draft_path), root, xml_path, cfg)
    write_json(qc_dir / "cvat_export_stats.json", stats)
    print(f"images={stats['images']} positives={stats['positive_shapes']} xml={xml_path} upload_images={roi_dir / 'images'}")


if __name__ == "__main__":
    main()
