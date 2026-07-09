from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.cvat_io import import_cvat_xml
from hand_autolabel.formats import cfg_path, load_yaml_config, read_jsonl, repo_root_from_config, resolve_path, write_json, write_jsonl
from hand_autolabel.quality_checks import summarize_label_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import reviewed CVAT XML back to crop-level JSONL.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--reviewed-xml", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--draft-jsonl", default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    review_dir = cfg_path(cfg, root, "review_dir")
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    labels_dir = cfg_path(cfg, root, "labels_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    xml_path = resolve_path(root, args.reviewed_xml) if args.reviewed_xml else review_dir / "cvat_reviewed.xml"
    manifest_path = resolve_path(root, args.manifest) if args.manifest else roi_dir / "hand_roi_crops_manifest.jsonl"
    draft_path = resolve_path(root, args.draft_jsonl) if args.draft_jsonl else labels_dir / "hand_landmarks_autolabel_draft.jsonl"
    output_path = resolve_path(root, args.output_jsonl) if args.output_jsonl else labels_dir / "hand_landmarks_reviewed.jsonl"
    rows, import_stats = import_cvat_xml(xml_path, read_jsonl(manifest_path), read_jsonl(draft_path), cfg)
    write_jsonl(output_path, rows)
    stats = summarize_label_rows(rows, cfg)
    stats.update(import_stats)
    stats["output_jsonl"] = str(output_path)
    write_json(qc_dir / "cvat_import_stats.json", stats)
    print(f"reviewed={len(rows)} positive={stats['positive']} negative={stats['negative']} output={output_path}")


if __name__ == "__main__":
    main()
