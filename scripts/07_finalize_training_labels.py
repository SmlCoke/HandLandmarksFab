from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand_autolabel.formats import cfg_path, index_by, load_yaml_config, merge_label_with_manifest, read_jsonl, repo_root_from_config, resolve_path, write_json, write_jsonl
from hand_autolabel.quality_checks import label_issues, summarize_label_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize reviewed crop-level labels for Hand Landmarker training.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--reviewed-jsonl", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-jsonl", default=None)
    return parser.parse_args()


def _training_row(row):
    out = dict(row)
    present = bool((out.get("hand_presence") or {}).get("present", False))
    if not present:
        out["hand_id"] = None
        out["handedness"] = {"label": "unknown", "score": None}
        out["landmarks_crop_norm"] = []
        out["landmarks_crop_px"] = []
        out["landmarks_image_px"] = []
    out["hand_presence_loss_weight"] = 1.0
    out["landmark_loss_weight"] = 1.0 if present else 0.0
    label = str((out.get("handedness") or {}).get("label", "unknown")).lower()
    out["handedness_loss_weight"] = 1.0 if present and label in {"left", "right"} else 0.0
    out["source"] = out.get("source", "cvat_reviewed")
    return out


def main() -> None:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    cfg = load_yaml_config(config_path)
    root = repo_root_from_config(config_path)
    labels_dir = cfg_path(cfg, root, "labels_dir")
    roi_dir = cfg_path(cfg, root, "roi_crops_dir")
    reviewed_dir = cfg_path(cfg, root, "reviewed_dir")
    qc_dir = cfg_path(cfg, root, "qc_dir")
    reviewed_path = resolve_path(root, args.reviewed_jsonl) if args.reviewed_jsonl else reviewed_dir / "hand_landmarks_reviewed.jsonl"
    manifest_path = resolve_path(root, args.manifest) if args.manifest else roi_dir / "hand_roi_crops_manifest.jsonl"
    output_path = resolve_path(root, args.output_jsonl) if args.output_jsonl else labels_dir / "hand_training_labels.jsonl"
    manifest_by_crop = index_by(read_jsonl(manifest_path), "crop_id")

    rows = []
    skipped = []
    for raw in read_jsonl(reviewed_path):
        manifest = manifest_by_crop.get(str(raw.get("crop_id")), {})
        row = merge_label_with_manifest(raw, manifest, cfg) if manifest else dict(raw)
        warnings, errors, needs_review = label_issues(row, cfg)
        present = bool((row.get("hand_presence") or {}).get("present", False))
        if present and len(row.get("landmarks_crop_norm") or []) != 21:
            skipped.append({"crop_id": row.get("crop_id"), "errors": errors or ["positive_without_21_landmarks"]})
            continue
        if not present and (row.get("landmarks_crop_norm") or []):
            skipped.append({"crop_id": row.get("crop_id"), "errors": errors or ["negative_has_landmarks"]})
            continue
        out = _training_row(row)
        out["needs_review"] = bool(needs_review or row.get("needs_review", False))
        rows.append(out)

    write_jsonl(output_path, rows)
    stats = summarize_label_rows(rows, cfg)
    stats.update({"skipped": skipped, "output_jsonl": str(output_path)})
    write_json(qc_dir / "final_training_label_stats.json", stats)
    print(f"training_rows={len(rows)} positive={stats['positive']} negative={stats['negative']} skipped={len(skipped)} output={output_path}")


if __name__ == "__main__":
    main()
