from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_autolabel.dataset_v3 import parse_capture_source_id, require_safe_id
from hand_autolabel.formats import load_yaml_config, read_jsonl, resolve_path
from hand_autolabel.image_io import read_image
from hand_autolabel.onnx_runtime import create_onnx_session, onnx_provider_for
from hand_autolabel.palm_decode import decode_onnx_outputs, normalize_feature_levels
from hand_autolabel.palm_onnx import palm_model_contract, preprocess_for_onnx
from hand_autolabel.quality_checks import RTMPOSE_CONNECTION_PAIRS
from hand_autolabel.roi_geometry import build_roi_rect_from_palm, roi_corners_px


SCORE_THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
NMS_THRESHOLDS = (0.10, 0.20, 0.30)
SCALE_VALUES = (1.5, 1.6, 1.7, 1.8)
MATCH_COST_LIMIT = 0.5


def parse_dataset_spec(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError("dataset must be <dataset_id>:<proposal_variant>")
    dataset_id, proposal_variant = value.split(":", 1)
    return (
        require_safe_id(dataset_id, "dataset_id"),
        require_safe_id(proposal_variant, "proposal_variant"),
    )


def _published_labels(
    dataset_root: Path,
    dataset_id: str,
    proposal_variant: str,
    source: Mapping[str, Any],
) -> Path:
    matches = [
        item
        for item in source.get("published_variants", [])
        if str(item.get("proposal_variant")) == proposal_variant
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{dataset_id}/{source.get('capture_source_id')} must contain one published {proposal_variant}"
        )
    path = (dataset_root / str(matches[0].get("labels_relpath") or "")).resolve()
    try:
        path.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"published labels escape dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _gold_points(row: Mapping[str, Any]) -> np.ndarray | None:
    if (
        row.get("human_reviewed") is not True
        or not bool((row.get("hand_presence") or {}).get("present"))
        or bool(row.get("ignore_for_training"))
    ):
        return None
    points = row.get("landmarks_image_px") or []
    if len(points) != 21:
        return None
    try:
        array = np.asarray([[float(point["x"]), float(point["y"])] for point in points], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    return array if np.isfinite(array).all() else None


def load_gold(
    dataset_root: Path,
    dataset_specs: Sequence[tuple[str, str]],
) -> tuple[Dict[tuple[Path, str], List[Dict[str, Any]]], List[Dict[str, Any]], List[float]]:
    groups: Dict[tuple[Path, str], List[Dict[str, Any]]] = defaultdict(list)
    summaries: List[Dict[str, Any]] = []
    old_occupancy: List[float] = []
    for dataset_id, proposal_variant in dataset_specs:
        manifest_path = dataset_root / "EValSource" / dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(manifest.get("scope")) != "eval":
            raise ValueError(f"{manifest_path} is not an Eval manifest")
        source_count = valid_count = 0
        for source in manifest.get("capture_sources", []):
            source_id = str(source.get("capture_source_id") or "")
            parts = parse_capture_source_id(source_id)
            source_dir = dataset_root / "EValSource" / dataset_id / source_id
            labels_path = _published_labels(
                dataset_root, dataset_id, proposal_variant, source
            )
            source_count += 1
            for row in read_jsonl(labels_path):
                points = _gold_points(row)
                if points is None:
                    continue
                old_points = row.get("landmarks_crop_px") or []
                if len(old_points) != 21:
                    continue
                old_array = np.asarray(
                    [[float(point["x"]), float(point["y"])] for point in old_points],
                    dtype=np.float32,
                )
                if not np.isfinite(old_array).all():
                    continue
                occupancy = max(float(np.ptp(old_array[:, 0])), float(np.ptp(old_array[:, 1]))) / 255.0
                old_occupancy.append(occupancy)
                groups[(source_dir, str(row["image"]))].append(
                    {
                        "points": points,
                        "distance": parts["distance"],
                    }
                )
                valid_count += 1
        summaries.append(
            {
                "dataset_id": dataset_id,
                "proposal_variant": proposal_variant,
                "sources": source_count,
                "gold_hands": valid_count,
            }
        )
    return groups, summaries, old_occupancy


class PalmRunner:
    def __init__(self, cfg: Mapping[str, Any], model_path: Path) -> None:
        preference = onnx_provider_for(cfg, "palm")
        self.session, self.provider, self.fallback_reason = create_onnx_session(
            model_path, preference
        )
        self.contract = palm_model_contract(self.session, cfg, model_path)
        self.palm_cfg = cfg["palm"]
        self.levels = normalize_feature_levels(self.palm_cfg)

    def outputs(self, image_path: Path) -> List[np.ndarray]:
        image = read_image(image_path)
        if image is None:
            raise ValueError(f"unreadable image: {image_path}")
        if image.shape[:2] != (720, 1280):
            raise ValueError(
                f"Eos-2.0 HLMF audit accepts upright 1280x720 only: {image_path} has {image.shape[:2]}"
            )
        tensor = preprocess_for_onnx(
            image,
            int(self.palm_cfg["input_width"]),
            int(self.palm_cfg["input_height"]),
            self.contract["input_type"],
        )
        return self.session.run(None, {self.contract["input_name"]: tensor})

    def decode(
        self,
        outputs: Sequence[np.ndarray],
        score_threshold: float,
        nms_threshold: float,
    ) -> List[Dict[str, Any]]:
        detections, _ = decode_onnx_outputs(
            outputs,
            self.levels,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_threshold,
            max_detections=int(self.palm_cfg["max_detections"]),
            negative_candidate_threshold=min(0.05, score_threshold),
            output_layout=str(self.palm_cfg["onnx_output_layout"]),
        )
        return detections


def analyze_test_images(runner: PalmRunner, test_images: Path) -> List[Dict[str, Any]]:
    paths = sorted(
        path
        for path in test_images.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"no top-level TIFF files in {test_images}")
    outputs = [runner.outputs(path) for path in paths]
    rows: List[Dict[str, Any]] = []
    for nms_threshold in NMS_THRESHOLDS:
        for score_threshold in SCORE_THRESHOLDS:
            counts = np.asarray(
                [
                    len(runner.decode(item, score_threshold, nms_threshold))
                    for item in outputs
                ],
                dtype=np.int32,
            )
            rows.append(
                {
                    "score": score_threshold,
                    "nms": nms_threshold,
                    "images": len(paths),
                    "zero": int(np.sum(counts == 0)),
                    "one": int(np.sum(counts == 1)),
                    "two": int(np.sum(counts == 2)),
                    "detections": int(np.sum(counts)),
                }
            )
    return rows


def _detection_pixels(detection: Mapping[str, Any]) -> Dict[str, Any]:
    bbox = detection["bbox_norm"]
    keypoints = detection["keypoints_norm"]
    return {
        "bbox_px": [
            float(bbox[0]) * 1280.0,
            float(bbox[1]) * 720.0,
            float(bbox[2]) * 1280.0,
            float(bbox[3]) * 720.0,
        ],
        "keypoints_px": {
            "p0": [float(keypoints["p0"][0]) * 1280.0, float(keypoints["p0"][1]) * 720.0],
            "p9": [float(keypoints["p9"][0]) * 1280.0, float(keypoints["p9"][1]) * 720.0],
        },
    }


def match_gold(
    runner: PalmRunner,
    groups: Mapping[tuple[Path, str], Sequence[Mapping[str, Any]]],
) -> tuple[List[Dict[str, Any]], float]:
    matched: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for (source_dir, image_name), gold_items in groups.items():
        detections = runner.decode(
            runner.outputs(source_dir / "images" / image_name), 0.05, 0.10
        )
        pairs: List[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            pixels = _detection_pixels(detection)
            p0 = np.asarray(pixels["keypoints_px"]["p0"], dtype=np.float32)
            p9 = np.asarray(pixels["keypoints_px"]["p9"], dtype=np.float32)
            for gold_index, gold in enumerate(gold_items):
                points = np.asarray(gold["points"], dtype=np.float32)
                span = max(float(np.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1]))), 1.0)
                cost = float(
                    (np.linalg.norm(p0 - points[0]) + np.linalg.norm(p9 - points[9]))
                    / (2.0 * span)
                )
                pairs.append((cost, detection_index, gold_index))
        used_detections: set[int] = set()
        used_gold: set[int] = set()
        for cost, detection_index, gold_index in sorted(pairs):
            if (
                cost > MATCH_COST_LIMIT
                or detection_index in used_detections
                or gold_index in used_gold
            ):
                continue
            used_detections.add(detection_index)
            used_gold.add(gold_index)
            detection = detections[detection_index]
            gold = gold_items[gold_index]
            matched.append(
                {
                    "detection": _detection_pixels(detection),
                    "score": float(detection["score"]),
                    "points": np.asarray(gold["points"], dtype=np.float32),
                    "distance": str(gold["distance"]),
                    "cost": cost,
                }
            )
    return matched, time.perf_counter() - started


def analyze_scales(
    matched: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    thresholds = cfg["quality"]["rtmpose_train_connection_length_thresholds_px"]
    rows: List[Dict[str, Any]] = []
    for scale in SCALE_VALUES:
        all_inside = point_inside = gate_flagged = 0
        occupancy: List[float] = []
        margins: List[float] = []
        for item in matched:
            rect = build_roi_rect_from_palm(
                item["detection"],
                1280,
                720,
                scale_x=scale,
                scale_y=scale,
                shift_x=0.0,
                shift_y=-0.1,
            )
            corners = roi_corners_px(rect)
            transform = cv2.getAffineTransform(
                np.asarray([corners[0], corners[1], corners[3]], dtype=np.float32),
                np.asarray([[0, 0], [255, 0], [0, 255]], dtype=np.float32),
            )
            points = cv2.transform(item["points"][None, :, :], transform)[0]
            inside = (
                (points[:, 0] >= 0.0)
                & (points[:, 0] <= 255.0)
                & (points[:, 1] >= 0.0)
                & (points[:, 1] <= 255.0)
            )
            all_inside += int(inside.all())
            point_inside += int(inside.sum())
            occupancy.append(
                max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1]))) / 255.0
            )
            margins.append(
                float(
                    np.min(
                        np.column_stack(
                            [points[:, 0], points[:, 1], 255.0 - points[:, 0], 255.0 - points[:, 1]]
                        )
                    )
                )
            )
            distance_thresholds = thresholds[str(item["distance"])]
            if any(
                float(np.linalg.norm(points[start] - points[end]))
                > float(distance_thresholds[f"{start}-{end}"])
                for start, end in RTMPOSE_CONNECTION_PAIRS
            ):
                gate_flagged += 1
        rows.append(
            {
                "scale": scale,
                "all_inside_rate": all_inside / len(matched),
                "point_inside_rate": point_inside / (len(matched) * 21),
                "occupancy_p50": float(np.quantile(occupancy, 0.50)),
                "occupancy_p95": float(np.quantile(occupancy, 0.95)),
                "margin_p01": float(np.quantile(margins, 0.01)),
                "gate_flagged": gate_flagged,
                "gate_flagged_rate": gate_flagged / len(matched),
            }
        )
    return rows


def _percent(value: float) -> str:
    return f"{value * 100.0:.3f}%"


def render_report(
    cfg: Mapping[str, Any],
    runner: PalmRunner,
    dataset_summaries: Sequence[Mapping[str, Any]],
    groups: Mapping[tuple[Path, str], Sequence[Mapping[str, Any]]],
    old_occupancy: Sequence[float],
    test_rows: Sequence[Mapping[str, Any]],
    matched: Sequence[Mapping[str, Any]],
    elapsed: float,
    scale_rows: Sequence[Mapping[str, Any]],
) -> str:
    total_gold = sum(len(items) for items in groups.values())
    selected_test = [
        row for row in test_rows if row["nms"] == 0.10 and row["score"] in {0.20, 0.25, 0.30, 0.50}
    ]
    score_rows = []
    for threshold in SCORE_THRESHOLDS:
        count = sum(float(item["score"]) >= threshold for item in matched)
        score_rows.append((threshold, count, count / total_gold))
    lines = [
        "# Eos-2.0 HLMF 兼容回放",
        "",
        "> 该文件由只读分析工具生成；它不修改数据仓库，也不等价于新的正式 Eos-2.0 Eval。",
        "",
        "## 结论",
        "",
        f"当前配置为 score `{cfg['palm']['score_threshold']}`、NMS `{cfg['palm']['nms_iou_threshold']}`、ROI scale `{cfg['hand_roi']['scale_x']}/{cfg['hand_roi']['scale_y']}`。代表性 Eos-2.0 Gold 发布后应使用连接长度统计工具重算 near/mid 阈值并回放 gold/draft；far 在模型不支持期间只保留不可达的历史值。",
        "",
        "## 模型契约",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| model/provider | `{runner.contract['model_id']}` / `{runner.provider}` |",
        f"| input | `{runner.contract['input_shape']}` float32，灰度、INTER_AREA、/255 |",
        f"| outputs | `{runner.contract['output_shapes']}` |",
        f"| anchors | {runner.contract['anchor_count']} |",
        "",
        "## 数据",
        "",
        "| dataset | variant | sources | gold hands |",
        "|---|---|---:|---:|",
    ]
    for summary in dataset_summaries:
        lines.append(
            f"| {summary['dataset_id']} | `{summary['proposal_variant']}` | {summary['sources']} | {summary['gold_hands']} |"
        )
    lines.extend(
        [
            f"| `/root/Test/Eos-2.0/testcase/inputs` | top-level TIFF | - | {test_rows[0]['images']} images |",
            "",
            "## score 回放（NMS=0.10）",
            "",
            "| score | zero/one/two images | detections | legacy-gold associations | association coverage |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    score_lookup = {threshold: (count, rate) for threshold, count, rate in score_rows}
    for row in selected_test:
        count, rate = score_lookup[float(row["score"])]
        lines.append(
            f"| {row['score']:.2f} | {row['zero']}/{row['one']}/{row['two']} | {row['detections']} | {count} | {_percent(rate)} |"
        )
    lines.extend(
        [
            "",
            "Association 使用 Eos p0/p9 与人工 0/9 点的尺度归一化距离做一对一匹配，只用于新旧 ROI 兼容分析，不应解释为 Palm 官方 recall。",
            "",
            "## ROI scale 回放",
            "",
            "| scale | hands fully inside | points inside | occupancy P50/P95 | margin P01 (px) | old gate flagged |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scale_rows:
        lines.append(
            f"| {row['scale']:.1f} | {_percent(row['all_inside_rate'])} | {_percent(row['point_inside_rate'])} | {row['occupancy_p50']:.3f}/{row['occupancy_p95']:.3f} | {row['margin_p01']:.2f} | {row['gate_flagged']} ({_percent(row['gate_flagged_rate'])}) |"
        )
    lines.extend(
        [
            "",
            f"旧 ROI occupancy P50/P95 为 `{np.quantile(old_occupancy, 0.5):.3f}/{np.quantile(old_occupancy, 0.95):.3f}`。本次匹配 {len(matched)}/{total_gold} 条 gold，推理与匹配耗时 {elapsed:.1f}s。",
            "",
            "## 重跑",
            "",
            "```bash",
            "python -B tools/analyze_eos2_adaptation.py \\",
            "  --dataset-root /root/autodl-tmp/DatesetFab \\",
            "  --dataset FullEnhanceVal0801:eos-1.0 \\",
            "  --dataset FullEnhanceVal0808:eos_1.0-gate_r2 \\",
            "  --test-images /root/Test/Eos-2.0/testcase/inputs \\",
            "  --config configs/autolabel.yaml \\",
            "  --output /tmp/eos2_compatibility_replay.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def validate_policy(cfg: Mapping[str, Any]) -> None:
    expected = {
        "model_id": "eos-2.0",
        "input_width": 384,
        "input_height": 224,
        "score_threshold": 0.25,
        "nms_iou_threshold": 0.10,
        "max_detections": 2,
        "negative_candidate_threshold": 0.15,
    }
    for key, value in expected.items():
        if cfg["palm"].get(key) != value:
            raise ValueError(f"configured palm.{key}={cfg['palm'].get(key)!r}, expected {value!r}")
    if float(cfg["hand_roi"]["scale_x"]) != 1.8 or float(cfg["hand_roi"]["scale_y"]) != 1.8:
        raise ValueError("Eos-2.0 compatibility policy requires hand_roi scale_x/scale_y=1.8")
    if cfg["quality"].get("rtmpose_train_connection_length_gate_enabled") is not True:
        raise ValueError("RTMPose connection-length gate must remain enabled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Eos-2.0 HLMF compatibility replay")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    try:
        output.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("analysis output must not be inside HAND_DATASET_ROOT")
    cfg = load_yaml_config(args.config.resolve())
    validate_policy(cfg)
    model_path = args.model.resolve() if args.model else resolve_path(REPO_ROOT, cfg["paths"]["palm_model_onnx"])
    runner = PalmRunner(cfg, model_path)
    specs = [parse_dataset_spec(value) for value in args.dataset]
    groups, summaries, old_occupancy = load_gold(dataset_root, specs)
    test_rows = analyze_test_images(runner, args.test_images.resolve())
    matched, elapsed = match_gold(runner, groups)
    if not matched:
        raise ValueError("no Eos-2.0 detections matched legacy gold")
    scale_rows = analyze_scales(matched, cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_report(
            cfg,
            runner,
            summaries,
            groups,
            old_occupancy,
            test_rows,
            matched,
            elapsed,
            scale_rows,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
