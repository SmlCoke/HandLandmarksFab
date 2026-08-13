from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_autolabel.dataset_v3 import parse_capture_source_id, require_safe_id
from hand_autolabel.formats import load_yaml_config, read_jsonl
from hand_autolabel.quality_checks import (
    RTMPOSE_CONNECTION_DISTANCES,
    RTMPOSE_CONNECTION_PAIRS,
    rtmpose_connection_lengths_px,
    validate_rtmpose_connection_thresholds,
)

QUANTILE = 0.9995
SAFETY_FACTOR = 1.05


def parse_dataset_spec(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise ValueError("dataset must be <dataset_id>:<proposal_variant>")
    dataset_id, proposal_variant = value.split(":", 1)
    return (
        require_safe_id(dataset_id, "dataset_id"),
        require_safe_id(proposal_variant, "proposal_variant"),
    )


def _valid_gold_row(row: Mapping[str, Any]) -> bool:
    return (
        row.get("human_reviewed") is True
        and bool((row.get("hand_presence") or {}).get("present"))
        and not bool(row.get("ignore_for_training"))
    )


def _source_label_file(
    dataset_root: Path,
    dataset_id: str,
    proposal_variant: str,
    source: Mapping[str, Any],
) -> Path | None:
    matches = [
        item
        for item in source.get("published_variants", [])
        if str(item.get("proposal_variant")) == proposal_variant
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"{dataset_id}/{source.get('capture_source_id')} must have exactly one "
            f"published variant {proposal_variant!r}"
        )
    labels_relpath = str(matches[0].get("labels_relpath") or "")
    path = (dataset_root / labels_relpath).resolve()
    try:
        path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"published label path escapes dataset root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _threshold_stats(values: Iterable[float]) -> Dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        raise ValueError("each distance/connection requires at least two finite values")
    p50, p95, p9995 = np.quantile(array, [0.5, 0.95, QUANTILE], method="linear")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "variance": float(array.var(ddof=1)),
        "p50": float(p50),
        "p95": float(p95),
        "p9995": float(p9995),
        "max": float(array.max()),
        "threshold": int(math.ceil(float(p9995) * SAFETY_FACTOR)),
    }


def _preserved_threshold_stats(threshold: float) -> Dict[str, float | int | None]:
    return {
        "n": 0,
        "mean": None,
        "variance": None,
        "p50": None,
        "p95": None,
        "p9995": None,
        "max": None,
        "threshold": int(threshold),
    }


def analyze_datasets(
    dataset_root: Path,
    dataset_specs: Sequence[tuple[str, str]],
    *,
    fallback_thresholds: Mapping[str, Mapping[tuple[int, int], float]] | None = None,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    values: Dict[str, Dict[tuple[int, int], List[float]]] = {
        distance: {pair: [] for pair in RTMPOSE_CONNECTION_PAIRS}
        for distance in RTMPOSE_CONNECTION_DISTANCES
    }
    sources: List[Dict[str, Any]] = []
    gold_entries: List[Dict[str, Any]] = []
    excluded = defaultdict(int)

    for dataset_id, proposal_variant in dataset_specs:
        manifest_path = (
            dataset_root / "EValSource" / dataset_id / "dataset_manifest.json"
        )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if str(manifest.get("scope")) != "eval":
            raise ValueError(f"{manifest_path} is not an Eval dataset manifest")
        for source in manifest.get("capture_sources", []):
            source_id = str(source.get("capture_source_id") or "")
            source_parts = parse_capture_source_id(source_id)
            distance = source_parts["distance"]
            if distance not in values:
                raise ValueError(f"unsupported distance {distance!r} in {source_id}")
            labels_file = _source_label_file(
                dataset_root, dataset_id, proposal_variant, source
            )
            if labels_file is None:
                continue
            valid_rows = 0
            total_rows = 0
            for row in read_jsonl(labels_file):
                total_rows += 1
                if not _valid_gold_row(row):
                    excluded["not_eligible_gold"] += 1
                    continue
                try:
                    lengths = rtmpose_connection_lengths_px(
                        row.get("landmarks_crop_px") or []
                    )
                except ValueError:
                    excluded["invalid_landmarks"] += 1
                    continue
                valid_rows += 1
                for pair, length in lengths.items():
                    values[distance][pair].append(length)
                gold_entries.append(
                    {
                        "dataset_id": dataset_id,
                        "proposal_variant": proposal_variant,
                        "source_id": source_id,
                        "distance": distance,
                        "row": row,
                        "lengths": lengths,
                    }
                )
            sources.append(
                {
                    "dataset_id": dataset_id,
                    "proposal_variant": proposal_variant,
                    "source_id": source_id,
                    "distance": distance,
                    "split": source_parts["split"],
                    "performer": source_parts["performer"],
                    "total_rows": total_rows,
                    "valid_rows": valid_rows,
                    "labels_file": labels_file,
                }
            )

    if not sources:
        variants = ", ".join(f"{dataset}:{variant}" for dataset, variant in dataset_specs)
        raise ValueError(f"no published sources found for requested variants: {variants}")

    preserved_distances: List[str] = []
    stats: Dict[str, Dict[tuple[int, int], Dict[str, float | int | None]]] = {}
    for distance in RTMPOSE_CONNECTION_DISTANCES:
        if values[distance][RTMPOSE_CONNECTION_PAIRS[0]]:
            stats[distance] = {
                pair: _threshold_stats(values[distance][pair])
                for pair in RTMPOSE_CONNECTION_PAIRS
            }
            continue
        if fallback_thresholds is None or distance not in fallback_thresholds:
            raise ValueError(
                f"distance {distance!r} has no published samples; "
                "configured fallback thresholds are required"
            )
        preserved_distances.append(distance)
        stats[distance] = {
            pair: _preserved_threshold_stats(fallback_thresholds[distance][pair])
            for pair in RTMPOSE_CONNECTION_PAIRS
        }
    thresholds = {
        distance: {
            pair: int(stats[distance][pair]["threshold"])
            for pair in RTMPOSE_CONNECTION_PAIRS
        }
        for distance in RTMPOSE_CONNECTION_DISTANCES
    }

    replay = {
        distance: {
            "gold": 0,
            "gold_flagged": 0,
            "draft_joined": 0,
            "draft_flagged": 0,
            "draft_flagged_human_modified": 0,
        }
        for distance in RTMPOSE_CONNECTION_DISTANCES
    }
    drafts_by_source: Dict[tuple[str, str, str], Dict[str, Mapping[str, Any]]] = {}
    for source in sources:
        key = (
            source["dataset_id"],
            source["proposal_variant"],
            source["source_id"],
        )
        draft_path = (
            dataset_root
            / "EValSource"
            / source["dataset_id"]
            / source["source_id"]
            / "02_roi_crops"
            / source["proposal_variant"]
            / "hand_landmarks_autolabel_draft.jsonl"
        )
        if not draft_path.is_file():
            raise FileNotFoundError(draft_path)
        drafts: Dict[str, Mapping[str, Any]] = {}
        for row in read_jsonl(draft_path):
            if str(row.get("source")) != "rtmpose_m_hand5_onnx":
                continue
            try:
                rtmpose_connection_lengths_px(row.get("landmarks_crop_px") or [])
            except ValueError:
                continue
            drafts[str(row.get("crop_id"))] = row
        drafts_by_source[key] = drafts

    for entry in gold_entries:
        distance = entry["distance"]
        replay[distance]["gold"] += 1
        if any(
            entry["lengths"][pair] > thresholds[distance][pair]
            for pair in RTMPOSE_CONNECTION_PAIRS
        ):
            replay[distance]["gold_flagged"] += 1
        key = (
            entry["dataset_id"],
            entry["proposal_variant"],
            entry["source_id"],
        )
        row = entry["row"]
        draft = drafts_by_source[key].get(str(row.get("crop_id")))
        if draft is None:
            continue
        draft_lengths = rtmpose_connection_lengths_px(
            draft.get("landmarks_crop_px") or []
        )
        replay[distance]["draft_joined"] += 1
        flagged = any(
            draft_lengths[pair] > thresholds[distance][pair]
            for pair in RTMPOSE_CONNECTION_PAIRS
        )
        if flagged:
            replay[distance]["draft_flagged"] += 1
            if row.get("human_modified_landmark_ids"):
                replay[distance]["draft_flagged_human_modified"] += 1

    return {
        "dataset_root": dataset_root,
        "dataset_specs": list(dataset_specs),
        "sources": sources,
        "excluded": dict(excluded),
        "stats": stats,
        "thresholds": thresholds,
        "replay": replay,
        "preserved_distances": preserved_distances,
    }


def verify_config_thresholds(
    analysis: Mapping[str, Any], cfg: Mapping[str, Any]
) -> None:
    configured = validate_rtmpose_connection_thresholds(cfg)
    mismatches: List[str] = []
    for distance in RTMPOSE_CONNECTION_DISTANCES:
        for pair in RTMPOSE_CONNECTION_PAIRS:
            expected = float(analysis["thresholds"][distance][pair])
            actual = configured[distance][pair]
            if actual != expected:
                mismatches.append(
                    f"{distance}.{pair[0]}-{pair[1]}:{actual:g}!={expected:g}"
                )
    if mismatches:
        raise ValueError("config threshold mismatch: " + ", ".join(mismatches))


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def render_report(analysis: Mapping[str, Any], command: str) -> str:
    sources = list(analysis["sources"])
    total_valid = sum(int(source["valid_rows"]) for source in sources)
    total_gold_flagged = sum(
        int(analysis["replay"][distance]["gold_flagged"])
        for distance in RTMPOSE_CONNECTION_DISTANCES
    )
    total_draft_flagged = sum(
        int(analysis["replay"][distance]["draft_flagged"])
        for distance in RTMPOSE_CONNECTION_DISTANCES
    )
    total_draft_modified = sum(
        int(analysis["replay"][distance]["draft_flagged_human_modified"])
        for distance in RTMPOSE_CONNECTION_DISTANCES
    )
    lines = [
        "# RTMPose 连接对长度质量门控统计",
        "",
        "## 结论",
        "",
        "遮挡时 RTMPose 可能把不可见关键点预测到无关位置，使相邻骨骼异常变长。本门控以人工复核 Eval 的 crop 像素长度为基准，过滤这类高置信异常；它不能发现所有遮挡或仍落在正常长度范围内的乱飞点。规则仅用于 RTMPose Train runtime，长度为 0 不拒绝。",
        "",
        f"本次共统计 **{total_valid:,}** 条有效 gold hand。阈值采用 `ceil(P99.95 × 1.05)`；gold 回放保留 **{total_valid-total_gold_flagged:,}/{total_valid:,}**，RTMPose 草标命中 **{total_draft_flagged:,}** 条，其中 **{total_draft_modified:,}** 条后来确有人工修点。",
        "",
        "## 数据集",
        "",
        "| Dataset | Variant | 距离 | 来源数 | 有效 hand |",
        "|---|---|---:|---:|---:|",
    ]
    grouped: Dict[tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"sources": 0, "valid": 0}
    )
    for source in sources:
        key = (source["dataset_id"], source["proposal_variant"], source["distance"])
        grouped[key]["sources"] += 1
        grouped[key]["valid"] += int(source["valid_rows"])
    for key in sorted(grouped):
        item = grouped[key]
        lines.append(
            f"| `{key[0]}` | `{key[1]}` | {key[2]} | {item['sources']} | {item['valid']:,} |"
        )
    lines.extend(["", "<details>", "<summary>完整 capture source 清单</summary>", ""])
    for source in sorted(sources, key=lambda item: (item["dataset_id"], item["source_id"])):
        lines.append(
            f"- `{source['dataset_id']}/{source['source_id']}`：{source['valid_rows']:,} 条"
        )
    lines.extend(["", "</details>", "", "## 统计方法", ""])
    lines.extend(
        [
            "- 连接对：`0-1-2-3-4`、`0-5-6-7-8`、`0-9-10-11-12`、`0-13-14-15-16`、`0-17-18-19-20`，共 20 条。",
            "- 长度：在 `256×256` Hand ROI 中计算两点欧氏距离；按 capture source 的 `near/mid/far` 分组。",
            "- 样本：仅保留已发布、`human_reviewed=true`、presence 为 hand、未忽略且 21 点有限的标签。",
            "- 阈值：每组取经验 P99.95，乘 1.05 泛化裕量后向上取整。样本最大值容易受残留极端值影响；当前分布偏态明显，因此不直接采用最大值或正态假设下的 `μ+3σ`。",
        ]
    )
    if analysis.get("preserved_distances"):
        joined = "/".join(analysis["preserved_distances"])
        lines.append(
            f"- `{joined}` 本次没有该模型支持的已发布样本，统计列记为 `—`，阈值保留 YAML 中的历史值；不把无样本的旧阈值伪装成新统计结果。"
        )
    for distance in RTMPOSE_CONNECTION_DISTANCES:
        lines.extend(
            [
                "",
                f"## {distance} 统计",
                "",
                "| 连接 | N | 均值 | 方差 | P50 | P95 | P99.95 | 最大值 | 阈值 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pair in RTMPOSE_CONNECTION_PAIRS:
            stat = analysis["stats"][distance][pair]
            lines.append(
                f"| {pair[0]}-{pair[1]} | {stat['n']:,} | {_fmt(stat['mean'])} | "
                f"{_fmt(stat['variance'])} | {_fmt(stat['p50'])} | {_fmt(stat['p95'])} | "
                f"{_fmt(stat['p9995'])} | {_fmt(stat['max'])} | {stat['threshold']} |"
            )
    lines.extend(
        [
            "",
            "## 回放验证",
            "",
            "| 距离 | Gold | Gold 保留 | 保留率 | 对齐草标 | 门控命中 | 命中且人工修点 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    total = defaultdict(int)
    for distance in RTMPOSE_CONNECTION_DISTANCES:
        item = analysis["replay"][distance]
        retained = item["gold"] - item["gold_flagged"]
        rate = f"{retained / item['gold']:.3%}" if item["gold"] else "—"
        lines.append(
            f"| {distance} | {item['gold']:,} | {retained:,} | {rate} | "
            f"{item['draft_joined']:,} | {item['draft_flagged']:,} | "
            f"{item['draft_flagged_human_modified']:,} |"
        )
        for key, value in item.items():
            total[key] += int(value)
    retained = total["gold"] - total["gold_flagged"]
    lines.append(
        f"| 合计 | {total['gold']:,} | {retained:,} | {retained/total['gold']:.3%} | "
        f"{total['draft_joined']:,} | {total['draft_flagged']:,} | "
        f"{total['draft_flagged_human_modified']:,} |"
    )
    lines.extend(
        [
            "",
            "## 运行与重新统计",
            "",
            "运行时由 `quality.rtmpose_train_connection_length_gate_enabled` 控制，默认开启；严格大于对应阈值才拒绝，关闭时不解析距离或阈值。",
            "",
            "```bash",
            command,
            "```",
            "",
            "出现以下任一情况时重新统计：新增有代表性的人工复核 Eval 数据；更新 Eos 后 proposal/ROI 几何发生变化；调整 ROI 构造参数。重算前应确认所有输入来源均已人工复核并发布；重算后审查回放保留率、更新 YAML 阈值并运行完整测试。增加数据集时追加 `--dataset <dataset_id>:<proposal_variant>`。该工具只读取数据仓库，只写指定报告。",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze reviewed Eval hand-connection lengths and verify gate thresholds."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "autolabel.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "assets"
        / "quality_gate"
        / "rtmpose_connection_length_distribution.md",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the existing report exactly matches recomputed content.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = [parse_dataset_spec(value) for value in args.dataset]
    cfg = load_yaml_config(args.config)
    configured = validate_rtmpose_connection_thresholds(cfg)
    analysis = analyze_datasets(
        args.dataset_root, specs, fallback_thresholds=configured
    )
    verify_config_thresholds(analysis, cfg)
    dataset_args = " ".join(
        f"--dataset {dataset_id}:{variant}" for dataset_id, variant in specs
    )
    command = (
        "python -B tools/analyze_rtmpose_connection_lengths.py \\\n"
        f"  --dataset-root {args.dataset_root} \\\n"
        f"  {dataset_args} \\\n"
        f"  --config {args.config} --output {args.output}"
    )
    report = render_report(analysis, command)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != report:
            raise SystemExit(f"report is stale: {args.output}")
        print(f"report and config match: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output} from {sum(s['valid_rows'] for s in analysis['sources'])} gold hands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
