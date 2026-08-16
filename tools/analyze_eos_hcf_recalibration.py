from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_autolabel.formats import load_yaml_config, read_jsonl
from hand_autolabel.handedness_classifier import (
    HAND_CLASSIFIER_INPUT_NAME,
    HAND_CLASSIFIER_OUTPUT_NAMES,
    decode_hand_classifier_batch,
    preprocess_hand_classifier_images,
)
from hand_autolabel.image_io import read_image
from hand_autolabel.onnx_runtime import create_onnx_session
from hand_autolabel.palm_decode import (
    decode_onnx_outputs,
    normalize_feature_levels,
    select_detections,
)
from hand_autolabel.palm_onnx import palm_model_contract, preprocess_for_onnx
from hand_autolabel.quality_checks import RTMPOSE_CONNECTION_PAIRS
from hand_autolabel.roi_geometry import (
    build_roi_rect_from_palm,
    crop_image_by_roi,
    roi_corners_px,
)
from hand_autolabel.rtmpose_hand_labeler import (
    RTMPOSE_INPUT_NAME,
    RTMPOSE_OUTPUT_NAMES,
    decode_simcc_batch,
    preprocess_rtmpose_images,
)


SOURCE_VARIANTS = {
    "complex-mid-bright-random-val-s01-peak": "eos_2.0-rtmpose-hcf0813-gate",
    "complex-mid-bright-random-val-s01-soar": "eos_2.0-rtmpose-hcf0813-gate",
    "complex-near-bright-random-val-s01-peak": "eos_2.0-rtmpose-hcf0813-gate",
    "complex-mid-dark-random-val-s01-peak": "eos_2.0-rtmpose-gate",
    "white-mid-bright-random-val-s01-soar": "eos_2.0-rtmpose-gate",
    "complex-mid-bright-random-test-s01-peak": "eos_2.0-rtmpose-gate",
    "complex-near-bright-random-test-s01-peak": "eos_2.0-rtmpose-gate",
}

PALM_SCORE_VALUES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PALM_NMS_VALUES = (0.05, 0.10, 0.20, 0.30)
PALM_MAX_VALUES = (1, 2, 3)
PRESENCE_VALUES = (
    0.001,
    0.005,
    0.010,
    0.025,
    0.050,
    0.100,
    0.200,
    0.300,
    0.400,
    0.500,
    0.600,
    0.700,
    0.800,
    0.900,
    0.950,
    0.975,
    0.990,
)
HANDEDNESS_VALUES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
ROI_SCALES = (1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
ROI_SHIFT_Y_VALUES = (-0.20, -0.15, -0.10, -0.05, 0.0)
QUANTILE = 0.9995
SAFETY_FACTOR = 1.05
MATCH_COST_LIMIT = 0.50


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values -= values.max(axis=1, keepdims=True)
    exps = np.exp(values)
    return exps / exps.sum(axis=1, keepdims=True)


def _distance_from_source(source_id: str) -> str:
    parts = source_id.split("-")
    if len(parts) != 7:
        raise ValueError(source_id)
    return parts[1]


def _published_labels(
    dataset_root: Path, source_id: str, variant: str
) -> Path:
    manifest = json.loads(
        (dataset_root / "EValSource" / "FullEnhanceVal0801" / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        item for item in manifest["capture_sources"] if item["capture_source_id"] == source_id
    )
    matches = [
        item
        for item in source.get("published_variants", [])
        if item.get("proposal_variant") == variant
    ]
    if len(matches) != 1:
        raise ValueError(f"{source_id}: expected one published {variant}")
    return dataset_root / matches[0]["labels_relpath"]


def _points(row: Mapping[str, Any], key: str) -> np.ndarray | None:
    raw = row.get(key) or []
    if len(raw) != 21:
        return None
    try:
        points = np.asarray([[float(item["x"]), float(item["y"])] for item in raw], dtype=np.float32)
    except (KeyError, TypeError, ValueError):
        return None
    return points if np.isfinite(points).all() else None


def load_gold(
    dataset_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], list[dict[str, Any]]],
    set[tuple[str, str]],
]:
    rows: list[dict[str, Any]] = []
    hands_by_image: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    reviewed_images: set[tuple[str, str]] = set()
    for source_id, variant in SOURCE_VARIANTS.items():
        labels = _published_labels(dataset_root, source_id, variant)
        for row in read_jsonl(labels):
            if row.get("human_reviewed") is not True or bool(row.get("ignore_for_training")):
                continue
            present = bool((row.get("hand_presence") or {}).get("present"))
            image_points = _points(row, "landmarks_image_px") if present else None
            crop_points = _points(row, "landmarks_crop_px") if present else None
            item = {
                "source_id": source_id,
                "variant": variant,
                "distance": _distance_from_source(source_id),
                "image": str(row["image"]),
                "crop_path": dataset_root / str(row["crop_path"]),
                "present": present,
                "handedness": str((row.get("handedness") or {}).get("label") or "unknown"),
                "image_points": image_points,
                "crop_points": crop_points,
                "human_modified_landmark_ids": list(row.get("human_modified_landmark_ids") or []),
            }
            rows.append(item)
            reviewed_images.add((source_id, item["image"]))
            if present and image_points is not None:
                hands_by_image[(source_id, item["image"])].append(item)
    return rows, hands_by_image, reviewed_images


def _detection_pixels(candidate: Mapping[str, Any]) -> dict[str, Any]:
    bbox = candidate["bbox_norm"]
    keypoints = candidate["keypoints_norm"]
    return {
        "score": float(candidate["score"]),
        "head": str(candidate["head"]),
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


def _match(
    detections: Sequence[Mapping[str, Any]], hands: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        p0 = np.asarray(detection["keypoints_px"]["p0"], dtype=np.float32)
        p9 = np.asarray(detection["keypoints_px"]["p9"], dtype=np.float32)
        for hand_index, hand in enumerate(hands):
            points = np.asarray(hand["image_points"], dtype=np.float32)
            span = max(float(np.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1]))), 1.0)
            cost = float(
                (np.linalg.norm(p0 - points[0]) + np.linalg.norm(p9 - points[9]))
                / (2.0 * span)
            )
            pairs.append((cost, detection_index, hand_index))
    used_detections: set[int] = set()
    used_hands: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for cost, detection_index, hand_index in sorted(pairs):
        if cost > MATCH_COST_LIMIT or detection_index in used_detections or hand_index in used_hands:
            continue
        used_detections.add(detection_index)
        used_hands.add(hand_index)
        matches.append((detection_index, hand_index, cost))
    return (
        matches,
        [index for index in range(len(detections)) if index not in used_detections],
        [index for index in range(len(hands)) if index not in used_hands],
    )


def _rates(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _decode(
    outputs: Sequence[np.ndarray], levels: Sequence[Mapping[str, Any]], score: float, nms: float, maximum: int
) -> list[dict[str, Any]]:
    detections, _ = decode_onnx_outputs(
        outputs,
        levels,
        score_threshold=score,
        nms_iou_threshold=nms,
        max_detections=maximum,
        negative_candidate_threshold=score,
        output_layout="nchw",
    )
    return [_detection_pixels(item) for item in detections]


def _candidate_pool(
    outputs: Sequence[np.ndarray], levels: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    candidates, _ = decode_onnx_outputs(
        outputs,
        levels,
        score_threshold=min(PALM_SCORE_VALUES),
        nms_iou_threshold=1.0,
        max_detections=0,
        negative_candidate_threshold=min(PALM_SCORE_VALUES),
        output_layout="nchw",
    )
    return candidates


def _decode_pool(
    candidates: Sequence[Mapping[str, Any]], score: float, nms: float, maximum: int
) -> list[dict[str, Any]]:
    eligible = [item for item in candidates if float(item["score"]) >= score]
    return [
        _detection_pixels(item)
        for item in select_detections(eligible, nms, maximum)
    ]


def palm_analysis(
    cfg: Mapping[str, Any],
    palm_model: Path,
    dataset_root: Path,
    hands_by_image: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    reviewed_images: set[tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gpu, gpu_provider, gpu_fallback = create_onnx_session(palm_model, "cuda")
    cpu, cpu_provider, cpu_fallback = create_onnx_session(palm_model, "cpu")
    contract = palm_model_contract(gpu, cfg, palm_model)
    levels = normalize_feature_levels(cfg["palm"])
    image_entries: list[tuple[str, str, Path]] = []
    for source_id in SOURCE_VARIANTS:
        source_dir = dataset_root / "EValSource" / "FullEnhanceVal0801" / source_id
        image_entries.extend(
            (source_id, path.name, path)
            for path in sorted((source_dir / "images").iterdir())
            if path.suffix.lower() in {".tif", ".tiff"}
            and (source_id, path.name) in reviewed_images
        )
    candidate_pools: dict[tuple[str, str], list[dict[str, Any]]] = {}
    tensors: list[np.ndarray] = []
    started = time.perf_counter()
    for source_id, image_name, image_path in image_entries:
        image = read_image(image_path)
        tensor = preprocess_for_onnx(
            image,
            int(cfg["palm"]["input_width"]),
            int(cfg["palm"]["input_height"]),
            contract["input_type"],
        )
        if len(tensors) < 128:
            tensors.append(tensor)
        item_outputs = gpu.run(None, {contract["input_name"]: tensor})
        candidate_pools[(source_id, image_name)] = _candidate_pool(item_outputs, levels)
    full_elapsed = time.perf_counter() - started

    consistency = {"samples": len(tensors), "max_abs_raw": 0.0, "decoded_mismatches": 0}
    for tensor in tensors:
        out_cpu = cpu.run(None, {contract["input_name"]: tensor})
        out_gpu = gpu.run(None, {contract["input_name"]: tensor})
        consistency["max_abs_raw"] = max(
            consistency["max_abs_raw"],
            max(float(np.max(np.abs(a - b))) for a, b in zip(out_cpu, out_gpu)),
        )
        dec_cpu = _decode(out_cpu, levels, 0.25, 0.10, 2)
        dec_gpu = _decode(out_gpu, levels, 0.25, 0.10, 2)
        if len(dec_cpu) != len(dec_gpu):
            consistency["decoded_mismatches"] += 1
            continue
        if any(
            abs(float(a["score"]) - float(b["score"])) > 1e-5
            or np.max(np.abs(np.asarray(a["bbox_px"]) - np.asarray(b["bbox_px"]))) > 1e-3
            for a, b in zip(dec_cpu, dec_gpu)
        ):
            consistency["decoded_mismatches"] += 1

    def benchmark(session: Any, tensor: np.ndarray, repeats: int = 100) -> dict[str, float]:
        for _ in range(10):
            session.run(None, {contract["input_name"]: tensor})
        values = []
        for _ in range(repeats):
            tick = time.perf_counter()
            session.run(None, {contract["input_name"]: tensor})
            values.append(time.perf_counter() - tick)
        median = statistics.median(values)
        return {"median_ms": median * 1000.0, "images_per_second": 1.0 / median}

    total_gold_hands = sum(len(items) for items in hands_by_image.values())
    sweeps = []
    for nms in PALM_NMS_VALUES:
        for maximum in PALM_MAX_VALUES:
            for score in PALM_SCORE_VALUES:
                tp = fp = fn = detections_count = 0
                by_distance = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
                for source_id, image_name, _ in image_entries:
                    detections = _decode_pool(
                        candidate_pools[(source_id, image_name)], score, nms, maximum
                    )
                    hands = hands_by_image.get((source_id, image_name), [])
                    matches, unmatched_detections, unmatched_hands = _match(detections, hands)
                    distance = _distance_from_source(source_id)
                    tp += len(matches)
                    fp += len(unmatched_detections)
                    fn += len(unmatched_hands)
                    detections_count += len(detections)
                    by_distance[distance]["tp"] += len(matches)
                    by_distance[distance]["fp"] += len(unmatched_detections)
                    by_distance[distance]["fn"] += len(unmatched_hands)
                item = {"score": score, "nms": nms, "max_detections": maximum, "detections": detections_count}
                item.update(_rates(tp, fp, fn))
                item["by_distance"] = {
                    key: _rates(value["tp"], value["fp"], value["fn"])
                    for key, value in by_distance.items()
                }
                sweeps.append(item)

    selected: list[dict[str, Any]] = []
    selected_score, selected_nms, selected_max = 0.25, 0.10, 2
    for source_id, image_name, image_path in image_entries:
        detections = _decode_pool(
            candidate_pools[(source_id, image_name)], selected_score, selected_nms, selected_max
        )
        hands = hands_by_image.get((source_id, image_name), [])
        matches, unmatched_detections, unmatched_hands = _match(detections, hands)
        for detection_index, hand_index, cost in matches:
            selected.append(
                {
                    "source_id": source_id,
                    "distance": _distance_from_source(source_id),
                    "image": image_name,
                    "image_path": image_path,
                    "detection": detections[detection_index],
                    "hand": hands[hand_index],
                    "cost": cost,
                }
            )

    result = {
        "model": str(palm_model),
        "contract": contract,
        "feature_levels": levels,
        "providers": {
            "gpu": gpu_provider,
            "gpu_fallback": gpu_fallback,
            "cpu": cpu_provider,
            "cpu_fallback": cpu_fallback,
        },
        "images": len(image_entries),
        "gold_hands": total_gold_hands,
        "full_gpu_elapsed_seconds": full_elapsed,
        "consistency": consistency,
        "benchmark": {
            "cpu": benchmark(cpu, tensors[0]),
            "gpu": benchmark(gpu, tensors[0]),
        },
        "sweeps": sweeps,
    }
    return result, selected


def _transform_points(rect: Mapping[str, Any], points: np.ndarray) -> tuple[np.ndarray, float, float]:
    corners = roi_corners_px(rect)
    transform = cv2.getAffineTransform(
        np.asarray([corners[0], corners[1], corners[3]], dtype=np.float32),
        np.asarray([[0, 0], [255, 0], [0, 255]], dtype=np.float32),
    )
    transformed = cv2.transform(points[None, :, :], transform)[0]
    inside = (
        (transformed[:, 0] >= 0.0)
        & (transformed[:, 0] <= 255.0)
        & (transformed[:, 1] >= 0.0)
        & (transformed[:, 1] <= 255.0)
    )
    margin = float(
        np.min(
            np.column_stack(
                [
                    transformed[:, 0],
                    transformed[:, 1],
                    255.0 - transformed[:, 0],
                    255.0 - transformed[:, 1],
                ]
            )
        )
    )
    return transformed, float(inside.mean()), margin


def roi_geometry_analysis(selected: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid = []
    adopted_projected: list[dict[str, Any]] = []
    for scale in ROI_SCALES:
        for shift_y in ROI_SHIFT_Y_VALUES:
            all_inside = 0
            point_rates = []
            margins = []
            occupancies = []
            for item in selected:
                rect = build_roi_rect_from_palm(
                    item["detection"], 1280, 720, scale_x=scale, scale_y=scale, shift_x=0.0, shift_y=shift_y
                )
                projected, point_rate, margin = _transform_points(rect, item["hand"]["image_points"])
                all_inside += int(point_rate == 1.0)
                point_rates.append(point_rate)
                margins.append(margin)
                occupancies.append(max(float(np.ptp(projected[:, 0])), float(np.ptp(projected[:, 1]))) / 255.0)
                if scale == 1.8 and shift_y == -0.1:
                    adopted_projected.append({**dict(item), "rect": rect, "projected_points": projected})
            grid.append(
                {
                    "scale_x": scale,
                    "scale_y": scale,
                    "shift_x": 0.0,
                    "shift_y": shift_y,
                    "hands_fully_inside_rate": all_inside / len(selected),
                    "points_inside_rate": float(np.mean(point_rates)),
                    "occupancy_p50": float(np.quantile(occupancies, 0.50)),
                    "occupancy_p95": float(np.quantile(occupancies, 0.95)),
                    "margin_p01": float(np.quantile(margins, 0.01)),
                }
            )
    return {"matched_hands": len(selected), "grid": grid}, adopted_projected


def connection_analysis(projected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {
        distance: {pair: [] for pair in RTMPOSE_CONNECTION_PAIRS} for distance in ("near", "mid")
    }
    for item in projected:
        points = item["projected_points"]
        for pair in RTMPOSE_CONNECTION_PAIRS:
            values[item["distance"]][pair].append(float(np.linalg.norm(points[pair[0]] - points[pair[1]])))
    stats: dict[str, Any] = {}
    for distance in values:
        stats[distance] = {}
        for pair, raw in values[distance].items():
            array = np.asarray(raw, dtype=np.float64)
            p50, p95, p9995 = np.quantile(array, [0.5, 0.95, QUANTILE], method="linear")
            stats[distance][f"{pair[0]}-{pair[1]}"] = {
                "n": len(array),
                "mean": float(array.mean()),
                "variance": float(array.var(ddof=1)),
                "p50": float(p50),
                "p95": float(p95),
                "p9995": float(p9995),
                "max": float(array.max()),
                "threshold": int(math.ceil(float(p9995) * SAFETY_FACTOR)),
            }
    flagged = defaultdict(int)
    for item in projected:
        for pair in RTMPOSE_CONNECTION_PAIRS:
            threshold = stats[item["distance"]][f"{pair[0]}-{pair[1]}"]["threshold"]
            if float(np.linalg.norm(item["projected_points"][pair[0]] - item["projected_points"][pair[1]])) > threshold:
                flagged[item["distance"]] += 1
                break
    return {"stats": stats, "gold_flagged": dict(flagged)}


def _batched(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _classifier_outputs(session: Any, crops: Sequence[np.ndarray], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    handedness = []
    presence = []
    for batch in _batched(crops, batch_size):
        tensor = preprocess_hand_classifier_images(batch)
        outputs = session.run(list(HAND_CLASSIFIER_OUTPUT_NAMES), {HAND_CLASSIFIER_INPUT_NAME: tensor})
        handedness.append(outputs[0])
        presence.append(outputs[1])
    return np.concatenate(handedness), np.concatenate(presence)


def _classifier_thresholds(
    records: Sequence[Mapping[str, Any]], handedness_logits: np.ndarray, presence_logits: np.ndarray
) -> dict[str, Any]:
    handedness_probs = _softmax(handedness_logits)
    presence_probs = _softmax(presence_logits)[:, 1]
    truth_presence = np.asarray([bool(item["present"]) for item in records])
    presence_rows = []
    for threshold in PRESENCE_VALUES:
        predicted = presence_probs >= threshold
        tp = int(np.sum(predicted & truth_presence))
        fp = int(np.sum(predicted & ~truth_presence))
        fn = int(np.sum(~predicted & truth_presence))
        tn = int(np.sum(~predicted & ~truth_presence))
        by_source = {}
        for source_id in SOURCE_VARIANTS:
            mask = np.asarray([item["source_id"] == source_id for item in records])
            pos = mask & truth_presence
            neg = mask & ~truth_presence
            by_source[source_id] = {
                "hands": int(pos.sum()),
                "hand_retained": int(np.sum(predicted & pos)),
                "no_hands": int(neg.sum()),
                "no_hand_rejected": int(np.sum(~predicted & neg)),
            }
        presence_rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "hand_retention": tp / int(truth_presence.sum()),
                "no_hand_rejection": tn / int((~truth_presence).sum()) if (~truth_presence).sum() else None,
                "by_source": by_source,
            }
        )

    valid_handedness = np.asarray(
        [item["present"] and item["handedness"] in {"Left", "Right"} for item in records]
    )
    truth_handedness = np.asarray([0 if item["handedness"] == "Left" else 1 for item in records])
    predicted_handedness = np.argmax(handedness_logits, axis=1)
    handedness_score = np.max(handedness_probs, axis=1)
    handedness_rows = []
    valid_count = int(valid_handedness.sum())
    for threshold in HANDEDNESS_VALUES:
        covered = valid_handedness & (handedness_score >= threshold)
        correct = covered & (predicted_handedness == truth_handedness)
        handedness_rows.append(
            {
                "threshold": threshold,
                "valid": valid_count,
                "covered": int(covered.sum()),
                "coverage": float(covered.sum()) / valid_count,
                "correct": int(correct.sum()),
                "accuracy_in_coverage": float(correct.sum()) / int(covered.sum()) if covered.sum() else None,
                "errors_in_coverage": int(covered.sum() - correct.sum()),
            }
        )
    return {
        "presence": presence_rows,
        "handedness": handedness_rows,
        "raw_handedness_accuracy": float(
            np.mean(predicted_handedness[valid_handedness] == truth_handedness[valid_handedness])
        ),
    }


def hcf_analysis(
    hcf_model: Path, gold_rows: Sequence[Mapping[str, Any]], projected: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, Any]:
    cpu, cpu_provider, cpu_fallback = create_onnx_session(hcf_model, "cpu")
    gpu, gpu_provider, gpu_fallback = create_onnx_session(hcf_model, "cuda")
    formal_records = []
    formal_crops = []
    for row in gold_rows:
        crop = read_image(row["crop_path"])
        if crop is None:
            raise ValueError(row["crop_path"])
        formal_records.append(row)
        formal_crops.append(crop)
    started = time.perf_counter()
    gpu_h, gpu_p = _classifier_outputs(gpu, formal_crops, batch_size)
    gpu_full_elapsed = time.perf_counter() - started
    cpu_h, cpu_p = _classifier_outputs(cpu, formal_crops, batch_size)
    cpu_full_elapsed = time.perf_counter() - started - gpu_full_elapsed

    gpu_h_prob = _softmax(gpu_h)
    gpu_p_prob = _softmax(gpu_p)
    cpu_h_prob = _softmax(cpu_h)
    cpu_p_prob = _softmax(cpu_p)
    crossings = {
        "presence_argmax": int(np.sum(np.argmax(gpu_p, axis=1) != np.argmax(cpu_p, axis=1))),
        "handedness_argmax": int(np.sum(np.argmax(gpu_h, axis=1) != np.argmax(cpu_h, axis=1))),
        "presence_0.025": int(np.sum((gpu_p_prob[:, 1] >= 0.025) != (cpu_p_prob[:, 1] >= 0.025))),
        "presence_0.5": int(np.sum((gpu_p_prob[:, 1] >= 0.5) != (cpu_p_prob[:, 1] >= 0.5))),
        "handedness_0.8": int(
            np.sum((np.max(gpu_h_prob, axis=1) >= 0.8) != (np.max(cpu_h_prob, axis=1) >= 0.8))
        ),
    }

    def benchmark(session: Any, crops: Sequence[np.ndarray], repeats: int = 80) -> dict[str, float]:
        batch = list(crops[:batch_size])
        tensor = preprocess_hand_classifier_images(batch)
        for _ in range(10):
            session.run(list(HAND_CLASSIFIER_OUTPUT_NAMES), {HAND_CLASSIFIER_INPUT_NAME: tensor})
        elapsed = []
        for _ in range(repeats):
            tick = time.perf_counter()
            session.run(list(HAND_CLASSIFIER_OUTPUT_NAMES), {HAND_CLASSIFIER_INPUT_NAME: tensor})
            elapsed.append(time.perf_counter() - tick)
        median = statistics.median(elapsed)
        return {"median_ms": median * 1000.0, "images_per_second": batch_size / median}

    reconstructed_records = []
    reconstructed_crops = []
    image_cache: dict[Path, np.ndarray] = {}
    for item in projected:
        path = Path(item["image_path"])
        if path not in image_cache:
            image_cache[path] = read_image(path)
        crop, _ = crop_image_by_roi(image_cache[path], item["rect"], 256, 256)
        reconstructed_crops.append(crop)
        reconstructed_records.append(
            {"source_id": item["source_id"], "present": True, "handedness": item["hand"]["handedness"]}
        )
    reconstructed_h, reconstructed_p = _classifier_outputs(gpu, reconstructed_crops, batch_size)
    return {
        "model": str(hcf_model),
        "model_id": f"hand-classifier-{hcf_model.parent.name}",
        "providers": {
            "cpu": cpu_provider,
            "cpu_fallback": cpu_fallback,
            "gpu": gpu_provider,
            "gpu_fallback": gpu_fallback,
        },
        "formal_gold": {
            "records": len(formal_records),
            "hands": sum(bool(item["present"]) for item in formal_records),
            "no_hands": sum(not bool(item["present"]) for item in formal_records),
            "thresholds": _classifier_thresholds(formal_records, gpu_h, gpu_p),
        },
        "reconstructed_eos21_hands": {
            "records": len(reconstructed_records),
            "thresholds": _classifier_thresholds(reconstructed_records, reconstructed_h, reconstructed_p),
        },
        "consistency": {
            "max_logits_abs": max(float(np.max(np.abs(gpu_h - cpu_h))), float(np.max(np.abs(gpu_p - cpu_p)))),
            "max_probability_abs": max(
                float(np.max(np.abs(gpu_h_prob - cpu_h_prob))), float(np.max(np.abs(gpu_p_prob - cpu_p_prob)))
            ),
            "crossings": crossings,
        },
        "benchmark": {
            "batch_size": batch_size,
            "cpu": benchmark(cpu, formal_crops),
            "gpu": benchmark(gpu, formal_crops),
            "full_cpu_seconds": cpu_full_elapsed,
            "full_gpu_seconds": gpu_full_elapsed,
        },
    }


def rtmpose_replay(
    model: Path, projected: Sequence[Mapping[str, Any]], connection: Mapping[str, Any], batch_size: int
) -> dict[str, Any]:
    session, provider, fallback = create_onnx_session(model, "cpu")
    crops = []
    records = []
    image_cache: dict[Path, np.ndarray] = {}
    for item in projected:
        path = Path(item["image_path"])
        if path not in image_cache:
            image_cache[path] = read_image(path)
        crop, _ = crop_image_by_roi(image_cache[path], item["rect"], 256, 256)
        crops.append(crop)
        records.append(item)
    predictions = []
    for batch in _batched(crops, batch_size):
        tensor = preprocess_rtmpose_images(batch)
        outputs = session.run(list(RTMPOSE_OUTPUT_NAMES), {RTMPOSE_INPUT_NAME: tensor})
        predictions.extend(decode_simcc_batch(outputs[0], outputs[1], split_ratio=2.0))
    flagged = defaultdict(int)
    flagged_bad = defaultdict(int)
    bad = defaultdict(int)
    errors = []
    for item, (points, _) in zip(records, predictions):
        gold = item["projected_points"]
        mean_error = float(np.linalg.norm(points - gold, axis=1).mean())
        errors.append(mean_error)
        is_bad = mean_error >= 10.0
        bad[item["distance"]] += int(is_bad)
        is_flagged = False
        for pair in RTMPOSE_CONNECTION_PAIRS:
            threshold = connection["stats"][item["distance"]][f"{pair[0]}-{pair[1]}"]["threshold"]
            if float(np.linalg.norm(points[pair[0]] - points[pair[1]])) > threshold:
                is_flagged = True
                break
        flagged[item["distance"]] += int(is_flagged)
        flagged_bad[item["distance"]] += int(is_flagged and is_bad)
    return {
        "provider": provider,
        "fallback": fallback,
        "records": len(records),
        "mean_point_error_px": float(np.mean(errors)),
        "p95_mean_point_error_px": float(np.quantile(errors, 0.95)),
        "bad_ge_10px": dict(bad),
        "connection_flagged": dict(flagged),
        "flagged_and_bad": dict(flagged_bad),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the canonical seven FullEnhanceVal0801 Gold sources to recalibrate "
            "Eos, HCF, ROI geometry, and RTMPose connection-length gates without writing "
            "to HAND_DATASET_ROOT."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--palm-model", type=Path, required=True)
    parser.add_argument("--hcf-model", type=Path, required=True)
    parser.add_argument("--rtmpose-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
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
        raise ValueError("output must not be inside HAND_DATASET_ROOT")
    cfg = load_yaml_config(args.config.resolve())
    cfg["palm"]["model_id"] = "eos-2.1"
    cfg["paths"]["palm_model_onnx"] = str(args.palm_model)
    gold_rows, hands_by_image, reviewed_images = load_gold(dataset_root)
    palm, selected = palm_analysis(
        cfg,
        args.palm_model.resolve(),
        dataset_root,
        hands_by_image,
        reviewed_images,
    )
    roi, projected = roi_geometry_analysis(selected)
    connection = connection_analysis(projected)
    hcf = hcf_analysis(args.hcf_model.resolve(), gold_rows, projected, args.batch_size)
    rtmpose = rtmpose_replay(args.rtmpose_model.resolve(), projected, connection, args.batch_size)
    result = {
        "source_variants": SOURCE_VARIANTS,
        "gold": {
            "rows": len(gold_rows),
            "hands": sum(bool(item["present"]) for item in gold_rows),
            "no_hands": sum(not bool(item["present"]) for item in gold_rows),
            "images_with_hands": len(hands_by_image),
        },
        "palm": palm,
        "roi": roi,
        "connection": connection,
        "hcf": hcf,
        "rtmpose_replay": rtmpose,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
