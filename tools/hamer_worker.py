from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _read_requests(path: Path) -> list[Dict[str, str]]:
    requests: list[Dict[str, str]] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid request JSONL at line {line_number}"
                ) from exc
            if not isinstance(item, Mapping):
                raise ValueError(f"request line {line_number} must be an object")
            crop_id = str(item.get("crop_id") or "")
            crop_path = str(item.get("crop_path") or "")
            handedness = str(item.get("handedness") or "").strip().lower()
            if not crop_id or not crop_path:
                raise ValueError(
                    f"request line {line_number} requires crop_id and crop_path"
                )
            if handedness not in {"left", "right"}:
                raise ValueError(
                    f"request line {line_number} handedness must be left or right"
                )
            if crop_id in seen:
                raise ValueError(f"duplicate crop_id in request: {crop_id}")
            seen.add(crop_id)
            requests.append(
                {
                    "crop_id": crop_id,
                    "crop_path": crop_path,
                    "handedness": handedness,
                }
            )
    if not requests:
        raise ValueError("HaMeR request is empty")
    return requests


def _validate_keypoints(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 21:
        raise ValueError("HaMeR inference must return 21 2D keypoints")
    output: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("HaMeR keypoint must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("HaMeR keypoints must be finite")
        output.append([x, y])
    return output


def _write_results(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )


def run_hamer(
    repository: Path,
    checkpoint: Path,
    requests: Iterable[Mapping[str, str]],
    *,
    device: str,
    rescale: float,
) -> list[Dict[str, Any]]:
    repository = Path(repository).resolve()
    checkpoint = Path(checkpoint).resolve()
    if not (repository / "predict_hand_keypoints.py").is_file():
        raise FileNotFoundError(
            f"HaMeR repository is invalid: {repository}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HaMeR checkpoint does not exist: {checkpoint}")
    os.chdir(repository)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    try:
        import torch
        import predict_hand_keypoints as predict
    except Exception as exc:
        raise RuntimeError(
            "HaMeR runtime imports failed; use the repository .hamer environment"
        ) from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("hamer.device=cuda but torch.cuda is unavailable")

    model, model_cfg = predict.load_hamer(str(checkpoint))
    model = model.to(device)
    model.eval()
    output: list[Dict[str, Any]] = []
    for request in requests:
        image = predict.read_roi(Path(request["crop_path"]))
        handedness = str(request["handedness"])
        result = predict.infer_one(
            model,
            model_cfg,
            image,
            1 if handedness == "right" else 0,
            device,
            rescale,
        )
        output.append(
            {
                "crop_id": request["crop_id"],
                "keypoints_2d": _validate_keypoints(result.get("keypoints_2d")),
                "flipped": bool(result.get("flipped")),
                "bbox_size": float(result.get("bbox_size")),
            }
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch HaMeR Hand ROI worker")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--rescale", type=float, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if not math.isfinite(args.rescale) or args.rescale <= 0.0:
            raise ValueError("--rescale must be a positive finite number")
        requests = _read_requests(Path(args.request))
        output = run_hamer(
            Path(args.repository),
            Path(args.checkpoint),
            requests,
            device=args.device,
            rescale=float(args.rescale),
        )
        _write_results(Path(args.output), output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
