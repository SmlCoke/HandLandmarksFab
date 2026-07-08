# File purpose: Run batch hand-landmark inference with Keras weights using ROI crops and render results.
import argparse
import shutil
from pathlib import Path

import cv2
try:
    import h5py
except Exception:
    h5py = None
import numpy as np
from tqdm.auto import tqdm

try:
    from model_2d_NHWC import hand_landmark_2d_model
except Exception:
    from model_2d_NHWC.model_2d_NHWC import hand_landmark_2d_model
from roi_utils import (
    clamp01,
    crop_image_by_rect,
    project_points_from_rect,
    rect_from_bbox_and_hand_points,
)

INPUT_W = 256
INPUT_H = 256
LANDMARK_DIM = 42
ROI_NUM_VALUES = 8
ROI_NUM_VALUES_WITH_SCORE = 9
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# MediaPipe-like hand drawing style (copied from MediaPipe default drawing_styles).
MP_RED = (48, 48, 255)
MP_GREEN = (48, 255, 48)
MP_BLUE = (192, 101, 21)
MP_YELLOW = (0, 204, 255)
MP_GRAY = (128, 128, 128)
MP_PURPLE = (128, 64, 128)
MP_PEACH = (180, 229, 255)

MP_CONN_THICKNESS = 3
MP_LANDMARK_RADIUS = 5
# Fixed hand drawing colors: red keypoints + yellow connections.
DRAW_KEYPOINT_COLOR = (0, 0, 255)
DRAW_CONNECTION_COLOR = (0, 255, 255)

PALM_CONNECTIONS = {
    frozenset((0, 1)),
    frozenset((0, 5)),
    frozenset((5, 9)),
    frozenset((9, 13)),
    frozenset((13, 17)),
    frozenset((0, 17)),
}
THUMB_CONNECTIONS = {
    frozenset((1, 2)),
    frozenset((2, 3)),
    frozenset((3, 4)),
}
INDEX_CONNECTIONS = {
    frozenset((5, 6)),
    frozenset((6, 7)),
    frozenset((7, 8)),
}
MIDDLE_CONNECTIONS = {
    frozenset((9, 10)),
    frozenset((10, 11)),
    frozenset((11, 12)),
}
RING_CONNECTIONS = {
    frozenset((13, 14)),
    frozenset((14, 15)),
    frozenset((15, 16)),
}
PINKY_CONNECTIONS = {
    frozenset((17, 18)),
    frozenset((18, 19)),
    frozenset((19, 20)),
}

LANDMARK_COLOR = {
    0: MP_RED,
    1: MP_RED,
    2: MP_PEACH,
    3: MP_PEACH,
    4: MP_PEACH,
    5: MP_PURPLE,
    6: MP_PURPLE,
    7: MP_PURPLE,
    8: MP_PURPLE,
    9: MP_YELLOW,
    10: MP_YELLOW,
    11: MP_YELLOW,
    12: MP_YELLOW,
    13: MP_GREEN,
    14: MP_GREEN,
    15: MP_GREEN,
    16: MP_GREEN,
    17: MP_BLUE,
    18: MP_BLUE,
    19: MP_BLUE,
    20: MP_BLUE,
}



def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch infer hand landmarks from frames + anchor/points text."
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default=r"D:\IC_Innovation_Challenge\train_with_annotations\infer\frames_gray",
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--points_file",
        type=str,
        default=r"D:\IC_Innovation_Challenge\train_with_annotations\infer\frames_anchor_points_model_gray.txt",
        help="Text file containing ROI bbox + wrist/middle points per frame.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=r"D:\IC_Innovation_Challenge\train_with_annotations\hand\hand_landmarks_best_gray.weights.h5",
        help="Model weights path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"D:\IC_Innovation_Challenge\train_with_annotations\infer\frames_hand_model_gray",
        help="Output directory for rendered results.",
    )
    parser.add_argument(
        "--roi_scale_x",
        type=float,
        default=1.5,
        help="ROI width scale for palm-to-hand expansion.",
    )
    parser.add_argument(
        "--roi_scale_y",
        type=float,
        default=1.5,
        help="ROI height scale for palm-to-hand expansion.",
    )
    parser.add_argument(
        "--roi_shift_y",
        type=float,
        default=-0.1,
        help="ROI center Y shift for palm-to-hand expansion.",
     )
    parser.add_argument(
        "--hand_flag_thr",
        type=float,
        default=0.5,
        help="Threshold to treat hand_flag as hand-present.",
    )
    parser.add_argument(
        "--handedness_thr",
        type=float,
        default=0.5,
        help="Threshold to treat handedness as Right (below is Left).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of ROI crops to run through the landmark model per predict call.",
    )
    parser.add_argument(
        "--frame_chunk_size",
        type=int,
        default=256,
        help="Number of source frames to stage before flushing batched ROI inference.",
    )
    parser.add_argument(
        "--clean_out_dir",
        dest="clean_out_dir",
        action="store_true",
        help="Clear existing files in out_dir before saving new inference results (default: enabled).",
    )
    parser.add_argument(
        "--no_clean_out_dir",
        dest="clean_out_dir",
        action="store_false",
        help="Do not clear out_dir; append/overwrite only by filename.",
    )
    parser.set_defaults(clean_out_dir=True)
    return parser.parse_args()


def imread_unicode(path):
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def imwrite_unicode(path, image):
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def ensure_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_gray(img):
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess_crop(crop):
    gray = ensure_gray(crop)
    gray = cv2.resize(gray, (INPUT_W, INPUT_H))
    gray = gray.astype(np.float32) / 255.0
    return gray[np.newaxis, np.newaxis, :, :]

def decode_landmarks(coords_pred):
    coords = np.asarray(coords_pred).reshape(-1)
    expected_dim = 42
    if coords.shape[0] != expected_dim:
        raise RuntimeError(f"Invalid landmark output shape. expected={expected_dim}, got={coords.shape[0]}")

    coord_scale = 1.0
    if float(np.max(np.abs(coords))) > 2.0:
        coord_scale = float(INPUT_W)

    pts = []
    for i in range(0, 42, 2):
        # Keep raw normalized coordinates to avoid artificial edge sticking.
        pts.append((float(coords[i] / coord_scale), float(coords[i + 1] / coord_scale)))
    return pts


def _select_landmark_output(pred):
    expected_dim = 42
    if not isinstance(pred, (list, tuple)):
        arr = np.asarray(pred)
        if arr.size % expected_dim == 0:
            return arr
        raise RuntimeError("Model output is not list/tuple and does not match landmark dimension.")

    for item in pred:
        arr = np.asarray(item)
        if arr.size % expected_dim == 0:
            return arr

    got_shapes = [tuple(np.asarray(item).shape) for item in pred]
    raise RuntimeError(f"Cannot find landmark output with dim={expected_dim}, got outputs={got_shapes}")


def _first_scalar(arr_like):
    arr = np.asarray(arr_like).reshape(-1)
    if arr.size == 0:
        return None
    return float(arr[0])


def decode_model_outputs(pred):
    coords_pred = _select_landmark_output(pred)
    hand_flag_score = None
    handedness_score = None

    if isinstance(pred, (list, tuple)):
        if len(pred) >= 3:
            hand_flag_score = _first_scalar(pred[1])
            handedness_score = _first_scalar(pred[2])
        else:
            scalar_items = []
            for item in pred:
                arr = np.asarray(item)
                if arr.size % LANDMARK_DIM == 0:
                    continue
                scalar = _first_scalar(arr)
                if scalar is not None:
                    scalar_items.append(scalar)
            if scalar_items:
                hand_flag_score = scalar_items[0]
            if len(scalar_items) >= 2:
                handedness_score = scalar_items[1]

    return coords_pred, hand_flag_score, handedness_score


def bbox_to_px(bbox_norm, image_width, image_height):
    xmin, ymin, xmax, ymax = [clamp01(v) for v in bbox_norm]
    px1 = int(xmin * image_width)
    py1 = int(ymin * image_height)
    px2 = int(xmax * image_width)
    py2 = int(ymax * image_height)
    px1 = max(0, min(px1, image_width - 1))
    py1 = max(0, min(py1, image_height - 1))
    px2 = max(px1 + 1, min(px2, image_width))
    py2 = max(py1 + 1, min(py2, image_height))
    return px1, py1, px2, py2


def _connection_color(a, b):
    _ = (a, b)
    return DRAW_CONNECTION_COLOR


def _format_raw_score(score):
    if score is None:
        return "NA"
    try:
        v = float(score)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    # Keep raw confidence readable while avoiding forced 0/1 rounding.
    return f"{v:.6g}"


def draw_result(
    out,
    bbox_norm,
    roi_corners_px,
    pts_abs,
    hand_idx,
    palm_det_score,
    hand_flag_score,
    handedness_score,
    handedness_thr,
):
    h, w = out.shape[:2]
    px1, py1, px2, py2 = bbox_to_px(bbox_norm, w, h)
    cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 255), 1)
    roi_poly = np.round(roi_corners_px).astype(np.int32).reshape((-1, 1, 2))
    roi_xy = roi_poly.reshape(-1, 2)
    roi_x = int(np.clip(np.min(roi_xy[:, 0]), 0, w - 1))
    roi_y = int(np.clip(np.min(roi_xy[:, 1]), 0, h - 1))
    roi_x2 = int(np.clip(np.max(roi_xy[:, 0]), 0, w - 1))
    cv2.polylines(out, [roi_poly], True, (0, 255, 0), 2)

    info_lines = []
    score_text = "NA" if palm_det_score is None else f"{float(palm_det_score):.2f}"
    hand_flag_text = _format_raw_score(hand_flag_score)
    label_texts = [score_text, f"hf:{hand_flag_text}"]
    label_y = roi_y - 8 if roi_y - 8 > 12 else roi_y + 18
    for label in label_texts:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        box_tl = (roi_x, label_y - th - baseline - 2)
        box_br = (roi_x + tw + 4, label_y + baseline)
        cv2.rectangle(out, box_tl, box_br, (0, 255, 0), thickness=-1)
        cv2.putText(
            out,
            label,
            (roi_x + 2, label_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        label_y += th + baseline + 8

    if handedness_score is None:
        pass
    else:
        right_p = float(handedness_score)
        left_p = 1.0 - right_p

        if right_p >= left_p:
            lr_text = "R"
            lr_prob = right_p
        else:
            lr_text = "L"
            lr_prob = left_p
        lr_label = f"{lr_text}:{lr_prob:.2f}"
        (tw, th), baseline = cv2.getTextSize(lr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x = max(0, roi_x2 - tw - 4)
        label_y = roi_y - 8 if roi_y - 8 > 12 else roi_y + 18
        box_tl = (label_x, label_y - th - baseline - 2)
        box_br = (label_x + tw + 4, label_y + baseline)
        cv2.rectangle(out, box_tl, box_br, (0, 255, 0), thickness=-1)
        cv2.putText(
            out,
            lr_label,
            (label_x + 2, label_y - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    ty = max(12, roi_y - 30)
    for i, text in enumerate(info_lines):
        cv2.putText(
            out,
            text,
            (roi_x, ty + i * 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    pts_int = []
    for x, y in pts_abs:
        xi = int(np.clip(x, 0.0, float(w - 1)))
        yi = int(np.clip(y, 0.0, float(h - 1)))
        pts_int.append((xi, yi))

    for a, b in HAND_CONNECTIONS:
        if a >= len(pts_int) or b >= len(pts_int):
            continue
        cv2.line(
            out,
            pts_int[a],
            pts_int[b],
            _connection_color(a, b),
            MP_CONN_THICKNESS,
            cv2.LINE_AA,
        )

    for idx, (xi, yi) in enumerate(pts_int):
        cv2.circle(out, (xi, yi), MP_LANDMARK_RADIUS, DRAW_KEYPOINT_COLOR, -1, cv2.LINE_AA)
    return out


def draw_frame_summary(out, pred_hand_count, pred_left_count, pred_right_count, roi_count):
    if pred_hand_count > 0:
        text = f"hand={pred_hand_count} L={pred_left_count} R={pred_right_count} roi={roi_count}"
        color = (0, 220, 0)
    else:
        text = f"hand=0 L=0 R=0 roi={roi_count}"
        color = (0, 180, 255)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def _slice_model_prediction(pred, index):
    if isinstance(pred, (list, tuple)):
        return [np.asarray(item)[index : index + 1] for item in pred]
    return np.asarray(pred)[index : index + 1]


def predict_and_draw_roi_jobs(model, jobs, batch_size, hand_flag_thr, handedness_thr, progress=None):
    if not jobs:
        return 0

    failed = 0
    batch_size = max(1, int(batch_size))
    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        x = np.concatenate([job["inp"] for job in batch_jobs], axis=0).astype(np.float32)
        pred = model.predict(x, verbose=0)

        for i, job in enumerate(batch_jobs):
            frame_state = job["frame_state"]
            try:
                item_pred = _slice_model_prediction(pred, i)
                coords_pred, hand_flag_score, handedness_score = decode_model_outputs(item_pred)
                coords_pred = np.squeeze(coords_pred)
                pts_crop = decode_landmarks(coords_pred)
                pts_abs = project_points_from_rect(
                    pts_crop,
                    job["rect"],
                    job["image_width"],
                    job["image_height"],
                )
                draw_result(
                    frame_state["out"],
                    job["bbox"],
                    job["roi_corners_px"],
                    pts_abs,
                    hand_idx=job["hand_idx"],
                    palm_det_score=job["palm_det_score"],
                    hand_flag_score=hand_flag_score,
                    handedness_score=handedness_score,
                    handedness_thr=handedness_thr,
                )
                frame_state["roi_count"] += 1
                if hand_flag_score is None or float(hand_flag_score) >= float(hand_flag_thr):
                    frame_state["pred_hand_count"] += 1
                    if handedness_score is not None:
                        if float(handedness_score) >= float(handedness_thr):
                            frame_state["pred_right"] += 1
                        else:
                            frame_state["pred_left"] += 1
            except Exception as ex:
                print(f"Failed on {frame_state['img_path'].name}: {ex}")
                failed += 1
            finally:
                if progress is not None:
                    progress.update(1)
    return failed


def list_images(frames_dir):
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def prepare_out_dir(out_dir: Path, clean_out_dir: bool):
    out_dir = out_dir.resolve()
    if str(out_dir) == out_dir.anchor:
        raise RuntimeError(f"Refuse to clean drive root directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not clean_out_dir:
        return
    removed = 0
    for child in out_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    print(f"[out_dir] cleaned {removed} existing item(s): {out_dir}")


def parse_points_file(points_file):
    text = points_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    records = {}
    for raw in lines:
        line = raw.strip().lstrip("\ufeff")
        if not line:
            continue

        low = line.lower()
        if not low.startswith("frame:"):
            continue

        tail = line.split(":", 1)[1].strip()
        parts = tail.split()
        # Supported formats:
        # frame: <image_name> [8 floats] [8 floats] ...  (bbox4 + wrist/middle4)
        # frame: <image_name> [9 floats] [9 floats] ...  (bbox4 + score1 + wrist/middle4)
        if len(parts) < 1:
            continue

        frame_name = parts[0]
        try:
            nums = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if len(nums) == 0:
            continue

        group_size = None
        if len(nums) % ROI_NUM_VALUES_WITH_SCORE == 0:
            group_size = ROI_NUM_VALUES_WITH_SCORE
        elif len(nums) % ROI_NUM_VALUES == 0:
            group_size = ROI_NUM_VALUES
        else:
            continue

        group_count = len(nums) // group_size
        hands = []
        for g in range(group_count):
            base = g * group_size
            chunk = nums[base : base + group_size]
            bbox = tuple(clamp01(v) for v in chunk[:4])
            det_score = float(chunk[4]) if group_size == ROI_NUM_VALUES_WITH_SCORE else None
            off = 5 if group_size == ROI_NUM_VALUES_WITH_SCORE else 4
            pts = {
                0: (clamp01(chunk[off]), clamp01(chunk[off + 1])),
                9: (clamp01(chunk[off + 2]), clamp01(chunk[off + 3])),
            }
            hands.append({"bbox": bbox, "points": pts, "score": det_score})

        if hands:
            records[frame_name] = hands

    return records


def resolve_record(image_name, records):
    if image_name in records:
        return records[image_name]

    image_stem = Path(image_name).stem
    for k, v in records.items():
        if Path(k).stem == image_stem:
            return v
    return None

def _read_keras3_h5_layers(weights_path):
    if h5py is None:
        return None
    with h5py.File(weights_path, "r") as f:
        if "layers" not in f:
            return None
        layers_group = f["layers"]
        source = {}
        for lname in layers_group.keys():
            g = layers_group[lname]
            if "vars" not in g:
                continue
            v = g["vars"]
            keys = sorted(v.keys(), key=lambda s: int(s))
            arrays = [np.array(v[k]) for k in keys]
            if arrays:
                source[lname] = arrays
        return source


def _load_weights_from_keras3_h5_manual(model, weights_path):
    source = _read_keras3_h5_layers(weights_path)
    if not source:
        return False

    used_source = set()
    loaded_target = set()

    # Pass 1: exact layer-name + shape mapping.
    for layer in model.layers:
        if not layer.weights:
            continue
        expected_shapes = [tuple(w.shape) for w in layer.weights]
        arrays = source.get(layer.name)
        if arrays is None:
            continue
        if [tuple(a.shape) for a in arrays] != expected_shapes:
            continue
        layer.set_weights(arrays)
        used_source.add(layer.name)
        loaded_target.add(layer.name)

    # Pass 2: unique shape signature mapping.
    for layer in model.layers:
        if not layer.weights:
            continue
        if layer.name in loaded_target:
            continue
        expected_shapes = [tuple(w.shape) for w in layer.weights]
        candidates = []
        for src_name, arrays in source.items():
            if src_name in used_source:
                continue
            if [tuple(a.shape) for a in arrays] == expected_shapes:
                candidates.append((src_name, arrays))
        if len(candidates) == 1:
            src_name, arrays = candidates[0]
            layer.set_weights(arrays)
            used_source.add(src_name)
            loaded_target.add(layer.name)

    # Success when every layer with weights got at least one assignment,
    # or only non-critical heads are missing.
    loaded_count = 0
    target_count = 0
    missing_layers = []
    for layer in model.layers:
        if not layer.weights:
            continue
        target_count += 1
        if layer.name in loaded_target:
            loaded_count += 1
        else:
            missing_layers.append(layer.name)

    if loaded_count == target_count:
        return True

    return False


def load_weights_compat(model, model_path):
    model_path = Path(model_path)
    try:
        model.load_weights(str(model_path))
        return
    except ValueError as e:
        err_msg = str(e).lower()
        retry = (
            "could not be loaded" in err_msg
            or "expected 2 variables, but received 0 variables" in err_msg
            or "layer count mismatch" in err_msg
            or "found 0 saved layers" in err_msg
        )
        if not retry:
            raise

    try:
        if _load_weights_from_keras3_h5_manual(model, str(model_path)):
            print(f"Loaded weights via manual Keras3 H5 reader: {model_path}")
            return
    except Exception:
        pass

    tmpdir = Path(".legacy_h5_tmp")
    tmpdir.mkdir(parents=True, exist_ok=True)

    candidates = []
    path_l = str(model_path).lower()
    if path_l.endswith(".weights.h5"):
        candidates.append(Path(tmpdir) / (model_path.stem.replace(".weights", "") + ".h5"))
    elif path_l.endswith(".h5"):
        candidates.append(Path(tmpdir) / (model_path.stem + ".weights.h5"))
    else:
        candidates.append(Path(tmpdir) / (model_path.name + ".h5"))
        candidates.append(Path(tmpdir) / (model_path.name + ".weights.h5"))

    last_err = None
    for alias in candidates:
        try:
            shutil.copyfile(model_path, alias)
            model.load_weights(str(alias))
            print(f"Loaded weights via alias: {alias.name}")
            return
        except Exception as ex:
            try:
                if _load_weights_from_keras3_h5_manual(model, str(alias)):
                    print(f"Loaded weights via manual alias reader: {alias.name}")
                    return
            except Exception:
                pass
            last_err = ex
    if last_err is not None:
        raise last_err


def main():
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    points_file = Path(args.points_file)
    weights_path = Path(args.weights)
    out_dir = Path(args.out_dir)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not points_file.exists():
        raise FileNotFoundError(f"Points file not found: {points_file}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    images = list_images(frames_dir)
    if not images:
        print(f"No images found in {frames_dir}")
        return

    records = parse_points_file(points_file)
    prepare_out_dir(out_dir, clean_out_dir=bool(args.clean_out_dir))

    model = hand_landmark_2d_model(input_size=(1, INPUT_H, INPUT_W))
    load_weights_compat(model, weights_path)

    total = len(images)
    ok_count = 0
    miss_meta = 0
    fail_count = 0
    saved_no_hand = 0
    frames_pred_with_hands = 0
    pred_left_total = 0
    pred_right_total = 0
    batch_size = max(1, int(args.batch_size))
    frame_chunk_size = max(1, int(args.frame_chunk_size))

    try:
        frame_pbar = tqdm(total=total, desc="infer_frames", unit="frame", dynamic_ncols=True)
        try:
            for chunk_start in range(0, len(images), frame_chunk_size):
                chunk = images[chunk_start : chunk_start + frame_chunk_size]
                frame_states = []
                roi_jobs = []

                for img_path in chunk:
                    img = imread_unicode(img_path)
                    if img is None:
                        fail_count += 1
                        frame_pbar.update(1)
                        continue

                    out = ensure_bgr(img).copy()
                    h, w = out.shape[:2]
                    frame_state = {
                        "img_path": img_path,
                        "out": out,
                        "pred_hand_count": 0,
                        "pred_left": 0,
                        "pred_right": 0,
                        "roi_count": 0,
                    }
                    frame_states.append(frame_state)

                    rec_list = resolve_record(img_path.name, records)
                    if rec_list is None:
                        miss_meta += 1
                        continue

                    for hand_idx, rec in enumerate(rec_list):
                        bbox = rec["bbox"]
                        hand_points = rec["points"]
                        palm_det_score = rec.get("score")
                        try:
                            rect = rect_from_bbox_and_hand_points(
                                bbox,
                                hand_points,
                                w,
                                h,
                                scale_x=args.roi_scale_x,
                                scale_y=args.roi_scale_y,
                                shift_y=args.roi_shift_y,
                            )
                            crop, roi_corners_px = crop_image_by_rect(img, rect, INPUT_W, INPUT_H)
                            roi_jobs.append(
                                {
                                    "frame_state": frame_state,
                                    "inp": preprocess_crop(crop),
                                    "bbox": bbox,
                                    "rect": rect,
                                    "roi_corners_px": roi_corners_px,
                                    "hand_idx": hand_idx,
                                    "palm_det_score": palm_det_score,
                                    "image_width": w,
                                    "image_height": h,
                                }
                            )
                        except Exception as ex:
                            print(f"Failed on {img_path.name}: {ex}")
                            fail_count += 1

                if roi_jobs:
                    with tqdm(
                        total=len(roi_jobs),
                        desc="roi_infer",
                        unit="roi",
                        leave=False,
                        dynamic_ncols=True,
                    ) as roi_pbar:
                        fail_count += predict_and_draw_roi_jobs(
                            model,
                            roi_jobs,
                            batch_size=batch_size,
                            hand_flag_thr=args.hand_flag_thr,
                            handedness_thr=args.handedness_thr,
                            progress=roi_pbar,
                        )

                for frame_state in frame_states:
                    out = frame_state["out"]
                    draw_frame_summary(
                        out,
                        pred_hand_count=frame_state["pred_hand_count"],
                        pred_left_count=frame_state["pred_left"],
                        pred_right_count=frame_state["pred_right"],
                        roi_count=frame_state["roi_count"],
                    )
                    if frame_state["pred_hand_count"] > 0:
                        frames_pred_with_hands += 1
                    pred_left_total += frame_state["pred_left"]
                    pred_right_total += frame_state["pred_right"]

                    out_path = out_dir / frame_state["img_path"].name
                    if imwrite_unicode(out_path, out):
                        ok_count += 1
                        if frame_state["pred_hand_count"] <= 0:
                            saved_no_hand += 1
                    else:
                        fail_count += 1
                    frame_pbar.update(1)
                    frame_pbar.set_postfix(saved=ok_count, hand_frames=frames_pred_with_hands)
        finally:
            frame_pbar.close()
    finally:
        pass
    print(
        f"Done. total={total}, saved={ok_count}, saved_no_hand={saved_no_hand}, "
        f"pred_hand_frames={frames_pred_with_hands}, pred_left_total={pred_left_total}, "
        f"pred_right_total={pred_right_total}, missing_meta={miss_meta}, failed={fail_count}"
    )
    print(f"Frames dir: {frames_dir}")
    print(f"Points file: {points_file}")
    print("Model type: 2d")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
