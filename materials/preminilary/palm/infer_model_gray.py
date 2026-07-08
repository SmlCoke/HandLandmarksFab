import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import mixed_precision

from model import palm_detection_model
from anchor_utils import ANCHORS_14, ANCHORS_7, NUM_POINTS

MODEL_PATH = r"D:\IC_Innovation_Challenge\train_with_annotations\palm\palm_mono_gray_best.weights.h5"

INPUT_DIR = Path(r"D:\IC_Innovation_Challenge\train_with_annotations\infer\frames_gray")
POINTS_OUTPUT_FILE = Path(r"D:\IC_Innovation_Challenge\train_with_annotations\infer\frames_anchor_points_model_gray.txt")
BOXED_OUTPUT_DIR = Path(r"D:\IC_Innovation_Challenge\train_with_annotations\infer\frames_palm_model_gray")

INPUT_SIZE = 224
SCORE_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.30
FORCE_KEEP_ONE_BOX_WHEN_ALL_BELOW_THRESHOLD = False
DEBUG_PRINT = False
PRINT_DETECTIONS = False
MAX_DETECTIONS = 2
PREFER_HEAD14_FOR_KEYPOINT = True
CROSS_HEAD_SUPPRESS_IOU = 0.35
APPLY_BOX_FILTER = False
MIN_BOX_AREA = 0.01
MAX_BOX_AREA = 0.40
MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 2.20
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
INFER_BATCH_SIZE = 128
PREPROCESS_WORKERS = 12


def read_image_with_pil_orientation(path: Path):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return None

    try:
        with Image.open(path) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img)
            img = np.array(pil_img)
    except OSError:
        return None

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def read_image_unicode(path: Path):
    img = read_image_with_pil_orientation(path)
    if img is not None:
        return img

    arr = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return img

    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    return read_image_with_pil_orientation(path)


def write_image_unicode(path: Path, image):
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def preprocess(img):
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE))
    if np.issubdtype(gray.dtype, np.integer):
        max_value = float(np.max(gray)) if gray.size else 0.0
        scale = 255.0 if max_value <= 255.0 else float(np.iinfo(gray.dtype).max)
        gray = gray.astype(np.float32) / scale
    else:
        gray = gray.astype(np.float32)
        max_value = float(np.nanmax(gray)) if gray.size else 0.0
        if max_value > 1.0:
            gray = gray / max_value
    gray = np.clip(gray, 0.0, 1.0)
    gray = gray[np.newaxis, np.newaxis, :, :]
    return gray


def load_image_and_preprocess(path: Path):
    img = read_image_unicode(path)
    if img is None:
        return None, None
    return img, preprocess(img)


def draw_boxes_on_image(img, boxes, keypoints=None, scores=None):
    vis = img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 1:
        vis = cv2.cvtColor(vis[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif vis.ndim == 3 and vis.shape[2] == 4:
        vis = cv2.cvtColor(vis, cv2.COLOR_BGRA2BGR)

    h, w = vis.shape[:2]
    for det_idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        px1 = int(np.clip(round(float(x1) * (w - 1)), 0, w - 1))
        py1 = int(np.clip(round(float(y1) * (h - 1)), 0, h - 1))
        px2 = int(np.clip(round(float(x2) * (w - 1)), 0, w - 1))
        py2 = int(np.clip(round(float(y2) * (h - 1)), 0, h - 1))
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 2)

        if scores is not None and det_idx < len(scores):
            score_text = f"{float(scores[det_idx]):.2f}"
            text_x = px1
            text_y = py1 - 8 if py1 - 8 > 12 else py1 + 18
            (tw, th), baseline = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            box_tl = (text_x, text_y - th - baseline - 2)
            box_br = (text_x + tw + 4, text_y + baseline)
            cv2.rectangle(vis, box_tl, box_br, (0, 255, 0), thickness=-1)
            cv2.putText(
                vis,
                score_text,
                (text_x + 2, text_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        if keypoints is not None and det_idx < len(keypoints):
            for kp in keypoints[det_idx]:
                kx, ky = kp
                pkx = int(np.clip(round(float(kx) * (w - 1)), 0, w - 1))
                pky = int(np.clip(round(float(ky) * (h - 1)), 0, h - 1))
                cv2.circle(vis, (pkx, pky), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(vis, (pkx, pky), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return vis


def decode_head(reg_pred, cls_pred, anchors, head_name=""):
    cls = cls_pred[0].transpose(1, 2, 0).reshape(-1)
    vals_per_anchor = 4 + NUM_POINTS * 2
    reg = reg_pred[0].transpose(1, 2, 0).reshape(-1, vals_per_anchor)

    if DEBUG_PRINT:
        top_show = min(10, cls.shape[0])
        top_idx = np.argsort(cls)[::-1][:top_show]
        top_scores = [float(cls[i]) for i in top_idx]
        print(
            f"[{head_name}] anchors={len(anchors)} "
            f"score(min/mean/max)=({float(np.min(cls)):.6f}/"
            f"{float(np.mean(cls)):.6f}/{float(np.max(cls)):.6f}) "
            f"top{top_show}={top_scores}"
        )

    mask = cls >= SCORE_THRESHOLD
    if not np.any(mask):
        return np.array([]), np.array([]), [], []

    indices = np.where(mask)[0]
    boxes, scores, keypoints, heads = [], [], [], []

    for idx in indices:
        score = cls[idx]
        anc_cx, anc_cy, anc_w, anc_h = anchors[idx]
        dx, dy, dw, dh = reg[idx, :4]

        cx = anc_cx + dx * anc_w
        cy = anc_cy + dy * anc_h
        w_box = anc_w * np.exp(dw)
        h_box = anc_h * np.exp(dh)

        x1 = np.clip(cx - w_box / 2, 0.0, 1.0)
        y1 = np.clip(cy - h_box / 2, 0.0, 1.0)
        x2 = np.clip(cx + w_box / 2, 0.0, 1.0)
        y2 = np.clip(cy + h_box / 2, 0.0, 1.0)

        kps = []
        for i in range(4, 4 + NUM_POINTS * 2, 2):
            kx = np.clip(anc_cx + reg[idx, i] * anc_w, 0.0, 1.0)
            ky = np.clip(anc_cy + reg[idx, i + 1] * anc_h, 0.0, 1.0)
            kps.append((kx, ky))

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        keypoints.append(kps)
        heads.append(head_name)

    return np.array(boxes), np.array(scores), keypoints, heads


def decode_head_top1(reg_pred, cls_pred, anchors, head_name=""):
    cls = cls_pred[0].transpose(1, 2, 0).reshape(-1)
    vals_per_anchor = 4 + NUM_POINTS * 2
    reg = reg_pred[0].transpose(1, 2, 0).reshape(-1, vals_per_anchor)
    if cls.size == 0:
        return None

    idx = int(np.argmax(cls))
    score = float(cls[idx])
    anc_cx, anc_cy, anc_w, anc_h = anchors[idx]
    dx, dy, dw, dh = reg[idx, :4]

    cx = anc_cx + dx * anc_w
    cy = anc_cy + dy * anc_h
    w_box = anc_w * np.exp(dw)
    h_box = anc_h * np.exp(dh)

    x1 = float(np.clip(cx - w_box / 2, 0.0, 1.0))
    y1 = float(np.clip(cy - h_box / 2, 0.0, 1.0))
    x2 = float(np.clip(cx + w_box / 2, 0.0, 1.0))
    y2 = float(np.clip(cy + h_box / 2, 0.0, 1.0))

    kps = []
    for i in range(4, 4 + NUM_POINTS * 2, 2):
        kx = float(np.clip(anc_cx + reg[idx, i] * anc_w, 0.0, 1.0))
        ky = float(np.clip(anc_cy + reg[idx, i + 1] * anc_h, 0.0, 1.0))
        kps.append((kx, ky))

    return [x1, y1, x2, y2], score, kps, head_name


def iou_single_to_many(box, boxes):
    if len(boxes) == 0:
        return np.array([], dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


def filter_boxes(boxes, scores, keypoints, heads=None):
    if len(boxes) == 0:
        if heads is None:
            return boxes, scores, keypoints
        return boxes, scores, keypoints, heads

    keep = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        ratio = bw / (bh + 1e-6)

        if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
            continue
        if ratio < MIN_ASPECT_RATIO or ratio > MAX_ASPECT_RATIO:
            continue
        keep.append(i)

    if len(keep) == 0:
        if heads is None:
            return np.array([]), np.array([]), []
        return np.array([]), np.array([]), [], []

    boxes = boxes[keep]
    scores = scores[keep]
    keypoints = [keypoints[i] for i in keep]
    if heads is None:
        return boxes, scores, keypoints
    heads = [heads[i] for i in keep]
    return boxes, scores, keypoints, heads


def infer_one_image_from_preds(reg14_pred, cls14_pred, reg7_pred, cls7_pred):
    boxes14, scores14, kps14, heads14 = decode_head(reg14_pred, cls14_pred, ANCHORS_14, head_name="head14")
    boxes7, scores7, kps7, heads7 = decode_head(reg7_pred, cls7_pred, ANCHORS_7, head_name="head7")

    boxes = []
    scores = []
    keypoints = []
    heads = []

    if len(boxes14) > 0:
        boxes.extend(boxes14.tolist())
        scores.extend(scores14.tolist())
        keypoints.extend(kps14)
        heads.extend(heads14)

    if len(boxes7) > 0:
        boxes.extend(boxes7.tolist())
        scores.extend(scores7.tolist())
        keypoints.extend(kps7)
        heads.extend(heads7)

    forced_fallback = None
    if len(boxes) == 0 and FORCE_KEEP_ONE_BOX_WHEN_ALL_BELOW_THRESHOLD:
        cand14 = decode_head_top1(reg14_pred, cls14_pred, ANCHORS_14, head_name="head14")
        cand7 = decode_head_top1(reg7_pred, cls7_pred, ANCHORS_7, head_name="head7")
        cands = [c for c in [cand14, cand7] if c is not None]
        if cands:
            forced_fallback = max(cands, key=lambda c: float(c[1]))
            box_f, score_f, kps_f, head_f = forced_fallback
            boxes = [box_f]
            scores = [score_f]
            keypoints = [kps_f]
            heads = [head_f]

    if len(boxes) > 0:
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        heads = np.array(heads)

        if PREFER_HEAD14_FOR_KEYPOINT:
            selected_idx = []

            idx14 = np.where(heads == "head14")[0]
            if len(idx14) > 0:
                keep14_local = nms(boxes[idx14], scores[idx14], NMS_IOU_THRESHOLD)
                keep14 = idx14[keep14_local].tolist()
                selected_idx.extend(keep14)

            idx7 = np.where(heads == "head7")[0]
            if len(idx7) > 0:
                order7 = idx7[np.argsort(scores[idx7])[::-1]]
                selected_boxes = boxes[selected_idx] if len(selected_idx) > 0 else np.zeros((0, 4), dtype=np.float32)
                for i7 in order7:
                    if len(selected_boxes) > 0:
                        ious = iou_single_to_many(boxes[i7], selected_boxes)
                        if np.any(ious > CROSS_HEAD_SUPPRESS_IOU):
                            continue
                    selected_idx.append(int(i7))
                    selected_boxes = boxes[selected_idx]
                    if MAX_DETECTIONS > 0 and len(selected_idx) >= MAX_DETECTIONS:
                        break

            if len(selected_idx) > 0:
                selected_idx = sorted(selected_idx, key=lambda i: float(scores[i]), reverse=True)
                if MAX_DETECTIONS > 0:
                    selected_idx = selected_idx[:MAX_DETECTIONS]
                keep = selected_idx
            else:
                keep = nms(boxes, scores, NMS_IOU_THRESHOLD)
                if MAX_DETECTIONS > 0 and len(keep) > MAX_DETECTIONS:
                    keep = keep[:MAX_DETECTIONS]
        else:
            keep = nms(boxes, scores, NMS_IOU_THRESHOLD)
            if MAX_DETECTIONS > 0 and len(keep) > MAX_DETECTIONS:
                keep = keep[:MAX_DETECTIONS]

        boxes = boxes[keep]
        scores = scores[keep]
        heads = heads[keep].tolist()
        keypoints = [keypoints[i] for i in keep]
        if APPLY_BOX_FILTER:
            boxes, scores, keypoints, heads = filter_boxes(boxes, scores, keypoints, heads=heads)

        if len(boxes) == 0 and forced_fallback is not None:
            box_f, score_f, kps_f, _ = forced_fallback
            boxes = np.array([box_f], dtype=np.float32)
            scores = np.array([score_f], dtype=np.float32)
            keypoints = [kps_f]
    else:
        boxes = np.array([], dtype=np.float32)
        scores = np.array([], dtype=np.float32)
        keypoints = []

    return len(boxes), boxes, scores, keypoints


def infer_batch_grays(model, gray_tensors):
    if not gray_tensors:
        return []

    gray_batch = np.concatenate(gray_tensors, axis=0)
    reg14_pred, cls14_pred, reg7_pred, cls7_pred = model.predict(gray_batch, verbose=0)

    results = []
    for i in range(gray_batch.shape[0]):
        results.append(
            infer_one_image_from_preds(
                reg14_pred[i:i + 1],
                cls14_pred[i:i + 1],
                reg7_pred[i:i + 1],
                cls7_pred[i:i + 1],
            )
        )
    return results


def fmt_float(v):
    return f"{float(v):.6f}"


def format_points_flat(kps):
    items = []
    for x, y in kps:
        items.append(fmt_float(x))
        items.append(fmt_float(y))
    return " ".join(items)


def list_images(input_dir: Path):
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def best_effort_remove_tree(target_dir: Path):
    locked_paths = []
    if not target_dir.exists():
        return locked_paths

    for p in sorted(target_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            else:
                p.rmdir()
        except PermissionError:
            locked_paths.append(str(p))
        except OSError:
            pass

    try:
        target_dir.rmdir()
    except OSError:
        pass
    return locked_paths


def reset_output_dir(output_dir: Path):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    old_dir = output_dir.with_name(f"{output_dir.name}__old__{int(time.time() * 1000)}")
    renamed = False
    try:
        output_dir.rename(old_dir)
        renamed = True
    except OSError:
        renamed = False

    if renamed:
        output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                shutil.rmtree(old_dir)
                return
            except PermissionError:
                time.sleep(0.2)

        locked = best_effort_remove_tree(old_dir)
        if locked:
            preview = ", ".join(locked[:3])
            print(f"[WARN] Some old output files are locked and kept temporarily: {preview}")
        return

    for _ in range(3):
        try:
            shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return
        except PermissionError:
            time.sleep(0.2)

    locked = best_effort_remove_tree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if locked:
        preview = ", ".join(locked[:3])
        print(f"[WARN] Could not remove locked files in output dir: {preview}")


def reset_points_output_file(output_file: Path):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not output_file.exists():
        output_file.touch()
        return

    backup = output_file.with_name(f"{output_file.name}.old.{int(time.time() * 1000)}")
    renamed = False
    try:
        output_file.rename(backup)
        renamed = True
    except OSError:
        renamed = False

    output_file.touch()

    if renamed:
        for _ in range(3):
            try:
                backup.unlink()
                return
            except PermissionError:
                time.sleep(0.2)
            except OSError:
                return
        print(f"[WARN] Old points file is locked and kept temporarily: {backup}")
        return

    for _ in range(3):
        try:
            output_file.write_text("", encoding="utf-8")
            return
        except PermissionError:
            time.sleep(0.2)


def resolve_model_path(path_str: str):
    p = Path(path_str)
    if p.exists():
        return p
    fallback = Path("palm_mono_best.weights.h5")
    if fallback.exists():
        print(f"MODEL_PATH not found, fallback to: {fallback}")
        return fallback
    raise FileNotFoundError(f"Model file not found: {path_str}")


def load_weights_compat(model, model_path: Path):
    try:
        model.load_weights(str(model_path))
        print(f"Loaded weights: {model_path}")
        return
    except ValueError as e:
        err_msg = str(e)
        should_retry_legacy = str(model_path).lower().endswith(".weights.h5") and (
            "could not be loaded" in err_msg.lower()
            or "expected 2 variables, but received 0 variables" in err_msg.lower()
        )
        if not should_retry_legacy:
            raise

    with tempfile.TemporaryDirectory(prefix="legacy_h5_") as tmpdir:
        legacy_alias = Path(tmpdir) / (model_path.stem.replace(".weights", "") + ".h5")
        shutil.copyfile(model_path, legacy_alias)
        model.load_weights(str(legacy_alias))
    print(f"Loaded legacy H5 weights via alias: {model_path}")


def configure_runtime_fp32():
    mixed_precision.set_global_policy("float32")
    tf.keras.backend.set_floatx("float32")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"TF={tf.__version__}, GPUs={len(gpus)}, policy={mixed_precision.global_policy().name}")


def main():
    configure_runtime_fp32()

    model_path = resolve_model_path(MODEL_PATH)
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in: {INPUT_DIR}")
        return

    reset_output_dir(BOXED_OUTPUT_DIR)
    reset_points_output_file(POINTS_OUTPUT_FILE)

    model = palm_detection_model()
    load_weights_compat(model, model_path)

    total = len(image_paths)
    ok_count = 0
    fail_count = 0
    det_images = 0
    total_detections = 0

    with POINTS_OUTPUT_FILE.open("a", encoding="utf-8") as fp:
        pbar = tqdm(total=total, desc="Model infer", ncols=100)
        with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
            for start in range(0, total, INFER_BATCH_SIZE):
                batch_paths = image_paths[start:start + INFER_BATCH_SIZE]
                prep_results = list(pool.map(load_image_and_preprocess, batch_paths))

                valid_paths = []
                valid_imgs = []
                valid_grays = []

                for in_path, (img, gray) in zip(batch_paths, prep_results):
                    if gray is None:
                        fail_count += 1
                        fp.write(f"frame: {in_path.name}\n")
                        continue
                    valid_paths.append(in_path)
                    valid_imgs.append(img)
                    valid_grays.append(gray)

                if valid_grays:
                    try:
                        batch_results = infer_batch_grays(model, valid_grays)
                    except Exception:
                        for in_path in valid_paths:
                            fail_count += 1
                            fp.write(f"frame: {in_path.name}\n")
                        pbar.update(len(batch_paths))
                        continue
                else:
                    batch_results = []

                for in_path, img, (det_num, all_boxes, all_scores, all_kps) in zip(valid_paths, valid_imgs, batch_results):
                    if det_num > 0:
                        line_parts = [f"frame: {in_path.name}"]
                        for box, score, kps in zip(all_boxes, all_scores, all_kps):
                            x1, y1, x2, y2 = box
                            line_parts.append(
                                f"{fmt_float(x1)} {fmt_float(y1)} {fmt_float(x2)} {fmt_float(y2)} "
                                f"{fmt_float(score)} "
                                f"{format_points_flat(kps)}"
                            )
                        fp.write(" ".join(line_parts) + "\n")
                    else:
                        fp.write(f"frame: {in_path.name}\n")

                    boxed_img = draw_boxes_on_image(img, all_boxes, all_kps, all_scores)
                    out_img_path = BOXED_OUTPUT_DIR / in_path.name
                    write_ok = write_image_unicode(out_img_path, boxed_img)
                    if not write_ok:
                        print(f"[WARN] Failed to save boxed image: {out_img_path}")

                    ok_count += 1
                    total_detections += det_num
                    if det_num > 0:
                        det_images += 1

                pbar.update(len(batch_paths))
        pbar.close()

    print(f"Done. total={total}, processed={ok_count}, failed={fail_count}")
    print(f"images_with_detections={det_images}, total_detections={total_detections}")
    print(f"Points file: {POINTS_OUTPUT_FILE}")
    print(f"Boxed images dir: {BOXED_OUTPUT_DIR}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
