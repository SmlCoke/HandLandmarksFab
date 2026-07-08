import numpy as np

INPUT_SIZE = 224
FEATURE_SIZE_14 = 14
FEATURE_SIZE_7 = 7
NUM_POINTS = 2

DEFAULT_ANCHORS_14 = [[0.10, 0.10], [0.18, 0.18]]
DEFAULT_ANCHORS_7 = [[0.25, 0.25], [0.40, 0.40]]
NUM_ANCHORS_14 = len(DEFAULT_ANCHORS_14)
NUM_ANCHORS_7 = len(DEFAULT_ANCHORS_7)

IOU_POS_THRESHOLD = 0.45
REG_OFFSET_CLIP = 3.0


def generate_anchors(feature_size, default_anchors):
    anchors = []
    step = 1.0 / feature_size
    for y in range(feature_size):
        for x in range(feature_size):
            cx = x * step + step / 2
            cy = y * step + step / 2
            for w, h in default_anchors:
                anchors.append([cx, cy, w, h])
    return np.asarray(anchors, dtype=np.float32)


ANCHORS_14 = generate_anchors(FEATURE_SIZE_14, DEFAULT_ANCHORS_14)
ANCHORS_7 = generate_anchors(FEATURE_SIZE_7, DEFAULT_ANCHORS_7)
ANCHORS_MULTI = np.concatenate([ANCHORS_14, ANCHORS_7], axis=0)


def iou(box1, box2):
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2

    x1min = cx1 - w1 / 2.0
    y1min = cy1 - h1 / 2.0
    x1max = cx1 + w1 / 2.0
    y1max = cy1 + h1 / 2.0

    x2min = cx2 - w2 / 2.0
    y2min = cy2 - h2 / 2.0
    x2max = cx2 + w2 / 2.0
    y2max = cy2 + h2 / 2.0

    inter_w = max(0.0, min(x1max, x2max) - max(x1min, x2min))
    inter_h = max(0.0, min(y1max, y2max) - max(y1min, y2min))
    inter = inter_w * inter_h
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / (union + 1e-6)


def iou_with_anchors(gt_box, anchors):
    """Vectorized IoU between one gt box and all anchors."""
    cx, cy, w, h = gt_box

    x1min = cx - w / 2.0
    y1min = cy - h / 2.0
    x1max = cx + w / 2.0
    y1max = cy + h / 2.0

    acx = anchors[:, 0]
    acy = anchors[:, 1]
    aw = anchors[:, 2]
    ah = anchors[:, 3]

    x2min = acx - aw / 2.0
    y2min = acy - ah / 2.0
    x2max = acx + aw / 2.0
    y2max = acy + ah / 2.0

    inter_w = np.maximum(0.0, np.minimum(x1max, x2max) - np.maximum(x1min, x2min))
    inter_h = np.maximum(0.0, np.minimum(y1max, y2max) - np.maximum(y1min, y2min))
    inter = inter_w * inter_h

    union = (w * h) + (aw * ah) - inter
    return inter / (union + 1e-6)


def _encode_label_by_level(gt_box, gt_kps, anchors, feature_size, num_anchors):
    num_reg = 4 + NUM_POINTS * 2
    cls_label = np.zeros((len(anchors),), dtype=np.float32)
    reg_label = np.zeros((len(anchors), num_reg), dtype=np.float32)

    gt_box = np.asarray(gt_box, dtype=np.float32)
    ious = iou_with_anchors(gt_box, anchors)

    pos_indices = np.where(ious > IOU_POS_THRESHOLD)[0]
    if pos_indices.size == 0:
        pos_indices = np.array([int(np.argmax(ious))], dtype=np.int64)

    cls_label[pos_indices] = 1.0

    cx, cy, w, h = gt_box
    pos_anchors = anchors[pos_indices]

    acx = pos_anchors[:, 0]
    acy = pos_anchors[:, 1]
    aw = pos_anchors[:, 2]
    ah = pos_anchors[:, 3]

    dx = (cx - acx) / aw
    dy = (cy - acy) / ah
    dw = np.log(w / aw + 1e-6)
    dh = np.log(h / ah + 1e-6)
    dx = np.clip(dx, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)
    dy = np.clip(dy, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)
    dw = np.clip(dw, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)
    dh = np.clip(dh, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)

    kps_arr = np.asarray(gt_kps, dtype=np.float32).reshape(NUM_POINTS, 2)
    kx = kps_arr[:, 0]
    ky = kps_arr[:, 1]

    kx_enc = (kx[None, :] - acx[:, None]) / aw[:, None]
    ky_enc = (ky[None, :] - acy[:, None]) / ah[:, None]
    kx_enc = np.clip(kx_enc, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)
    ky_enc = np.clip(ky_enc, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)

    kps_enc = np.empty((len(pos_indices), NUM_POINTS * 2), dtype=np.float32)
    kps_enc[:, 0::2] = kx_enc
    kps_enc[:, 1::2] = ky_enc

    reg_vals = np.concatenate(
        [dx[:, None], dy[:, None], dw[:, None], dh[:, None], kps_enc],
        axis=1,
    ).astype(np.float32)
    reg_label[pos_indices] = reg_vals

    reg_out = reg_label.reshape(
        feature_size, feature_size, num_anchors * num_reg
    ).transpose(2, 0, 1)

    cls_out = cls_label.reshape(
        feature_size, feature_size, num_anchors
    ).transpose(2, 0, 1)

    return reg_out, cls_out


def encode_label(gt_box, gt_kps):
    reg_14, cls_14 = _encode_label_by_level(
        gt_box, gt_kps, ANCHORS_14, FEATURE_SIZE_14, NUM_ANCHORS_14
    )
    reg_7, cls_7 = _encode_label_by_level(
        gt_box, gt_kps, ANCHORS_7, FEATURE_SIZE_7, NUM_ANCHORS_7
    )
    return reg_14, cls_14, reg_7, cls_7
