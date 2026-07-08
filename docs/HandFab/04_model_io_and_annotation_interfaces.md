# 模型定义脚本输入输出与标注接口

本文只关注两个模型定义脚本：

- `preminilary/palm/model.py`
- `hand/root/hand/model_2d_NHWC.py`

目标是说明它们的输入、输出，以及这些输入输出分别对应哪些标注文件和辅助脚本。本轮工程原则仍然是：训练和数据管线全部重写，只保留模型结构边界；Palm Detector 使用初赛冻结 ONNX 或 MediaPipe 官方模型生成 ROI，不重新训练。

## 0. 统一图像坐标系

本轮脚本读取到的输入图像已经是：

```text
image: 1280x720 upright grayscale TIFF
```

本文所有 bbox、关键点、ROI 坐标默认都属于这张 `1280x720` 正向图。训练工程不再包含原始图片旋转脚本。

## 1. Palm Detector: `preminilary/palm/model.py`

### 1.1 模型入口

函数：

```python
palm_detection_model(input_size=(1, 224, 224), num_iterations=(7, 7, 6, 5, 4))
```

模型输入：

```text
shape: (batch, 1, 224, 224)
layout: NCHW / channels_first
type: float32
value: 灰度图，通常归一化到 [0, 1]
```

输入图像来源：

```text
1280x720 正向灰度图
-> resize / pad / normalize 到 224x224
-> 增加 channel 维度
-> (1, 224, 224)
```

### 1.2 模型原始输出

模型输出顺序：

```python
[regressor_14, classificator_14, regressor_7, classificator_7]
```

输出形状：

| 输出名 | shape | 含义 |
|---|---:|---|
| `regressor_14` | `(batch, 16, 14, 14)` | 14x14 head 的 bbox + 2 个 Palm 关键点回归 |
| `classificator_14` | `(batch, 2, 14, 14)` | 14x14 head 的 2 anchors 分类分数 |
| `regressor_7` | `(batch, 16, 7, 7)` | 7x7 head 的 bbox + 2 个 Palm 关键点回归 |
| `classificator_7` | `(batch, 2, 7, 7)` | 7x7 head 的 2 anchors 分类分数 |

每个 anchor 的回归通道数是：

```text
4 bbox values + 2 keypoints * 2 xy values = 8
```

每个 head 有 `2` 个 anchors，所以回归输出通道数是：

```text
2 anchors * 8 values = 16
```

这些输出还不是最终 bbox。它们需要经过 anchor decode、score threshold、NMS，才得到最终 Palm ROI。

### 1.3 四个输出变量的逐维含义

Palm 模型有两个检测尺度：

- `14x14` head：更密的特征图，通常对较小或定位更细的手掌更敏感。
- `7x7` head：更粗的特征图，通常覆盖更大的候选区域。

每个网格位置有 `2` 个 anchors。anchor 是一个预设候选框，包含：

```text
anchor = [anchor_cx, anchor_cy, anchor_w, anchor_h]
```

这些值都使用归一化坐标。当前 anchor 设计来自 `preminilary/palm/anchor_utils.py`：

```text
14x14 head anchors per grid cell:
  anchor 0 size = 0.10 x 0.10
  anchor 1 size = 0.18 x 0.18

7x7 head anchors per grid cell:
  anchor 0 size = 0.25 x 0.25
  anchor 1 size = 0.40 x 0.40
```

#### `regressor_14`

```text
shape = (batch, 16, 14, 14)
```

维度含义：

| 维度 | 含义 |
|---:|---|
| `batch` | 一次推理的图片数量 |
| `16` | 每个网格位置的回归通道数 |
| `14` | 特征图高度，共 14 行 |
| `14` | 特征图宽度，共 14 列 |

`16` 个通道的来源：

```text
2 anchors * (4 bbox values + 2 keypoints * 2 xy values)
= 2 * 8
= 16
```

对每个网格位置 `(gy, gx)`：

```text
channels 0..7   -> anchor 0 的回归值
channels 8..15  -> anchor 1 的回归值
```

每个 anchor 的 8 个回归值顺序是：

```text
[dx, dy, dw, dh, k0_dx, k0_dy, k1_dx, k1_dy]
```

含义：

| 值 | 含义 |
|---|---|
| `dx, dy` | bbox 中心相对 anchor 中心的偏移 |
| `dw, dh` | bbox 宽高相对 anchor 宽高的 log-scale |
| `k0_dx, k0_dy` | Palm 关键点 0 相对 anchor 中心的偏移 |
| `k1_dx, k1_dy` | Palm 关键点 1 相对 anchor 中心的偏移 |

decode 公式：

```text
cx = anchor_cx + dx * anchor_w
cy = anchor_cy + dy * anchor_h
w  = anchor_w * exp(dw)
h  = anchor_h * exp(dh)

xmin = cx - w / 2
ymin = cy - h / 2
xmax = cx + w / 2
ymax = cy + h / 2

k0_x = anchor_cx + k0_dx * anchor_w
k0_y = anchor_cy + k0_dy * anchor_h
k1_x = anchor_cx + k1_dx * anchor_w
k1_y = anchor_cy + k1_dy * anchor_h
```

#### `classificator_14`

```text
shape = (batch, 2, 14, 14)
```

维度含义：

| 维度 | 含义 |
|---:|---|
| `batch` | 一次推理的图片数量 |
| `2` | 每个网格位置的 2 个 anchors |
| `14` | 特征图高度，共 14 行 |
| `14` | 特征图宽度，共 14 列 |

对每个网格位置 `(gy, gx)`：

```text
channel 0 -> anchor 0 的 palm score
channel 1 -> anchor 1 的 palm score
```

模型里对分类输出做了 `sigmoid`，所以这里的值已经是 `[0, 1]` 范围内的置信度。

#### `regressor_7`

```text
shape = (batch, 16, 7, 7)
```

维度含义：

| 维度 | 含义 |
|---:|---|
| `batch` | 一次推理的图片数量 |
| `16` | 每个网格位置的回归通道数 |
| `7` | 特征图高度，共 7 行 |
| `7` | 特征图宽度，共 7 列 |

`regressor_7` 和 `regressor_14` 的通道组织完全相同：

```text
channels 0..7   -> anchor 0 的 [dx, dy, dw, dh, k0_dx, k0_dy, k1_dx, k1_dy]
channels 8..15  -> anchor 1 的 [dx, dy, dw, dh, k0_dx, k0_dy, k1_dx, k1_dy]
```

区别是 `7x7` head 的网格更粗、anchor 尺寸更大：

```text
anchor 0 size = 0.25 x 0.25
anchor 1 size = 0.40 x 0.40
```

decode 公式与 `regressor_14` 相同。

#### `classificator_7`

```text
shape = (batch, 2, 7, 7)
```

维度含义：

| 维度 | 含义 |
|---:|---|
| `batch` | 一次推理的图片数量 |
| `2` | 每个网格位置的 2 个 anchors |
| `7` | 特征图高度，共 7 行 |
| `7` | 特征图宽度，共 7 列 |

对每个网格位置 `(gy, gx)`：

```text
channel 0 -> anchor 0 的 palm score
channel 1 -> anchor 1 的 palm score
```

这些 score 与 `regressor_7` 中同一 `(gy, gx, anchor_id)` 的回归值一一对应。

### 1.4 flatten 后的候选顺序

推理 decode 时通常会把输出从 NCHW 转成 HWC，再 flatten：

```python
cls = cls_pred[0].transpose(1, 2, 0).reshape(-1)
reg = reg_pred[0].transpose(1, 2, 0).reshape(-1, 8)
```

flatten 后的候选顺序是：

```text
for gy in rows:
  for gx in cols:
    for anchor_id in [0, 1]:
      candidate
```

因此：

```text
14x14 head 候选数 = 14 * 14 * 2 = 392
7x7 head 候选数  = 7 * 7 * 2 = 98
总候选数          = 490
```

每个候选由两部分组成：

```text
score: classificator_* 中对应 anchor 的值
reg:   regressor_* 中对应 anchor 的 8 个回归值
```

decode 后得到：

```text
[xmin, ymin, xmax, ymax], score, [(k0_x, k0_y), (k1_x, k1_y)]
```

随后再做 score threshold、NMS、max detections，最终导出给 Hand 使用的 Palm ROI。

### 1.5 多 anchor 候选如何变成最多两只手

Palm Detector 的原始输出共有 490 个 anchor 候选：

```text
14x14 head: 14 * 14 * 2 = 392
7x7 head:   7 * 7 * 2 = 98
```

这些候选不是最终检测结果。后处理流程应是：

```text
raw anchors
-> score threshold
-> bbox/keypoint decode
-> head 内 NMS
-> 跨 head suppression
-> 按 score 排序
-> 最多保留 2 个 detections
```

因此，一张图最多两只手的约束来自后处理中的 `max_detections=2`，不是模型结构天然只输出两只手。如果画面中出现多个高分候选，最终保留的是经过 NMS 后分数最高的最多两个。

Palm Detector 本身不能判断左手/右手。它的分类分数只表示 palm confidence，没有 handedness 语义。左右手判断应由 Hand Landmarker 的 `handedness` 输出承担，或者由额外后处理启发式估计；但启发式在镜像、交叉手、手背/手心变化时并不可靠。

### 1.6 Palm 解码后的对接结果

给 Hand Landmarker 对接时，Palm 最终应导出：

```text
image_name xmin ymin xmax ymax p0_x p0_y p9_x p9_y [第二只手重复 8 个值]
```

含义：

- `image_name`：`1280x720` 正向 TIFF 文件名。
- `xmin ymin xmax ymax`：Palm bbox，归一化到 `1280x720` 正向图。
- `p0_x p0_y`：Palm/hand 关键点 0，通常对应 wrist。
- `p9_x p9_y`：Palm/hand 关键点 9，通常对应 middle MCP。

这个文件建议命名为：

```text
points_pred_palm.txt
```

Palm ROI 文件可以由两种方式生成：

- `preminilary/palm/model_opt.onnx` 推理生成。
- MediaPipe 官方模型推理生成，并在导出阶段规范化为同一字段。

注意：这里的 `xmin ymin xmax ymax` 是 palm bbox，不是最终 Hand Landmarker 的输入框。它通常只能覆盖手掌附近，不保证包含完整手指。

### 1.7 Palm 对应的辅助脚本

当前 AetherSign 路线下，Palm 不需要新的训练标注文件。

需要的辅助脚本：

| 脚本 | 作用 |
|---|---|
| `export_palm_rois.py` | 读取 `1280x720` 正向 TIFF，调用冻结 Palm ONNX 或 MediaPipe 官方模型，导出 `points_pred_palm.txt` |
| `palm_decode.py` 或同等模块 | 从 Palm raw outputs 解码 bbox、score、2 个关键点 |
| `palm_nms.py` 或同等模块 | 对 490 个 anchor 候选做 threshold、NMS、top-k |

可以抽取的旧函数来源：

- `preminilary/palm/infer_model_gray.py`：Palm 预处理、推理输出整理、NMS、可视化/导出中的核心函数。
- `preminilary/palm/anchor_utils.py`：anchor 常量和 decode 所需逻辑。
- `preminilary/palm/model_opt.onnx`：冻结 Palm 模型本体。

不需要继承 Palm 训练脚本、数据集构建脚本或旧实验文档。

## 2. Hand Landmarker: `hand/root/hand/model_2d_NHWC.py`

### 2.1 模型入口

函数：

```python
hand_landmark_2d_model(input_size=(1, 256, 256), num_iterations=8)
```

模型输入：

```text
shape: (batch, 1, 256, 256)
layout at input: NCHW
internal layout: NHWC
type: float32
value: Palm ROI 裁剪后的灰度 hand crop，通常归一化到 [0, 1]
```

脚本内部第一层会执行：

```python
Permute((2, 3, 1), name="input_nchw_to_nhwc")
```

也就是说，对外输入仍是 `(1, 256, 256)`，但网络主体按 NHWC 执行。

输入图像来源：

```text
1280x720 正向 TIFF
-> 使用 Palm ROI 构造 rotated hand rect
-> crop / warp 到 256x256
-> 灰度归一化
-> (1, 256, 256)
```

这里的 `256x256` 输入不是把 Palm bbox 直接缩放得到的。正确流程是先用 Palm bbox 和两个 Palm 关键点构造一个更大的 rotated Hand ROI：bbox 给出手掌位置和基础尺度，`p0/p9` 给出手腕到中指掌指关节方向，用于估计旋转角；随后对 ROI 做放大、平移和仿射采样。这样裁出的 crop 才应覆盖完整手掌和手指。

训练时，GT 的 21 个正向图关键点也必须用同一个 rotated ROI 几何投影到 `256x256` ROI 坐标系；推理时，模型输出的 ROI 坐标再用同一个几何关系投影回 `1280x720` 正向图。训练与推理的 ROI 参数必须一致。

### 2.2 模型输出

模型输出顺序：

```python
[landmarks, hand_flag, handedness]
```

输出形状：

| 输出名 | shape | 含义 |
|---|---:|---|
| `landmarks` | `(batch, 1, 1, 42)` | 21 个 2D hand landmarks |
| `hand_flag` | `(batch, 1, 1, 1)` | ROI 内是否存在有效手 |
| `handedness` | `(batch, 1, 1, 1)` | 左右手分类 |

`landmarks` 的 42 维顺序是：

```text
x0 y0 x1 y1 ... x20 y20
```

这些坐标属于 `256x256` Hand ROI 坐标系，通常按 `[0,1]` 表示。推理后需要再通过 ROI 几何投影回 `1280x720` 正向图坐标系。

21 点顺序：

| id | 含义 |
|---:|---|
| 0 | wrist |
| 1-4 | thumb |
| 5-8 | index |
| 9-12 | middle |
| 13-16 | ring |
| 17-20 | pinky |

`hand_flag`：

- `1`：ROI 中有有效手。
- `0`：ROI 中无有效手或应被忽略。

`handedness`：

- 建议统一 `0=Left`、`1=Right`。
- 如果某些样本左右手不确定，可在训练脚本中对 handedness loss 降权或跳过。

### 2.3 Hand 对应的标注文件

Hand 的主人工标注文件建议是：

```text
hand_landmarks.jsonl
```

建议字段：

```json
{
  "image": "seq001_f000123.tiff",
  "width": 1280,
  "height": 720,
  "orientation": "upright_1280x720",
  "hands": [
    {
      "hand_id": "h0",
      "handedness": "Right",
      "landmarks": [
        {"id": 0, "x": 519.9, "y": 335.2, "visible": 1}
      ]
    }
  ]
}
```

其中 `x/y` 是 `1280x720` 正向图上的像素坐标。

训练派生文件建议是：

```text
annotations_hand.txt
```

格式：

```text
annotation_id image_name hand_flag handedness x0 y0 x1 y1 ... x20 y20 [第二只手继续 44 个值]
```

说明：

- `x0 y0 ... x20 y20` 是 `1280x720` 正向图归一化坐标。
- 这个文件监督 Hand 模型的 `landmarks`、`hand_flag`、`handedness`。
- 一张图有两只手时，可以在同一行追加第二只手。

Hand 还需要 Palm 辅助文件：

```text
points_pred_palm.txt
```

这个文件由 Palm ROI 生成脚本生成，用于为每张图提供 ROI。它不是 Hand landmark 人工标注，但它决定 Hand 输入 crop 的位置和几何。

### 2.4 Hand 对应的辅助脚本

| 脚本 | 作用 |
|---|---|
| `convert_hand_annotations.py` | 把人工标注导出为 `annotations_hand.txt` |
| `visualize_hand_annotations.py` | 在 `1280x720` 正向图上画 21 点，检查标注质量 |
| `export_palm_rois.py` | 用冻结 Palm ONNX 或 MediaPipe 官方模型生成 `points_pred_palm.txt` |
| `build_hand_roi_cache.py` | 用 Palm ROI 裁剪 `256x256` hand crop，并把 21 点投影到 ROI 坐标系 |
| `train_hand_landmarker.py` | 加载 `model_2d_NHWC.py` 训练 Hand |
| `eval_hand_landmarker.py` | 在固定 ROI 来源上评估 Hand 的 landmark 精度 |
| `infer_palm_hand_pipeline.py` | `1280x720` 正向图 -> Palm -> Hand -> 正向图 21 点 |

## 3. 两个模型之间的数据关系

整体数据流：

```text
1280x720 正向 TIFF
-> Palm Detector 或 MediaPipe 官方模型
-> points_pred_palm.txt
-> 构造 rotated hand rect
-> crop 256x256 Hand ROI
-> Hand Landmarker
-> ROI 21 点
-> 投影回 1280x720 正向图
```

其中 `points_pred_palm.txt` 中的 Palm bbox 只负责启动 ROI 几何，不能跳过 rotated hand rect 这一步。

训练时的文件关系：

```text
hand_landmarks.jsonl
  -> convert_hand_annotations.py
  -> annotations_hand.txt

1280x720 正向 TIFF images
  -> export_palm_rois.py
  -> points_pred_palm.txt

annotations_hand.txt + points_pred_palm.txt + 1280x720 正向 TIFF images
  -> build_hand_roi_cache.py
  -> train/eval samples for model_2d_NHWC.py
```

## 4. 最小结论

- `preminilary/palm/model.py` 定义 Palm Detector 结构：输入 `(batch,1,224,224)`，输出两个尺度的 anchor 分类和 bbox/keypoint 回归。
- 当前不训练 Palm；使用 `preminilary/palm/model_opt.onnx` 或 MediaPipe 官方模型生成 `points_pred_palm.txt`。
- `hand/root/hand/model_2d_NHWC.py` 定义 Hand Landmarker 结构：输入 `(batch,1,256,256)`，输出 `42` 维 21 点、`hand_flag`、`handedness`。
- Hand 的人工标注核心是 `hand_landmarks.jsonl`，派生为 `annotations_hand.txt`。
- Hand 的 ROI 辅助输入是 `points_pred_palm.txt`，坐标统一属于 `1280x720` 正向图。
