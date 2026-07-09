# Hand Landmarker 训练接口文档

本文说明当前标注流水线与 `materials/preminilary/hand/model.py` 之间的训练接口。目标是给后续训练脚本使用，不沿用 `materials/preminilary/hand/` 下的旧推理或 ROI 脚本。

## 1. 训练需要哪些文件

### 必需文件

训练 Hand Landmarker 时只需要以下核心文件：

```text
materials/preminilary/hand/model.py
data/05_labels/hand_training_labels.jsonl
data/02_roi_crops/images/*.png
configs/autolabel.yaml
```

含义如下：

| 文件 | 用途 |
|---|---|
| `materials/preminilary/hand/model.py` | Hand Landmarker 网络结构定义。当前只使用这个文件。 |
| `data/05_labels/hand_training_labels.jsonl` | 最终训练标注，一行一个 ROI crop。 |
| `data/02_roi_crops/images/*.png` | 训练输入图片。每行标注中的 `crop_path` 指向其中一张 `256x256` crop。 |
| `configs/autolabel.yaml` | 路径、图像尺寸、ROI 尺寸等配置来源。 |

`materials/preminilary/hand/` 下其他文件，例如 `infer_frames_with_roi.py`、`roi_utils.py`，属于旧脚本或旧推理辅助逻辑；当前训练接口不依赖它们。

### 可选诊断文件

以下文件不直接参与训练，但可以用于检查数据质量：

```text
data/qc/final_training_label_stats.json
data/04_visualization/review_index.csv
data/04_visualization/crop_images/*.png
data/04_visualization/global_images/*.png
```

`data/03_reviewed/hand_landmarks_reviewed.jsonl` 是人工复核后的中间结果。训练脚本应读取 `data/05_labels/hand_training_labels.jsonl`，不要直接读取 reviewed 文件。

`data/02_roi_crops/hand_roi_crops_manifest.jsonl` 已经被 `07_finalize_training_labels.py` 合并进最终训练标注；训练时通常不需要再次读取。

## 2. `model.py` 的输入输出

模型入口：

```python
from materials.preminilary.hand.model import hand_landmark_2d_model

model = hand_landmark_2d_model(input_size=(1, 256, 256))
```

输入：

```text
shape: (batch, 1, 256, 256)
layout: NCHW
type: float32
value: 灰度 crop，建议归一化到 [0, 1]
```

`model.py` 内部会用 `Permute((2, 3, 1))` 把 NCHW 转成 NHWC 后执行卷积。

输出顺序：

```python
[landmarks, hand_flag, handedness]
```

这里需要特别强调：**`hand_flag` 的语义就是本仓库标注系统中的 `hand_presence`**，也就是“当前 `256x256` Hand ROI crop 内是否存在有效手”。它不是 Palm Detector 的 `palm_valid`，也不是 Palm confidence。

输出含义：

| 输出 | shape | 监督来源 |
|---|---:|---|
| `landmarks` | `(batch, 1, 1, 42)` | `landmarks_crop_norm` |
| `hand_flag` | `(batch, 1, 1, 1)` | `hand_presence.present`，即本系统的手存在性标签 |
| `handedness` | `(batch, 1, 1, 1)` | `handedness.label` |

`landmarks` 的 42 维顺序固定为：

```text
x0 y0 x1 y1 ... x20 y20
```

也就是 `landmarks_crop_norm` 中 21 个点按 `id=0..20` 排序后展开。

`hand_flag` 建议使用 BCE：

```text
target = 1.0 if hand_presence.present=true else 0.0
```

因此后续训练、评估、推理文档中如果出现 `hand_flag`、`hand presence`、`hand_presence.present`，三者应理解为同一个二分类语义：ROI 内有无有效手。

`handedness` 建议使用 BCE，并保持以下映射：

```text
Left  -> 0.0
Right -> 1.0
```

如果 `handedness.label=unknown`，该样本的 `handedness_loss_weight` 会是 `0.0`，训练脚本应跳过 handedness loss。

## 3. 最终训练标注字段

`data/05_labels/hand_training_labels.jsonl` 一行代表一个 ROI crop。关键字段如下。

| 字段 | 含义 |
|---|---|
| `crop_id` | crop 唯一 ID。 |
| `crop_path` | 训练输入 crop 图片路径。 |
| `image` | 对应原始 `1280x720` 图像文件名。 |
| `palm_det_id` | 上游 Palm candidate ID。 |
| `palm_valid` | 该 crop 是否来自有效 Palm detection。 |
| `palm_score` | 上游 Palm candidate 分数。 |
| `hand_presence.present` | 当前 crop 内是否有有效手；这是 Hand Landmarker 正负样本的主要判断字段，对应 `model.py` 输出中的 `hand_flag`。 |
| `handedness.label` | `Left`、`Right` 或 `unknown`。 |
| `landmarks_crop_norm` | 21 点在 `256x256` crop 内的归一化坐标，用于 landmark 训练。 |
| `landmarks_crop_px` | 21 点在 crop 内的像素坐标，用于可视化和 QC。 |
| `landmarks_image_px` | 21 点反投影回原图的像素坐标，用于可视化和 QC。 |
| `roi_rect` / `roi_corners_px` | crop 对应的原图 ROI 几何。 |
| `needs_review` | 该样本是否有 QC warning/error 或人工仍需复查。 |
| `hand_presence_loss_weight` | hand flag loss 权重。 |
| `landmark_loss_weight` | landmark loss 权重。 |
| `handedness_loss_weight` | handedness loss 权重。 |

三个 landmark 字段中的点结构统一为：

```json
{"id": 0, "x": 0.5357, "y": 0.4884}
```

当前不包含 `z` 或 `visible`。

## 4. 四个训练控制字段

### `needs_review`

`needs_review` 是质量标记，不是 loss 权重。

它表示该样本在最终检查中存在值得人工复查的情况，例如：

- crop 点越界。
- 一个 crop 中曾检测到多只手。
- handedness 分数偏低。
- 高分 Palm 结果被标为无手。
- 标注结构曾出现 warning/error。

`07_finalize_training_labels.py` 不会因为 `needs_review=true` 自动删除样本，也不会自动修改 loss weight。训练脚本可以自行决定：

- 过滤掉 `needs_review=true`。
- 只过滤严重异常。
- 保留但降低采样权重。
- 单独训练一版包含困难样本的模型做对比。

### `hand_presence_loss_weight`

当前总是：

```text
hand_presence_loss_weight = 1.0
```

正样本和负样本都应参与 `hand_flag` 训练。这个 head 的目标是判断当前 ROI crop 内是否有有效手，监督信号就是 `hand_presence.present`。

### `landmark_loss_weight`

规则：

```text
hand_presence.present=true  -> landmark_loss_weight = 1.0
hand_presence.present=false -> landmark_loss_weight = 0.0
```

无手样本没有 21 点，不应参与 landmark loss。

### `handedness_loss_weight`

规则：

```text
hand_presence.present=true 且 handedness.label in {Left, Right}
  -> handedness_loss_weight = 1.0

其他情况
  -> handedness_loss_weight = 0.0
```

无手样本、左右手未知样本都不参与 handedness loss。

## 5. 样本类型与训练方式

训练脚本可以用 `hand_presence.present`、`palm_valid`、`needs_review` 和三个 loss weight 判断样本类型。

### 标准正样本

```text
hand_presence.present=true
palm_valid=true
needs_review=false
```

训练方式：

```text
hand_flag target = 1
landmark loss weight = 1
handedness loss weight = 1, 如果 handedness.label 是 Left/Right
```

这是最干净的正样本，监督三个输出 head。

### 低分 Palm 候选中的正样本

```text
hand_presence.present=true
palm_valid=false
```

含义：Palm 分数低，但人工复核或 teacher 认为 crop 中确实有手。

训练方式与普通正样本相同：

```text
hand_flag target = 1
landmark loss weight = 1
handedness loss weight = 1, 如果 label 已知
```

这类样本对 Hand Landmarker 是正样本。它只说明上游 Palm 分数低，不影响 Hand 训练的正负判断。

### Palm hard negative

```text
hand_presence.present=false
palm_valid=true
```

含义：Palm 有效检测触发了 ROI，但人工复核认为 crop 内没有有效手。

训练方式：

```text
hand_flag target = 0
landmark loss weight = 0
handedness loss weight = 0
landmarks = []
```

这类样本对 hand flag 很有价值，可以帮助模型拒绝错误 ROI。

### 低分负样本

```text
hand_presence.present=false
palm_valid=false
```

含义：低分 Palm candidate，且 crop 内确实没有有效手。

训练方式：

```text
hand_flag target = 0
landmark loss weight = 0
handedness loss weight = 0
```

这类样本可以参与 hand flag 训练，但数量过多时建议在 dataloader 中采样或配比，避免负样本压倒正样本。

### handedness 未知的正样本

```text
hand_presence.present=true
handedness.label=unknown
```

训练方式：

```text
hand_flag target = 1
landmark loss weight = 1
handedness loss weight = 0
```

它仍然可以训练 hand flag 和 landmark，只是不训练 handedness。

### `needs_review=true` 的样本

```text
needs_review=true
```

含义：样本被 QC 标记为需要复查，但不一定不能训练。

训练策略建议：

1. 第一版正式训练可以先过滤 `needs_review=true`，得到更干净的 baseline。
2. 对 `needs_review=true` 做人工复查后再纳入训练。
3. 如果保留它们，建议记录单独实验配置，方便比较。

### 被 `07_finalize_training_labels.py` 跳过的严重异常

以下样本不会进入 `hand_training_labels.jsonl`：

```text
hand_presence.present=true  但 landmarks_crop_norm 不是 21 点
hand_presence.present=false 但仍然带 landmarks
```

它们会写入：

```text
data/qc/final_training_label_stats.json
```

训练脚本不需要处理这些样本，因为它们已经不在最终训练文件里。

## 6. 推荐 dataloader 逻辑

训练脚本读取每一行 JSONL 后建议执行：

1. 读取 `crop_path` 指向的 PNG。
2. 转灰度，resize/校验为 `256x256`。
3. 归一化到 `[0,1]`，形成 `(1,256,256)`。
4. `hand_flag_target = float(hand_presence.present)`。
5. 如果 `landmark_loss_weight > 0`，将 `landmarks_crop_norm` 按 id 排序并展开为 42 维。
6. 如果 `handedness_loss_weight > 0`，将 `Left/Right` 映射为 `0/1`。
7. 将三个 loss weight 一并传给 loss 计算。

伪代码：

```python
present = bool(row["hand_presence"]["present"])
hand_flag_target = 1.0 if present else 0.0

landmark_weight = float(row["landmark_loss_weight"])
if landmark_weight:
    points = sorted(row["landmarks_crop_norm"], key=lambda p: p["id"])
    landmark_target = [v for p in points for v in (p["x"], p["y"])]
else:
    landmark_target = [0.0] * 42

handedness_weight = float(row["handedness_loss_weight"])
if handedness_weight:
    handedness_target = 1.0 if row["handedness"]["label"] == "Right" else 0.0
else:
    handedness_target = 0.0
```

关键原则：样本是否是 Hand Landmarker 正样本，只看 `hand_presence.present`，也就是 `model.py` 中 `hand_flag` 的训练目标；样本是否来自高分 Palm，只看 `palm_valid`，它用于分析和采样，不直接改变 `hand_flag`、landmark 或 handedness 的监督语义。
