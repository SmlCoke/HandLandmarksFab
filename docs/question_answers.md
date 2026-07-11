# question.md 问题解答

本文面向当前仓库版本，解释半自动标注系统中仍然需要理解的关键问题。已经在代码中解决、且不再影响当前流程的历史建议不再保留。

## 1. 当前数据流总览

当前默认输出目录如下：

```text
data/
  images/
  01_palm/
    palm_detections.jsonl
  02_roi_crops/
    images/
    hand_roi_crops_manifest.jsonl
    hand_landmarks_autolabel_draft.jsonl
    cvat_autolabel.xml
  03_reviewed/
    cvat_reviewed.xml
    hand_landmarks_reviewed.jsonl
  04_visualization/
    global_images/
    crop_images/
    review_index.csv
  05_labels/
    hand_training_labels.jsonl
  qc/
```

核心流程是：

```text
原图
  -> Palm detection
  -> 256x256 rotated ROI crop
  -> MediaPipe 自动标注 draft
  -> CVAT skeleton/no_hand 人工复核
  -> reviewed JSONL
  -> training JSONL
```

`data/05_labels/hand_training_labels.jsonl` 是后续训练 Hand Landmarker 的主输入标注文件。

## 2. `compatible_bbox_expand` 的含义

`compatible_bbox_expand` 只用于 `palm.backend=mediapipe_official`。

MediaPipe Tasks HandLandmarker 对外给出 21 个手部关键点和 handedness，但不暴露官方内部 Palm ROI。因此我们从公开的 21 点中派生一个 palm-compatible detection，让后续 ROI crop 流程继续使用统一 Palm schema。

当前做法是取 palm 相关点：

```text
0 wrist
1 thumb CMC
5 index MCP
9 middle MCP
13 ring MCP
17 pinky MCP
```

先计算这些点的外接框：

```text
xmin = min(x_i)
ymin = min(y_i)
xmax = max(x_i)
ymax = max(y_i)
w = xmax - xmin
h = ymax - ymin
```

然后按 `compatible_bbox_expand=e` 扩大：

```text
xmin' = xmin - e * w
xmax' = xmax + e * w
ymin' = ymin - e * h
ymax' = ymax + e * h
```

最后 clamp 到 `[0,1]`。默认 `0.25` 表示在 palm 近似外接框四周各扩展 25% 的宽高。

它不是官方内部 Palm ROI，也不用于 `aethersign_onnx` 后端。ONNX 后端直接使用 Palm 模型解码出的 bbox。

## 3. Palm detection 如何变成 Hand ROI

`01_export_palm_detections.py` 输出：

```text
palm bbox + p0/p9 + score
```

其中：

- `bbox_px=[x1,y1,x2,y2]` 是原图 `1280x720` 坐标系下的 palm bbox。
- `p0` 是 wrist。
- `p9` 是 middle MCP。

`02_build_hand_roi_crops.py` 使用 `hand_autolabel/roi_geometry.py` 中的 `build_roi_rect_from_palm()` 复刻板端 ROI 几何。

步骤如下：

1. 对 palm bbox 排序并限制到图内：

```text
raw_width  = max(1, x2 - x1)
raw_height = max(1, y2 - y1)
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
```

2. 由 wrist 到 middle MCP 估计旋转：

```text
dx = p9_x - p0_x
dy = p9_y - p0_y
rotation = normalize(pi/2 - atan2(-dy, dx))
```

这里与板端保持一致。图像 y 轴向下，所以公式中是 `atan2(-dy, dx)`。

3. 按旋转坐标系平移 ROI 中心：

```text
cx' = cx + raw_width  * shift_x * cos(rotation)
         - raw_height * shift_y * sin(rotation)

cy' = cy + raw_width  * shift_x * sin(rotation)
         + raw_height * shift_y * cos(rotation)
```

默认：

```text
shift_x = 0.0
shift_y = -0.1
```

4. 使用 palm bbox 长边构造扩大后的正方形 ROI：

```text
long_side = max(raw_width, raw_height)
roi_width  = long_side * scale_x
roi_height = long_side * scale_y
```

默认：

```text
scale_x = 1.8
scale_y = 1.8
```

5. 根据中心、宽高和旋转角计算四角，并把旋转 ROI 仿射采样成 `256x256` crop。

如果旋转/平移/扩大后的 ROI 超出原图边界，crop 中超出原图的部分会被填充为黑色。Python 实现使用 `cv2.BORDER_CONSTANT, borderValue=0`，与板端采样越界返回 0 的行为一致。

这些黑色区域会作为 crop 图片的一部分进入 MediaPipe 自动标注、CVAT 人工复核、训练和板端推理。系统不会额外 mask 掉黑边，因为训练分布应尽量模拟板端推理分布。

## 4. `roi_rect`、`roi_corners_px` 和反投影

`data/02_roi_crops/hand_roi_crops_manifest.jsonl` 每行表示一个 crop。核心几何字段是：

```json
"roi_rect": {
  "x_center": 786.59,
  "y_center": 592.32,
  "width": 352.55,
  "height": 352.55,
  "rotation_rad": -2.59
},
"roi_corners_px": [
  [845.37, 834.58],
  [544.33, 651.10],
  [727.82, 350.06],
  [1028.86, 533.54]
]
```

`roi_rect` 是 ROI 的参数化描述，属于原图 `1280x720` 坐标系：

- `x_center/y_center`：ROI 中心点。
- `width/height`：ROI 在原图像素坐标中的宽高。
- `rotation_rad`：ROI 相对原图坐标系的旋转角，单位弧度。

`roi_corners_px` 是由 `roi_rect` 计算出的四个角点，顺序固定为：

```text
C0 = top_left
C1 = top_right
C2 = bottom_right
C3 = bottom_left
```

这些角点也属于原图 `1280x720` 坐标系，允许越界。

### 从 crop 坐标反投影回原图

对于 crop 归一化坐标：

```text
(u, v), u in [0,1], v in [0,1]
```

反投影公式是：

```text
P_image = C0 + u * (C1 - C0) + v * (C3 - C0)
```

展开为：

```text
x_image = C0_x + u * (C1_x - C0_x) + v * (C3_x - C0_x)
y_image = C0_y + u * (C1_y - C0_y) + v * (C3_y - C0_y)
```

如果输入是 crop 像素坐标 `(x_crop, y_crop)`，先转成归一化坐标：

```text
u = x_crop / (crop_width  - 1)
v = y_crop / (crop_height - 1)
```

当前 `crop_width=crop_height=256`，所以分母是 `255`。

这就是 `landmarks_crop_norm` 和 `landmarks_crop_px` 反投影为 `landmarks_image_px` 的方法。

## 5. `palm_valid` 与 `hand_presence.present`

`palm_valid` 和 `hand_presence.present` 是两个层级的语义，不能互相替代。

`palm_valid` 表示这个 crop 的 Palm 来源：

```text
palm_valid=true   -> 来自有效 Palm detection
palm_valid=false  -> 来自低分 negative candidate
```

`hand_presence.present` 表示当前 `256x256` crop 内是否有有效手：

```text
hand_presence.present=true   -> crop 内有手
hand_presence.present=false  -> crop 内无有效手
```

因此可能出现：

```text
palm_valid=false, hand_presence.present=true
```

这表示 Palm 分数低，但 crop 里确实有手。对于 Hand Landmarker 来说，它仍然是正样本。

也可能出现：

```text
palm_valid=true, hand_presence.present=false
```

这表示 Palm 有效检测触发了 ROI，但人工复核认为 crop 内没有有效手。这是 hard negative。

训练时应以 `hand_presence.present` 决定 Hand Landmarker 的正负，以 `palm_valid` 辅助分析样本来源和异常类型。

## 6. CVAT skeleton 与 `no_hand` tag

当前 CVAT 复核对象是 `data/02_roi_crops/images/` 下的 crop 图片，不是原始大图。每张 crop 最多一只手。

当前导出规则：

- 有手样本导出一个 `hand_landmarks` skeleton shape。
- skeleton 内有 21 个子 points，CVAT 子点 label 为 `1` 到 `21`。
- 无手样本导出一个 `no_hand` tag。
- `04_export_cvat_xml.py` 不复制图片；CVAT 中直接上传 `data/02_roi_crops/images/`，再导入 `data/02_roi_crops/cvat_autolabel.xml`。

CVAT 编号与内部 id 的对应关系是：

```text
CVAT 1  -> internal id 0
CVAT 2  -> internal id 1
...
CVAT 21 -> internal id 20
```

也就是说，CVAT 上看到的是 1-based 编号；JSONL、训练和 MediaPipe 语义里使用 0-based id。

人工复核后的 XML 保存为：

```text
data/03_reviewed/cvat_reviewed.xml
```

`05_import_cvat_xml.py` 导入规则：

- 有且仅有一个 `hand_landmarks` skeleton，且 21 个子点都存在：`hand_presence.present=true`。
- 无 skeleton 且有 `no_hand` tag：`hand_presence.present=false`。
- 无 skeleton 且无 `no_hand` tag：按无手导入，并写 QC warning `missing_no_hand_tag`。
- 同时存在 `no_hand` 和 skeleton：按无手保守导入，并写 QC error `conflicting_no_hand_and_skeleton`。
- 旧式顶层 `hand_landmarks` points shape 不再支持，会写 QC error `legacy_points_shape_not_supported`。

## 7. 三份 JSONL 的关系

当前系统中有三份 crop 级 hand label JSONL：

```text
data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
data/03_reviewed/hand_landmarks_reviewed.jsonl
data/05_labels/hand_training_labels.jsonl
```

它们一行都对应一个 crop。landmark 点字段统一为：

```json
{"id": 0, "x": 0.5357, "y": 0.4884}
```

不再保留 `z` 或 `visible`：

- `z` 是 MediaPipe 的相对深度输出，本项目当前只训练二维 landmark，不使用它。
- `visible` 在当前系统里始终为 1，没有形成单点 loss mask，因此删除。

### draft

`hand_landmarks_autolabel_draft.jsonl` 是 MediaPipe teacher 对每个 ROI crop 的自动标注草稿。它保留完整 manifest 元数据，例如 `crop_path`、`roi_rect`、`roi_corners_px`、`palm_valid`、`palm_score`。

### reviewed

`hand_landmarks_reviewed.jsonl` 是从 CVAT reviewed XML 导回后的人工复核结果。CVAT XML 本身只保存图片名、skeleton 点和 tag，因此导入时必须结合 draft 与 manifest 恢复 ROI 几何、原图反投影坐标和样本来源字段。

### training

`hand_training_labels_pretrain.jsonl` 和 `hand_training_labels_finetune.jsonl` 是 07A 生成的阶段化训练标注文件。

## 8. 07A / 07B 做了什么

07A 负责有噪声训练集，07B 负责完整人工复核的 Val/Test：

```text
输入:
  一个或多个训练来源的 manifest、pseudo labels
  可选 human Gold labels
  data/02_roi_crops/hand_roi_crops_manifest.jsonl

输出:
  hand_train_catalog_{stage}.jsonl
  hand_training_labels_{stage}.jsonl
  hand_training_excluded_{stage}.jsonl
  qc/finalize_train_{stage}_report.json
```

07A 会执行 manifest 权威校验、四象限样本分型、Gold 覆盖、pseudo 质量分层、重复 ROI 降采样，并写入采样与 loss 权重。07B 则要求 manifest、CVAT 和 reviewed JSONL 一一覆盖，对非 ignored Gold 执行严格结构校验，不按 Palm/teacher 分数过滤，也不降采样。

### 8.1 合并 manifest 元数据

脚本先按 `crop_id` 读取 ROI manifest。对于 reviewed 中每一行，它会用 manifest 补齐或确认：

- `image`
- `crop_path`
- `palm_det_id`
- `palm_valid`
- `palm_score`
- `width/height`
- `source_image_width/source_image_height`
- `roi_rect`
- `roi_corners_px`

这样最终训练文件单独拿出来也能知道每个 crop 来自哪张原图、哪个 Palm 候选、怎样反投影回原图。

### 8.2 执行 QC 检查

脚本调用 `label_issues()` 检查每个样本是否有结构或质量问题。典型检查包括：

- `hand_presence.present=true` 但 `landmarks_crop_norm` 不是 21 点。
- `hand_presence.present=false` 但仍然带 landmarks。
- `landmarks_crop_px` 中存在越界点。
- 一个 crop 中 MediaPipe 曾检测到多只手。
- 正样本 handedness 分数过低。
- 负样本来自高分 palm，可能是 hard negative 或漏标。

QC 结果会体现在两个地方：

- 行内 `needs_review`。
- `data/qc/final_training_label_stats.json` 中的 warnings/errors/skipped 统计。

### 8.3 跳过结构上不能训练的样本

有两类样本不会写入最终训练 JSONL：

```text
hand_presence.present=true  但 landmarks_crop_norm 不是 21 点
hand_presence.present=false 但仍然带 landmarks
```

前者无法监督 21 点输出；后者正负语义冲突。脚本会把它们写入 `final_training_label_stats.json` 的 `skipped`，而不是强行进入训练。

### 8.4 规范负样本

如果 `hand_presence.present=false`，脚本会强制清空所有手部监督字段：

```json
"hand_id": null,
"handedness": {"label": "unknown", "score": null},
"landmarks_crop_norm": [],
"landmarks_crop_px": [],
"landmarks_image_px": []
```

这样可以保证负样本只监督 hand presence，不会意外参与 landmark 或 handedness 训练。

### 8.5 写入 loss weight

最终训练文件会新增三个 loss weight：

```json
"hand_presence_loss_weight": 1.0,
"landmark_loss_weight": 1.0,
"handedness_loss_weight": 1.0
```

规则如下：

- `hand_presence_loss_weight` 当前总是 `1.0`。正样本和负样本都监督 hand flag。
- `landmark_loss_weight=1.0` 仅当 `hand_presence.present=true`，否则为 `0.0`。
- `handedness_loss_weight=1.0` 仅当 `hand_presence.present=true` 且 `handedness.label` 是 `Left` 或 `Right`，否则为 `0.0`。

因此负样本不会参与 landmark loss 和 handedness loss；handedness 不确定的正样本仍可参与 hand presence 与 landmark 训练，但不参与 handedness 训练。

### 8.6 统一 landmark schema

最终输出前会清洗三个 landmark 字段：

```text
landmarks_crop_norm
landmarks_crop_px
landmarks_image_px
```

每个点只保留：

```json
{"id": 0, "x": ..., "y": ...}
```

这一步保证 draft、reviewed、training 三份 JSONL 的 landmark 点结构一致。

### 8.7 `needs_review` 如何影响训练

07A 会把 `needs_review=true` 转化为具体 `quality_flags`，按质量策略将样本保留、hold 或排除，并输出独立的 sampling/supervision/quality 权重。

后续训练脚本可以根据策略选择：

- 过滤 `needs_review=true`。
- 只过滤严重错误，保留轻微越界或低置信样本。
- 使用更低的采样权重或 loss 权重。

正式训练前应查看 `qc/finalize_train_{stage}_report.json`、全量 catalog 和 `data/04_visualization/review_index.csv`，确认各类排除原因和采样分布。

## 9. 可视化如何使用

`06_visualize_autolabels.py` 生成两类可视化：

```text
data/04_visualization/global_images/*.png
data/04_visualization/crop_images/*.png
```

`global_images/` 在原图上绘制：

- Palm bbox。
- Palm p0/p9。
- rotated Hand ROI 四边形。
- 反投影到原图上的 21 点骨架。

`crop_images/` 在 `256x256` crop 图上直接绘制 `landmarks_crop_px`。

同时生成：

```text
data/04_visualization/review_index.csv
```

它把每个 crop 的 `crop_overlay_path`、`global_overlay_path`、`hand_presence`、`handedness`、`needs_review`、`palm_valid`、`palm_score` 汇总在一起，方便人工定位需要复查的样本。
