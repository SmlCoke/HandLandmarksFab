# question.md 问题解答

本文解释当前半自动标注系统的数据流、几何关系、负样本语义和后续建议。本文面向人工阅读，不修改当前代码。

## 1. `compatible_bbox_expand` 的含义

`compatible_bbox_expand` 只用于 `palm.backend=mediapipe_official`。

MediaPipe Tasks HandLandmarker 对外只给出 21 个手部关键点和 handedness，不暴露官方内部 Palm ROI。因此我们需要从公开的 21 点中派生一个“palm-compatible detection”，让后续 ROI crop 流程继续使用统一的 Palm schema。

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

## 2. Palm detection 如何变成 Hand ROI

`01_export_palm_detections.py` 输出的是：

```text
palm bbox + p0/p9 + score
```

其中：

- `bbox_px=[x1,y1,x2,y2]` 是原图 `1280x720` 坐标系下的 palm bbox。
- `p0` 是 wrist。
- `p9` 是 middle MCP。

`02_build_hand_roi_crops.py` 使用 `hand_autolabel/roi_geometry.py` 中的 `build_roi_rect_from_palm()` 复刻板端 `materials/preminilary/device/src/hand_landmarker.cpp` 的 ROI 几何。

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

这里与板端保持一致。注意图像 y 轴向下，所以公式中是 `atan2(-dy, dx)`。

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

5. 根据中心、宽高和旋转角计算四角，并把这块旋转 ROI 仿射采样成 `256x256` crop。

### 是否会出现黑色区域

会。

如果旋转/平移/扩大后的 ROI 超出原图边界，crop 中超出原图的部分会被填充为黑色。当前 Python 实现使用：

```python
cv2.warpAffine(..., borderMode=cv2.BORDER_CONSTANT, borderValue=0)
```

这与板端逻辑一致。板端 `SampleBilinear()` 在采样点远离图像或像素越界时返回 `0`，因此超出原图的区域也是黑色。

这些黑色区域会作为 `256x256` crop 图片的一部分直接参与后续 MediaPipe 标注、人工复核、训练和板端推理。系统不会额外 mask 掉这些黑边，因为训练分布必须尽量模拟板端推理分布。

如果黑色区域过多，应该通过 QC 标记或人工复核处理，而不是在 crop 阶段自动裁掉。

## 3. `roi_rect` 和 `roi_corners_px` 的含义与反投影

`hand_roi_crops_manifest.jsonl` 每行表示一个 crop。核心几何字段是：

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

`roi_rect` 是 ROI 的参数化描述，全部属于原图 `1280x720` 坐标系：

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

它们也属于原图 `1280x720` 坐标系，允许越界。

### 从 crop 坐标反投影回原图

MediaPipe 在 `256x256` crop 上输出归一化点：

```text
(u, v),  u in [0,1], v in [0,1]
```

反投影公式是双线性仿射插值：

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

这就是 `landmarks_crop_norm` 和 `landmarks_crop_px` 反投影为 `landmarks_image_px` 的方法。它与 crop 生成时的三点仿射关系互为正反方向。

## 4. `hand_landmarks_mediapipe_raw.jsonl` 和 `hand_landmarks_autolabel_draft.jsonl`

当前这两份文件内容完全相同：

```text
data/labels/hand_landmarks_mediapipe_raw.jsonl
data/labels/hand_landmarks_autolabel_draft.jsonl
```

原本的设计意图是：

- `hand_landmarks_mediapipe_raw.jsonl`：保存 MediaPipe teacher 的直接输出。
- `hand_landmarks_autolabel_draft.jsonl`：保存准备进入 CVAT 的自动标注草稿，包含恢复元数据所需的字段。

但当前实现已经在 `03_run_mediapipe_on_rois.py` 中直接把 MediaPipe 输出和 manifest 元数据合并成完整 crop 级标注，所以两份文件实际重复。

可以简化为只保留一份，推荐保留：

```text
hand_landmarks_autolabel_draft.jsonl
```

如果后续仍想保留 raw，则 raw 应该变成真正的“裸 MediaPipe 输出”，不包含 ROI manifest 的冗余字段。否则保留两份没有实际价值。

### `hand_presence.present` 是否完全取决于 `palm_valid`

不是。

`palm_valid` 表示这个 crop 来源于 Palm 正检还是低分负候选：

```text
palm_valid=true   -> 来自有效 Palm detection
palm_valid=false  -> 来自 negative_candidates
```

`hand_presence.present` 表示 MediaPipe 或人工复核认为该 `256x256` crop 内是否有手：

```text
hand_presence.present=true   -> crop 内有手
hand_presence.present=false  -> crop 内无手
```

两者是不同层级的语义。实际可能出现：

```text
palm_valid=false, hand_presence.present=true
```

这表示 Palm 分数低，但 crop 里确实有手。当前烟测中已经出现这种情况。

也可能出现：

```text
palm_valid=true, hand_presence.present=false
```

这表示 Palm 有效检测触发了 ROI，但 Hand teacher 或人工复核认为 crop 内没有有效手。这是 hard negative。

## 5. `crop_id`、`palm_det_id`、`hand_id` 的唯一性

当前系统中：

- 一个 `palm_det_id` 只生成一个 `crop_id`。
- 一个 `crop_id` 最多生成一个 `hand_id`。
- 如果 `hand_presence.present=false`，则 `hand_id=null`。

当前 ID 形式是：

```text
palm_det_id = image_stem:palm0 或 image_stem:neg0
crop_id     = palm_det_id:crop0
hand_id     = palm_det_id:hand0
```

在现有约束下，每个 Palm detection 只构造一个 crop，每个 crop 最多一只手，所以 `:crop0` 和 `:hand0` 的数字后缀确实没有必要。

可以简化为：

```text
crop_id = palm_det_id:crop
hand_id = crop_id:hand
```

这样更符合当前数据结构，也更容易人工阅读。

唯一需要注意的是：如果未来打算从同一个 Palm detection 派生多个不同 scale/shift 的 crops，或者允许一个 crop 中多个 hand shapes，那么数字后缀会重新有用。但本项目当前明确“一 palm 一 crop，一 crop 最多一手”，因此建议采纳这个简化。

## 6. `cvat_upload_images/` 与 `roi_crops/images/*.png`

当前 `04_export_cvat_xml.py` 会把：

```text
data/roi_crops/images/*.png
```

复制一份到：

```text
data/review/cvat_upload_images/
```

这两份图片内容应完全一致，只是路径不同。

从节省磁盘空间的角度看，确实不需要复制。CVAT XML 中保存的是图片文件名，而不是仓库内的绝对路径。只要上传到 CVAT 的图片文件名与 XML 中的 `<image name="...">` 对得上，就可以直接上传 `data/roi_crops/images/*.png`。

因此这个建议合理：后续可以删除 `cvat_upload_images/` 复制逻辑，让 `04_export_cvat_xml.py` 只生成 XML，并在 README 中说明直接上传 ROI crop 图片目录。

## 7. CVAT 中负样本如何处理

当前导出到 CVAT 的对象是 crop 图片，不是原始大图。每张 crop 最多一只手。

导出规则：

- 如果草稿中 `hand_presence.present=true` 且有 21 个点，则导出一个 `hand_landmarks` points shape。
- 否则导出一个 `no_hand` tag。

人工复核时可以遇到以下情况。

### 情况 A：负样本确实没有手

例如：

```text
palm_valid=false
hand_presence.present=false
```

人工确认没有手，则保持 `no_hand` tag，不添加 points。

导回 JSONL 后：

```json
"hand_presence": {"present": false},
"hand_id": null,
"landmarks_crop_norm": [],
"landmarks_crop_px": [],
"landmarks_image_px": []
```

### 情况 B：负样本其实有手

例如：

```text
palm_valid=false
hand_presence.present=false
```

但人工看到 crop 里其实有手。这时应删除或忽略 `no_hand`，添加一个 `hand_landmarks` points shape，标满 21 点。

导回 JSONL 后：

```json
"hand_presence": {"present": true},
"hand_id": "...:hand",
"landmarks_crop_norm": [21 points],
"landmarks_crop_px": [21 points],
"landmarks_image_px": [21 points]
```

`palm_valid=false` 会被保留，表示它来自低分 Palm 候选，但对于 Hand Landmarker 训练来说，它已经是一个正样本。

### 情况 C：正样本其实没有手

例如：

```text
palm_valid=true
hand_presence.present=true
```

但人工发现 crop 中没有有效手。这时应删除 points，添加或保留 `no_hand` tag。

导回 JSONL 后会变成：

```json
"hand_presence": {"present": false},
"hand_id": null,
"landmarks_crop_norm": []
```

这类样本是非常有价值的 hard negative，因为它来自 Palm 正检，却不应参与 landmark 训练。

### 情况 D：同时有 `no_hand` 和 points

当前导入逻辑中，只要存在 `no_hand` tag，就会按无手处理。这可以避免错误 points 被误当成正样本。

更严格的后续实现可以把“同时有 no_hand 和 points”写入 QC error，要求人工重新检查。

## 8. 异常样本如何参与训练

训练最小单位是 crop，不是原图，也不是 palm detection。

### `palm_valid=true, hand_presence.present=true`

这是标准正样本。

训练方式：

```text
hand_presence target = 1
landmark loss weight = 1
handedness loss weight = 1，前提是 label 为 Left/Right
```

它监督 Hand Landmarker 的三类输出：是否有手、21 点、左右手。

### `palm_valid=true, hand_presence.present=false`

这是 Palm 正检触发的 hard negative。

训练方式：

```text
hand_presence target = 0
landmark loss weight = 0
handedness loss weight = 0
landmarks = []
```

它不监督 landmarks 和 handedness，但应该参与 hand presence 训练，让 Hand Landmarker 学会拒绝错误 ROI。

### `palm_valid=false, hand_presence.present=false`

这是低分 Palm 负候选且人工/MediaPipe 确认无手。

训练方式：

```text
hand_presence target = 0
landmark loss weight = 0
handedness loss weight = 0
```

这类样本可以参与训练，但要注意数量不要远超正样本，否则 hand presence head 可能偏向预测无手。后续训练脚本可以做负样本采样或 loss reweight。

### `palm_valid=false, hand_presence.present=true`

这是低分 Palm 候选，但 crop 内确实有手。

训练方式：

```text
hand_presence target = 1
landmark loss weight = 1
handedness loss weight = 1，前提是 label 为 Left/Right
```

它对 Hand Landmarker 是正样本。它也提示 Palm Detector 可能漏检或低估了该手，但本工具链当前不训练 Palm，因此这里只把它作为 Hand 正样本使用。

### `hand_presence.present=true` 但点数不是 21

这是错误样本，不应进入最终训练。

当前 `07_finalize_training_labels.py` 会跳过这类正样本，并写入 QC。

### 点越界或 handedness 低置信度

当前实现会保留样本并标记 `needs_review`。建议人工复核后再用于正式训练。

训练阶段可以选择：

- 直接过滤 `needs_review=true`。
- 或者只过滤严重越界样本。
- 或者降低这类样本的 landmark/handedness loss 权重。

## 9. 输出结构调整建议

建议是合理的，尤其是把流水线阶段编号后，数据流会更清楚。

但建议中的目录有两个小问题：

1. `03_reviewd` 应修正为 `03_reviewed`。
2. 示例树没有列出 `05_labels/`，但文字中提到了，应补上。

建议目标结构为：

```text
data/
  images/
  01_palm/
    palm_detections.jsonl
  02_roi_crops/
    images/
    hand_roi_crops_manifest.jsonl
    hand_landmarks_mediapipe_raw.jsonl
    hand_landmarks_autolabel_draft.jsonl
    cvat_autolabel.xml
  03_reviewed/
    cvat_reviewed.xml
    hand_landmarks_reviewed.jsonl
  04_visualization/
    crop_images/
    global_images/
    review_index.csv
  05_labels/
    hand_training_labels.jsonl
  qc/
```

其中 `hand_landmarks_autolabel_draft.jsonl` 虽然 question.md 没列出，但当前 `05_import_cvat_xml.py` 需要它恢复 CVAT XML 中丢失的字段，所以应该放在 `02_roi_crops/`。

`data/review/` 可以逐步废弃，不再作为主要输出目录。CVAT 上传图片可以直接使用 `02_roi_crops/images/`。

## 10. 可视化如何做

当前 `06_visualize_autolabels.py` 主要生成原图级可视化：

```text
data/review/overlay_images/*.png
```

它读取：

- 原图：`data/images/*.tiff`
- Palm 结果：`palm_detections.jsonl`
- ROI manifest：`hand_roi_crops_manifest.jsonl`
- Reviewed labels：`hand_landmarks_reviewed.jsonl`

然后在原图上画：

1. Palm bbox。
2. Palm p0/p9。
3. rotated Hand ROI 四边形。
4. 反投影到原图上的 21 点骨架。
5. `hand_presence`、`handedness`、score、source 等文本。

### crop 坐标如何反投影到原图

仍然使用第 3 节公式。

对于 crop 内点：

```text
(x_crop, y_crop)
```

先归一化：

```text
u = x_crop / 255
v = y_crop / 255
```

然后使用 ROI 四角点：

```text
C0 = top_left
C1 = top_right
C3 = bottom_left
```

反投影：

```text
P_image = C0 + u * (C1 - C0) + v * (C3 - C0)
```

这样得到：

```text
(x_image, y_image)
```

也就是 `landmarks_image_px`。原图可视化直接使用 `landmarks_image_px` 绘制骨架。

### crop 小图可视化

question.md 建议增加：

```text
04_visualization/crop_images/
```

这是合理的。它应在 `256x256` crop 图上直接绘制 `landmarks_crop_px`，无需反投影。这样可以同时检查：

- MediaPipe/人工点在 crop 内是否合理。
- 反投影到原图后是否仍然与手对齐。

后续实现时建议同时生成 crop 级和 global 级两类可视化。
