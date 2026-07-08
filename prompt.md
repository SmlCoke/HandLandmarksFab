# Prompt

项目背景：

请先阅读当前仓库目录下的：`project-8.md`，了解我们的项目背景。

现在，我们需要重新训练 Hand Landmarker 模型（注意，此时的 Hand Landmarker 模型是 Palm Detector 接收 1280x720 图输入推理然后经过后处理得到的 Hand ROI）。为此，首先需要为 Hand Landmarker 模型准备训练数据集，因此，首先我们需要构造一个半自动化数据标注体系。你现在要为我编写“半自动化数据标注体系”的所有脚本。请先阅读我提供的项目文档和核心参考代码，再开始写代码。你需要阅读的文件有：

```
docs/HandFab/01_hand_landmarker_annotation_format.md: Hand Landmarker 的训练标注文件格式文档
docs/HandFab/04_model_io_and_annotation_interfaces.md: Palm Detector 和 Hand Landmarker 的输入输出接口文档

materials/preminilary/palm/model.py: Palm Detector 的模型定义
materials/preminilary/palm/model_opt.onnx: Palm Detector 的 ONNX 模型，对应板端表现较好的初赛模型
materials/preminilary/palm/anchor_utils.py
materials/preminilary/palm/infer_model_gray.py

materials/preminilary/hand/model.py: Hand Landmarker 的模型定义
materials/preminilary/hand/roi_utils.py
materials/preminilary/hand/infer_frames_with_roi.py

materials/preminilary/device/src/palm_detector.cpp: 板端调度程序的 palm_detector.cpp
materials/preminilary/device/src/hand_landmarker.cpp: 板端调度程序的 hand_landmarker.cpp
materials/preminilary/device/include/: 板端调度程序的头文件
materials/preminilary/device/main.cpp: 板端调度程序的 main.cpp

```

本任务的目标是：对已经准备好的 `1280x720` 正向灰度 `.tiff` 图片（人像端正），自动生成可供人工复核的 Hand Landmarker 标注草稿，包括 Palm 检测结果、Hand ROI crop、MediaPipe 自动关键点结果、ROI 与关键点的匹配关系、可视化质检图，以及最终可导出的训练标注文件。

如果文档与代码有冲突，以本提示词中的最终方案和 `materials/preminilary/device` 的 ROI 几何为准。

> 例如文档中提到了生成 .txt 类型的标注文件，但是本项目所有标注文件统一为 `.jsonl`，不再生成旧 txt 文件。

## I. 项目关键约定

1. 输入图片已经是 `1280x720` 正向灰度 `.tiff`。不要写原始图片旋转脚本，不要在脚本中假设还需要从 `720x1280` 旋转到 `1280x720`。
2. 所有 bbox、landmark、ROI 坐标都属于 `1280x720` 正向图坐标系，除非字段名明确写着 `crop` 或 `roi`。
3. Palm Detector 只负责输出 `palm bbox + p0/p9 + score`。Palm 不判断左手/右手。
4. 左手/右手信息统一使用字段名 `handedness`，不要使用 `handedness_score` 这种平级字段。推荐结构为 `{"label": "Right", "score": 0.91}`。
5. 手是否存在统一使用字段名 `hand_presence`，不要使用 `hand_flag`、`presence_flag` 或其他名字。推荐结构为 `{"present": true, "score": null}`。
6. 没有手的 ROI crop 需要保留为负样本候选，但后续训练中只能监督 `hand_presence.present=false`，不能参与 landmark loss 和 handedness loss。
7. 官方 MediaPipe HandLandmarker 的高层 Python Tasks API 不要假设能输出内部 Palm ROI。MediaPipe 在本任务中主要作为“关键点老师”：对 Hand ROI crop 进行识别，输出 21 点和 handedness 等草稿。
8. 官方 MediaPipe 文档说明，HandLandmarkerResult 对外包含三类结果：handedness、image landmarks、world landmarks。image landmarks 是 21 个 `x/y/z`，其中 `x/y` 归一化到输入图宽高；world landmarks 是以米为单位的 3D 坐标（这一个数据我们不用）。请按这个定义解析结果。
9. 如果当前 mediapipe Python API 不暴露 hand presence score，立刻终止工作，马上向我汇报。不要伪造 hand presence score。
10. 本份文档中提供的输出数据的值、图片文件的名字（例如"seq001_f000123"）均为示例，请不要硬编码在脚本中。
11. 
## II. Palm 与 Hand ROI 约定

Palm 后端需要支持两种方式，输出文件格式必须完全相同：

```text
palm.backend = mediapipe_official
palm.backend = aethersign_onnx
```

默认使用：

```yaml
palm:
  backend: mediapipe_official
```

### `mediapipe_official` backend

默认后端，用官方 MediaPipe HandLandmarker 在整张 `1280x720` 图上运行，最多检测 2 只手，然后派生出与 Palm detection schema 兼容的结果：

- `p0` 使用 hand landmark `id=0`，也就是 wrist（手腕）。
- `p9` 使用 hand landmark `id=9`，也就是 middle MCP（中指根）。
- `bbox` 可以由 palm 相关点或 21 点外接框派生，但必须在输出中写明 `bbox_source`。建议优先使用 palm 相关点 `0, 1, 5, 9, 13, 17` 的外接框，并适度扩大到可作为 palm bbox 的近似。
- `score` 可以使用 handedness score 或可用的 detection confidence；如果没有可靠 palm score，写入最合理的可用 confidence，并在 `score_source` 中说明来源。
- 这不是官方内部 Palm ROI；它只是 `palm-compatible detection`，目的是让后续 ROI crop 流程具有统一接口。不要在代码或 README 中声称官方高层 API 暴露了内部 Palm ROI。

### `aethersign_onnx` backend

使用 `materials/preminilary/palm/model_opt.onnx` 和配套 decode/NMS 逻辑。对于当前 ONNX 模型，Palm 原始输出会产生：

```text
14x14 head: 14 * 14 * 2 = 392
7x7 head:   7 * 7 * 2 = 98
total: 490 anchors
```

后处理流程必须是：

```text
raw outputs
-> decode 490 anchor candidates
-> score threshold
-> NMS
-> cross-head suppression
-> score 排序
-> 最多保留 2 个有效 Palm detections
```

“最多 2 个”不等于“一定 2 个真实手”。没有手时 `detections=[]`；只有一只手时 `detections` 长度为 1；两只手时长度为 2。不要为了固定格式强行写两个有效 bbox。低分候选如需保留，只能写入 `negative_candidates`，不能写入 `detections`。

### Hand ROI 构造

NMS 或官方 backend 得到的是 Palm detection，不是最终 Hand Landmarker 输入。最终 Hand 输入必须由：

```text
Palm bbox + p0/p9 + roi_scale/roi_shift
-> 构造 rotated/expanded Hand ROI
-> 从 1280x720 原图仿射裁剪成 256x256 crop
```

Hand ROI 几何必须尽量与 `materials/preminilary/device/src/hand_landmarker.cpp` 一致。至少要复现：

```text
rotation: 由 wrist -> middle MCP 方向估计
roi_scale_x: 默认 1.8
roi_scale_y: 默认 1.8
roi_shift_x: 默认 0.0
roi_shift_y: 默认 -0.1
output crop: 256x256
```

这些参数必须写入配置文件。

## III. 请创建的目录结构

请在当前工作区创建：

```text
HandLandmarkerFab/
  README.md
  requirements.txt
  configs/
    autolabel.yaml
  hand_autolabel/
    __init__.py
    image_io.py
    formats.py
    palm_mediapipe.py
    palm_onnx.py
    palm_decode.py
    nms.py
    roi_geometry.py
    mediapipe_roi_labeler.py
    cvat_io.py
    projection.py
    visualization.py
    quality_checks.py
  scripts/
    00_validate_images.py
    01_export_palm_detections.py
    02_build_hand_roi_crops.py
    03_run_mediapipe_on_rois.py
    04_export_cvat_xml.py
    05_import_cvat_xml.py
    06_visualize_autolabels.py
    07_finalize_training_labels.py
  data/
    images/
    palm/
    roi_crops/
    labels/
    review/
    qc/
```

脚本不要使用绝对路径。所有路径从 `configs/autolabel.yaml` 读取。

### `data/` 子目录含义

- `data/images/`：输入原图目录。这里只放已经是 `1280x720` 正向的 `.tiff` 灰度图。
- `data/palm/`：Palm 或 palm-compatible detection 的输出目录，例如 `palm_detections.jsonl` 和统计报告。
- `data/roi_crops/`：由 Palm detection 构造出的 `256x256` Hand ROI crop，以及 crop manifest。
- `data/labels/`：MediaPipe 自动标注、CVAT 复核后标注、最终训练标注 JSONL。
- `data/review/`：人工复核辅助文件，包括 CVAT 上传用 XML、CVAT 导出的 reviewed XML、overlay 可视化图和 review index。
- `data/qc/`：所有自动质检统计、错误报告和日志。

## IV. 配置文件要求

请创建 `HandLandmarkerFab/configs/autolabel.yaml`，至少包含：

```yaml
paths:
  images_dir: data/images
  palm_model_onnx: ../materials/preminilary/palm/model_opt.onnx
  palm_outputs_dir: data/palm
  roi_crops_dir: data/roi_crops
  labels_dir: data/labels
  review_dir: data/review
  qc_dir: data/qc

image:
  width: 1280
  height: 720
  channels: 1
  orientation: upright

palm:
  backend: mediapipe_official
  input_size: 224
  score_threshold: 0.50
  nms_iou_threshold: 0.30
  cross_head_suppress_iou: 0.35
  max_detections: 2
  keep_low_score_candidates_for_negatives: true
  negative_candidate_threshold: 0.15

hand_roi:
  output_width: 256
  output_height: 256
  scale_x: 1.8
  scale_y: 1.8
  shift_x: 0.0
  shift_y: -0.1

mediapipe:
  num_hands: 1
  min_hand_detection_confidence: 0.5
  min_hand_presence_confidence: 0.5
  min_tracking_confidence: 0.5

cvat:
  label_name: hand_landmarks
  no_hand_label_name: no_hand
  xml_version: "1.1"

review:
  draw_palm_bbox: true
  draw_hand_roi: true
  draw_landmarks: true
```

### `palm` 参数含义

- `backend`：Palm detection 来源。`mediapipe_official` 表示用官方 MediaPipe 整图结果派生 palm-compatible detections；`aethersign_onnx` 表示用我们冻结的 `model_opt.onnx` 输出真实 Palm detections。
- `input_size`：`aethersign_onnx` Palm 模型输入尺寸，当前为 `224x224`。`mediapipe_official` 后端可以不使用这个值。
- `score_threshold`：Palm 候选进入有效 detections 的最低分数。低于此阈值的候选不能作为正样本。
- `nms_iou_threshold`：同一 head 内做 NMS 时的 IoU 阈值。IoU 高于该阈值的重叠候选会被抑制。
- `cross_head_suppress_iou`：`14x14` 与 `7x7` 两个 head 之间做跨 head 抑制时的 IoU 阈值。
- `max_detections`：每张原图最多保留多少个有效 Palm detections。本项目为最多 2 个，不保证一定有 2 个。
- `keep_low_score_candidates_for_negatives`：是否保留低分候选作为未来负样本候选。它们不进入 `detections`，只能进入 `negative_candidates`。
- `negative_candidate_threshold`：低分负候选的最低分数。低于该阈值的候选通常直接丢弃。

### `mediapipe` 参数含义

这些参数来自官方 MediaPipe HandLandmarkerOptions：

- `num_hands`：最多检测几只手。对每个 `256x256` ROI crop 运行时建议为 `1`，因为每个 crop 理论上最多一只手；对整图 official backend 派生 Palm detections 时可以临时覆盖为 `2`。
- `min_hand_detection_confidence`：Palm detection 被认为成功的最低置信度阈值。
- `min_hand_presence_confidence`：Hand landmark 模型内部 hand presence score 的最低阈值。官方文档说明，在视频/直播模式中，如果 hand presence 低于该阈值，会重新触发 palm detection。
- `min_tracking_confidence`：视频/直播模式下的跟踪成功阈值，本质上是当前帧与上一帧手框的 IoU 阈值。当前半自动标注主要使用 IMAGE 模式，这个参数通常影响不大，但保留在配置里保持接口完整。

## V. 需要生成的文件格式

### 1. Palm 检测主文件

生成：

```text
HandLandmarkerFab/data/palm/palm_detections.jsonl
```

每行一张原图。`detections` 只保存有效检测，长度可以是 0、1、2，由阈值和 NMS 结果决定，不一定有两个 bbox。

```json
{
  "image": "seq001_f000123.tiff",
  "width": 1280,
  "height": 720,
  "detections": [
    {
      "palm_det_id": "seq001_f000123:palm0",
      "valid": true,
      "score": 0.93,
      "score_source": "palm_score",
      "bbox_norm": [0.30, 0.40, 0.48, 0.67],
      "bbox_px": [384.0, 288.0, 614.4, 482.4],
      "bbox_source": "aethersign_onnx_palm_bbox",
      "keypoints_norm": {
        "p0": [0.35, 0.62],
        "p9": [0.42, 0.47]
      },
      "keypoints_px": {
        "p0": [448.0, 446.4],
        "p9": [537.6, 338.4]
      },
      "source": "aethersign_onnx",
      "head": "head14"
    }
  ],
  "negative_candidates": []
}
```

### 2. Hand ROI crop manifest

生成：

```text
HandLandmarkerFab/data/roi_crops/hand_roi_crops_manifest.jsonl
```

每行一个 ROI crop。这个文件是后续所有 crop 级标注的几何基准。

```json
{
  "crop_id": "seq001_f000123:palm0:crop0",
  "image": "seq001_f000123.tiff",
  "palm_det_id": "seq001_f000123:palm0",
  "palm_valid": true,
  "palm_score": 0.93,
  "crop_path": "data/roi_crops/images/seq001_f000123_palm0.png",
  "roi_rect": {
    "x_center": 512.0,
    "y_center": 360.0,
    "width": 360.0,
    "height": 360.0,
    "rotation_rad": 0.42
  },
  "roi_corners_px": [
    [310.0, 230.0],
    [660.0, 380.0],
    [510.0, 730.0],
    [160.0, 580.0]
  ],
  "output_size": [256, 256]
}
```

### 3. MediaPipe ROI 自动关键点文件

生成：

```text
HandLandmarkerFab/data/labels/hand_landmarks_mediapipe_raw.jsonl
```

每行一个 ROI crop 的自动标注结果。实际文件中，`landmarks_crop_norm`、`landmarks_crop_px`、`landmarks_image_px` 必须各有 21 个点；下面示例只展示少量点以说明字段结构。

```json
{
  "crop_id": "seq001_f000123:palm0:crop0",
  "image": "seq001_f000123.tiff",
  "palm_det_id": "seq001_f000123:palm0",
  "hand_id": "seq001_f000123:palm0:hand0",
  "hand_presence": {
    "present": true,
    "score": null
  },
  "handedness": {
    "label": "Right",
    "score": 0.91
  },
  "landmarks_crop_norm": [
    {"id": 0, "x": 0.45, "y": 0.78, "z": -0.0001, "visible": 1},
    {"id": 1, "x": 0.47, "y": 0.70, "z": -0.02, "visible": 1}
  ],
  "landmarks_crop_px": [
    {"id": 0, "x": 115.2, "y": 199.7, "visible": 1},
    {"id": 1, "x": 120.3, "y": 179.2, "visible": 1}
  ],
  "landmarks_image_px": [
    {"id": 0, "x": 520.3, "y": 410.8, "visible": 1},
    {"id": 1, "x": 530.1, "y": 395.4, "visible": 1}
  ],
  "source": "mediapipe_hand_landmarker"
}
```

字段解释：

- `landmarks_crop_norm`：关键点在 `256x256` crop 内的归一化坐标，用于训练 Hand Landmarker 的直接监督。
- `landmarks_crop_px`：关键点在 `256x256` crop 内的像素坐标，用于 CVAT、可视化和调试。
- `landmarks_image_px`：关键点反投影回原始 `1280x720` tiff 图上的像素坐标，用于原图可视化和质量检查。
- `source`：标注来源，例如 `mediapipe_hand_landmarker`、`cvat_reviewed`、`manual_fixed`，方便追踪数据来源。

如果 MediaPipe 在 crop 中没有检测到手：

```json
{
  "crop_id": "seq001_f000123:palm1:crop0",
  "image": "seq001_f000123.tiff",
  "palm_det_id": "seq001_f000123:palm1",
  "hand_id": null,
  "hand_presence": {
    "present": false,
    "score": null
  },
  "handedness": {
    "label": "unknown",
    "score": null
  },
  "landmarks_crop_norm": [],
  "landmarks_crop_px": [],
  "landmarks_image_px": [],
  "source": "mediapipe_hand_landmarker"
}
```

### 4. CVAT XML 文件

本项目的人工复核必须借助 CVAT。上传到 CVAT 的图片是 `data/roi_crops/images/` 下的 crop 图，每张 crop 最多一只手，不上传原始 `1280x720` tiff。

需要生成自动标注 XML：

```text
HandLandmarkerFab/data/review/cvat_autolabel.xml
```

人工在 CVAT 网站复核后导出：

```text
HandLandmarkerFab/data/review/cvat_reviewed.xml
```

XML 格式要求：

- 使用 `CVAT for images 1.1`。
- label 名称默认 `hand_landmarks`。
- 每只有手 crop 使用一个 `points` shape 保存 21 个点，点顺序必须是 MediaPipe 21 点顺序。
- 人工复核的流程是：CVAT 自动标注 -> 人工复核 -> CVAT 导出 XML -> 脚本解析 XML 生成最终训练标注 JSONL。因此文件格式转换链路为：`.jsonl -> .xml -> .jsonl`，其中：
    - `.jsonl -> .xml`: 保留最关键的图片名字、关键点坐标即可，其他没法被 cvat 识别的字段可以丢弃。
    - `.xml -> .jsonl`: 此过程还必须使用原始 `hand_landmarks_autolabel_draft.jsonl` 共同解析，才能恢复 在 `.jsonl -> .xml` 阶段被删除的字段。

必须实现 JSONL 与 CVAT XML 互转：

```text
hand_landmarks_autolabel_draft.jsonl -> cvat_autolabel.xml
cvat_reviewed.xml -> hand_landmarks_reviewed.jsonl
```

### 5. 人工复核后的主标注文件

主标注文件以 crop 为最小单元，而不是以原始 tiff 为最小单元。原因是 Hand Landmarker 的训练输入就是 `256x256` Hand ROI crop；每个 crop 理论上最多一只手，结构更简单，也不需要额外的 Palm-Hand 匹配文件。

脚本先生成草稿：

```text
HandLandmarkerFab/data/labels/hand_landmarks_autolabel_draft.jsonl
```

CVAT 人工复核后生成最终主标注：

```text
HandLandmarkerFab/data/labels/hand_landmarks_reviewed.jsonl
```

每行一个 crop：

```json
{
  "crop_id": "seq001_f000123:palm0:crop0",
  "image": "seq001_f000123.tiff",
  "crop_path": "data/roi_crops/images/seq001_f000123_palm0.png",
  "palm_det_id": "seq001_f000123:palm0",
  "hand_id": "seq001_f000123:palm0:hand0",
  "width": 256,
  "height": 256,
  "source_image_width": 1280,
  "source_image_height": 720,
  "hand_presence": {
    "present": true,
    "score": null
  },
  "handedness": {
    "label": "Right",
    "score": 0.91
  },
  "landmarks_crop_norm": [
    {"id": 0, "x": 0.45, "y": 0.78, "visible": 1}
  ],
  "landmarks_crop_px": [
    {"id": 0, "x": 115.2, "y": 199.7, "visible": 1}
  ],
  "landmarks_image_px": [
    {"id": 0, "x": 520.3, "y": 410.8, "visible": 1}
  ],
  "roi_rect": {
    "x_center": 512.0,
    "y_center": 360.0,
    "width": 360.0,
    "height": 360.0,
    "rotation_rad": 0.42
  },
  "roi_corners_px": [
    [310.0, 230.0],
    [660.0, 380.0],
    [510.0, 730.0],
    [160.0, 580.0]
  ],
  "source": "cvat_reviewed"
}
```

实际文件中，有手样本的 landmark 列表必须是完整 21 点；无手样本 landmark 列表为空。

不要再单独生成 `palm_hand_matches.jsonl`。`crop_id`、`palm_det_id`、`hand_id` 已经在每条 crop 标注中表达了绑定关系，额外匹配文件会造成冗余。

### 6. 最终训练标注文件

最终训练标注文件仍然使用 JSONL：

```text
HandLandmarkerFab/data/labels/hand_training_labels.jsonl
```

它可以与 `hand_landmarks_reviewed.jsonl` 内容基本一致，但需要经过严格校验并补齐训练所需字段：

- 有手样本：`hand_presence.present=true`，必须有 21 个 `landmarks_crop_norm`。
- 无手样本：`hand_presence.present=false`，landmarks 为空，训练时 landmark loss weight 为 0，handedness loss weight 为 0。
- 所有样本保留 `landmarks_image_px`，方便在原始 tiff 上可视化。


## VI. 每个脚本职责

### `00_validate_images.py`

输入：

```text
data/images/
configs/autolabel.yaml
```

输出：

```text
data/qc/image_validation_report.json
```

要求：

- 检查图片是否可读取。
- 检查是否为单通道灰度或可转灰度。
- 检查是否为 `1280x720`。
- 不允许自动旋转图片。

### `01_export_palm_detections.py`

输入：

```text
data/images/
configs/autolabel.yaml
```

输出：

```text
data/palm/palm_detections.jsonl
data/qc/palm_detection_stats.json
```

要求：

- 根据 `palm.backend` 选择 `mediapipe_official` 或 `aethersign_onnx`。
- 默认使用 `mediapipe_official`。
- 两种 backend 输出完全相同的 `palm_detections.jsonl` schema。
- `aethersign_onnx` backend 使用 ONNX Runtime 加载 `model_opt.onnx`，输入预处理、decode、NMS 尽量与 `materials/preminilary/palm/infer_model_gray.py` 和板端 `palm_detector.cpp` 一致。
- `mediapipe_official` backend 使用官方 HandLandmarker 整图检测结果派生 palm-compatible detections，不要声称获得了官方内部 ROI。
- `detections` 长度只能是 0、1、2。
- 低分候选只能作为 `negative_candidates`，不得当正样本。

### `02_build_hand_roi_crops.py`

输入：

```text
data/images/
data/palm/palm_detections.jsonl
configs/autolabel.yaml
```

输出：

```text
data/roi_crops/images/*.png
data/roi_crops/hand_roi_crops_manifest.jsonl
data/qc/roi_crop_stats.json
```

要求：

- 使用 Palm bbox + p0/p9 构造 rotated/expanded Hand ROI。
- ROI 几何参考 `materials/preminilary/device/src/hand_landmarker.cpp`。
- 输出 crop 为 `256x256` 灰度图。
- 保存 ROI 四角点，方便反投影。
- 不要把 Palm bbox 直接 resize 成 Hand 输入。

### `03_run_mediapipe_on_rois.py`

输入：

```text
data/roi_crops/images/
data/roi_crops/hand_roi_crops_manifest.jsonl
configs/autolabel.yaml
```

输出：

```text
data/labels/hand_landmarks_mediapipe_raw.jsonl
data/labels/hand_landmarks_autolabel_draft.jsonl
data/qc/mediapipe_roi_stats.json
```

要求：

- 对每个 ROI crop 独立运行官方 MediaPipe HandLandmarker。
- `num_hands` 对 ROI crop 默认设为 1。
- 如果检测到手，保存 21 点 crop 坐标、反投影回原图的坐标、handedness。
- 如果未检测到手，保存 `hand_presence.present=false`，landmarks 为空。
- 不要假设 MediaPipe 输出内部 ROI。
- 不要伪造 hand presence score；如果 API 没有提供，立刻停止工作并向我汇报

### `04_export_cvat_xml.py`

输入：

```text
data/roi_crops/hand_roi_crops_manifest.jsonl
data/labels/hand_landmarks_autolabel_draft.jsonl
configs/autolabel.yaml
```

输出：

```text
data/review/cvat_autolabel.xml
data/review/cvat_upload_images/
data/qc/cvat_export_stats.json
```

要求：

- 将 crop 图片准备为 CVAT 可上传的数据。
- 将自动标注草稿导出为 `CVAT for images 1.1` XML。
- 其余要求同上

### `05_import_cvat_xml.py`

输入：

```text
data/review/cvat_reviewed.xml
data/roi_crops/hand_roi_crops_manifest.jsonl
configs/autolabel.yaml
```

输出：

```text
data/labels/hand_landmarks_reviewed.jsonl
data/qc/cvat_import_stats.json
```

要求：

- 读取 CVAT for images 1.1 XML。
- 每张 crop 最多一只手。
- 有 `hand_landmarks` points 且点数为 21 时，解析为 `hand_presence.present=true`。
- 无 landmarks 或带 `no_hand` tag 时，解析为 `hand_presence.present=false`。
- 根据 ROI 几何把 crop 坐标反投影回原始 `1280x720` 图，写入 `landmarks_image_px`。
- 如果点数不是 21、点越界、属性缺失，应写入 QC 报告并标记需要人工检查。

### `06_visualize_autolabels.py`

输入：

```text
data/images/
data/palm/palm_detections.jsonl
data/roi_crops/hand_roi_crops_manifest.jsonl
data/labels/hand_landmarks_reviewed.jsonl
```

输出：

```text
data/review/overlay_images/*.png
data/review/review_index.csv
data/qc/visualization_stats.json
```

要求：

- 在原图上绘制 Palm bbox。
- 绘制 Palm p0/p9。
- 绘制 rotated Hand ROI 四边形。
- 绘制反投影后的 21 点骨架。
- 标注 hand_presence、handedness、score、source。

### `07_finalize_training_labels.py`

输入：

```text
data/labels/hand_landmarks_reviewed.jsonl
data/roi_crops/hand_roi_crops_manifest.jsonl
configs/autolabel.yaml
```

输出：

```text
data/labels/hand_training_labels.jsonl
data/qc/final_training_label_stats.json
```

要求：

- 输出 crop 级训练标注 JSONL。
- 不生成旧 txt。
- 正样本必须有 21 个 `landmarks_crop_norm`。
- 负样本保留 `hand_presence.present=false`，landmarks 为空。
- 输出统计：正样本数、负样本数、左右手比例、点越界数、需要复查样本数。

## VII. 质量检查要求

请至少实现这些自动质检：

- 图片尺寸不是 `1280x720` 时报警。
- Palm bbox 越界或面积异常时报警。
- `detections` 数量超过 `max_detections` 时视为错误。
- Hand ROI 四角全部远离图像时报警。
- MediaPipe 或 CVAT 输出点越界比例过高时报警。
- 21 点顺序必须固定为 MediaPipe 手部 21 点。
- 同一 crop 超过 1 只手时报警。
- `handedness.score` 低于阈值时标记人工复核。
- `hand_presence.present=false` 但 Palm score 很高时标记人工复核。
- `hand_presence.present=true` 但 landmarks 数不是 21 时视为错误。

## VIII. 实现风格要求

- 使用 Python 3.8+。
- 使用 `pathlib` 管理路径。
- 使用 `argparse`，所有脚本必须能命令行运行。
- 路径必须从 yaml 配置读取，不要硬编码绝对路径。
- argparse 仍可提供命令行参数；如果与 yaml 冲突，命令行参数优先。
- 图像读取要支持中文路径，或至少给出清晰错误。
- 尽量把通用逻辑放在 `hand_autolabel/` 包里，脚本只负责调度。
- 对每个输出文件都写入统计报告。
- 对关键几何函数写简单单元测试或自检函数，尤其是 ROI crop 和 landmark 反投影。
- README 中写清楚完整运行顺序和 CVAT 复核步骤。

## IX. 预期运行顺序

README 中请给出类似命令：

```bash
python scripts/00_validate_images.py --config configs/autolabel.yaml
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml
python scripts/02_build_hand_roi_crops.py --config configs/autolabel.yaml
python scripts/03_run_mediapipe_on_rois.py --config configs/autolabel.yaml
python scripts/04_export_cvat_xml.py --config configs/autolabel.yaml

# 将 data/review/cvat_upload_images/ 和 data/review/cvat_autolabel.xml 上传到 CVAT
# 在 CVAT 人工复核 crop 图上的 21 点
# 从 CVAT 导出 CVAT for images 1.1 XML，保存为 data/review/cvat_reviewed.xml

python scripts/05_import_cvat_xml.py --config configs/autolabel.yaml
python scripts/06_visualize_autolabels.py --config configs/autolabel.yaml
python scripts/07_finalize_training_labels.py --config configs/autolabel.yaml
```

并且需要解释 `scripts/` 下每一个脚本的输入输出内容，如果有 `jsonl` 文件，需要详细解释每行的具体含义。

## X. 最终交付物

请最终交付：

```text
HandLandmarkerFab/
  README.md
  requirements.txt
  configs/autolabel.yaml
  hand_autolabel/*.py
  scripts/*.py
```

并说明：

- 哪些脚本已实现。
- 每个脚本输入输出。
- 如何运行一个小样例。
- 如何安装依赖。
- 如何把自动标注导入 CVAT。
- 如何从 CVAT XML 导回最终 JSONL。
- 当前无法自动完成、必须人工复核的部分。

## XI. 外部文档事实

请注意这些官方 MediaPipe 事实：

- HandLandmarkerOptions 中，`num_hands` 是最大检测手数。
- `min_hand_detection_confidence` 是 palm detection 成功所需的最低置信度。
- `min_hand_presence_confidence` 是 hand landmark 模型 hand presence score 的阈值。
- `min_tracking_confidence` 是视频/流模式中的 tracking IoU 阈值。
- HandLandmarkerResult 对外包含 handedness、image landmarks、world landmarks，不要假设它暴露内部 Palm ROI。


## XII. 备注

### 12.1 环境说明

本机 conda 环境：

```json
"Anaconda PowerShell": {
    "source": "PowerShell",
    "icon": "python",
    "args":[
        "-ExecutionPolicy",
        "ByPass",
        "-NoExit",
        "-Command",
        "& 'D:\\Anaconda\\shell\\condabin\\conda-hook.ps1'; conda activate 'D:\\Anaconda'"
    ]
    },
```

进入 conda 环境后，执行：

```powershell
conda activate anfab
```

后，即可进入拥有 `mediapipe` 工具的环境。

本机硬件配置：

CPU: Intel(R) Core(TM) i9-13900
RAM: 16 GB
GPU: NVIDIA GeForce RTX 4060 8GB
OS: Windows 11

可以租到的服务器配置：

OS: Linux
CPU: AMD EPYC 等多核性能 CPU, Xeon(R) Platinum 8470Q, Xeon(R) Silver 4214R 等
GPU: RTX 5090, RTX 4090, RTX 3090, V100, RTX 3060 等，显存从 12GB 到 48 GB 不等

### 12.2 运行说明

`data/images/` 下已经准备好了部分 `.tiff` 文件，供你进行小样本测试。请按照 README 中的运行顺序，依次执行每个脚本，并检查输出文件是否符合预期。