# HandLandmarkerFab 半自动标注工具链

本仓库实现用于重新训练 Hand Landmarker 的半自动标注流水线。已参考 `project-8.md`、`docs/HandFab/*` 和 `materials/preminilary/device` 中的板端 ROI 几何；如文档与本任务冲突，以 `prompt.md` 的最终方案和板端 `hand_landmarker.cpp` 为准。

输入图片必须已经是 `1280x720` 正向灰度 `.tiff`。脚本不会旋转原图。

## I. 目录结构

```
HandLandmarkerFab/
├─ configs/                  # 配置文件
│
├─ data/
│  ├─ images/                # 原始图片
│  ├─ palm/                  # Palm 检测结果
│  ├─ roi_crops/             # ROI crop 图片和 manifest
│  ├─ labels/                # MediaPipe 自动标注和复核结果
│  ├─ review/                # CVAT 上传、复核和可视化
│  └─ qc/                    # QC 报告
│
├─ materials/                # 板端模型和预训练模型
│
├─ scripts/                  # 半自动标注脚本
│  ├─ 00_validate_images.py
│  ├─ 01_export_palm_detections.py
│  ├─ 02_build_hand_roi_crops.py
│  ├─ 03_run_mediapipe_on_rois.py
│  ├─ 04_export_cvat_xml.py
│  ├─ 05_import_cvat_xml.py
│  ├─ 06_visualize_autolabels.py
│  └─ 07_finalize_training_labels.py
│
├─ hand_autolabel/
│  ├─ __init__.py
│  ├─ cvai_io.py
│  ├─ formats.py
│  ├─ image_io.py
│  ├─ mediapipe_roi_label.py
│  ├─ nms.py
│  ├─ palm_decode.py
│  ├─ palm_mediapipe.py
│  ├─ palm_onnx.py
│  ├─ projection.py
│  ├─ quality_checks.py
│  ├─ roi_gemotry.py
│  └─ visualization.py
│
├─ requirements.txt          # Python 依赖
└─ README.md                 # 本文档
```

## II. 半自动标注工程运行流程

## III. Quick Start

### 3.1 依赖安装

```powershell
D:\Anaconda\envs\anfab\python.exe -m pip install -r requirements.txt
```

说明：

- `mediapipe` 用于 `mediapipe_official` Palm 后端和 ROI 自动关键点标注。
- `onnxruntime` 仅在 `palm.backend=aethersign_onnx` 时需要。
- 当前 MediaPipe Python API 的 `HandLandmarkerResult` 不暴露 hand presence score。因此本工具链按用户确认后的方案处理：`hand_presence` 只写 `{"present": true/false}`，不写 `score` 字段，也不伪造分数。

### 3.2 配置

主配置文件是 `configs/autolabel.yaml`。所有路径均相对仓库根目录解析。当前工作区本身就是 `HandLandmarkerFab`，因此 `palm_model_onnx` 配置为：

```yaml
paths:
  palm_model_onnx: materials/preminilary/palm/model_opt.onnx
```

`palm.backend` 支持：

- `mediapipe_official`：默认。使用官方 MediaPipe 的公开手部关键点/handedness 结果，派生 palm-compatible detection。它不是官方内部 Palm ROI。
- `aethersign_onnx`：使用 `materials/preminilary/palm/model_opt.onnx`，按 14x14 与 7x7 共 490 anchors 解码、阈值、NMS、跨 head 抑制，最多输出 2 个有效 palm detections。

MediaPipe 只使用最新 Tasks API 写法。请在配置中填写 `mediapipe.model_asset_path`，指向官方 `hand_landmarker.task`；`.task` 模型文件可单独下载后放入仓库或本地路径。若该字段为空，`mediapipe_official` 和 ROI MediaPipe 自动标注会直接报错，不会尝试其他 MediaPipe API。

### 3.3 运行顺序

```powershell
python scripts/00_validate_images.py --config configs/autolabel.yaml
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml
python scripts/02_build_hand_roi_crops.py --config configs/autolabel.yaml
python scripts/03_run_mediapipe_on_rois.py --config configs/autolabel.yaml
python scripts/04_export_cvat_xml.py --config configs/autolabel.yaml

# 将 data/review/cvat_upload_images/ 和 data/review/cvat_autolabel.xml 上传到 CVAT
# 在 CVAT 中复核每张 256x256 crop 的 21 点
# 导出 CVAT for images 1.1 XML，保存为 data/review/cvat_reviewed.xml

python scripts/05_import_cvat_xml.py --config configs/autolabel.yaml
python scripts/06_visualize_autolabels.py --config configs/autolabel.yaml
python scripts/07_finalize_training_labels.py --config configs/autolabel.yaml
```

### 3.4 脚本输入输出

#### `00_validate_images.py`

- 输入：`data/images/`、`configs/autolabel.yaml`
- 输出：`data/qc/image_validation_report.json`
- 检查图片是否可读、是否为 `1280x720`、是否灰度或可转灰度；不旋转图片。

#### `01_export_palm_detections.py`

- 输入：`data/images/`、`configs/autolabel.yaml`
- 输出：`data/palm/palm_detections.jsonl`、`data/qc/palm_detection_stats.json`
- 每行是一张原图，包含 `detections` 和 `negative_candidates`。有效检测最多 2 个。

#### `palm_detections.jsonl` 每行含义：

- `image/width/height`：原图文件名和尺寸。
- `detections`：有效 palm detections；每项包含 `palm_det_id`、`score`、`bbox_norm/bbox_px`、`keypoints_norm/keypoints_px.p0/p9`、`source/head`。
- `negative_candidates`：低分候选，只能作为负样本候选，不作为正样本。

#### `02_build_hand_roi_crops.py`

- 输入：`data/images/`、`data/palm/palm_detections.jsonl`
- 输出：`data/roi_crops/images/*.png`、`data/roi_crops/hand_roi_crops_manifest.jsonl`、`data/qc/roi_crop_stats.json`
- 使用 palm bbox + p0/p9 构造 rotated/expanded ROI，参数与板端默认一致：`scale_x=1.8`、`scale_y=1.8`、`shift_y=-0.1`，输出 `256x256` 灰度 crop。

#### `hand_roi_crops_manifest.jsonl` 每行含义：

- `crop_id`：crop 级唯一 ID。
- `image/palm_det_id`：绑定回原图和 Palm detection。
- `crop_path`：ROI crop 图片路径。
- `roi_rect/roi_corners_px`：反投影和可视化所需的 ROI 几何。
- `palm_valid/palm_score`：该 crop 来自有效 detection 还是低分负候选。

#### `03_run_mediapipe_on_rois.py`

- 输入：`data/roi_crops/images/`、`data/roi_crops/hand_roi_crops_manifest.jsonl`
- 输出：`data/labels/hand_landmarks_mediapipe_raw.jsonl`、`data/labels/hand_landmarks_autolabel_draft.jsonl`、`data/qc/mediapipe_roi_stats.json`
- 每个 crop 独立运行 MediaPipe，最多一只手；有手则保存 21 点 crop 坐标和原图反投影坐标，无手则保留负样本。

`hand_landmarks_autolabel_draft.jsonl` 每行含义：

- `crop_id/image/crop_path/palm_det_id`：crop 与上游 Palm 的绑定关系。
- `hand_presence`：只包含 `present`，不包含 score。
- `handedness`：`{"label": "Left/Right/unknown", "score": ...}`。
- `landmarks_crop_norm`：21 点在 crop 内归一化坐标。
- `landmarks_crop_px`：21 点在 `256x256` crop 内像素坐标。
- `landmarks_image_px`：21 点反投影回 `1280x720` 原图坐标。

#### `04_export_cvat_xml.py`

- 输入：`hand_roi_crops_manifest.jsonl`、`hand_landmarks_autolabel_draft.jsonl`
- 输出：`data/review/cvat_autolabel.xml`、`data/review/cvat_upload_images/`、`data/qc/cvat_export_stats.json`
- 将 crop 图片复制到 CVAT 上传目录，并导出 `CVAT for images 1.1` XML。有手 crop 写一个 `points` shape，无手 crop 写 `no_hand` tag。

#### `05_import_cvat_xml.py`

- 输入：`data/review/cvat_reviewed.xml`、`hand_roi_crops_manifest.jsonl`、`hand_landmarks_autolabel_draft.jsonl`
- 输出：`data/labels/hand_landmarks_reviewed.jsonl`、`data/qc/cvat_import_stats.json`
- 解析 CVAT 复核后的 21 点，并用 manifest 恢复 ROI 几何、反投影坐标和原始元数据。

#### `06_visualize_autolabels.py`

- 输入：原图、Palm JSONL、ROI manifest、reviewed JSONL
- 输出：`data/review/overlay_images/*.png`、`data/review/review_index.csv`、`data/qc/visualization_stats.json`
- 在原图上绘制 Palm bbox、p0/p9、rotated ROI 和反投影后的 21 点骨架。

#### `07_finalize_training_labels.py`

- 输入：`hand_landmarks_reviewed.jsonl`、ROI manifest
- 输出：`data/labels/hand_training_labels.jsonl`、`data/qc/final_training_label_stats.json`
- 严格校验训练样本。正样本必须有 21 个 `landmarks_crop_norm`；负样本 landmarks 为空，并写入 loss weight：`landmark_loss_weight=0`、`handedness_loss_weight=0`。

### 3.5 CVAT 复核

1. 运行到 `04_export_cvat_xml.py`。
2. 在 CVAT 创建 image task，上传 `data/review/cvat_upload_images/` 中的 crop 图片。
3. 导入 `data/review/cvat_autolabel.xml` 作为初始标注。
4. 每张 crop 最多保留一个 `hand_landmarks` points shape，点数必须为 21，点顺序必须是 MediaPipe 21 点顺序。
5. 无手 crop 删除 points，并保留或添加 `no_hand` tag。
6. 导出 `CVAT for images 1.1` XML，保存为 `data/review/cvat_reviewed.xml`。
7. 运行 `05_import_cvat_xml.py` 和 `07_finalize_training_labels.py`。

### 3.6 小样例

当前 `data/images/` 已有一批 `.tiff`。先运行：

```powershell
python scripts/00_validate_images.py --config configs/autolabel.yaml
```

然后选择 Palm 后端：

```powershell
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml --backend mediapipe_official
```

若要使用 ONNX 后端，请先确认已安装 `onnxruntime`，再改为：

```powershell
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml --backend aethersign_onnx
```

### 3.7 需要人工复核的部分

- MediaPipe 自动 21 点只是草稿，遮挡、交叉手、边缘手、强暗光/反光样本必须人工复核。
- handedness 低置信度、`present=false` 但 Palm score 很高、点越界、同一 crop 多手等情况会写入 QC 报告并标记 `needs_review`。
- MediaPipe 不提供 hand presence score，本项目不会自动补造这个分数。
