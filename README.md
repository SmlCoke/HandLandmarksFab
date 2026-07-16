# HandLandmarkerFab 半自动标注工具链

本仓库实现用于重新训练 Hand Landmarker 的半自动标注流水线。

项目背景及前置参考材料：

- `project-8.md`、`docs/HandFab/*` 
- `materials/preminilary/device` 中的板端 ROI 几何；

如文档与本任务冲突，以本文档为准。

常规 00～06 输入图片必须已经是 `1280x720` 正向灰度 `.tiff`，脚本不会旋转原图。唯一例外是 3.8 节的 Dragon external-Gold adapter：它只读 JPEG EXIF 并生成派生 ROI，不修改原图。

## I. 目录结构

```
HandLandmarkerFab/
├─ configs/                  # 配置文件
│
├─ data/
│  ├─ images/                # 原始图片
│  ├─ 01_palm/               # Palm 检测结果
│  ├─ 02_roi_crops/          # ROI crop 图片、Mediapipe Hand Landmark 标注初稿、转化的 xml
│  ├─ 03_reviewed/           # 人工复核结果
│  ├─ 04_visualization/      # 可视化结果
│  ├─ 05_labels/             # 最终训练标签
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
│  ├─ 07A_finalize_training_labels.py
│  ├─ 07B_finalize_evaluation_labels.py
│  └─ 08_finetune_gold.py       # Dragon、finetune CVAT Gold 与 Gold 聚合
│
├─ hand_autolabel/
│  ├─ __init__.py
│  ├─ cvat_io.py
│  ├─ finalization.py
│  ├─ finetune_gold.py       # registry、Dragon、strict Gold source 与 aggregate 内核
│  ├─ formats.py
│  ├─ image_io.py
│  ├─ mediapipe_roi_labeler.py
│  ├─ mediapipe_roi_visualization.py
│  ├─ nms.py
│  ├─ palm_decode.py
│  ├─ palm_mediapipe.py
│  ├─ palm_onnx.py
│  ├─ projection.py
│  ├─ quality_checks.py
│  ├─ roi_geometry.py
│  └─ visualization.py
│
├─ requirements.txt          # Python 依赖
└─ README.md                 # 本文档
```

## II. 半自动标注工程运行流程

## III. Quick Start

### 3.1 依赖安装

```powershell
conda create -n anfab python=3.11
conda activate anfab
python -m pip install -r requirements.txt
```

说明: 

- `mediapipe` 用于 `mediapipe_official` Palm 后端和 ROI 自动关键点标注。
- `onnxruntime` 仅在 `palm.backend=aethersign_onnx` 时需要。
- 当前 MediaPipe Python API 的 `HandLandmarkerResult` 不暴露 hand presence score。因此本工具链按用户确认后的方案处理: `hand_presence` 只写 `{"present": true/false}`，不写 `score` 字段，也不伪造分数。

### 3.2 配置

主配置文件是 `configs/autolabel.yaml`。配置中的相对路径均以仓库根目录解析；绝对路径保持不变。

Val/Test 的三份 00–06 配置使用 `${HAND_DATA_ROOT:../autodl-tmp}` 作为数据根目录；该根目录下应包含 `vals_data/`、`vali_data/` 和 `test_data/`。未设置 `HAND_DATA_ROOT` 时仍回退到原来的 `../autodl-tmp`。可按使用场景选择以下方式设置：

```powershell
# 当前 PowerShell 会话；直接运行 Python 和 make 都会读取
$env:HAND_DATA_ROOT = "D:/datasets/hand"

# 仅覆盖这一次 make 调用
make validate_images_vals HAND_DATA_ROOT=D:/datasets/hand

# 每台机器持久设置：复制模板后编辑 Makefile.local（该文件已被 Git 忽略）
Copy-Item Makefile.local.example Makefile.local
```

在 Linux/macOS shell 中可使用 `export HAND_DATA_ROOT=/data/hand`。Windows 路径建议使用正斜杠；若路径含空格，优先写入 `Makefile.local`。

`finalize_train.yaml`、`finalize_val.yaml`、`finalize_test.yaml` 将 `HAND_DATA_ROOT` 解释为统一的 HLML 数据根目录；其下应包含 `peak_train_data/`、`soar_train_data/`、`eval_sources/` 以及各合并输出目录。Makefile 默认使用 `../autodl-tmp/TrainFab/HLML-2.0`，并将该变量导出给 Python。配置解析同时支持 `${NAME:default}` 与 shell 风格的 `${NAME:-default}`。

**关键字段及其含义**

- `palm` 字段:
    - `backend`: Palm detection 来源。可取值: 
        - `mediapipe_official` **默认**。表示使用官方 MediaPipe 的公开手部关键点/handedness 结果，派生 palm-compatible detection。它不是官方内部 Palm ROI。
        - `aethersign_onnx` 按 14x14 与 7x7 共 490 anchors 解码、阈值、NMS、跨 head 抑制，最多输出 2 个有效 palm detections。
    - `input_size`: `aethersign_onnx` Palm 模型输入尺寸，当前为 `224x224`。`mediapipe_official` 后端可以不使用这个值。
    - `score_threshold`: **Palm 候选进入有效 `detections` 的最低分数，默认 0.5**。低于此阈值的候选不能作为正样本，而标记为**负样本候选**: `negative_candidates`。
    - `nms_iou_threshold`: 同一 head 内做 NMS 时的 IoU 阈值，默认 0.3。**IoU 高于该阈值的重叠候选会被抑制**。
    - `cross_head_suppress_iou`: `14x14` 与 `7x7` **两个 head 之间做跨 head 抑制时的 IoU 阈值**，默认0.35。
    - `max_detections`: 每张原图最多保留多少个有效 Palm detections。本项目为最多 2 个，但不保证一定有 2 个。
    - `keep_low_score_candidates_for_negatives`: 是否保留低分候选作为未来负样本候选。它们不进入 `detections`，只能进入 `negative_candidates`。
    - `negative_candidate_threshold`: 低分负候选的最低分数，默认 0.15。**低于该阈值的候选通常直接丢弃**。
    - `compatible_bbox_expand`: 只用于 `palm.backend=mediapipe_official`, 默认 0.25。ediaPipe Tasks HandLandmarker 对外只给出 21 个手部关键点和 handedness，不暴露官方内部 Palm ROI。因此我们需要从公开的 21 点中派生一个“palm-compatible detection”，让后续 ROI crop 流程继续使用统一的 Palm schema。派生逻辑就是对 [0,1,5,9,13,17] 6 个关键点构建外接框，并且按照 `compatible_bbox_expand=e` 进行扩大（在 palm 近似外接框四周各扩展 `e` 倍的宽高）
- `mediapipe` 字段
    - `mediapipe.model_asset_path`: 指向官方 `hand_landmarker.task`；`.task` **模型文件可单独下载后放入仓库或本地路径**。若该字段为空，`mediapipe_official` 和 ROI MediaPipe 自动标注会直接报错，不会尝试其他 MediaPipe API。
    - `num_hands`: 最多检测几只手。对每个 `256x256` ROI crop 运行时建议为 `1`，因为每个 crop 理论上最多一只手；对整图 official backend 派生 Palm detections 时可以临时覆盖为 `2`。
    - `min_hand_detection_confidence`: **Palm detection 被认为成功的最低置信度阈值**。
    - `min_hand_presence_confidence`: **Hand landmark 模型内部 hand presence score 的最低阈值**。官方文档说明，在视频/直播模式中，**如果 hand presence 低于该阈值，会重新触发 palm detection**。
    - `min_tracking_confidence`: 视频/直播模式下的跟踪成功阈值，**本质上是当前帧与上一帧手框的 IoU 阈值**。当前半自动标注主要使用 IMAGE 模式，这个参数通常影响不大，但保留在配置里保持接口完整。

**不同配置文件职责**

| 配置 | 作用 |
|---|---|
| `configs/autolabel_train.yaml` | 单个 Train 来源运行 00–06 |
| `configs/finalize_train.yaml` | 07A 合并 pretrain 来源、自动 namespace、分型和降采样；不承载 finetune Gold |
| `configs/autolabel_val.yaml` | shared Val，即 `vals_data`，运行 00–06 |
| `configs/autolabel_vali.yaml` | independent Val，即 `vali_data`，运行 00–06 |
| `configs/finalize_val.yaml` | 07B 合并 Peak shared + Soar shared + 当前路线 independent Val |
| `configs/autolabel_test.yaml` | shared Test 运行 00–06 |
| `configs/finalize_test.yaml` | 07B 汇总并冻结最终 Test |
| `configs/dragon_gold.yaml` | 将 Dragon legacy 标注转换为 finetune-only external Gold |
| `configs/finetune_gold.yaml` | b/c/e Gold subset 的专用 04 与 strict 05 |
| `configs/finalize_finetune.yaml` | 自动发现并认证 a/b/c/e Gold source，生成 HLMF Gold aggregate |

### 3.3 运行顺序

首先，将待处理图片放入 `data/images` 中，然后：

```powershell
python scripts/00_validate_images.py --config configs/autolabel.yaml
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml
python scripts/02_build_hand_roi_crops.py --config configs/autolabel.yaml
python scripts/03_run_mediapipe_on_rois.py --config configs/autolabel.yaml
python scripts/04_export_cvat_xml.py --config configs/autolabel.yaml

# 将 data/02_roi_crops/images 和 data/02_roi_crops/cvat_autolabel.xml 上传到 CVAT
# 在 CVAT 中复核每张 256x256 crop 的 21 点
# 导出 CVAT for images 1.1 XML，保存为 data/03_reviewed/cvat_reviewed.xml

python scripts/05_import_cvat_xml.py --config configs/autolabel.yaml
python scripts/06_visualize_autolabels.py --config configs/autolabel.yaml

# Train 使用 07A，Val/Test 使用 07B；完整命令见下文第 (7) 节
```

### 3.4 脚本输入输出

#### (0) `00_validate_images.py`

- **输入**: `data/images/`、`configs/autolabel.yaml`
- **输出**: `data/qc/image_validation_report.json`
- **功能**: 检查图片是否可读、是否为 `1280x720`、是否灰度或可转灰度；不旋转图片。

#### (1) `01_export_palm_detections.py`

- **输入**: `data/images/`、`configs/autolabel.yaml`
- **输出**: `data/01_palm/palm_detections.jsonl`、`data/qc/palm_detection_stats.json`
- **功能**: 对每张图进行 plam detection, 从生成的 392 个 14x14 anchors 和 98 个 7x7 anchors 中**解码**、根据**置信度阈值**进行筛选、**NMS**、**跨 head 抑制**，最多输出 2 个有效 palm detections。**低 score 候选会被保留为负样本候选**。

**输出标注** `palm_detections.jsonl` 每行含义: 

- `image/width/height`: 原图文件名和尺寸。

- 当前 Palm detection 级信息: 
    - `detections`: 有效 palm detections；每项包含:
        - `palm_det_id`: palm detection 级唯一 ID。
        - `score`: palm detection 分数。
        - `bbox_norm/bbox_px`: 边界框坐标 (`norm`代表相对原始 tiff 图的归一化值，`px`代表原图中的像素坐标)，顺序分别为：`[xmin, ymin, xmax, ymax]`。
        - `keypoints_norm/keypoints_px.p0/p9`: 关键点坐标。
        - `source`: 检测模型来源。
        - `head`: 14x14 或 7x7。
    - `negative_candidates`: 低分候选，只能作为负样本候选，不作为正样本。

#### (2) `02_build_hand_roi_crops.py`

- **输入**: `data/images/`、`data/01_palm/palm_detections.jsonl`
- **输出**: `data/02_roi_crops/images/*.png`、`data/02_roi_crops/hand_roi_crops_manifest.jsonl`、`data/qc/roi_crop_stats.json`
- **功能**: 使用 palm bbox + p0/p9 进行旋转、平移、扩张构造 Hand ROI，参数与板端默认一致: `scale_x=1.8`、`scale_y=1.8`、`shift_y=-0.1`，输出 `256x256` 灰度 crop。注意，由于存在旋转、平移和扩张取样操作，因此旋转/平移/扩大后的 ROI 超出原图边界，**crop 中超出原图的部分会被填充为黑色**: `cv2.warpAffine(..., borderMode=cv2.BORDER_CONSTANT, borderValue=0)`。这**与板端逻辑一致**。板端 `SampleBilinear()` 在采样点**远离图像或像素越界时返回 0**，因此超出原图的区域也是黑色。

`hand_roi_crops_manifest.jsonl` 每行都对应 `data/02_roi_crops/images/` 下的着一张 `256x256` ROI crop，含义: 

- 前序 image 级信息: 
    - `image`: 绑定回原图。
- 前序 Palm detection 级信息: 
    - `palm_det_id`: 绑定回原 Palm detection。
    - `palm_valid`: 该 crop 来自有效有效检测结果 `detections` 还是低分负候选 `negative_candidates`。
    - `palm_score`: Palm detection 的分数。
- **当前 Hand ROI 级信息**: 
  - `crop_id`: crop 级唯一 ID。
  - `crop_path`: ROI crop 图片路径。
  - `roi_rect`: 属于原图 1280x720 坐标系，x_center/y_center 是 ROI 中心点, width/height 是 ROI 在原图像素坐标中的宽高, rotation_rad 是 ROI 相对原图坐标系的旋转角，单位弧度。
  - `roi_corners_px`: 属于原图  1280x720 坐标系，是由 roi_rect 计算出的四个角点，顺序固定为: C0 = top_left, C1 = top_right, C2 = bottom_right, C3 = bottom_left
  - `output_size`: 输出尺寸大小，默认 `[256,256]`。

#### (3) `03_run_mediapipe_on_rois.py`

- **输入**: `data/02_roi_crops/images/`、`data/02_roi_crops/hand_roi_crops_manifest.jsonl`
- **输出**: `data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl`、`data/qc/mediapipe_roi_stats.json`
- **功能**: 每个 crop 独立运行 MediaPipe，最多一只手；有手则保存 21 点 crop 坐标和原图反投影坐标，无手则保留负样本。

通过 Makefile 变量可以在 03 完成后立即生成 ROI 关键点可视化，默认关闭：

```powershell
make run_mediapipe_train VISUALIZE_MEDIAPIPE_ROIS=1
```

`1/true/yes/on` 均表示开启。可视化只读取 `02_roi_crops/hand_landmarks_autolabel_draft.jsonl` 和 `02_roi_crops/images/`，并按 draft 中 `crop_path` 的文件名在当前 `images/` 下重定位图片；它不读取原始图或 `01_palm/`。结果写入 `02_roi_crops/hand_landmarks_visualization/*.png`，其中包含 21 点编号、骨架、handedness、点数和越界点数；`present=false` 的 ROI 也会输出并标红。该功能只增加 PNG，不修改或增加任何 JSONL/XML 字段。

`hand_landmarks_autolabel_draft.jsonl` 每行代表一个 crop 小图的标注信息，具体含义: 

- 前序 image 级信息: 
    - `image`: 绑定回原图。
- 前序 Palm detection 级信息: 
    - `palm_det_id`: 绑定回原 Palm detection。
    - `palm_valid`: 该 crop 来自有效有效检测结果 `detections` 还是低分负候选 `negative_candidates`。
    - `palm_score`: Palm detection 的分数。
- 前序 Hand ROI 级信息: 
    - `crop_id`: crop 级唯一 ID。
    - `crop_path`: ROI crop 中间图片路径。
    - `roi_rect`:
    - `roi_corners_px`:
- **当前 Landmarks 级信息**: 
    - `hand_presence`: 只包含 `present`，不包含 score。`present` 取值只有 `true/false`，即有手/无手。该变量来自于官方 MediaPipe Hand Landmarker，后续可以人工复核时进行修改。
    - `handedness`: `{"label": "Left/Right/unknown", "score": ...}`。
    - `landmarks_crop_norm`: 21 点在 crop 内归一化坐标。
    - `landmarks_crop_px`: 21 点在 `256x256` crop 内像素坐标。
    - `landmarks_image_px`: 21 点反投影回 `1280x720` 原图坐标。

#### (4) `04_export_cvat_xml.py`

- **输入**: `hand_roi_crops_manifest.jsonl`、`hand_landmarks_autolabel_draft.jsonl`
- **输出**: `data/02_roi_crops/cvat_autolabel.xml`、`data/03_reviewed/cvat_upload_images/`、`data/qc/cvat_export_stats.json`
- **功能**: 将 crop 图片复制到 CVAT 上传目录，并导出 `CVAT for images 1.1` XML。有手 crop 写一个 `points` shape，无手 crop 写 `no_hand` tag。

#### (5) `05_import_cvat_xml.py`

- **输入**: `data/03_reviewed/cvat_reviewed.xml`、`hand_roi_crops_manifest.jsonl`、`hand_landmarks_autolabel_draft.jsonl`
- **输出**: `data/03_reviewed/hand_landmarks_reviewed.jsonl`、`data/qc/cvat_import_stats.json`
- **功能**: 解析 CVAT 复核后的 21 点，**并用之前的 `.jsonl` 文件恢复 ROI 几何、反投影坐标和原始元数据等在 cvat 1.1 标注文件中被删除的信息**。

#### (6) `06_visualize_autolabels.py`

- **输入**: 原图、Palm JSONL、ROI manifest、reviewed JSONL
- **输出**: `data/04_visualization/crop_images`, `data/04_visualization/global_images`, `data/04_visualization/review_index.csv`
- **功能**: 在原图上绘制 Palm bbox、p0/p9、rotated ROI 和反投影后的 21 点骨架。

#### (7) `07A_finalize_training_labels.py` / `07B_finalize_evaluation_labels.py`

正式数据集按用途分为三类路由：

| 流程 | 数据集 | 主要处理 | 命令 |
|---|---|---|---|
| 07A | Pretrain | 合并多个 pseudo 来源并为 ID 加 namespace；结构校验、样本分型、伪标签 QC、同原图重复 ROI 降采样 | `python scripts/07A_finalize_training_labels.py --config configs/finalize_train.yaml --stage pretrain` |
| 08 finalize | Finetune Gold | 自动发现已发布的 a/b/c/e Gold source；认证 descriptor、逐图 SHA 和全覆盖 Gold；跨 source 冲突检查后输出 HLMF Gold aggregate | `make finalize_train_finetune` |
| 07B | Val/Test | 汇总一个或多个已人工复核的来源；要求 manifest、CVAT XML、reviewed JSONL 一一覆盖；严格校验 Gold；分离 `ignore_for_training`；不按 teacher/Palm 分数筛样本、不降采样 | `python scripts/07B_finalize_evaluation_labels.py --config configs/finalize_val.yaml --split val` 或使用 `configs/finalize_test.yaml --split test` |

Val/Test 的来源和 namespace 规则：

| 最终数据集 | 07B 输入 | namespace 处理 | 最终输出目录 |
|---|---|---|---|
| Val | Peak `vals_data` + Soar `vals_data` + 当前路线自己的 `vali_data` | 07B 按每个 source 的 `dataset_id` 强制生成 namespace | `../autodl-tmp/val_merged/` |
| Test | Peak `test_data` + Soar `test_data`，最终 100% 共享 | 同上；原始文件名允许重复 | `../autodl-tmp/test_merged/` |
| Pretrain | `configs/finalize_train.yaml` 中的多个来源 | 07A 按 `dataset_id` 自动生成全局 namespace | `train_pretrain_merged/` |
| Finetune Gold | `finetune/<ID>/sources/gold/*/finetune_source.json` | 08 按 descriptor 自动发现；跨 source 再做 parent/global/SHA 冲突门禁 | `finetune/<ID>/hmlf_gold_merged/` |

两人的 shared Val 分别完成 05 导入，当前路线的 independent Val 也完成 05 导入后，再将三个 source 登记到 `configs/finalize_val.yaml` 并运行一次 `make finalize_val`。两人的 Test source 同理登记到 `configs/finalize_test.yaml` 后运行一次 `make finalize_test`。07B 不写死比例，而是在报告中按实际有效 ROI 数量统计 owner/source/partition 构成。

完整的数据制作命令、配置示例和人工复核步骤见：[数据集制作操作手册](docs/flow_and_interface/dataset_preparation_workflow.md)。

07A 按 `palm_valid × hand_presence` 划分四种训练样本：

| `sample_type` | `palm_valid` | presence | 用途 |
|---|---:|---:|---|
| `POS_RUNTIME` | true | true | 部署路径正样本 |
| `POS_LOW_PALM` | false | true | Palm 漏检但 teacher/Gold 找到手 |
| `NEG_RUNTIME_CANDIDATE` | true | false | Palm 误检形成的 hard negative |
| `NEG_LOW_PALM_CANDIDATE` | false | false | 低分背景负样本 |

| 输出 | 说明 |
|---|---|
| `train_pretrain_merged/05_labels/hand_train_catalog_pretrain.jsonl` | 07A pretrain 全量审计目录 |
| `train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl` | pretrain loader 使用的 canonical 清单 |
| `train_pretrain_merged/05_labels/hand_training_excluded_pretrain.jsonl` | pretrain 未入选样本及原因 |
| `finetune/<ID>/hmlf_gold_merged/05_labels/*_finetune.jsonl` | HLMF 聚合后的 Gold-only 输入；HLML 后续再与 replay 合并 |
| `hand_validation_labels.jsonl` / `hand_test_labels.jsonl` | 07B 严格 Gold 主评测集 |
| `hand_val_ignored.jsonl` / `hand_test_ignored.jsonl` | 07B 排除的歧义/不可可靠标注样本 |
| `qc/finalize_*_report.json` | 覆盖率、样本分布、排除原因和输出 SHA-256；fatal 时不覆盖已有 canonical 文件 |

07A Train 合并完成后可选择绘制 canonical 清单中全部 `selection_action=include` 的 ROI，默认关闭：

```powershell
make finalize_train_pretrain VISUALIZE_FINALIZED_TRAIN_ROIS=1
```

可视化是 07A 的最后一步，不新增流程脚本；07B Val/Test 不执行该功能。输出位于与对应 `05_labels/`、`qc/` 同级的 `hand_landmarks_visualization/<dataset_id>/*.png`，按所有配置来源分别保存。绘制前会检查每个 source 的 `crop_images_dir` 以及每条入选记录对应的 ROI 图片；任一来源或图片缺失都会输出 `ERROR` 并使命令失败。该步骤只读取 canonical JSONL 并写 PNG，不修改任何 JSONL/XML。

07A JSONL 新增的主要训练接口字段如下：

| 字段组 | 字段 | 训练端用法 |
|---|---|---|
| 来源与追踪 | `dataset_id`、`source_crop_id`、`global_crop_id`、`source_group_id`、`annotation_provenance`、`supervision_tier` | 防止多人数据 ID 冲突，区分 pseudo 与 human Gold |
| 分类与质量 | `sample_type`、`quality_tier`、`quality_flags`、`selection_action` | 只读取 `selection_action=include` 的 canonical 文件；QC/抽检使用 catalog |
| 去重 | `duplicate_cluster_id`、`duplicate_cluster_size`、`duplicate_rank` | 追溯同一原图的重复 ROI 代表样本 |
| 采样 | `sampling_bucket`、`sampling_weight` | loader 先按 Gold/pseudo，再按四种 sample type 做两级采样；该权重不乘入 loss |
| loss | `supervision_loss_weight`、三个 `*_quality_weight`、三个 `*_loss_weight` | `*_loss_weight` 仅为 0/1 head mask；有效 head 权重 = mask × supervision × 对应 quality weight |

`pretrain` 继续使用原来的 MediaPipe pseudo 流程。Finetune 不再把 Gold 手工追加到 `configs/finalize_train.yaml`：HLMF 只发布经过认证的 Gold aggregate，HLML 再把该 aggregate 与 pretrain replay 合并。Val 用于调参和选 checkpoint，Test 只用于最终冻结评测。

### 3.5 CVAT 复核

1. 运行到 `04_export_cvat_xml.py`。
2. 在 CVAT 创建 image task，上传 `data/02_roi_crops/images/` 中的 crop 图片，然后：
   1. "Add Label"，创建 "tag"，命名为 "no_hand"。进行标注时，如果发现对应图片没有手，则打上这个 tag。如果发现没有手，但是有错误的标注，请删除标注并打上 "no_hand" tag。
   2. "Add Label"，创建 "Left" 和 "Right" 两个 tag，分别对应左手和右手。进行标注时，注意甄别左右手，打上对应的 label。
   3. "Add Label"，创建 "tag"，命名为 "ignore_for_training"，进行标注时，如果发现 MediaPipe 自动标注的 21 点有严重错误，或者遮挡、交叉手、边缘手、强暗光/反光样本等情况，并且时间紧迫不值得人工修复，请打上这个 tag。在 07 阶段，所有打上这个 tag 的样本都会被忽略，不参与训练。
   4. "Setup skeleton"，创建 21 关键点，命名为 "hand_landmarks"，按照 MediaPipe 21 点顺序创建 21 个关键点。
   > 注意：`configs/cvat_label.json` 是上述已经创建好的标注工具的 json 格式，如果不想自己手动创建，可以直接在 new task 界面的 `raw` 栏目中，将 json 文件中的内容复制进去即可。
3. 将 `data/02_roi_crops/cvat_autolabel.xml` 作为初始标注上传到 CVAT。
4. 每张 crop 最多保留一个 `hand_landmarks` points shape，点数必须为 21，点顺序必须是 MediaPipe 21 点顺序。
5. 无手 crop 删除 points，并保留或添加 `no_hand` tag。
6. 导出 `CVAT for images 1.1` XML，保存为 `data/03_reviewed/cvat_reviewed.xml`。
7. 运行 `05_import_cvat_xml.py`，然后根据数据集用途运行 `07A_finalize_training_labels.py` 或 `07B_finalize_evaluation_labels.py`。

### 3.6 小样例

当前 `data/images/` 已有一批 `.tiff`。先运行: 

```powershell
python scripts/00_validate_images.py --config configs/autolabel.yaml
```

然后选择 Palm 后端: 

```powershell
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml --backend mediapipe_official
```

若要使用 ONNX 后端，请先确认已安装 `onnxruntime`，再改为: 

```powershell
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml --backend aethersign_onnx
```

### 3.7 需要人工复核的部分

- MediaPipe 自动 21 点只是草稿，遮挡、交叉手、边缘手、强暗光/反光样本必须人工复核（即 CVAT 自行复核）。
- handedness 低置信度、`present=false` 但 Palm score 很高、点越界、同一 crop 多手等情况会写入 QC 报告并标记 `needs_review`。最终标注文件 `data\05_labels\hand_training_labels.jsonl` 亦会为每个样本的信息中标明这个字段，**在训练阶段，建议直接忽略所有 `needs_review:true` 的样本，即不让它们参与训练**。

### 3.8 Finetune Gold 制作与人工操作

这一节只用于 HLMF→HLML v1.0 finetune。原有 Train/Val/Test 的 00～07 流程保持不变。所有新命令都使用独立的 `HAND_FINETUNE_ID`，不会覆盖 pretrain 产物。

先设置中央数据根和 finetune ID：

```powershell
$env:HAND_DATA_ROOT = "D:/datasets/HLML-2.0"
$env:HAND_FINETUNE_ID = "v2-finetune-r1"
$env:HAND_PRETRAIN_ID = "v2-pretrain-r3"
```

Linux 服务器使用同名 `export`。下文所有目录均位于 `${HAND_DATA_ROOT}/finetune/${HAND_FINETUNE_ID}/`。

#### 3.8.1 导入 Dragon external Gold

Dragon 原始目录应直接包含 `images/`、`annotations_hand.txt`、`annotations_palm.txt` 和 `README.md`。不需要建立另一个 HLMF 仓库，也不要旋转或修改原 JPEG。

```powershell
$env:DRAGON_RAW_ROOT = "D:/CICIEC/datasets/HandViolenceEnhanced0716/dragon"
make prepare_dragon_gold
```

程序自动验证三份输入 SHA、读取 JPEG EXIF Orientation=6、转成逻辑 1280×720、严格建立 Hand–Palm 唯一对应、按 1.8/1.8 和 `shift_y=-0.1` 裁剪 ROI、投影 21 点并生成 source descriptor。Dragon 不提供 handedness，因此所有行写 `unknown`，训练时 handedness mask 为 0。缺失的 Palm score 以 `0.5` 兼容哨兵保存，同时明确写入 `palm_score_observed=false`，不能把它理解为真实置信度。

查看：

```text
finetune/<ID>/sources/gold/dragon_gold_0716_v1/
├─ finetune_source.json
├─ 02_roi_crops/images/
├─ source_images/             # 3565 张参与匹配的原 JPEG 审计副本/硬链接
├─ 03_reviewed/hand_landmarks_reviewed.jsonl
├─ 03_reviewed/ignored.jsonl
└─ qc/
   ├─ gold_source_report.json
   ├─ crop_images_sha256.jsonl
   ├─ source_images_sha256.jsonl
   └─ overlays/                 # 固定 64 张抽检图
```

当前 Dragon 版本的硬验收值为：5191 条匹配 Gold、5189 条可训练、2 条关键点越出 ROI 并进入 ignored；任一数字或输入 SHA 不符都会停止发布。参与匹配的 3565 张原 JPEG 会以不重编码的 hardlink/copy 放进 source package 的 `source_images/`，descriptor 使用相对只读根，HLML 会逐张复核 SHA；因此整个 Gold source 可以跨机器搬运，不依赖原始绝对路径。

#### 3.8.2 导出 b/c 人工 Gold 子集

先发布一次只读的 pretrain source lookup。这个动作不会改写已经冻结的 pretrain labels：

```powershell
make build_pretrain_source_registry
```

结果是 `train_pretrain_merged/qc/pretrain_source_registry.jsonl` 及其 report。每行认证一个 `global_crop_id` 对应的原 manifest、MediaPipe draft、物理 ROI 和 SHA；HLML 用它恢复 b/c 的父数据，而不猜测目录。registry 中路径是生成机器上的绝对路径；中央数据根迁移后应重新运行本命令，不要手改 JSONL。

随后 HLML 自动选择 b（从人工删除的困难样本中回看）或 c（teacher–student 分歧样本），并生成：

```text
finetune/<ID>/mining/<source_id>/selection_request.jsonl
```

每行包含父数据集/ROI/global ID、父 manifest/draft/crop 的绝对路径，以及 manifest SHA、draft SHA 和图片 SHA。不要手写候选 ID，也不要把裸图片目录交给 HLMF。

```powershell
# b 示例
make export_finetune_gold `
  FINETUNE_SOURCE_ID=negative_removed_gold `
  FINETUNE_SOURCE_MODE=selection_subset

# c 示例
make export_finetune_gold `
  FINETUNE_SOURCE_ID=disagreement_gold `
  FINETUNE_SOURCE_MODE=selection_subset
```

程序不会重跑 MediaPipe 03；它从已认证的 parent manifest/draft 恢复 ROI，逐图核对 SHA，保留 `parent_*` 身份，并剥离 teacher 自动给出的 Left/Right。

#### 3.8.3 导出新录制 e source

新录制数据先在一个独立 HLMF raw source root 中正常跑完 00～03，确保存在：

```text
<raw-root>/02_roi_crops/images/
<raw-root>/02_roi_crops/hand_roi_crops_manifest.jsonl
<raw-root>/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
```

然后进入同一套 finetune strict CVAT 流程：

```powershell
make export_finetune_gold `
  FINETUNE_SOURCE_ID=new_recorded_gold `
  FINETUNE_SOURCE_MODE=native_existing `
  FINETUNE_RAW_SOURCE_ROOT=D:/datasets/new_recorded_raw
```

`native_existing` 不需要、也不会伪造 HLML selection request。

#### 3.8.4 CVAT 中必须完成的人工内容

导出后，每个 source 的 task 位于：

```text
finetune/<ID>/cvat/<source_id>/
├─ 02_roi_crops/images/       # 上传这些图片
├─ cvat_autolabel.xml         # 导入为初始标注
├─ task_descriptor.json
└─ qc/
   ├─ cvat_export_stats.json
   └─ crop_images_sha256.jsonl
```

使用 `configs/cvat_label.json` 创建任务标签。人工逐图复核时，所有未打 `ignore_for_training` 的图片必须满足：

1. 恰好保留一个完整的 21 点 `hand_landmarks` skeleton，或明确添加一个 `no_hand` tag；二者同时存在或都不存在都会失败。
2. Positive 必须明确选择且只选择一个 `Left`、`Right` 或 `unknown_handedness`。无法可靠判断左右手时选择 `unknown_handedness`，不要猜测。
3. Negative 只能有 `no_hand`，不能同时带 handedness。
4. 严重模糊、遮挡或无法可靠标注的图片可以添加 `ignore_for_training`。

从 CVAT 导出 `CVAT for images 1.1`，文件名必须是 `reviewed.xml`，放回 task 根目录：

```text
finetune/<ID>/cvat/<source_id>/reviewed.xml
```

然后执行：

```powershell
# 只导入一个 source
make import_finetune_gold FINETUNE_SOURCE_ID=negative_removed_gold

# 所有 task 的 reviewed.xml 都放回后，省略 ID 可先全量预检、再批量发布
make import_finetune_gold
```

程序负责 strict 05、覆盖率检查、逐图 SHA、全量 Gold/ignored sidecar、QC 和原子发布。批量模式先预检所有待发布 task，任一 `reviewed.xml` 缺失或不合格时不会开始发布；成功后另写 `cvat/batch_import_report.json`。任何图片漏回收、缺 presence/handedness 决策、重复 skeleton 或 artifact 被修改时，`sources/gold/<source_id>` 都不会出现半成品。

#### 3.8.5 聚合全部 HLMF Gold

Dragon 以及计划启用的 b/c/e source 都发布完成后运行：

```powershell
make finalize_train_finetune
```

该命令自动发现 `sources/gold/*/finetune_source.json`，不需要手工编辑 source 列表。它会认证 descriptor 和每个 artifact 的 SHA，对 `parent_global_crop_id → global_crop_id → ROI SHA → 归一化像素 SHA` 做跨 source 去重；同一身份标签冲突立即失败，标签一致则按 source role 决定唯一 owner。

最终交给 HLML 的 HLMF Gold-only 结果为：

```text
finetune/<ID>/hmlf_gold_merged/
├─ hmlf_gold_aggregate.json
├─ 05_labels/
│  ├─ hand_train_catalog_finetune.jsonl
│  ├─ hand_training_labels_finetune.jsonl
│  └─ hand_training_excluded_finetune.jsonl
└─ qc/finalize_train_finetune_report.json
```

先查看 aggregate 的 `counts`、`duplicate_count`、`conflict_count` 和 `source_descriptors`。HLMF 到此只聚合 Gold；pretrain replay 由 HLML 单独制作并在 HLML 的 finetune curation 阶段合并，不能拷贝进这个目录。
