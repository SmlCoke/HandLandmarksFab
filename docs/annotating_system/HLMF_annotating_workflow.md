# HLMF 3.0 标注工作流

## 1. 系统边界与身份

HLMF 从 Eos Palm Detector 的 proposal 开始工作。程序原样使用 bbox、p0、p9，构造固定 `256×256` Hand ROI；Palm 几何和 ROI 不允许人工修改。Hand landmark 教师可以是 MediaPipe Tasks 或 RTMPose-m Hand5，人工只复核 Hand ROI 内的 21 点、handedness、`no_hand` 和 `ignore_for_training`。

所有持久数据写入 `HAND_DATASET_ROOT`，不绑定训练 run ID。每个来源由下列七段 ID 唯一描述：

```text
<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>
```

`split` 只能是 `train|val|test`，在注册来源时从 `capture_source_id` 解析并写入 `source.json`、raw manifest 和 dataset manifest；发布阶段不会随机划分。train 来源位于 `PretrainSource`，val/test 来源位于 `EValSource`。

每次 proposal 配置使用一个 `PROPOSAL_VARIANT`。同一来源/变体在 Registry 中为 `active` 或 `retired`；retired 名称永久不能复用。

## 2. 环境、模型和配置

```bash
cd /root/HandLandmarksFab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
make compile
make test
```

代码依赖以 `requirements.txt` 为准。本轮继续使用现有 ONNX Runtime、OpenCV 和 FFmpeg 支持，无新增依赖。

Eos、MediaPipe Task、RTMPose 和 HCF ONNX 按仓库既有策略被 Git 忽略，需要在执行环境单独部署。HCF 默认路径为：

```text
models/hand_classifier/model.onnx
```

关键配置：

```yaml
hand_landmark:
  backend: mediapipe_tasks
rtmpose:
  model_onnx_path: models/rtmpose/rtmpose-m_hand5_256x256.onnx
  simcc_split_ratio: 2.0
hand_classifier:
  model_onnx_path: models/hand_classifier/model.onnx
quality:
  handedness_review_threshold: 0.7
  rtmpose_train_boundary_coordinate_reject_threshold: 3
visualization:
  roi_enabled: false
  original_image_enabled: false
  original_video_enabled: true
  train_max_samples: 200
```

配置原则：MediaPipe 保持全局默认；命令行后端只覆盖当前执行。SimCC split ratio 与模型绑定为 `2.0`。handedness 阈值越高，Train 被送入人工复核/忽略的行越多；边界阈值表示 42 个 x/y 值中允许出现多少个精确边界值，当前达到 3 个才拒绝。

## 3. 来源注册与图像检查

输入目录：

```text
HAND_DATASET_ROOT/PretrainSource/<dataset_id>/<capture_source_id>/images/*.tif[f]
HAND_DATASET_ROOT/EValSource/<dataset_id>/<capture_source_id>/images/*.tif[f]
```

`images/` 必须平铺。命令：

```bash
make source-check \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0
```

处理：校验来源 ID、TIFF、尺寸和方向，生成稳定 raw ID，登记 dataset/source/raw image，并在操作开始前拒绝 retired 变体名。

输出：

```text
<source>/source.json
<source>/raw_images.jsonl
<source>/qc/image_validation_report.json
HAND_DATASET_ROOT/Registry/registry.sqlite3
```

## 4. Train 自动标注与发布

```bash
make train-autolabel \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输入：已注册原图、Eos/RTMPose/HCF 模型和 `configs/autolabel.yaml`。

处理：Eos 生成 runtime 与 low-score candidate proposal；程序构造 ROI；runtime ROI 运行所选 Hand landmark 后端。RTMPose runtime ROI 同时运行 HCF；candidate 不运行 RTMPose/HCF。Train 自动完成质量分流和发布。

输出：

```text
<source>/01_palm/<variant>/palm_detections.jsonl
<source>/02_roi_crops/<variant>/images/*.png
<source>/02_roi_crops/<variant>/hand_roi_crops_manifest.jsonl
<source>/02_roi_crops/<variant>/hand_landmarks_autolabel_draft.jsonl
<source>/05_labels/<variant>/hand_training_labels.jsonl
<source>/05_labels/<variant>/candidate_negatives.jsonl
<source>/05_labels/<variant>/ignored.jsonl
<source>/qc/<variant>/*_report.json
```

## 5. RTMPose 与 HCF 推理契约

RTMPose 灰度 ROI 复制为 RGB，使用官方 mean/std，输出两个 `[N,21,512]` SimCC logits。坐标直接对原始 logits 取 argmax，再除以 `2.0`，不执行 softmax；输出夹到 `[0,255]`。默认关键点分数为 x/y 峰值较小者。

HCF 输入是灰度 `[N,1,256,256]`：转 float、除以 255，再以 `mean=0.485/std=0.229` 归一化。输出 `[N,2]` logits，argmax 映射 `0=Left、1=Right`，对应 softmax 概率写入 `handedness.score`。

两个模型都校验输入输出名称、形状和有限值。ONNX Runtime 可用 CUDA 时优先 CUDA，否则 CPU；实际 RTMPose/HCF provider、HCF 模型 ID 和推理数量写入 `qc/<variant>/mediapipe_report.json`（报告路径为兼容既有消费者而保留）。

RTMPose runtime ROI 固定输出 21 点，`hand_presence.present=true` 只用于现有发布路由。它不代表真实 presence 标签。low-score candidate 的关键点为空、handedness 为 `unknown/null`，进入 `candidate_negatives.jsonl` 人工链路。

## 6. Train 质量门控

质量门控只改变发布分流，不改变 Palm 或 ROI：

1. 所有 Train positive 若 handedness 分数低于 `quality.handedness_review_threshold`，整行进入 `ignored.jsonl`。
2. 仅对 `split=train`、`proposal_kind=runtime`、`source=rtmpose_m_hand5_onnx` 的行统计 21 点的 42 个 crop x/y 值。
3. 每个精确等于 `0.0` 或 `255.0` 的值计一次；计数达到 `rtmpose_train_boundary_coordinate_reject_threshold` 时，写入 `rtmpose_boundary_coordinate_values:<count>>=<threshold>`，并以 `ignore_reason=rtmpose_boundary_coordinate_gate` 进入 `ignored.jsonl`。
4. 当前阈值为 3，因此 1–2 个边界值通过。
5. Eval、MediaPipe 和 low-score candidate 不应用 RTMPose 边界门控。

Iris 第一阶段 geometry pretrain 可以使用 RTMPose 关键点，但必须忽略这些行的 presence。后续 multitask 训练和正式评估必须使用独立分类器或人工确认的真实 presence/handedness 标签。

## 7. Eval 自动标注、CVAT 与发布

```bash
make eval-autolabel \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx

make hand-cvat-export ... DATASET_SCOPE=eval PROPOSAL_VARIANT=eos-2.0
```

`eval-autolabel` 生成 Palm、ROI、草标和 QC，但不把教师结果当成正式评估真值；`hand-cvat-export` 从 ROI manifest、ROI images 和 draft 生成：

```text
<source>/03_reviewed/<variant>/cvat_autolabel.xml
<source>/qc/<variant>/cvat_export_report.json
```

在 CVAT Images 中使用 `Lexicographical` 排序。只允许调整 21 点、Left/Right/unknown、`no_hand`、`ignore_for_training`，不得改 Palm 或 ROI。将复核 XML 保存为：

```text
<source>/03_reviewed/<variant>/cvat_reviewed.xml
```

导入和发布：

```bash
make hand-cvat-import ... DATASET_SCOPE=eval PROPOSAL_VARIANT=eos-2.0
make source-publish ... DATASET_SCOPE=eval PROPOSAL_VARIANT=eos-2.0
```

输出 `hand_landmarks_reviewed.jsonl`、`hand_evaluation_labels.jsonl`、`ignored.jsonl` 和重建后的 dataset manifest。

Eval 限额在 `configs/datasets.yaml` 的 `evaluation_limits.max_raw_images_per_split` 与 `max_rois_per_split` 调整。限额按整个 val/test split 的 prospective dataset manifest 统计，不是单来源阈值。

## 8. Provenance

- MediaPipe：`label_origin=mediapipe`、`annotation_style=mediapipe_v1`。
- RTMPose：`label_origin=rtmpose`、`annotation_style=rtmpose_m_hand5_v1`、`teacher_model_id=rtmpose-m_hand5_256x256_onnx`。
- HCF：`handedness_teacher_model_id=hand-classifier-mobilenetv3-small-v1`。
- 人工复核记录 `human_reviewed`、`human_modified_landmark_ids` 和 `human_modified_handedness`；修点后使用 `*_human_corrected/project_consensus_v1`。
- 未推理 candidate：`unresolved/unlabeled_v1`，不伪装为教师标签。

## 9. 可视化、视频与清理

从既有 draft 重建 ROI 图：

```bash
make autolabel-visualize-roi ... PROPOSAL_VARIANT=eos-2.0
```

RTMPose 根据既有 QC 报告读取实际教师后端，并只抽样 runtime ROI；不会受后来修改 YAML 的影响。MediaPipe 保持原抽样行为。Train 最多按 `train_max_samples` 确定性均匀抽样；Val/Test 渲染全部适用行。

原图可视化：

```bash
make autolabel-visualize-original ... PROPOSAL_VARIANT=eos-2.0
make autolabel-visualize-original ... PROPOSAL_VARIANT=eos-2.0 ORIGINAL_VIDEO=false
```

输出按原图 stem 命名的 PNG，并默认按文件名字典序生成 30 FPS、`mp4v` 视频：

```text
<source>/visualizations/original_image_landmarks/<variant>/*.png
<source>/visualizations/original_image_landmarks/<variant>.mp4
```

只删除可重建可视化：

```bash
make autolabel-visualizations-clean ... PROPOSAL_VARIANT=eos-2.0
```

该命令删除 ROI/原图审核图、MP4 和对应 visualization QC 报告；不删除 ROI/draft，不改变 Registry，不写 tombstone。

## 10. 变体删除与 tombstone

```bash
make source-variant-delete \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 CONFIRM_DELETE=eos-2.0
```

`CONFIRM_DELETE` 必须与 `PROPOSAL_VARIANT` 完全相同。处理顺序是先把 `(capture_source_id, proposal_variant)` 标为 retired，再删除精确变体的 `01_palm`、`02_roi_crops`、`03_reviewed`、`05_labels`、`qc`、原图可视化目录与 MP4，最后重建 dataset manifest。

原始 `images/`、`raw_images.jsonl`、`source.json` 和 Registry 中的 ROI 元数据永久保留。同名变体不能再次标注或发布。若删除中断，使用同一确认命令继续执行，剩余清理是幂等的。

## 11. 负样本与困难样本

```bash
make negative-review NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS=/abs/candidate_negatives.jsonl
make negative-publish NEGATIVE_DATASET_ID=background-neg-0801

make hard-review SELECTION_ID=hard-0801 MINING_REQUEST=/abs/request.jsonl
make hard-publish SELECTION_ID=hard-0801
```

review 和 published 图片都使用普通独立复制，不创建硬链接。困难样本 published 目录拥有自己的图片副本，`selection.jsonl` 同时保存 `source_crop_relpath` 与 `published_relpath`；因此源变体被删除后，已发布负样本/困难样本仍可读取。

## 12. 批处理

```bash
make batch-train-autolabel DATASET_ID=FullEnhance0801 \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
make batch-eval-autolabel DATASET_ID=FullEnhanceVal0801 \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

脚本位于 `scripts/`，从 `<dataset>/<source>/source.json` 发现来源。`HAND_DATASET_ROOT`、`DATASET_ID`、`PROPOSAL_VARIANT`、`HAND_LANDMARK_BACKEND`、`PYTHON_BIN`、`REPO_DIR`、`LOG_DIR` 均通过环境变量传入。每个来源单独写日志；任一来源失败时脚本完成其余来源后返回非零。Train 脚本不会自动关机。

## 13. Registry 与验收

```bash
make registry-check HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
make compile
make test
make help
```

Registry 首次由新代码打开时，会把已有 ROI 的 `(capture_source_id, proposal_variant)` 自动回填为 active。`registry-check` 报告 active/retired 数量但不计算全链路 SHA-256。日常操作依靠数据集、来源和变体身份隔离，不在每一步重复哈希。
