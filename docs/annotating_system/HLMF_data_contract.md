# HLMF 3.0 数据契约

## 1. 仓库根目录

```text
HAND_DATASET_ROOT/
  PretrainSource/<dataset_id>/<capture_source_id>/
  EValSource/<dataset_id>/<capture_source_id>/
  GoldSource/NegativeSamples/<negative_dataset_id>/
  Selections/<selection_id>/
  Registry/registry.sqlite3
```

train 来源只能位于 `PretrainSource`，val/test 来源只能位于 `EValSource`。`capture_source_id` 固定为：

```text
<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>
```

split 在来源注册阶段写入所有 manifest，不在发布阶段随机生成。

## 2. Capture source 目录

```text
<source>/
  images/*.tif[f]
  source.json
  raw_images.jsonl
  01_palm/<proposal_variant>/
    palm_detections.jsonl
  02_roi_crops/<proposal_variant>/
    images/<roi_id>.png
    hand_roi_crops_manifest.jsonl
    hand_landmarks_autolabel_draft.jsonl
    hand_landmarks_roi_visualization/<roi_id>.png
  03_reviewed/<proposal_variant>/
    cvat_autolabel.xml
    cvat_reviewed.xml
    hand_landmarks_reviewed.jsonl
  05_labels/<proposal_variant>/
    hand_training_labels.jsonl | hand_evaluation_labels.jsonl
    candidate_negatives.jsonl
    ignored.jsonl
  qc/<proposal_variant>/*_report.json
  visualizations/original_image_landmarks/
    <proposal_variant>/<original_stem>.png
    <proposal_variant>.mp4
```

原始 `images/`、`source.json` 和 `raw_images.jsonl` 不属于 proposal variant。其余派生产物按精确 variant 隔离。

## 3. ID 与 Registry

- `raw_image_id`：同一来源原图的稳定 ID，不随 proposal variant 改变。
- `roi_id/crop_id`：由 raw ID、proposal variant、proposal slot 和 ROI contract version 稳定派生。
- Registry `proposal_variants` 主键：`(capture_source_id, proposal_variant)`。
- `status=active`：变体可按现有幂等流程重跑。
- `status=retired`：变体已永久退役，同一来源不得复用该名称。
- `retired_at`：UTC 退役时间。

打开 Registry 时，已有 ROI 的 source/variant 会自动回填为 active。删除变体只改变 status 并删除派生文件，既有 ROI Registry 元数据保留。

## 4. Palm 与 ROI manifest

Eos 产生 `proposal_kind=runtime|negative_candidate`。ROI manifest 至少保存 dataset/source/split、raw/ROI ID、proposal variant/slot/kind、Palm score、`palm_valid`、crop 路径与尺寸、ROI rect/corners 和 ROI contract version。

Palm bbox、p0、p9 与 ROI 几何是程序输出，CVAT 不得修改。所有 Hand ROI 固定为 `256×256`。

## 5. Hand landmark draft

公共字段包括：

```text
schema_version
dataset_id, capture_source_id, split
raw_image_id, roi_id/crop_id, palm_det_id
proposal_variant, proposal_slot, proposal_kind
crop_path/crop_relpath, roi_rect, roi_corners_px
hand_presence.present
handedness.label, handedness.score
landmarks_crop_norm[21]
landmarks_crop_px[21]
landmarks_image_px[21]
source
label_origin, annotation_style
teacher_model_id
handedness_teacher_model_id
teacher_detected
human_reviewed
human_modified_landmark_ids
human_modified_handedness
```

### MediaPipe

`source=mediapipe...`，provenance 为 `mediapipe/mediapipe_v1`，presence、handedness 和关键点来自 MediaPipe。现有行为保持不变。

### RTMPose + HCF runtime

- `source=rtmpose_m_hand5_onnx`；
- 固定 21 个关键点；
- `label_origin=rtmpose`；
- `annotation_style=rtmpose_m_hand5_v1`；
- `teacher_model_id=rtmpose-m_hand5_256x256_onnx`；
- `handedness_teacher_model_id=hand-classifier-mobilenetv3-small-v1`；
- `handedness.label=Left|Right`，score 为 HCF 对应 softmax 概率；
- `hand_presence.present=true` 仅为发布路由哨兵，不是真实 presence 标注。

HCF runtime 输入为 `[1,1,256,256]` float32；灰度除以 255 后按 `0.485/0.229` 归一化。模型接口名为 `input`/`output`，输出两类 logits（0 Left、1 Right）。

### 未推理 candidate

low-score candidate 的关键点为空，handedness 为 `{label: unknown, score: null}`，`handedness_teacher_model_id=null`，provenance 为 `unresolved/unlabeled_v1`。它不得伪装成 RTMPose、HCF 或 MediaPipe 标签。

## 6. QC provider 字段

既有 `qc/<variant>/mediapipe_report.json` 路径为兼容消费者而保留。报告增加：

```text
hand_landmark_backend
execution_provider
handedness_classifier_provider
handedness_classifier_model_id
handedness_runtime_rois_labeled
runtime_rois_labeled
negative_candidates_skipped
```

MediaPipe 的 HCF 字段为空/0。RTMPose runtime 记录两个实际 ONNX provider 和 HCF 推理数。

## 7. Train 发布分流

Train positive 若质量门控失败，整行进入 `ignored.jsonl` 且 `train_eligible=false`。

- handedness score 低于 `quality.handedness_review_threshold`：`ignore_reason=automatic_positive_failed_quality_gate`。
- RTMPose Train runtime 的 42 个 crop x/y 值中，精确为 `0.0` 或 `255.0` 的值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold`：quality error 为 `rtmpose_boundary_coordinate_values:<count>>=<threshold>`，`ignore_reason=rtmpose_boundary_coordinate_gate`。

当前边界阈值为 3；1–2 个边界值通过。边界门控不应用于 Eval、MediaPipe 或 negative candidate。Train candidate 进入 `candidate_negatives.jsonl`，不进入正样本。

RTMPose 行用于 geometry pretrain 时必须忽略 presence。正式 presence/handedness multitask 训练与 Val/Test 评估必须使用人工确认或独立真实标签。

## 8. Eval、CVAT 与发布

Eval draft 不是正式真值。CVAT frame 依据 ROI 图片字典序映射到 manifest；导入后产生 `hand_landmarks_reviewed.jsonl`。人工修改 handedness 时设置 `human_modified_handedness=true`，修点 ID 写入 `human_modified_landmark_ids`。

Val/Test 发布输出 `hand_evaluation_labels.jsonl` 和 `ignored.jsonl`，不发布 negative candidate。Eval 限额按整个 split 的 prospective dataset manifest 统计，配置位于：

```yaml
evaluation_limits:
  max_raw_images_per_split: 5000
  max_rois_per_split: 6000
```

需要增大时修改 `configs/datasets.yaml` 中对应值。

## 9. 可视化与视频

ROI 可视化目录是 `02_roi_crops/<variant>/hand_landmarks_roi_visualization/`。RTMPose 独立重建时从既有 QC 报告确定后端，只接受 `proposal_kind=runtime`；MediaPipe 保持原行为。

原图可视化 PNG 位于 `visualizations/original_image_landmarks/<variant>/`。默认视频位于同级 `<variant>.mp4`，PNG 按文件名字典序写入，默认 30 FPS、codec `mp4v`。`ORIGINAL_VIDEO=false` 时不生成视频。

`clean-autolabel-visualizations` 只删除上述 ROI/原图 PNG、MP4 和对应 visualization QC 报告，不修改 draft、发布标签、dataset manifest 或 Registry。

## 10. 变体删除契约

`delete-source-variant` 要求确认字符串与 variant 完全一致。目标仅限精确 source/variant：

```text
01_palm/<variant>
02_roi_crops/<variant>
03_reviewed/<variant>
05_labels/<variant>
qc/<variant>
visualizations/original_image_landmarks/<variant>
visualizations/original_image_landmarks/<variant>.mp4
```

必须保留：

```text
images/
raw_images.jsonl
source.json
Registry 中 raw/ROI 元数据和 retired tombstone
```

命令成功或中断后均可用同一确认命令继续幂等清理；dataset manifest 会重建并排除已删除发布变体。retired 名称不能重新注册 ROI、标注、复核或发布。

## 11. 负样本与困难样本

负样本目录：

```text
GoldSource/NegativeSamples/<negative_dataset_id>/review/images/
GoldSource/NegativeSamples/<negative_dataset_id>/published/images/
GoldSource/NegativeSamples/<negative_dataset_id>/published/negative_labels.jsonl
```

困难样本目录：

```text
Selections/<selection_id>/review/images/
Selections/<selection_id>/published/images/
Selections/<selection_id>/published/selection.jsonl
```

review 与 published 图片都是普通复制产生的独立文件，不是硬链接。困难样本记录保留原始 `source_crop_relpath`，并增加可直接读取的 `published_relpath`。删除源变体不会破坏已经发布的负样本/困难样本。

## 12. Dataset manifest 与完整性

dataset manifest 聚合 source、split 和 active published variant，保存 raw/ROI/label 数量和发布标签相对路径。删除变体后立即重建。

系统用 dataset/source/variant 身份和 SQLite 唯一约束隔离数据。日常流水线不在每一步计算 SHA-256；报告中的 `content_sha256` 保持 `not_computed`。
