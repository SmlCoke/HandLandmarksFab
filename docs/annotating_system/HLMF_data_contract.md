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

来源初始可以只有 `images/`；`source-check` 或自动标注流水线的第一步生成 `source.json` 与 `raw_images.jsonl`。自动标注批处理以数据集直接子目录中的 `images/` 为来源发现条件，不要求预注册。

原始 `images/`、`source.json` 和 `raw_images.jsonl` 不属于 proposal variant。其余派生产物按精确 variant 隔离。

## 3. ID 与 Registry

- `raw_image_id`：同一来源原图的稳定 ID，不随 proposal variant 改变。
- `roi_id/crop_id`：由 raw ID、proposal variant、proposal slot 和 ROI contract version 稳定派生。
- Registry `proposal_variants` 主键：`(capture_source_id, proposal_variant)`。
- `status=active`：变体可按现有幂等流程重跑。
- `status=retired`：变体已永久退役，同一来源不得复用该名称。
- `retired_at`：UTC 退役时间。

打开 Registry 时，已有 ROI 的 source/variant 会自动回填为 active。删除变体只改变 status 并删除派生文件，既有 ROI Registry 元数据保留。

ROI ID 不包含图片扩展名，但 `crop_path/crop_relpath` 是精确文件引用。Registry 会校验完整相对路径；文件从 `.png` 改为 `.tiff` 时，不能只改磁盘文件而不迁移 Registry 和所有 manifest/label 引用。

## 4. Palm 与 ROI manifest

Eos 产生 `proposal_kind=runtime|negative_candidate`。ROI manifest 至少保存 dataset/source/split、raw/ROI ID、proposal variant/slot/kind、Palm score、`palm_valid`、crop 路径与尺寸、ROI rect/corners 和 ROI contract version。

Palm bbox、p0、p9 与 ROI 几何是程序输出，CVAT 不得修改。所有 Hand ROI 固定为单通道 `uint8 256×256`，以无损 PNG 保存。模型输入契约是图片解码后的灰度像素数组，不是 PNG/TIFF 容器；相同 `uint8` 数组使用无损 PNG 或无损 TIFF 编解码后像素一致。板端从摄像头 `SSNE_Y_8` 内存构造 ROI，不读取 TIFF 文件作为 Hand Landmarker 输入。

原图若是 `uint16`、彩色、不同有效动态范围或经过有损编码，必须重新审查灰度转换与归一化，不能沿用当前 8-bit 单通道数据域结论。

## 5. Hand landmark draft

公共字段包括：

```text
schema_version
dataset_id, capture_source_id, split
raw_image_id, roi_id/crop_id, palm_det_id
proposal_variant, proposal_slot, proposal_kind
crop_path/crop_relpath, roi_rect, roi_corners_px
hand_presence.present, hand_presence.score
handedness.label, handedness.score
landmarks_crop_norm[21]
landmarks_crop_px[21]
landmarks_image_px[21]
source
label_origin, annotation_style
teacher_model_id
handedness_teacher_model_id
hand_presence_teacher_model_id
teacher_detected
human_reviewed
human_modified_landmark_ids
human_modified_handedness
human_modified_presence
rtmpose_geometry_rescue (optional)
```

### MediaPipe

`source=mediapipe...`，provenance 为 `mediapipe/mediapipe_v1`，presence、handedness 和关键点来自 MediaPipe。`hand_presence.score` 不是 MediaPipe 行的必需字段；现有行为保持不变。

### RTMPose + HCF runtime

- `source=rtmpose_m_hand5_onnx`；
- RTMPose 固定输出 21 个关键点；
- `label_origin=rtmpose`；
- `annotation_style=rtmpose_m_hand5_v1`；
- `teacher_model_id=rtmpose-m_hand5_256x256_onnx`；
- `handedness_teacher_model_id=hand-classifier-handedness-handpresence-0807`；
- `hand_presence_teacher_model_id=hand-classifier-handedness-handpresence-0807`；
- `handedness.label=Left|Right`，score 为 HCF 胜出类 softmax 概率；
- `hand_presence.present` 为 HCF presence argmax（0 no_hand、1 has_hand）；
- `hand_presence.score` 始终为 `P(has_hand)`，不是胜出类别置信度。

HCF runtime 输入为 `[N,1,256,256]` float32；灰度除以 255 后按 `mean=0.485/std=0.229` 归一化。模型必须暴露动态 batch 的 `input`，以及 `handedness`、`hand_presence` 两个 `[N,2]` float 输出。该 HCF 只用于 RTMPose runtime ROI，Train 和 Eval 都执行；输出是教师伪标签，不等价于人工真值。Eval 经 CVAT 导入后，人工 presence 的 `score` 可为 null，教师身份由 provenance 字段保留。

### RTMPose 的 MediaPipe TFLite 几何补救

该补救只处理 RTMPose Train runtime 中未通过边界或已开启连接长度门控的行。成功时：

- `source=mediapipe_hand_landmarker_full_tflite_rtmpose_rescue`；
- `label_origin=mediapipe`、`annotation_style=mediapipe_tflite_rescue_v1`；
- `teacher_model_id=mediapipe-hand-landmark-full-tflite`；
- presence、handedness 及两个 classifier teacher ID 仍为原 HCF 输出。

尝试过补救的行包含：

```json
{
  "rtmpose_geometry_rescue": {
    "attempted": true,
    "accepted": true,
    "trigger_errors": [],
    "result_errors": [],
    "model_id": "mediapipe-hand-landmark-full-tflite"
  }
}
```

补救失败时 `accepted=false`，保留原 RTMPose 坐标和 provenance。TFLite 的 handflag、handedness、world landmarks 不进入标签。

### 未推理 candidate

low-score candidate 的关键点为空，handedness 为 `{label: unknown, score: null}`，`hand_presence={present:false}`，`handedness_teacher_model_id=null`、`hand_presence_teacher_model_id=null`，provenance 为 `unresolved/unlabeled_v1`。它不运行 RTMPose/HCF，不得伪装成任何教师标签。

## 6. QC provider 字段

既有 `qc/<variant>/mediapipe_report.json` 路径为兼容消费者而保留。报告增加：

```text
hand_landmark_backend
execution_provider
execution_provider_fallback_reason
hand_classifier_provider
hand_classifier_provider_fallback_reason
onnx_batch_size
hand_classifier_model_id
hand_classifier_runtime_rois_labeled
runtime_rois_labeled
negative_candidates_skipped
mediapipe_tflite_rescue.enabled/model_id/attempted/accepted/rejected
```

MediaPipe 的 HCF 字段为空/0。RTMPose runtime 记录 RTMPose 与双头 HCF 的实际 ONNX provider、fallback 原因、batch size、HCF 模型 ID 和 HCF 推理数。Palm 的 `palm_detection_report.json.onnx_runtime` 记录 provider、fallback 原因及固定 batch 1 的原因。

`onnx_runtime.provider` 及 `onnx_runtime.model_providers.{palm,rtmpose,hand_classifier}` 只接受 `auto|cuda|cpu`；`auto` 为 CUDA 优先并允许 CPU fallback，`cuda` 在 CUDA provider 未激活时失败，`cpu` 固定 CPU。`onnx_runtime.batch_size` 必须是正整数。当前模型路径为 `models/hand_classifier/handedness-handpresence-0807/model.onnx`。

## 7. Train 发布分流

Train quality gate 失败的行进入 `ignored.jsonl` 且 `train_eligible=false`。

- RTMPose Train runtime 的 `hand_presence.score=P(has_hand)` 缺失或非有限：quality error 为 `rtmpose_hand_presence_score_missing|non_finite`，`ignore_reason=rtmpose_hand_presence_gate`。
- RTMPose Train runtime 的 `P(has_hand)` 严格低于 `quality.rtmpose_train_hand_presence_threshold`：quality error 为 `rtmpose_hand_presence_score_below_threshold:{score}<{threshold}`，`ignore_reason=rtmpose_hand_presence_gate`。等于阈值时通过。
- Train positive 的 handedness score 低于 `quality.handedness_review_threshold`：`ignore_reason=automatic_positive_failed_quality_gate`。
- RTMPose Train runtime 的 42 个 crop x/y 值中，精确为 `0.0` 或 `255.0` 的值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold`：quality error 为 `rtmpose_boundary_coordinate_values:<count>>=<threshold>`，`ignore_reason=rtmpose_boundary_coordinate_gate`。
- `quality.rtmpose_train_connection_length_gate_enabled` 为布尔开关，缺省及正式配置均为 `true`。开启时按 capture source 距离读取 `quality.rtmpose_train_connection_length_thresholds_px.<distance>`；任一连接长度严格超过阈值时，quality error 为 `rtmpose_connection_length_exceeded:<pair>:<length>><threshold>:distance=<distance>`，`ignore_reason=rtmpose_connection_length_gate`。21 点无效时 error 为 `rtmpose_connection_length_landmarks_invalid`。等于阈值和长度为 0 均通过；关闭时不解析距离或阈值。
- `quality.rtmpose_train_mediapipe_tflite_rescue_enabled` 缺省及正式配置均为 `true`。开启时，边界或已开启的连接长度门控失败会触发 TFLite 重预测；两项几何复检通过才替换关键点。关闭时不读取 `mediapipe_tflite` 配置、模型或独立环境。它不是新的门控，不改变既有 quality error 与 `ignore_reason`。

当前 presence 阈值为 `0.5`，边界阈值为 2；0–1 个边界值通过。Presence、边界和连接长度门控不应用于 Eval、MediaPipe 主链路或 Eos negative candidate；成功补救行仍属于 RTMPose Train runtime 链路，继续应用三条 RTMPose 专用门控。Train candidate 进入 `candidate_negatives.jsonl`，不进入正样本。

双头 HCF 的 presence/handedness 属于教师伪标签；正式 Val/Test 评估必须使用 CVAT 人工确认标签。

`source_publish_report.json.quality_gate_rejections` 固定包含 `hand_presence`、`boundary_coordinate`、`connection_length`、`handedness` 四个非负整数。计数按发布拒绝优先级互斥归因，因此一行最多计入一项；其他自动质量错误不计入四项。`quality_gate_counting_policy` 固定为 `exclusive_by_publish_routing_priority`。

## 8. Eval、CVAT 与发布

Eval draft 不是正式真值，RTMPose Train presence、边界和连接长度门控均不作用于 Eval。CVAT frame 依据 ROI 图片完整 basename（包含扩展名）的字典序映射到 manifest；导入后也按完整 basename 精确匹配。因此擅自更换 ROI 后缀会使既有 CVAT XML 无法直接导入，即使稳定 ROI ID 没有变化。导入后产生 `hand_landmarks_reviewed.jsonl`。人工改变 presence 时设置 `human_modified_presence=true`，改变 handedness 时设置 `human_modified_handedness=true`，修点 ID 写入 `human_modified_landmark_ids`。

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

`batch-autolabel-visualizations-clean` 从指定 scope/dataset 的直接子目录 `images/` 发现来源，并对每个来源应用相同的精确 variant 清理契约；批处理不扩大单来源删除范围。

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

`batch-source-variant-delete` 只遍历指定 scope/dataset 下存在 `source.json` 的注册来源，并对每个来源使用同一确认值执行上述契约；批处理末尾再次重建 dataset manifest。单来源失败会被汇总并使批处理返回非零，但不回滚已经完成的 retired tombstone 或文件删除。

## 11. 负样本与困难样本

负样本目录：

```text
GoldSource/NegativeSamples/<negative_dataset_id>/review/images/
GoldSource/NegativeSamples/<negative_dataset_id>/review/candidate_manifest.jsonl
GoldSource/NegativeSamples/<negative_dataset_id>/review/precheck_excluded.jsonl
GoldSource/NegativeSamples/<negative_dataset_id>/review/README.json
GoldSource/NegativeSamples/<negative_dataset_id>/published/images/
GoldSource/NegativeSamples/<negative_dataset_id>/published/negative_labels.jsonl
```

困难样本目录：

```text
Selections/<selection_id>/review/images/
Selections/<selection_id>/published/images/
Selections/<selection_id>/published/selection.jsonl
```

`negative_review.hand_presence_threshold` 必须是 `[0,1]` 内有限数，缺省及正式配置为 `0.5`。`candidate_manifest.jsonl` 仅含 `P(has_hand)<threshold` 的行；`precheck_excluded.jsonl` 保存其余行。两者的 `negative_review_precheck` 包含 `hand_presence_score`、`threshold`、`selected_for_human_review` 和 `model_id`。人工发布前仍必须复核所选图片。

review 与 published 图片都是普通复制产生的独立文件，不是硬链接。困难样本记录保留原始 `source_crop_relpath`，并增加可直接读取的 `published_relpath`。删除源变体不会破坏已经发布的负样本/困难样本。

## 12. Dataset manifest 与完整性

dataset manifest 只聚合含至少一个 `qc/<proposal_variant>/source_publish_report.json` 的 source 及其 published variant，保存 split、raw/ROI/label 数量和发布标签相对路径。每个 capture source 的 `quality_gate_rejections` 聚合其已发布 variants；顶层 `quality_gate_rejections` 为 dataset 合计，`quality_gate_rejections_by_capture_source_id` 为 source ID 到四项计数的映射，`quality_gate_counting_policy` 固定为互斥发布归因策略。仅有 `source.json`、自动标注 draft 或 CVAT 导出文件的来源不进入 manifest；未完成 CVAT 导入和 `source-publish` 的 Eval 来源因此不会被按 manifest 消费数据的下游 HLML 读取。

删除变体后立即重建 dataset manifest；数据集级批量删除在全部单来源操作后额外重建一次。

系统用 dataset/source/variant 身份和 SQLite 唯一约束隔离数据。日常流水线不在每一步计算 SHA-256；报告中的 `content_sha256` 保持 `not_computed`。
