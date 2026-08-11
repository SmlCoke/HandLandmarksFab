# HLMF 3.0 标注工作流

## 1. 系统边界与身份

HLMF 从 Eos Palm Detector 的 proposal 开始工作。程序原样使用 bbox、p0、p9，构造固定 `256×256` Hand ROI；Palm 几何和 ROI 不允许人工修改。Hand landmark 教师可以是 MediaPipe Tasks 或 RTMPose-m Hand5，人工只复核 Hand ROI 内的 21 点、handedness、`no_hand` 和 `ignore_for_training`。

所有持久数据写入 `HAND_DATASET_ROOT`，不绑定训练 run ID。每个来源由下列七段 ID 唯一描述：

```text
<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>
```

`distance` 是 Palm 模型能力门控字段。当前 Eos-2.0 只支持 `near|mid`；`far` 可以注册、保存和查看历史资产，但不能进入 Palm、ROI、Hand Landmark、CVAT 或发布阶段。Eos-1.0 的能力更弱，历史上仅适合作为 mid 兼容资产。新 Palm 模型必须先完成各距离覆盖评测，再更新配置中的支持列表。

`split` 只能是 `train|val|test`，在注册来源时从 `capture_source_id` 解析并写入 `source.json`、raw manifest 和 dataset manifest；发布阶段不会随机划分。train 来源位于 `PretrainSource`，val/test 来源位于 `EValSource`。

每次 proposal 配置使用一个 `PROPOSAL_VARIANT`。同一来源/变体在 Registry 中为 `active` 或 `retired`；retired 名称永久不能复用。

## 2. 环境、模型和配置

```bash
cd /root/HandLandmarksFab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
/root/miniconda3/envs/anfab/bin/python -m pip uninstall -y onnxruntime
/root/miniconda3/envs/anfab/bin/python -m pip install -r requirements.txt
make compile
make test
```

主环境依赖以 `requirements.txt` 为准。当前 GPU Runtime 与服务器 CUDA 11/cuDNN 8 的兼容组合为 `onnxruntime-gpu==1.18.0`、NumPy `<2`、OpenCV `<4.11`；安装后应运行 `pip check`。MediaPipe TFLite 补救不向 `anfab` 增加 TensorFlow，而是使用独立轻量环境：

```bash
/root/miniconda3/envs/anfab/bin/python -m venv \
  /root/miniconda3/envs/hlmf-mp-tflite
/root/miniconda3/envs/hlmf-mp-tflite/bin/python -m pip install \
  -r requirements-mediapipe-tflite.txt
```

Eos、MediaPipe Task、RTMPose 和 HCF ONNX 按仓库既有策略被 Git 忽略，需要在执行环境单独部署。当前默认 Eos-2.0、双头 HCF 与上一版归档路径为：

```text
models/palm_detector/eos-2.0/model_384x224_opt.onnx
models/hand_classifier/handedness-handpresence-0809/model.onnx
models/handedness-0806/
models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
```

代码只读取新路径；旧目录仅用于保留 handedness-only 模型和指标，不参与推理。

关键配置：

```yaml
palm:
  model_id: eos-2.0
  supported_capture_distances: [near, mid]
  input_width: 384
  input_height: 224
  score_threshold: 0.25
  nms_iou_threshold: 0.10
  max_detections: 2
hand_roi:
  scale_x: 1.8
  scale_y: 1.8
  shift_x: 0.0
  shift_y: -0.1
hand_landmark:
  backend: mediapipe_tasks
onnx_runtime:
  provider: auto
  model_providers:
    palm: auto
    rtmpose: cpu
    hand_classifier: auto
  batch_size: 64
rtmpose:
  model_onnx_path: models/rtmpose/rtmpose-m_hand5_256x256.onnx
  simcc_split_ratio: 2.0
mediapipe_tflite:
  model_asset_path: models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
  python_executable: /root/miniconda3/envs/hlmf-mp-tflite/bin/python
hand_classifier:
  model_onnx_path: models/hand_classifier/handedness-handpresence-0809/model.onnx
negative_review:
  hand_presence_threshold: 0.5
quality:
  handedness_review_threshold: 0.7
  rtmpose_train_hand_presence_threshold: 0.025
  rtmpose_train_boundary_coordinate_reject_threshold: 2
  rtmpose_train_mediapipe_tflite_rescue_enabled: true
  rtmpose_train_connection_length_gate_enabled: true
  # rtmpose_train_connection_length_thresholds_px 在配置中按 near/mid/far
  # 分别给出 20 条连接的 crop 像素阈值。
visualization:
  roi_enabled: false
  original_image_enabled: false
  original_video_enabled: true
  train_max_samples: 200
```

配置原则：Eos-2.0 固定使用灰度 `INTER_AREA 384×224`、`/255`、NCHW 输入；两个矩形 feature level 使用模型配套的 840 anchors，检测在 level 合并后执行全局 NMS。`0.25` 是本次确认的召回/候选量折中值；ROI scale `1.8/1.8` 用于保持旧 Eval 与连接门控兼容。MediaPipe Tasks 保持 Hand landmark 全局默认；命令行后端只覆盖当前执行。ONNX `auto` 表示 CUDA 可用时优先 GPU、否则回退 CPU；`cuda` 要求 GPU provider 必须激活，`cpu` 固定 CPU。性能与人工复核 Eval 回放表明 Eos-2.0 Palm/HCF 采用 `auto`，RTMPose 因 GPU 关键点精度轻微下降而固定 CPU；RTMPose/HCF 使用动态 batch 64，Palm 模型输入固定为 batch 1，详见 `assets/device_perf/onnx_cpu_gpu_benchmark.md`。SimCC split ratio 与模型绑定为 `2.0`。HCF 模型 ID 自动由 `model_onnx_path` 的父目录生成，切换版本时只需修改该路径，但版本目录必须使用安全名称。`negative_review.hand_presence_threshold=0.5` 只用于负样本预审，严格低于模型 argmax 分界的候选才进入人工 review。`rtmpose_train_hand_presence_threshold=0.025` 是针对 0809 校准的 RTMPose Train runtime 最小 `P(has_hand)`；低于阈值、缺失或非有限时整行拒绝，等于阈值通过。两项阈值用途不同，不应联动修改。handedness 阈值越高，Train 被忽略的低置信行越多；边界阈值表示 42 个 x/y 值中允许出现多少个精确边界值，当前达到 2 个即拒绝。连接长度门控默认开启；关闭时完全跳过距离解析和阈值校验。TFLite 补救也默认开启；关闭时不解析其模型和 Python 环境配置，原四条门控照常执行。

`palm.supported_capture_distances` 是与 Palm 权重绑定的必填能力资产；缺失、为空或格式非法时，所有模型相关阶段在写入前终止。

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

可在自动标注前只读检查当前 Palm 是否支持该距离：

```bash
make palm-distance-check \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak
```

支持时返回 0；不支持时返回专用状态并显示 model、实际 distance 和支持列表；配置或 source ID 非法时返回配置错误。`source-check` 本身不应用该限制，因为 far 原图和 raw/source 元数据仍应保留。

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

输入：已注册原图、Eos/RTMPose/HCF 模型、TFLite 补救资产和 `configs/autolabel.yaml`。

处理：Eos 生成 runtime 与 low-score candidate proposal；程序构造固定 `256×256` 灰度 ROI；RTMPose runtime ROI 同时运行 RTMPose 与 HCF。Train 行若未通过边界或已开启的连接长度门控，则把失败 ROI 一次性提交给独立 TFLite worker；补救点通过两项几何门控才替换 RTMPose 点。candidate 不运行 RTMPose/HCF/TFLite。Train 最后按原四条门控分流发布。

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

ROI 使用无损 PNG 保存，模型实际读取的是解码后的 `uint8` 灰度像素，而不是 PNG/TIFF 容器。板端输入为摄像头 `SSNE_Y_8` 内存并现场构造 ROI，也不会读取 TIFF 文件作为 Hand Landmarker 输入。对同一 `uint8` ROI，改用无损 TIFF 不改变像素域，只会改变文件路径和存储开销。

## 5. RTMPose 与 HCF 推理契约

RTMPose 灰度 ROI 复制为 RGB，使用官方 mean/std，输出两个 `[N,21,512]` SimCC logits。坐标直接对原始 logits 取 argmax，再除以 `2.0`，不执行 softmax；输出夹到 `[0,255]`。默认关键点分数为 x/y 峰值较小者。

双头 HCF 输入是灰度 `[N,1,256,256]`：转 float、除以 255，再以 `mean=0.485/std=0.229` 归一化。输出名称必须恰为 `handedness` 和 `hand_presence`，形状均为 `[N,2]`。handedness 的 argmax 映射 `0=Left、1=Right`，胜出类 softmax 概率写入 `handedness.score`；presence 的 argmax 映射 `0=no_hand、1=has_hand`，无论胜出类别为何，`hand_presence.score` 始终保存 `P(has_hand)`。

两个模型都校验输入输出名称、动态 batch、固定通道/空间/类别形状、float 类型和有限值。RTMPose/HCF 按 `onnx_runtime.batch_size` 批量处理 ROI；每个模型按独立 provider 配置创建会话。实际 provider、CPU fallback 原因、batch size、HCF 模型 ID 和推理数量写入 `qc/<variant>/mediapipe_report.json`（报告路径沿用现有位置）。Palm 实际 provider 写入 `palm_detection_report.json`。

RTMPose runtime ROI 固定输出 21 点，并在 Train 与 Eval 都运行一次双头 HCF；`hand_presence.present` 和 `hand_presence.score` 是 HCF 教师输出，不是人工真值。Eos low-score candidate 不运行 RTMPose/HCF，关键点为空、handedness 为 `unknown/null`、两个 HCF provenance ID 均为 null，继续进入 `candidate_negatives.jsonl` 人工链路。

TFLite worker 输入为灰度 ROI，经 `224×224` 双线性缩放、三通道复制和 `/255` 后送入 `hand_landmark_full.tflite`。坐标按 `raw/224×256` 解码；模型输出的 handflag、handedness 和 world landmarks 不进入 HLMF。补救成功行的关键点教师为 `mediapipe-hand-landmark-full-tflite`，HCF 的 presence/handedness 及其 teacher ID 原样保留。补救失败时保留 RTMPose 点，并在 `rtmpose_geometry_rescue` 中记录触发错误和 TFLite 结果错误。

## 6. Train 质量门控

质量门控只改变发布分流，不改变 Palm 或 ROI：

1. **Hand presence 置信度门控**：仅对 RTMPose Train runtime 读取 `hand_presence.score=P(has_hand)`。分数缺失、非有限或严格低于 `quality.rtmpose_train_hand_presence_threshold` 时，以 `ignore_reason=rtmpose_hand_presence_gate` 进入 `ignored.jsonl`；等于阈值通过。
2. **Handedness 置信度门控**：所有 Train positive 的 handedness 分数严格低于 `quality.handedness_review_threshold` 时，以 `ignore_reason=automatic_positive_failed_quality_gate` 进入 `ignored.jsonl`。该规则同时适用于 RTMPose 与 MediaPipe。
3. **边界坐标门控**：仅对 RTMPose Train runtime 统计 21 点的 42 个 crop x/y 值。精确为 `0.0` 或 `255.0` 的值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold` 时，写入 `rtmpose_boundary_coordinate_values:<count>>=<threshold>`，并以 `ignore_reason=rtmpose_boundary_coordinate_gate` 进入 `ignored.jsonl`。
4. **连接对长度门控**：仅在 `quality.rtmpose_train_connection_length_gate_enabled=true` 时对 RTMPose Train runtime 生效。程序按 capture source 的 `near/mid/far` 选择阈值，计算 20 条连接的 crop 像素欧氏距离；任一长度严格超过阈值时写入 `rtmpose_connection_length_exceeded:<pair>:<length>><threshold>:distance=<distance>`，并以 `ignore_reason=rtmpose_connection_length_gate` 进入 `ignored.jsonl`。等于阈值及长度为 0 均通过；关闭开关时不解析距离或阈值。

当前 RTMPose Train presence 阈值为 `0.025`，边界阈值为 2，因此 0–1 个边界值通过。Presence、边界和连接长度门控作用于 RTMPose Train runtime 链路，包括成功采用 TFLite 补救点的行；Eval、MediaPipe 主链路和 Eos low-score candidate 不应用这三条 RTMPose 专用门控。

TFLite 补救不是第五条门控。执行顺序为：RTMPose/HCF → 几何预检 → 必要时 TFLite 重预测并复检 → 四条质量门控发布分流。补救后的 presence/handedness 仍只来自 HCF；最终拒绝原因优先级保持 `presence → boundary → connection length → handedness/其他`。

`source-publish` 按上述发布优先级为每条 rejected 行只归因一次，并把四项互斥计数写入 `source_publish_report.json.quality_gate_rejections`。`dataset_manifest.json` 同时保存 dataset 合计、每个 `capture_source_id` 合计及 `quality_gate_counting_policy=exclusive_by_publish_routing_priority`；其他通用质量问题不计入四项统计。

Presence 阈值应在每次 HCF 更新后使用与正式标注隔离的人工复核 ROI 副本重新校准。0809 在 7,892 条 hand、15 条 no_hand 上沿用 `0.5` 只保留 97.149% 的 hand，最弱来源为 76.97%，说明同结构重训也不能继承旧权重的分数阈值。Train 门控最终采用 `0.025`：保留 99.582% 的 hand，最弱来源为 96.58%，同时拒绝全部人工 no_hand。`negative_review.hand_presence_threshold=0.5` 仍表示负候选的模型 argmax 分界，两项阈值用途不同。

连接长度阈值来自 `FullEnhanceVal0801:eos-1.0` 与 `FullEnhanceVal0808:eos_1.0-gate_r2` 的 9,868 条人工复核 gold hand，按距离和连接取 `ceil(P99.95 × 1.05)`。Eos-2.0 通过旧 gold 只读投影回放后暂时保留这些值；这只是兼容过渡，不是正式重算。完整分布、阈值与回放结果位于 `assets/quality_gate/rtmpose_connection_length_distribution.md`，Eos-2.0 回放依据位于 `assets/palm_detector/eos_2_0_adaptation.md`。首个代表性 Eos-2.0 Eval 人工复核并发布后，必须将下列 dataset 参数换成新 variant 后重新执行：

```bash
python -B tools/analyze_rtmpose_connection_lengths.py   --dataset-root /root/autodl-tmp/DatesetFab   --dataset FullEnhanceVal0801:eos-1.0   --dataset FullEnhanceVal0808:eos_1.0-gate_r2   --config configs/autolabel.yaml   --output assets/quality_gate/rtmpose_connection_length_distribution.md
```

输入是已人工复核并发布的 Eval manifests/labels；输出只写仓库内的统计报告，不修改数据仓库。重算后必须审查 gold/draft 回放、更新 YAML 阈值并运行完整测试。

HCF presence/handedness 是 Train 的教师伪标签；Eval draft 必须经过 CVAT 人工确认后才能成为正式真值。

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

`eval-autolabel` 生成 Palm、ROI、RTMPose 关键点以及双头 HCF 的 presence/handedness 草标和 QC，但不把任何教师结果当成正式评估真值；`hand-cvat-export` 从 ROI manifest、ROI images 和 draft 生成：

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

Dataset manifest 只聚合至少存在一个 `qc/<variant>/source_publish_report.json` 的来源。某个 Eval 来源即使已执行 `eval-autolabel` 或导出 CVAT，只要尚未导入人工复核结果并执行 `source-publish`，就不会进入 `dataset_manifest.json`；下游 HLML 按 manifest 中的发布标签路径读取数据，因此会忽略该来源。

## 8. Provenance

- MediaPipe：`label_origin=mediapipe`、`annotation_style=mediapipe_v1`。
- RTMPose：`label_origin=rtmpose`、`annotation_style=rtmpose_m_hand5_v1`、`teacher_model_id=rtmpose-m_hand5_256x256_onnx`。
- 双头 HCF：`handedness_teacher_model_id` 与 `hand_presence_teacher_model_id` 均由模型版本目录生成；当前为 `hand-classifier-handedness-handpresence-0809`。
- 人工复核记录 `human_reviewed`、`human_modified_landmark_ids`、`human_modified_handedness` 和 `human_modified_presence`；修点后使用 `*_human_corrected/project_consensus_v1`。
- 未推理 candidate：两个 HCF teacher ID 均为 null，provenance 为 `unresolved/unlabeled_v1`，不伪装为教师标签。

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

数据集级批量清理：

```bash
make batch-autolabel-visualizations-clean \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 PROPOSAL_VARIANT=eos-2.0
```

脚本从数据集的直接子目录 `images/` 发现全部来源，对每个来源执行同一精确变体的可视化清理，并在全部来源处理后汇总成功与失败数量。输入是 scope、dataset ID 和 proposal variant；输出是删除汇总，不改变 dataset manifest 或 Registry。

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

数据集级批量删除：

```bash
make batch-source-variant-delete \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  PROPOSAL_VARIANT=eos-2.0 CONFIRM_DELETE=eos-2.0
```

脚本从 `source.json` 发现全部已注册来源，逐来源执行永久退役与精确产物删除；`CONFIRM_DELETE` 不完全匹配时在任何删除前退出。所有来源处理结束后，无论是否有单来源失败，都会再执行一次 `dataset-manifest-rebuild`，使 manifest 与已完成的删除保持一致。批处理继续保留每个来源的原图、raw/source 元数据及 Registry ROI 元数据。

若直接继续某个 active 变体已有的 near/mid 草稿，可沿用原 variant 完成人工复核，但不要重新自动标注。若要整轮从头重跑，应先明确放弃并退役旧变体，再使用新名称；例如：

```bash
make batch-source-variant-delete \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  PROPOSAL_VARIANT=eos_2.0-rtmpose-gate \
  CONFIRM_DELETE=eos_2.0-rtmpose-gate
make batch-eval-autolabel \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_ID=FullEnhanceVal0801 \
  PROPOSAL_VARIANT=eos_2.0-rtmpose-gate-r2 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

第二条命令会跳过 far，只处理 near/mid。上述删除不是日常预检步骤，仅在确认放弃旧草稿时执行；retired 名称永久不能复用。

## 11. 负样本与困难样本

```bash
make negative-review NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS=/abs/candidate_negatives.jsonl
make negative-publish NEGATIVE_DATASET_ID=background-neg-0801

make hard-review SELECTION_ID=hard-0801 MINING_REQUEST=/abs/request.jsonl
make hard-publish SELECTION_ID=hard-0801
```

`negative-review` 从同一 `hand_classifier.model_onnx_path` 加载当前 HCF（现为 0809），批量计算每个候选的 `P(has_hand)` 并显示进度；仅严格低于 `negative_review.hand_presence_threshold` 的 ROI 被复制到 `review/images/`。`candidate_manifest.jsonl` 保存所选行及 `negative_review_precheck`，`precheck_excluded.jsonl` 保存未复制行及其分数，两者都记录实际 `model_id`；`README.json` 汇总模型 ID、阈值、数量、provider 与 batch。人工仍需删除有手或不确定图片，再执行 `negative-publish`；预审不是正式负标签，等于阈值的候选不进入 review。

review 和 published 图片都使用普通独立复制，不创建硬链接。困难样本 published 目录拥有自己的图片副本，`selection.jsonl` 同时保存 `source_crop_relpath` 与 `published_relpath`；因此源变体被删除后，已发布负样本/困难样本仍可读取。

## 12. 批处理

```bash
make batch-train-autolabel DATASET_ID=FullEnhance0801 \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
make batch-eval-autolabel DATASET_ID=FullEnhanceVal0801 \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

自动标注脚本从 `<dataset>/<source>/images/` 发现来源，不要求预先存在 `source.json`；每个单来源流水线首先执行 source check，因此新来源会在批处理中自动注册。只有数据集的直接子目录被视为 capture source，`images/` 内部不递归发现来源。

批处理会在任何来源写入前调用统一 Palm 距离预检。兼容来源进入处理队列；far 显示 `SKIPPED_UNSUPPORTED_DISTANCE`，汇总记录 discovered、supported、success、failed、skipped 以及完整 skipped source ID。只要至少存在一个兼容来源且没有真实执行失败，跳过 far 不会使批处理失败；全部来源均不兼容或预检配置非法时返回非零。

`HAND_DATASET_ROOT`、`DATASET_ID`、`PROPOSAL_VARIANT`、`HAND_LANDMARK_BACKEND`、`PYTHON_BIN`、`REPO_DIR`、`LOG_DIR` 均通过环境变量传入。每个自动标注来源单独写日志；任一来源失败时脚本完成其余来源后返回非零。Train 脚本不会自动关机。

可视化清理批处理同样从 `images/` 发现来源；永久变体删除批处理只处理已有 `source.json` 的注册来源，并额外接收 `DATASET_SCOPE` 与 `CONFIRM_DELETE`。所有批处理仅作用于指定 dataset ID 与 proposal variant。

## 13. Registry 与验收

```bash
make registry-check HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
make compile
make test
make help
```

Registry 首次由新代码打开时，会把已有 ROI 的 `(capture_source_id, proposal_variant)` 自动回填为 active。`registry-check` 报告 active/retired 数量但不计算全链路 SHA-256。日常操作依靠数据集、来源和变体身份隔离，不在每一步重复哈希。
