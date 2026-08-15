# HandLandmarkerFab（HLMF 3.0）

HandLandmarkerFab 是 Hand Landmarker 训练系统的上游数据制作仓库。系统默认从 Eos-2.0 Palm Detector 的 bbox、p0、p9 构造固定 `256×256` Hand ROI，再执行 RTMPose-m Hand5 + 双头 HCF + 质量门控 + MediaPipe Hand Landmarker TFLite rescue；MediaPipe Tasks 仍可作为显式覆盖。

当前 Eos-2.0 只在 near/mid 数据上训练，HLMF、后续 Iris/Muse 和端侧演示因此仅支持 near/mid。far 原始来源应保留，但不得进入 Eos-2.0 的 Palm→ROI→Landmark→复核→发布链路；单来源命令会硬拒绝，批处理会显式跳过。

本仓库只制作 Hand Landmarker 数据。Palm 几何不能人工修改，人工复核对象是程序生成的 Hand ROI、21 个关键点、handedness 以及 `no_hand/ignore_for_training` 状态。

Hand ROI 的模型输入契约是解码后的单通道 `uint8 256×256` 像素。仓库使用无损 PNG 保存 ROI；PNG/TIFF 是存储容器，不改变相同灰度数组的像素域。板端从 `SSNE_Y_8` 摄像头内存直接构造 ROI，并不把 TIFF 文件送入 Hand Landmarker。

## 入口文档

- [完整工作流](docs/annotating_system/HLMF_annotating_workflow.md)
- [快速开始](docs/annotating_system/HLMF_quick_start.md)
- [数据契约](docs/annotating_system/HLMF_data_contract.md)
- [当前状态](docs/annotating_system/HLMF_current_status.md)
- [常见问题与解答](docs/annotating_system/HLMF_qa.md)
- [Eos-2.0 审计与适配报告](assets/palm_detector/eos_2_0_adaptation.md)
- [HCF 0813 接入与校准报告](assets/hand_classifier/handedness_handpresence_0813.md)
- [Iris Eval 就绪度评估](assets/evaluation/iris_eval_readiness.md)
- [ONNX CPU/GPU 性能报告](assets/device_perf/onnx_cpu_gpu_benchmark.md)

## 当前教师后端

- Palm：默认 Eos-2.0，输入灰度 `[1,1,224,384]`，score `0.25`、全局 NMS IoU `0.10`、840 anchors；ROI scale 保持 `1.8/1.8`，`supported_capture_distances=[near,mid]`。默认 proposal variant 为 `eos-2.0`。
- 默认后端：`hand_landmark.backend: rtmpose_onnx`。
- 单次切换：`HAND_LANDMARK_BACKEND=mediapipe_tasks`。
- RTMPose：原始 SimCC logits 直接 argmax，除以固定 `2.0`；runtime ROI 总是输出 21 点。
- HCF：`models/hand_classifier/handedness-handpresence-0813/model.onnx`，输入灰度 `[N,1,256,256]`，输出 `handedness` 与 `hand_presence` 两个 `[N,2]` logits；模型 ID 从版本目录名生成，仅运行于 RTMPose runtime ROI。0813 使用 Eos-2.0 与历史 Eos-1.0 数据混合训练；旧模型仅作归档。
- 负样本 `negative-review` 预审核复用同一 `hand_classifier.model_onnx_path`，并在所选/排除清单及 README 中记录实际 HCF 模型 ID。
- MediaPipe TFLite 补救：仅当 RTMPose Train runtime 未通过边界或已开启的连接长度门控时，使用纯 Hand Landmarker TFLite 重预测 21 点；presence/handedness 仍只采用 HCF。
- Eos low-score candidate 不运行 RTMPose/HCF，保持 `unknown/null`，进入人工候选链路。

ONNX/TFLite 模型遵循仓库现有忽略策略，不纳入 Git；代码和配置通过 Git 同步，模型需在执行环境中单独部署。

ONNX Runtime 使用逐模型 provider：Eos-2.0 Palm 与 HCF 默认 `auto`（CUDA 可用时使用 GPU，否则回退 CPU），RTMPose 固定 CPU；后者是因为 GPU 虽更快，但在人工复核 Eval 上轻微降低了关键点精度。RTMPose/HCF 动态 batch 默认为 `64`，Palm 模型输入固定为 batch 1。可在 `onnx_runtime.model_providers` 中改为 `auto|cuda|cpu`，实测依据见性能报告。

## 常用命令

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab

make source-check DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice \
  PROPOSAL_VARIANT=eos-2.0

make palm-distance-check \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice

make train-autolabel DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx

make eval-autolabel DATASET_SCOPE=eval DATASET_ID=demo-eval \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx

make autolabel-visualize-roi DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.0
make autolabel-visualize-original DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.0
make autolabel-visualizations-clean DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.0

make source-variant-delete DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.0 CONFIRM_DELETE=eos-2.0

make batch-train-autolabel DATASET_ID=demo PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
make batch-eval-autolabel DATASET_ID=demo-eval PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx

make negative-review NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS=/abs/candidate_negatives.jsonl

make gold-autolabel DATASET_SCOPE=gold DATASET_ID=gold-demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s06-alice \
  PROPOSAL_VARIANT=eos-2.0

make hard-review HARD_DATASET_ID=hard-hands-r1 \
  MINING_REQUEST=/abs/hlmf_review_request.jsonl
# CVAT 精修并放回 review/cvat_reviewed.xml 后：
make hard-import HARD_DATASET_ID=hard-hands-r1
make hard-publish HARD_DATASET_ID=hard-hands-r1

make registry-check
make compile
make test
make help
```

`autolabel-visualize-original` 默认在 PNG 完成后生成 `visualizations/original_image_landmarks/<variant>.mp4`；传 `ORIGINAL_VIDEO=false` 可只生成 PNG。

`source-variant-delete` 只删除精确来源/变体的派生产物，保留 `images/`、`raw_images.jsonl`、`source.json`，并写入永久 retired tombstone。同一来源不能再次使用被 retired 的变体名。若只想释放可重建的可视化空间，使用 `autolabel-visualizations-clean`，它不会写 tombstone。

批量自动标注先按当前 Palm 模型能力预检全部 source：near/mid 正常执行，far 输出 `SKIPPED_UNSUPPORTED_DISTANCE` 并计入 skipped，不计作运行失败；若没有任何兼容来源则返回非零。`source-check`、历史可视化和精确变体清理仍允许处理 far，以保留原始资产和诊断能力。

现有 near/mid 草稿可继续沿用原 variant 复核；若要从头重跑，必须换用新 variant。以 `eos_2.0-rtmpose-gate` 为例，清理后应使用 `eos_2.0-rtmpose-gate-r2`，不能复用已 retired 的名称，也不应在已有 ROI/CVAT 资产上原地重跑。

## Train 质量边界

RTMPose Train runtime 行满足以下任一条件时整行进入 `ignored.jsonl`：

- `hand_presence.score=P(has_hand)` 缺失、非有限，或低于 `quality.rtmpose_train_hand_presence_threshold`（0813 复核后保持 `0.025`；负候选预审仍独立使用 `0.5`）；
- HCF handedness 分数低于 `quality.handedness_review_threshold`（当前 `0.7`）；
- 42 个 crop 坐标值中，精确边界值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold`（当前 `2`）；
- 开启连接长度门控时，任一指定连接的 crop 像素长度严格超过当前 `near/mid/far` 阈值。

连接长度门控由 `quality.rtmpose_train_connection_length_gate_enabled` 独立控制并默认开启；关闭时不解析距离或阈值。near/mid 阈值已按 6,095 条 Eos-2.0 人工复核 gold hand 的 `ceil(P99.95 × 1.05)` 重算；Eos-2.0 不支持 far，因此 far 只保留不可达的历史阈值，不宣称为新统计结果。完整统计见 `assets/quality_gate/rtmpose_connection_length_distribution.md`。

`quality.rtmpose_train_mediapipe_tflite_rescue_enabled` 默认开启。RTMPose 触发边界或已开启的连接长度门控时，程序批量调用 `models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite`；补救结果通过两项几何检查才替换关键点，否则保留原 RTMPose 点并继续拒绝。关闭后不读取 TFLite 模型或独立环境配置。该补救不是第五条门控，也不使用 TFLite 的 presence/handedness 输出。

Presence、边界和连接长度门控只作用于 RTMPose Train runtime；handedness 门控还适用于 MediaPipe Train positive。四条门控均不改变 Eval 发布，Eos low-score candidate 也不应用 RTMPose 门控。HCF presence 是教师伪标签，Eval 正式真值仍以 CVAT 人工复核为准。

发布报告会按既有发布优先级对四条门控做互斥计数：单来源写入 `source_publish_report.json`，dataset 总计及 `capture_source_id` 明细写入 `dataset_manifest.json`。

主环境由 `requirements.txt` 管理，当前使用 `onnxruntime-gpu==1.18.0`，并为其 ABI 约束 NumPy `<2`、OpenCV `<4.11`；既有 `anfab` 需移除 CPU 包 `onnxruntime` 后更新依赖。TFLite 补救仍使用独立 Python 3.11 环境和 `requirements-mediapipe-tflite.txt`。
