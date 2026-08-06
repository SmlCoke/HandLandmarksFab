# HandLandmarkerFab（HLMF 3.0）

HandLandmarkerFab 是 Hand Landmarker 训练系统的上游数据制作仓库。系统从 Eos Palm Detector 的 bbox、p0、p9 构造固定 `256×256` Hand ROI，再由 MediaPipe Tasks 或 RTMPose-m Hand5 生成 21 点草标；RTMPose runtime ROI 的 Left/Right 由独立 HCF ONNX 分类器给出。

本仓库只制作 Hand Landmarker 数据。Palm 几何不能人工修改，人工复核对象是程序生成的 Hand ROI、21 个关键点、handedness 以及 `no_hand/ignore_for_training` 状态。

## 入口文档

- [完整工作流](docs/annotating_system/HLMF_annotating_workflow.md)
- [快速开始](docs/annotating_system/HLMF_quick_start.md)
- [数据契约](docs/annotating_system/HLMF_data_contract.md)
- [当前状态](docs/annotating_system/HLMF_current_status.md)

## 当前教师后端

- 默认后端：`hand_landmark.backend: mediapipe_tasks`。
- 单次切换：`HAND_LANDMARK_BACKEND=rtmpose_onnx`。
- RTMPose：原始 SimCC logits 直接 argmax，除以固定 `2.0`；runtime ROI 总是输出 21 点。
- HCF：`models/hand_classifier/model.onnx`，输入灰度 `[N,1,256,256]`，输出 Left/Right 两类 logits；仅运行于 RTMPose runtime ROI。
- Eos low-score candidate 不运行 RTMPose/HCF，保持 `unknown/null`，进入人工候选链路。

ONNX 模型遵循仓库现有忽略策略，不纳入 Git；代码和配置通过 Git 同步，模型需在执行环境中单独部署。

## 常用命令

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab

make source-check DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice \
  PROPOSAL_VARIANT=eos-2.0

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

make registry-check
make compile
make test
make help
```

`autolabel-visualize-original` 默认在 PNG 完成后生成 `visualizations/original_image_landmarks/<variant>.mp4`；传 `ORIGINAL_VIDEO=false` 可只生成 PNG。

`source-variant-delete` 只删除精确来源/变体的派生产物，保留 `images/`、`raw_images.jsonl`、`source.json`，并写入永久 retired tombstone。同一来源不能再次使用被 retired 的变体名。若只想释放可重建的可视化空间，使用 `autolabel-visualizations-clean`，它不会写 tombstone。

## Train 质量边界

RTMPose Train runtime 行满足以下任一条件时整行进入 `ignored.jsonl`：

- HCF handedness 分数低于 `quality.handedness_review_threshold`（当前 `0.7`）；
- 42 个 crop 坐标值中，精确等于 `0.0` 或 `255.0` 的值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold`（当前 `3`）。

1–2 个边界值仍通过。该边界计数不用于 Eval、MediaPipe 或 low-score candidate。RTMPose 的 `hand_presence.present=true` 只是现有发布路由哨兵，不是真实 presence 标签；Iris geometry pretrain 必须忽略这些行的 presence，后续正式 multitask/评估应使用人工确认或独立真实标签。

依赖仍由现有 `requirements.txt` 管理，本次集成没有新增 Python 依赖。
