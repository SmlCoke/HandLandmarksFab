# HandLandmarkerFab（HLMF 3.0）

HandLandmarkerFab 是 Hand Landmarker 训练系统的上游数据制作仓库。系统从 Eos Palm Detector 的 bbox、p0、p9 构造固定 `256×256` Hand ROI，再由 MediaPipe Tasks 或 RTMPose-m Hand5 生成 21 点草标；RTMPose runtime ROI 的 Left/Right 与 hand presence 由独立双头 HCF ONNX 分类器给出。

本仓库只制作 Hand Landmarker 数据。Palm 几何不能人工修改，人工复核对象是程序生成的 Hand ROI、21 个关键点、handedness 以及 `no_hand/ignore_for_training` 状态。

Hand ROI 的模型输入契约是解码后的单通道 `uint8 256×256` 像素。仓库使用无损 PNG 保存 ROI；PNG/TIFF 是存储容器，不改变相同灰度数组的像素域。板端从 `SSNE_Y_8` 摄像头内存直接构造 ROI，并不把 TIFF 文件送入 Hand Landmarker。

## 入口文档

- [完整工作流](docs/annotating_system/HLMF_annotating_workflow.md)
- [快速开始](docs/annotating_system/HLMF_quick_start.md)
- [数据契约](docs/annotating_system/HLMF_data_contract.md)
- [当前状态](docs/annotating_system/HLMF_current_status.md)
- [常见问题与解答](docs/annotating_system/HLMF_qa.md)

## 当前教师后端

- 默认后端：`hand_landmark.backend: mediapipe_tasks`。
- 单次切换：`HAND_LANDMARK_BACKEND=rtmpose_onnx`。
- RTMPose：原始 SimCC logits 直接 argmax，除以固定 `2.0`；runtime ROI 总是输出 21 点。
- HCF：`models/handedness-handpresence-0807/model.onnx`，输入灰度 `[N,1,256,256]`，输出 `handedness` 与 `hand_presence` 两个 `[N,2]` logits；仅运行于 RTMPose runtime ROI。旧 handedness-only 资产保存在 `models/handedness-0806/`。
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

- `hand_presence.score=P(has_hand)` 缺失、非有限，或低于 `quality.rtmpose_train_hand_presence_threshold`（当前 `0.5`）；
- HCF handedness 分数低于 `quality.handedness_review_threshold`（当前 `0.7`）；
- 42 个 crop 坐标值中，精确等于 `0.0` 或 `255.0` 的值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold`（当前 `2`）。

Presence 阈值来自 7,907 条人工复核 Eval ROI：`0.5` 拒绝全部 15 条 no_hand，同时保留 7,856/7,892 条正样本（99.5438%），并与模型 `no_hand/has_hand` 的 argmax 决策边界一致。等于阈值时 quality gate 通过；0–1 个边界值通过。这些门控不作用于 Eval、MediaPipe 或 low-score candidate。HCF presence 是教师伪标签，Eval 正式真值仍以 CVAT 人工复核为准。

依赖仍由现有 `requirements.txt` 管理，本次集成没有新增 Python 依赖。
