# HandLandmarkerFab（HLMF 3.0）

HLMF 是 Hand Landmarker 的上游数据发布系统。它调用既有 Palm Detector（产品名 **Eos**）生成 proposal，再由程序生成固定的 `256×256` Hand ROI，并在 ROI 内运行可配置的 MediaPipe Tasks 或 RTMPose-m Hand5 教师。

HLMF **不制作 Palm Detector 训练数据**，不提供 Palm CVAT、Palm 标签导入、bbox/p0/p9 修改或人工绘制 Hand ROI 的入口。Palm 输出始终原样使用。唯一的人工复核对象是程序生成的 Hand ROI：21 点、handedness、`no_hand` 与 `ignore_for_training`。

公共配置按单一职责拆分：`autolabel.yaml` 只负责 Palm/ROI/Hand landmark 自动标注，`review.yaml` 只负责 Hand ROI CVAT 复核，`datasets.yaml` 只负责数据目录与发布策略，`cvat_label.json` 是 CVAT label schema。

## 模型产品命名

- Palm Detector：**Eos**，如第一缕微光划破黑暗，模型首先从灰度画面中发现并定位手掌，为后续链路指明方向。
- Hand Landmarker：**Iris**，模型连接离散关键点，将像素编织成完整、可解释的手部几何结构。
- Gloss Translator：**Muse**，模型为物理动作赋予语言与语义，将骨骼序列转化为人类可读的 Gloss。

当前自动标注使用冻结版本 `eos-1.0`，模型文件为 `models/palm_detector/eos-1.0/model_opt.onnx`。后续 Eos 版本统一放入 `models/palm_detector/eos-*/`，并使用相同版本名作为 `PROPOSAL_VARIANT`。

Hand landmark 默认后端为 `hand_landmark.backend: mediapipe_tasks`。已集成的可选后端 `rtmpose_onnx` 使用 `models/rtmpose/rtmpose-m_hand5_256x256.onnx`；单次运行可追加 `HAND_LANDMARK_BACKEND=rtmpose_onnx`，不改变 Palm、ROI、输出路径或 JSONL 字段集合。模型目录继续遵循仓库现有忽略策略，部署时必须单独准备 ONNX 文件。

## 文档入口

- [完整工作流](docs/annotating_system/HLMF_annotating_workflow.md)
- [Quick Start](docs/annotating_system/HLMF_quick_start.md)
- [数据契约](docs/annotating_system/HLMF_data_contract.md)
- [当前状态](docs/annotating_system/HLMF_current_status.md)

## 公共命令

```bash
make help
make source-check DATASET_SCOPE=pretrain DATASET_ID=demo CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice PROPOSAL_VARIANT=eos-1.0
make train-autolabel DATASET_SCOPE=pretrain DATASET_ID=demo CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice PROPOSAL_VARIANT=eos-1.0
make train-autolabel DATASET_SCOPE=pretrain DATASET_ID=demo CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice PROPOSAL_VARIANT=eos-1.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
make eval-autolabel DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=eos-1.0
make autolabel-visualize-roi DATASET_SCOPE=pretrain/eval DATASET_ID=demo CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-1.0
make autolabel-visualize-original DATASET_SCOPE=pretrain/eval DATASET_ID=demo CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-1.0
make hand-cvat-export DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=eos-1.0
make hand-cvat-import DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=eos-1.0
make source-publish DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=eos-1.0
make registry-check
make compile test
```

长期图片、标签和 registry 只写入 `HAND_DATASET_ROOT`。HLMF 不迁移或删除旧 schema 数据；3.0 只发布新契约数据。

RTMPose 对每个 Eos runtime ROI 固定输出 21 点；Eos 低分候选不会送入 RTMPose，仍以 `unresolved/unlabeled_v1` 进入 `candidate_negatives.jsonl` 人工审核链路。RTMPose 行中的 `hand_presence.present=true` 只是现有发布路由哨兵，`handedness=unknown/null`；HLML 只能在 Iris 第一阶段 geometry pretrain 中使用这些行，并必须屏蔽 presence 与 handedness loss。多任务训练和正式评估需要独立分类器或人工确认的真实标签。
