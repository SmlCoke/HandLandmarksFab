# HandLandmarkerFab（HLMF 3.0）

HLMF 是 Hand Landmarker 的上游数据发布系统。它调用既有 Palm Detector 生成 proposal，再由程序生成固定的 `256×256` Hand ROI，并在 ROI 内运行 MediaPipe Hand Landmarker。

HLMF **不制作 Palm Detector 训练数据**，不提供 Palm CVAT、Palm 标签导入、bbox/p0/p9 修改或人工绘制 Hand ROI 的入口。Palm 输出始终原样使用。唯一的人工复核对象是程序生成的 Hand ROI：21 点、handedness、`no_hand` 与 `ignore_for_training`。

公共配置按单一职责拆分：`autolabel.yaml` 只负责 Palm/ROI/MediaPipe 自动标注，`review.yaml` 只负责 Hand ROI CVAT 复核，`datasets.yaml` 只负责数据目录与发布策略，`cvat_label.json` 是 CVAT label schema。

## 文档入口

- [完整工作流](docs/annotating_system/HLMF_annotating_workflow.md)
- [Quick Start](docs/annotating_system/HLMF_quick_start.md)
- [数据契约](docs/annotating_system/HLMF_data_contract.md)
- [当前状态](docs/annotating_system/HLMF_current_status.md)

## 公共命令

```bash
make help
make source-check DATASET_SCOPE=pretrain DATASET_ID=demo CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice PROPOSAL_VARIANT=palm-v1
make train-autolabel DATASET_SCOPE=pretrain DATASET_ID=demo CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice PROPOSAL_VARIANT=palm-v1
make eval-autolabel DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
make hand-cvat-export DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
make hand-cvat-import DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
make source-publish DATASET_SCOPE=eval DATASET_ID=demo-eval CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
make registry-check
make compile test
```

长期图片、标签和 registry 只写入 `HAND_DATASET_ROOT`。HLMF 不迁移或删除旧 schema 数据；3.0 只发布新契约数据。
