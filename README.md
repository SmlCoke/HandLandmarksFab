# HandLandmarkerFab（HLMF 2.0）

HLMF 是 HLML 的上游数据集制作系统：它把原始图片转换为 Hand ROI、生成 MediaPipe 伪标签、导出/导入 CVAT 人工复核结果，并发布带 SHA-256 证据的训练标签。

HLMF 2.0 是一套新的、精简的操作契约，不兼容旧版多套 Train/Val/Test 配置。任意普通来源都使用同一份 `configs/autolabel.yaml`，只通过 `HLMF_SOURCE_ROOT` 指定当前来源目录。

## 文档入口

- [完整操作流程](docs/annotating_system/HLMF_annotating_workflow.md)：首次使用、理解数据契约和排错时阅读。
- [Quick Start](docs/annotating_system/HLMF_quick_start.md)：熟悉流程后直接照着运行。
- [目录与接口](docs/annotating_system/HLMF_data_contract.md)：查询输入、输出、Gold 和 HLML 交接格式。
- [当前数据状态](docs/annotating_system/HLMF_current_status.md)：服务器上已归档批次和进行中的人工任务。
- [当前下一步计划](docs/annotating_system/HLMF_next_step_plan.md)：当前批次的路径、数量、人工分工和执行顺序。

## 两个根目录

```text
HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
  可再生数据仓库

HAND_GOLD_ROOT=$HAND_DATASET_ROOT/GoldSource
  跨训练版本复用的 Gold 真源；结构为 domain/source-id/{source,task,published}

HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
  当前版本的聚合、mining/replay 和 HLML 训练结果
```

普通来源的 00～06 直接在 `HLMF_SOURCE_ROOT` 内工作。Gold 的 CVAT task 与 published source 也长期保存在 GoldSource，不再绑定某个 `HAND_FINETUNE_ID`；当前训练版本只生成可重建的认证聚合。

## 最短命令索引

```bash
conda activate anfab
make paths HLMF_SOURCE_ROOT=/path/to/source
make autolabel HLMF_SOURCE_ROOT=/path/to/source AUTOLABEL_ROLE=train
make export_cvat HLMF_SOURCE_ROOT=/path/to/source

# 人工在 CVAT 完成后放回 reviewed XML，再继续：
make import_cvat visualize HLMF_SOURCE_ROOT=/path/to/source

# 测试代码
make compile
make test
```

完整的 pretrain 聚合、Dragon 多批次 Gold 和多轮 finetune Gold 流程见完整操作文档；当前一次性任务目标只记录在下一步计划中。
