# HLMF Quick Start

本文只列通用命令。当前要使用的 ID、路径和数量见 [当前下一步计划](HLMF_next_step_plan.md)。

## 1. 初始化

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
make compile
make test
```

## 2. 普通来源 00～03

```bash
# 训练来源，使用默认阈值
make autolabel HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id> AUTOLABEL_ROLE=train

# 训练来源，覆盖本批低分负样本阈值
make autolabel HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id> \
  AUTOLABEL_ROLE=train \
  AUTOLABEL_OVERRIDES='{"palm":{"negative_candidate_threshold":<threshold>}}'

# Val/Test：负样本候选会被强制关闭
make autolabel HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<val-source> AUTOLABEL_ROLE=val
make autolabel HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<test-source> AUTOLABEL_ROLE=test
```

需要普通全量 CVAT 时：

```bash
make export_cvat
# 返回 XML 放入 $HLMF_SOURCE_ROOT/03_reviewed/cvat_reviewed.xml
make import_cvat
make visualize
```

## 3. 聚合 pretrain/Val/Test

```bash
make finalize_train_pretrain
make build_pretrain_source_registry
make finalize_val
make finalize_test
```

## 4. 发布每一批 Dragon Gold

```bash
make prepare_dragon_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  DRAGON_SOURCE_ROOT=$HAND_DATASET_ROOT/<dragon-batch-root> \
  DRAGON_BATCH_ID=<unique-dragon-batch-id>
```

N 批重复 N 次，每批换唯一 ID。

## 5. 导出新录制 Gold

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=<new-recorded-source-id> \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id> \
  FINETUNE_MAX_ITEMS=<task-limit>
```

## 6. 导出 HLML disagreement Gold

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=disagreement_gold_<round-id> \
  FINETUNE_SOURCE_MODE=selection_subset
```

按 `cvat/<source-id>/qc/cvat_job_plan.json` 分工，返回 XML 保存为 `cvat/<source-id>/reviewed.xml`，然后：

```bash
make import_finetune_gold HAND_FINETUNE_ID=<finetune-data-id>
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

## 7. 新快照继承旧 Gold

```bash
make seed_finetune_gold \
  BASE_FINETUNE_ID=<old-data-id> \
  HAND_FINETUNE_ID=<new-data-id>
```
