# HLMF 2.0 Quick Start

本文只保留操作。原理和排错见 [完整操作流程](HLMF_annotating_workflow.md)。

## A. 处理一个新来源

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab

export HLMF_SOURCE_ROOT=/root/autodl-tmp/DatesetFab/<source-name>
make paths
make validate_images
make palm_detection
make build_roi
make run_mediapipe
```

检查：

```text
$HLMF_SOURCE_ROOT/qc/
$HLMF_SOURCE_ROOT/02_roi_crops/images/
$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
```

普通全量 CVAT 复核：

```bash
make export_cvat
# 上传 02_roi_crops/images 和 02_roi_crops/cvat_autolabel.xml；完成后把返回 XML 保存为 03_reviewed/cvat_reviewed.xml
make import_cvat
make visualize
```

## B. 生成 HLML-3.0 的 pretrain/Val/Test 聚合

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0

make finalize_train_pretrain
make build_pretrain_source_registry
make finalize_val
make finalize_test
```

## C. Dragon Gold

```bash
export HAND_FINETUNE_ID=v3-finetune-r1
make prepare_dragon_gold
```

结果：

```text
$HAND_WORK_ROOT/finetune/$HAND_FINETUNE_ID/sources/gold/dragon_gold_0716_v1/
```

## D. 从上一数据快照继承 Gold

```bash
make seed_finetune_gold \
  BASE_FINETUNE_ID=v3-finetune-r1 \
  HAND_FINETUNE_ID=v3-finetune-r2
```

目标必须不存在；程序通过硬链接复用旧 Gold 并逐文件复验 SHA。

## E. 导出新录制 Gold（最多 300，人工总预算仍不得超过 800）

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-r2 \
  FINETUNE_SOURCE_ID=new_recorded_gold_r01 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=/root/autodl-tmp/DatesetFab/<new-source> \
  FINETUNE_MAX_ITEMS=300
```

## F. 导出 HLML 已选的 disagreement Gold

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-r2 \
  FINETUNE_SOURCE_ID=disagreement_gold_r02 \
  FINETUNE_SOURCE_MODE=selection_subset
```

查看每 100 张的分工计划：

```text
$HAND_WORK_ROOT/finetune/$HAND_FINETUNE_ID/cvat/<source_id>/qc/cvat_job_plan.json
```

CVAT 完成后把完整 task 的 `CVAT for images 1.1` XML 保存为 `reviewed.xml`，然后：

```bash
make import_finetune_gold HAND_FINETUNE_ID=v3-finetune-r2
make finalize_train_finetune HAND_FINETUNE_ID=v3-finetune-r2
```

## G. 代码检查

```bash
make compile
make test
```
