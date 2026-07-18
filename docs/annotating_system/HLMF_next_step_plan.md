# HLMF 当前下一步计划

更新时间：2026-07-18。本文只记录当前任务；通用方法见 [完整标注流程](HLMF_annotating_workflow.md)，最短命令见 [Quick Start](HLMF_quick_start.md)。

## 1. 当前事实

- HLML `pretrain-geometry` 正在训练；本轮 HLMF/HLML 更新不修改其 `HLML-3.0` 输入和 run。
- `v3-pretrain-r1` 的人工负样本删除复核已经完成。
- 新录制 TIFF 图片流已经在 `HandViolenceHard0718/peak` 跑完 00～03，自动标注使用 `negative_candidate_threshold=0.3`。
- `new_recorded_gold_r01` 已确定性导出 300 个 ROI，CVAT 人工标注正在进行。
- 本轮允许继续增加 `new_recorded_gold_r02`、`r03` 等独立批次；所有人工 Gold 的合计人工上限仍不超过当前团队可完成的 800 个 ROI。

## 2. 已归档的长期 Gold

规范仓库是：

```text
/root/autodl-tmp/DatesetFab/GoldSource/
├── new_recorded_gold/
├── disagreement_gold/
├── negative_removed_gold/
└── dragon/
```

历史 `disagreement_gold` 和 `negative_removed_gold` 都是人工精标数据，不应废弃。它们已作为独立 published 批次归档；后续 HLML 可逐批选择。Dragon 也保留，但因 H.264/I420/JPEG 与板端无损 TIFF 域不一致，本次选择应设为 disabled。

查看每批是否齐全：

```bash
find "$HAND_GOLD_ROOT" -path '*/published/finetune_source.json' -print
find "$HAND_GOLD_ROOT" -path '*/task/task_descriptor.json' -print
```

## 3. 完成 `new_recorded_gold_r01`

把 CVAT 完整导出 XML 上传到：

```text
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r01/task/reviewed.xml
```

然后只导入这一批，避免其他未完成人工任务阻塞 `--all`：

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_GOLD_ROOT=$HAND_DATASET_ROOT/GoldSource
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-r1
export HAND_FINETUNE_ID=v3-finetune-r1

make import_finetune_gold FINETUNE_SOURCE_ID=new_recorded_gold_r01
```

检查：

```text
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r01/published/finetune_source.json
.../published/qc/gold_source_report.json
```

## 4. 制作额外的新录制批次

每批使用新的图片流和唯一 ID，例如 `new_recorded_gold_r02`。直接把图片放在：

```text
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/source/images/
```

建议优先录制当前模型薄弱姿态：握拳及开合过程、数字 1、侧向张掌、手指遮挡、手腕旋转、画面边缘、不同距离和左右手。必须保存板端同域的无损 TIFF 图片流，不导出 H.264 视频再抽 JPEG。

运行：

```bash
make autolabel \
  HLMF_SOURCE_ROOT=$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/source \
  AUTOLABEL_ROLE=train \
  AUTOLABEL_OVERRIDES='{"palm":{"negative_candidate_threshold":<本批冻结阈值>}}'

make export_finetune_gold \
  FINETUNE_SOURCE_ID=new_recorded_gold_r02 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/source \
  FINETUNE_MAX_ITEMS=<本批人工上限>
```

阈值根据本批 Palm 分布决定，不沿用 `0.3` 作为永久常量。查看 `source/qc/*` 中的 `autolabel_runtime`，确认四阶段参数一致。

人工完成后上传到 `.../new_recorded_gold_r02/task/reviewed.xml`，再按 source ID 单批导入。

## 5. disagreement / negative-removed 新批次

HLML 生成 selection request 后，HLMF 只负责恢复认证 ROI、导出 CVAT 和发布：

```bash
make export_finetune_gold \
  FINETUNE_SOURCE_ID=disagreement_gold_<round-id> \
  FINETUNE_SOURCE_MODE=selection_subset
```

如当前计划决定继续采样 `negative_removed_gold`，也必须使用新的 source ID。HLML 会扫描 GoldSource 中所有历史 `task/` 和 `published/`，不会再次抽到已经标注或已进入待标任务的 ROI。

## 6. 全部人工批次完成后的聚合

所有计划使用的 task 都已导入后：

```bash
make finalize_train_finetune HAND_FINETUNE_ID=v3-finetune-r1
```

输出位于：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/finetune/v3-finetune-r1/hmlf_gold_merged/
```

这里是本次训练版本的认证聚合，不是 Gold 真源。Gold 真源始终位于 `DatesetFab/GoldSource`。

## 7. 交接 HLML

HLMF 聚合完成后，HLML 先生成逐批选择清单。当前建议启用：历史 `disagreement_gold`、历史 `negative_removed_gold`、本轮已经完成的每个 `new_recorded_gold_r*`；禁用 `dragon_gold_0716_v1`。具体命令和检查见 HLML 的 `HLML_next_step_plan.md`。
