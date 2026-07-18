# HLMF 最终 Finetune 数据制作计划

更新时间：2026-07-18。本文件是当前冲刺的执行计划，不是通用手册。通用原理见 [HLMF 完整标注流程](HLMF_annotating_workflow.md)。

目标：重新录制板端同域无损 TIFF，制作多个彼此独立的 `new_recorded_gold_r*` 批次；人工 Gold 总量最多 800 ROI；随后发布可能存在的 disagreement Gold，并为 HLML 最终 finetune 生成认证聚合。

## 1. 固定环境与本轮 ID

每次重新登录服务器先执行：

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_GOLD_ROOT=$HAND_DATASET_ROOT/GoldSource
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-r1
export HAND_FINETUNE_ID=v3-finetune-final-r1

make paths
make compile
make test
```

本轮不复用 `new_recorded_gold_r01`。建议制作两批：

```text
new_recorded_gold_r02：正面、握拳/张开过渡、数字 1 等主要失败姿态
new_recorded_gold_r03：侧向、旋转、遮挡、边缘、远近尺度和左右手
```

建议先按 `400 + 400` 规划，最终以两个 task descriptor 的实际数量之和为准，且不得超过 800。

## 2. 人工录制要求

对每一批单独录制、单独建目录。必须直接保存板端/部署链路的 `1280×720` 灰度无损 TIFF，不经过 H.264、I420/YUV 4:2:0 或 JPEG。

录制时画面内只出现一只手。不要依赖后续 ROI 自动把两只手拆开；如果第二只手进入画面或同一 Palm ROI，应当当场重录。每种姿态包含缓慢静止、轻微位移和自然过渡，不要用大量近重复连续帧凑数量。

第一批重点：

- 完全握拳、半握拳、握拳到张掌的连续变化；
- 数字 1、弯曲食指、拇指内收/外展；
- 正面张掌但手指间距和手腕角度不同；
- 左右手数量尽量平衡。

第二批重点：

- 手掌侧对镜头、手背/手心旋转、手腕旋转；
- 手指互相遮挡、局部接近画面边缘；
- 近、中、远三种尺度；
- 不同背景和亮度，但仍保证关键点肉眼可判断。

原图分别放入：

```text
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/source/images/
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/source/images/
```

两个批次不得混用图片，不复制 r01 图片。

## 3. 每批运行 HLMF 00～03

以下命令对 r02、r03 分别执行一次。`0.3` 只能作为同设备条件下的初始参考；先看本批 Palm 统计，再决定是否重建该批，不能把阈值当作永久常量。

```bash
export BATCH_ID=new_recorded_gold_r02
export HLMF_SOURCE_ROOT=$HAND_GOLD_ROOT/new_recorded_gold/$BATCH_ID/source

make autolabel \
  HLMF_SOURCE_ROOT=$HLMF_SOURCE_ROOT \
  AUTOLABEL_ROLE=train \
  AUTOLABEL_OVERRIDES='{"palm":{"negative_candidate_threshold":0.3}}'
```

r03 时只替换 `BATCH_ID`。检查：

```bash
test -f "$HLMF_SOURCE_ROOT/qc/image_validation_report.json"
test -f "$HLMF_SOURCE_ROOT/qc/palm_detection_stats.json"
test -f "$HLMF_SOURCE_ROOT/02_roi_crops/hand_roi_crops_manifest.jsonl"
test -f "$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl"

find "$HLMF_SOURCE_ROOT/02_roi_crops/images" -maxdepth 1 -type f | wc -l
```

人工快速浏览 `source/02_roi_crops/images/`，重点检查是否仍有单个 ROI 同时包含双手。只要这种情况成批出现，就不要导出 CVAT；作废该 source ID，修正录制方式后用新 ID 重新制作。

## 4. 冻结两个新录制 CVAT 任务

r02：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_SOURCE_ID=new_recorded_gold_r02 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/source \
  FINETUNE_MAX_ITEMS=400
```

r03：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=$HAND_FINETUNE_ID \
  FINETUNE_SOURCE_ID=new_recorded_gold_r03 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/source \
  FINETUNE_MAX_ITEMS=400
```

查看真实任务数量和分工边界：

```text
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/task/task_descriptor.json
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/task/qc/cvat_job_plan.json
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/task/task_descriptor.json
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/task/qc/cvat_job_plan.json
```

task 目录中的 `images/` 是冻结的所选 ROI 快照，与 source 能共用的图片使用硬链接，不占第二份图片数据块。

## 5. CVAT 人工标注与分工

每批建立一个 CVAT image task。上传该批 `task/images/` 和 `task/cvat_autolabel.xml`；不要跨批混合，也不要重命名图片。

按 `qc/cvat_job_plan.json` 的 job 边界分工。每张 ROI 只能选择一种结果：

- 清晰单手：完整修正 21 点并标 Left、Right 或 unknown；
- 确定无手：标 `no_hand`；
- 双手同时进入 ROI、严重截断、模糊或无法可靠标点：标 `ignore_for_training`。

不要尝试用一组 21 点同时描述两只手。团队先共同校准少量拳头、侧掌、数字 1 样本，再正式分工。

从完整 CVAT task 导出 `CVAT for images 1.1`，分别上传到：

```text
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/task/reviewed.xml
$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/task/reviewed.xml
```

## 6. 严格导入两个批次

逐批导入，便于定位错误：

```bash
make import_finetune_gold FINETUNE_SOURCE_ID=new_recorded_gold_r02
make import_finetune_gold FINETUNE_SOURCE_ID=new_recorded_gold_r03
```

成功后应满足：

```bash
test -f "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/published/finetune_source.json"
test ! -e "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r02/task"
test -f "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/published/finetune_source.json"
test ! -e "$HAND_GOLD_ROOT/new_recorded_gold/new_recorded_gold_r03/task"
```

`reviewed.xml`、自动标注 XML 和任务描述符已经转存至各自 `published/audit/`。不要手工恢复 task。

## 7. 接收 HLML 生成的 disagreement 余量

HLML 完成 multitask、`prepare-finetune-sources` 和 `prepare-finetune-round` 后，查看：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/finetune/v3-finetune-final-r1/mining/rounds/final_r01/disagreement_gold_final_r01/selection_report.json
```

如果报告的选中数为 0，跳过本节。如果大于 0：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-final-r1 \
  FINETUNE_SOURCE_ID=disagreement_gold_final_r01 \
  FINETUNE_SOURCE_MODE=selection_subset
```

按第 5 节完成 CVAT，上传到：

```text
$HAND_GOLD_ROOT/disagreement_gold/disagreement_gold_final_r01/task/reviewed.xml
```

再执行：

```bash
make import_finetune_gold FINETUNE_SOURCE_ID=disagreement_gold_final_r01
```

本轮人工总量是 r02 task 数 + r03 task 数 + disagreement task 数，必须不超过 800。

## 8. 生成最终 HLMF Gold 聚合

确认所有本轮 task 均已 published 后：

```bash
make finalize_train_finetune HAND_FINETUNE_ID=v3-finetune-final-r1
```

检查：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/finetune/v3-finetune-final-r1/hmlf_gold_merged/hmlf_gold_aggregate.json
/root/autodl-tmp/TrainFab/HLML-3.0/finetune/v3-finetune-final-r1/hmlf_gold_merged/qc/finalize_train_finetune_report.json
```

聚合应发现：

- `disagreement_gold_hlml2.0`；
- `negative_removed_gold_hlml2.0`；
- `dragon_gold_0716_v1`；
- `new_recorded_gold_r02`、`new_recorded_gold_r03`；
- `disagreement_gold_final_r01`（若实际制作）。

HLMF 聚合会认证全部 published；是否进入最终训练由 HLML 的逐批 `gold_selection.yaml` 决定。完成后切换到 `/root/HandLandmarkerLab`，继续 HLML 下一步计划第 7 节。
