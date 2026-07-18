# HLMF 当前下一步计划

更新时间：2026-07-18。本文只记录本次两天冲刺；通用方法见 [完整标注流程](HLMF_annotating_workflow.md)。

## 1. 当前目标和已有 Gold

- 当前 finetune 数据快照：`v3-finetune-r1`。
- 已有 Dragon 批次：`dragon_gold_0716_v1`，已发布 5,191 个 ROI，其中 5,189 个可训练、2 个 ignored；不要重复执行 prepare，也不要删除或覆盖。
- 本轮新增人工 Gold 总预算优先冻结为 800 个 Hand ROI；如果实际人手不足，在导出任何 disagreement 任务前统一降为 600。
- 800 方案：新录制来源最多 300，disagreement 使用剩余额度；600 方案：新录制来源最多 200，disagreement 使用剩余额度。
- 本轮不再制作新的 `negative_removed_gold`。其含义是“旧负样本候选经人工发现实际有手后，转为精标正样本”；当前新工作区没有必须补做的同类任务。

## 2. 人工录制 source e

建议总录制 20～40 分钟，分成多个短视频，至少覆盖：

1. 握拳及从张掌缓慢收拳、从拳头展开；
2. 数字 1，食指伸直、略弯、正面和侧面；
3. 张掌侧对镜头、手背/手心、手腕旋转；
4. 手指互相遮挡、两手靠近或短暂重叠；
5. 手靠近/远离镜头、位于画面中心和边缘；
6. 左右手、不同参与者、背景和明暗条件。

保持横屏 `1280×720`，手应大部分时间完整可见。不要长时间保持同一静止姿态。原视频放在：

```text
/root/autodl-tmp/DatesetFab/finetune_source_e_r01/raw_videos/
```

程序当前不负责抽视频帧。每个视频稀疏抽取约每 2 秒一帧，并用 session 前缀避免重名：

```bash
mkdir -p /root/autodl-tmp/DatesetFab/finetune_source_e_r01/images
ffmpeg -i raw_videos/person01_session01.mp4 \
  -vf "fps=1/2,format=gray" \
  images/person01_session01_%06d.tiff
```

如果原视频不是正向 `1280×720`，先按实际方向旋转/缩放后再导出；不要拉伸宽高比。最终以 `make validate_images` 为准。

## 3. 程序处理 source e

```bash
cd /root/HandLandmarksFab
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_FINETUNE_ID=v3-finetune-r1
export HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/finetune_source_e_r01

make validate_images
make palm_detection
make build_roi
make run_mediapipe
```

人工只需要检查：

```text
$HLMF_SOURCE_ROOT/qc/image_validation_report.json
$HLMF_SOURCE_ROOT/qc/palm_detection_stats.json
$HLMF_SOURCE_ROOT/qc/mediapipe_roi_stats.json
$HLMF_SOURCE_ROOT/02_roi_crops/images/
```

抽看 ROI 是否包含整手、困难手势是否真正出现。若大量无手或裁切异常，先修正录制/抽帧或 Palm 输入，不进入 CVAT。

## 4. 先导出新录制 Gold

只能在团队确认本轮采用哪档总预算后运行。800 方案：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_SOURCE_ID=new_recorded_gold_r01 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HLMF_SOURCE_ROOT \
  FINETUNE_MAX_ITEMS=300
```

若采用 600 方案，只把 `FINETUNE_MAX_ITEMS` 改为 `200`。查看实际选中了多少张：

```text
/root/autodl-tmp/TrainFab/HLML-3.0/finetune/v3-finetune-r1/cvat/new_recorded_gold_r01/task_descriptor.json
.../cvat/new_recorded_gold_r01/qc/cvat_job_plan.json
.../cvat/new_recorded_gold_r01/02_roi_crops/images/
```

任务生成后不要换预算、改图片或重跑同一 ID。

## 5. 等 HLML 生成 disagreement request

HLML 完成 geometry、负样本复核、multitask 和 `make prepare-finetune-sources` 后，运行本轮选择。800 方案命令由 HLML 仓库执行：

```bash
make prepare-finetune-round \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_ROUND_ID=r01 \
  FINETUNE_GOLD_BUDGET=800 \
  NEW_RECORDED_SOURCE_ID=new_recorded_gold_r01
```

600 方案把预算改为 `600`。HLML 会用 disagreement 自动补齐“总预算减 new-recorded 实际任务数”，并排除 Dragon、已有 Gold、当前 CVAT、Val/Test 和历史 request 重复项。

返回 HLMF 导出：

```bash
cd /root/HandLandmarksFab
make export_finetune_gold \
  HAND_FINETUNE_ID=v3-finetune-r1 \
  FINETUNE_SOURCE_ID=disagreement_gold_r01 \
  FINETUNE_SOURCE_MODE=selection_subset
```

检查：

```text
.../finetune/v3-finetune-r1/cvat/disagreement_gold_r01/task_descriptor.json
.../finetune/v3-finetune-r1/cvat/disagreement_gold_r01/qc/cvat_job_plan.json
```

## 6. 团队 CVAT 分工

人工量只包括 `new_recorded_gold_r01` 和 `disagreement_gold_r01`，两者合计不超过冻结的 600/800。开始前所有人共同标 10 张校准图；之后按各自 `cvat_job_plan.json` 的不重叠 job 工作。

每张 ROI：清楚时标完整 21 点和 handedness；确定无手时标 `no_hand`；模糊、严重截断或点无法可靠落在 ROI 内时标 `ignore_for_training`。负责人抽查每人约 5%，重点看腕点、拇指、左右手和握拳遮挡点。

每个完整 task 导出 `CVAT for images 1.1`，分别上传为：

```text
.../cvat/new_recorded_gold_r01/reviewed.xml
.../cvat/disagreement_gold_r01/reviewed.xml
```

## 7. 导入、聚合和交接

```bash
cd /root/HandLandmarksFab
make import_finetune_gold HAND_FINETUNE_ID=v3-finetune-r1
make finalize_train_finetune HAND_FINETUNE_ID=v3-finetune-r1
```

必须查看：

```text
.../finetune/v3-finetune-r1/sources/gold/new_recorded_gold_r01/qc/gold_source_report.json
.../finetune/v3-finetune-r1/sources/gold/disagreement_gold_r01/qc/gold_source_report.json
.../finetune/v3-finetune-r1/hmlf_gold_merged/qc/finalize_finetune_report.json
```

只有报告 `status=ok`、计数符合实际且无 blocking error，才交回 HLML 执行 finetune curate/gate/smoke。若时间只够完成一部分，必须在 CVAT 任务冻结前降低总预算；不要冻结大任务后只返回部分 XML。
