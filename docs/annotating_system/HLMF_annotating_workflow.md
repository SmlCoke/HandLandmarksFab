# HLMF 2.0 完整操作流程

## 1. 目标与边界

HLMF 负责“从图片到可认证训练标签”，不负责训练模型。HLML 负责读取 HLMF 发布的标签、筛选下一轮高价值 ROI、训练、评估和导出。

HLMF 2.0 不兼容旧版的多套 `autolabel_train/val/vali/test.yaml`。所有普通来源都走同一个 00～06 流程；来源差异只由 `HLMF_SOURCE_ROOT` 表达。

## 2. 初始化

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab
make compile
make test
```

服务器统一设置：

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=v3-pretrain-r1
export HAND_FINETUNE_ID=v3-finetune-r1
```

`DatesetFab` 是可再生数据仓库；不要为了运行 HLML 再复制出 `HLML-3.0/train_sources`。`HLML-3.0` 保存聚合标签和运行结果。

## 3. 制作一个普通来源

### 3.1 人工准备

新建独立来源目录，例如：

```text
/root/autodl-tmp/DatesetFab/new_recorded_0720_r01/images/
```

人工只做以下事情：

1. 把原始文件整理为正向的 `1280×720` 灰度 TIFF；不要在后续阶段重新编码。
2. 不把 Val/Test 或已经用于固定 infer 的图片混入训练来源。
3. 不同人员、背景、距离和手势使用独立 session/目录；避免长串近重复帧。

```bash
export HLMF_SOURCE_ROOT=/root/autodl-tmp/DatesetFab/new_recorded_0720_r01
make paths
```

### 3.2 阶段 00：图片检查

```bash
make validate_images
```

程序检查可读性、分辨率、通道和方向契约，结果位于：

```text
$HLMF_SOURCE_ROOT/qc/image_validation_report.json
```

失败时修正原始输入后重跑；不要跳过。

### 3.3 阶段 01：Palm 检测

```bash
make palm_detection
```

结果：

```text
$HLMF_SOURCE_ROOT/01_palm/palm_detections.jsonl
$HLMF_SOURCE_ROOT/qc/palm_detection_stats.json
```

### 3.4 阶段 02：Hand ROI

```bash
make build_roi
```

程序按板端一致的旋转、平移和缩放几何生成 `256×256` 灰度 ROI：

```text
$HLMF_SOURCE_ROOT/02_roi_crops/images/
$HLMF_SOURCE_ROOT/02_roi_crops/hand_roi_crops_manifest.jsonl
```

### 3.5 阶段 03：MediaPipe 伪标签

```bash
make run_mediapipe
```

可临时加 `VISUALIZE_MEDIAPIPE_ROIS=1`。输出：

```text
$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
$HLMF_SOURCE_ROOT/qc/mediapipe_roi_stats.json
```

### 3.6 阶段 04～06：普通 CVAT 复核

```bash
make export_cvat
```

把 ROI 图片和 `02_roi_crops/cvat_autolabel.xml` 上传到 CVAT。每张 ROI 完成 21 点/handedness、`no_hand` 或忽略决策。导出 `CVAT for images 1.1`，保存为 `03_reviewed/cvat_reviewed.xml` 后：

```bash
make import_cvat
make visualize
```

人工只编辑 CVAT 标注，不编辑 JSONL、SHA、descriptor 或聚合配置。

## 4. 人工负样本删除复核

HLML 的 pretrain curation 会生成待复核图片树。可 zip/7z 压缩、网盘传输、解压后人工删除所有含手、模糊或不易识别的图片，再保持原相对路径压缩上传。

只要压缩/解压工具没有重新编码图片、没有修改文件内容，同一文件的 SHA-256 不变。文件名、时间戳或压缩包自身 SHA 改变不影响图片内容 SHA。上传回服务器后由 HLML 的 `make pretrain-curate-reviewed` 逐文件复验。

## 5. 直接从 DatesetFab 聚合

### 5.1 pretrain

`configs/finalize_train.yaml` 直接读取：

- `HandViolence0708`；
- `HandViolenceEnhanced0714` 的 peak/soar/dragon 来源。

```bash
make finalize_train_pretrain
make build_pretrain_source_registry
```

输出只包含标签、注册表和 QC：

```text
$HAND_WORK_ROOT/train_pretrain_merged/
```

ROI 图片继续留在 `DatesetFab`。

### 5.2 Val/Test

```bash
make finalize_val
make finalize_test
```

程序直接读取 `$HAND_DATASET_ROOT/eval_sources`，输出：

```text
$HAND_WORK_ROOT/val_merged/
$HAND_WORK_ROOT/test_merged/
```

## 6. Finetune Gold 工作区

### 6.1 Dragon Gold

```bash
make prepare_dragon_gold
```

默认从 DatesetFab 的 Dragon 原始目录读取，发布为：

```text
$HAND_WORK_ROOT/finetune/$HAND_FINETUNE_ID/sources/gold/dragon_gold_0716_v1
```

### 6.2 从上一轮继承认证 Gold

```bash
make seed_finetune_gold \
  BASE_FINETUNE_ID=v3-finetune-r1 \
  HAND_FINETUNE_ID=v3-finetune-r2
```

程序要求目标工作区完全不存在；递归拒绝符号链接，优先为所有文件建立硬链接，复验每个文件 SHA，再生成 `qc/seed_finetune_gold_report.json`。旧工作区不修改。

如果从零重建 v3 数据，不需要 seed；先准备 Dragon 和各轮 Gold 即可。

## 7. 团队制作 600/800 个新 Gold ROI

### 7.1 冻结预算

正式导出前按参与人数选择 600 或 800。硬上限为 800；不要生成 800 后只完成一部分。

推荐 800 分配：

- 新录制 difficult source：最多 300；
- HLML disagreement：用 `800 - 新录制实际任务数` 自动补足。

600 预算对应新录制最多 200，其余由 disagreement 补足。

### 7.2 新录制来源限额导出

先按第 3 节完成来源的 00～03，然后：

```bash
make export_finetune_gold \
  FINETUNE_SOURCE_ID=new_recorded_gold_r01 \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HLMF_SOURCE_ROOT \
  FINETUNE_MAX_ITEMS=300
```

程序按 session、来源图片和稳定哈希做确定性分散抽样，不要求人工挑图。未给 `FINETUNE_MAX_ITEMS` 会 fail-closed。

### 7.3 disagreement 导出

HLML 运行 `make prepare-finetune-round` 后，HLMF 自动在 `mining/rounds` 找到唯一匹配 request：

```bash
make export_finetune_gold \
  FINETUNE_SOURCE_ID=disagreement_gold_r02 \
  FINETUNE_SOURCE_MODE=selection_subset
```

选择 request 已绑定原始 manifest、伪标签、ROI 及三份 SHA；任何上游内容变化都会阻止导出。

### 7.4 CVAT 分工

对每个 source 创建一个 CVAT image task。查看：

```text
cvat/<source_id>/qc/cvat_job_plan.json
```

按其中约 100 张/job 的边界创建和分配 job，不手工拆图片目录。团队开始前共同标 10 张校准图，统一 21 点编号、Left/Right 视角和 ignore 标准。正式任务不重复标注；负责人抽查每人约 5%。

完成后从完整 task 导出 XML，放到：

```text
cvat/<source_id>/reviewed.xml
```

### 7.5 导入和聚合

```bash
make import_finetune_gold
make finalize_train_finetune
```

导入是事务式的：先对所有待导入 task 做严格预检，再发布 Gold source。聚合允许多个相同 `source_kind` 的不同 `source_id`，按跨轮身份去重；标签冲突会 fail-closed。

结果：

```text
$HAND_WORK_ROOT/finetune/$HAND_FINETUNE_ID/hmlf_gold_merged/
```

## 8. 常见错误

- `source/task already exists`：source ID 是不可变轮次；换新 `source_id`，不要覆盖历史 Gold。
- `max_items is required`：native_existing 必须由程序冻结任务数量。
- `task exceeds hard limit`：预算超过 800；重新冻结为 600/800。
- `SHA mismatch`：文件内容在生成 descriptor 后变化；恢复原文件或重新建立新 source，不改 descriptor 绕过。
- CVAT import blocking errors：查看 task 的 QC 报告，修正 XML 中缺点、重复点、presence/handedness 冲突。
- `landmarks_out_of_crop`：如果点确实无法在 Hand ROI 内可靠表达，使用 `ignore_for_training`；不要把点硬拉回 ROI。

## 9. 与 HLML 交接

HLMF 完成下列输出后，切换到 `/root/HandLandmarkerLab`：

```text
train_pretrain_merged/
val_merged/
test_merged/
finetune/<id>/hmlf_gold_merged/
```

HLML 的具体命令见其 `docs/training_system/HLML_training_workflow.md`。
