# HLMF 2.0 目录与数据接口

## 1. 普通来源目录

把 `HLMF_SOURCE_ROOT` 设为一个来源的根目录。目录由程序逐步补全：

```text
<source>/
├── images/                         # 人工放置：1280×720、正向、灰度 TIFF
├── 01_palm/                        # 程序：Palm 检测
├── 02_roi_crops/
│   ├── images/                     # 程序：256×256 灰度 Hand ROI
│   ├── hand_roi_crops_manifest.jsonl
│   └── hand_landmarks_autolabel_draft.jsonl
├── 03_reviewed/                    # 人工 CVAT 返回结果/程序导入结果
├── 04_visualization/               # 程序：复核可视化
├── 05_labels/                      # 程序：来源级最终标签
└── qc/                             # 程序：每阶段报告
```

程序不旋转输入原图。需要先在图片进入 `images/` 前完成方向、分辨率和灰度转换。

## 2. 可再生数据仓库与工作区

`HAND_DATASET_ROOT` 指向 `/root/autodl-tmp/DatesetFab`。HLMF 2.0 的 pretrain/Val/Test 配置直接从其中读取来源目录，聚合输出写入 `HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0`：

```text
HLML-3.0/
├── train_pretrain_merged/          # pretrain 聚合标签、注册表、QC
├── val_merged/                     # Val Gold 聚合标签
├── test_merged/                    # Test Gold 聚合标签
└── finetune/<HAND_FINETUNE_ID>/
    ├── cvat/<source_id>/           # 允许物化图片的人工任务包
    ├── sources/gold/<source_id>/   # 导入并认证后的 Gold source
    └── hmlf_gold_merged/           # 所有轮次 Gold 聚合
```

聚合标签里的 `crop_path` 是指向 `DatesetFab` 或认证 Gold 包的绝对路径。HLML 会按允许根目录校验，不需要建立 `train_sources` 副本。

## 3. CVAT Gold 任务

每个 `source_id` 是不可变的一轮任务，例如：

```text
disagreement_gold_r02       source_kind=disagreement_gold
new_recorded_gold_r01       source_kind=new_recorded_gold
```

任务包内重要文件：

```text
cvat/<source_id>/
├── 02_roi_crops/images/
├── cvat_autolabel.xml
├── reviewed.xml                    # 人工放回
├── task_descriptor.json
└── qc/cvat_job_plan.json           # 每个 job 的范围、数量和 SHA
```

每轮人工预算和每个来源的限额由执行计划冻结，并在命令中显式传入；系统按配置的安全上限拒绝超量任务。`cvat_job_plan.json` 按配置的 segment size 规划 job。

Dragon 外部 Gold 同样使用不可变 `source_id`。`configs/dragon_gold.yaml` 只描述输入格式；每次运行通过 `DRAGON_SOURCE_ROOT` 和 `DRAGON_BATCH_ID` 指定一批，因此多批 Dragon 可以同时存在并参与最终聚合。

## 4. 人工标签决策

每张 ROI 只能做一种决定：

- 清楚可标：完整 21 点，并标 Left、Right 或 unknown。
- 确定无手：标 `no_hand`，不得同时标 handedness。
- 模糊、严重截断或无法可靠判断：标 `ignore_for_training`。

导入程序会检查点编号、重复/缺失点、presence 冲突、handedness 冲突、图片和任务描述符 SHA。标到 ROI 外的点不会被程序静默截断；确实无法在 ROI 内可靠表达的样本应使用 `ignore_for_training`。

## 5. 去重身份

Gold 聚合和 HLML 多轮抽样共同使用以下身份：

- `parent_global_crop_id`；
- `global_crop_id`；
- 来源图片身份；
- ROI 文件 SHA-256；
- 归一化灰度像素 SHA-256。

因此同一种 `source_kind` 可以有多轮不同 `source_id`。旧 Gold 不删除，新一轮只选择未进入历史 Gold、历史 CVAT、Val 或 Test 的 ROI。
