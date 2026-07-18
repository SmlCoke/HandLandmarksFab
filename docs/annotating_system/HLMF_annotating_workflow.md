# HLMF 完整标注流程

本文只描述可重复使用的 HLMF 操作方法。当前批次名称、数量、时间安排和负责人见 [当前下一步计划](HLMF_next_step_plan.md)。

## 1. 系统边界

HLMF 把原始图片变为可认证的 Hand ROI、MediaPipe 伪标签和人工 Gold 标签；HLML 读取这些产物完成选样、训练、评估和导出。普通图片来源共用一份 `configs/autolabel.yaml`，由 `HLMF_SOURCE_ROOT` 指定当前来源。HLMF 不负责训练模型，也不要求把 `DatesetFab` 中可直接引用的数据复制进训练工作区。

## 2. 初始化和目录

```bash
cd /root/HandLandmarksFab
git pull --ff-only
conda activate anfab
make compile
make test

export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
export HAND_GOLD_ROOT=$HAND_DATASET_ROOT/GoldSource
export HAND_WORK_ROOT=/root/autodl-tmp/TrainFab/HLML-3.0
export HAND_PRETRAIN_ID=<pretrain-id>
export HAND_FINETUNE_ID=<finetune-data-id>
```

- `HAND_DATASET_ROOT`：可再生数据仓库，保存原始图、来源级 ROI 和标注。
- `HAND_GOLD_ROOT`：跨训练版本复用的人工 Gold 源仓库。
- `HAND_WORK_ROOT`：当前训练版本的 mining、replay、Gold 聚合和运行结果；不再保存 Gold 真源。
- `HAND_FINETUNE_ID`：一份冻结的 finetune 数据快照，不等同于某次模型实验 ID。

## 3. 普通图片来源：00～06

### 3.1 人工准备输入

每个来源使用独立且长期稳定的目录：

```text
$HAND_DATASET_ROOT/<source-id>/images/
```

把图片整理为 `1280×720`、正向、灰度 TIFF。不同录制场次使用不同文件名前缀；不要混入 Val、Test 或固定 infer 图片。完成后：

```bash
export HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id>
make paths
```

### 3.2 选择数据角色和覆盖参数

同一份 `configs/autolabel.yaml` 提供默认值，运行时通过 Make 顶层参数区分数据性质：

- `AUTOLABEL_ROLE=train`：允许按配置保留低分 Palm 负样本候选；
- `AUTOLABEL_ROLE=val` 或 `test`：程序强制关闭低分负样本候选，只处理正常检测到的 ROI；
- `AUTOLABEL_OVERRIDES`：JSON 对象，严格覆盖配置中已经存在的任意局部字段。未知键、非法 JSON 或不合理阈值会直接失败。

训练批次使用默认值：

```bash
make autolabel \
  HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id> \
  AUTOLABEL_ROLE=train
```

单独覆盖本批次的低分负样本阈值：

```bash
make autolabel \
  HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<source-id> \
  AUTOLABEL_ROLE=train \
  AUTOLABEL_OVERRIDES='{"palm":{"negative_candidate_threshold":<threshold>}}'
```

Val/Test：

```bash
make autolabel HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<val-source> AUTOLABEL_ROLE=val
make autolabel HLMF_SOURCE_ROOT=$HAND_DATASET_ROOT/<test-source> AUTOLABEL_ROLE=test
```

`negative_candidate_threshold` 必须小于 `score_threshold`。推荐使用单个 `make autolabel` 命令，确保 00～03 全程使用同一 role/override；若逐步运行，则每条命令必须重复完全相同的顶层参数。

### 3.3 程序检查、Palm、ROI 和伪标签

```bash
make validate_images
make palm_detection
make build_roi
make run_mediapipe
```

依次查看：

```text
$HLMF_SOURCE_ROOT/qc/image_validation_report.json
$HLMF_SOURCE_ROOT/01_palm/palm_detections.jsonl
$HLMF_SOURCE_ROOT/qc/palm_detection_stats.json
$HLMF_SOURCE_ROOT/02_roi_crops/hand_roi_crops_manifest.jsonl
$HLMF_SOURCE_ROOT/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
$HLMF_SOURCE_ROOT/qc/mediapipe_roi_stats.json
```

`build_roi` 生成与训练/板端一致的 `256×256` 灰度 Hand ROI。`run_mediapipe` 只是生成可供复核或 pretrain 使用的教师伪标签，不会把它自动提升为人工 Gold。

四个 QC 报告均保存 `autolabel_runtime`，其中记录 role、显式 overrides、最终负样本开关和实际阈值。检查报告时不要只看数量，也要确认运行参数与本批计划一致。

### 3.4 可选的普通全量 CVAT 复核

```bash
make export_cvat
```

把 `02_roi_crops/images/` 和生成的 CVAT XML 上传到一个 CVAT image task。每张图只能作一种决定：

- 清楚可标：完整标 21 个点，并选择 Left、Right 或 unknown；
- 确定无手：标 `no_hand`；
- 模糊、严重截断或无法可靠判断：标 `ignore_for_training`。

从完整 task 导出 `CVAT for images 1.1`，保存为 `03_reviewed/cvat_reviewed.xml`，然后：

```bash
make import_cvat
make visualize
```

人工只编辑 CVAT 标注，不手改 JSONL、SHA256、descriptor 或聚合报告。

## 4. Pretrain、Val、Test 聚合

`configs/finalize_train.yaml`、`finalize_val.yaml` 和 `finalize_test.yaml` 定义允许进入各集合的来源。先在配置中登记新的来源；同一原始身份不得同时进入训练和 Val/Test。

```bash
make finalize_train_pretrain
make build_pretrain_source_registry
make finalize_val
make finalize_test
```

输出：

```text
$HAND_WORK_ROOT/train_pretrain_merged/
$HAND_WORK_ROOT/val_merged/
$HAND_WORK_ROOT/test_merged/
```

聚合目录主要保存标签、来源注册表和 QC；`crop_path` 可以直接指向 `DatesetFab`，无需复制 ROI。

## 5. 人工负样本删除复核

这一步由 HLML 的 `make pretrain-curate` 生成候选树。人工把候选树复制成 reviewed 树，逐图删除所有含手、手指、手腕、模糊或无法确认的图片，只保留明确背景，然后由 HLML 执行 `make pretrain-curate-reviewed`。

候选树可以 zip/7z 压缩后经网盘传输。正常压缩、解压不会改变文件内容 SHA256；不要用会重编码图片的软件，也不要编辑、改名或移动保留图片。HLML 会逐文件重新校验内容。

## 6. Dragon 外部 Gold：按批次独立发布

### 6.1 通用输入契约

每一批 Dragon 数据先放入自己的规范批次目录：

```text
$HAND_GOLD_ROOT/dragon/<dragon-batch-id>/
├── source/
│   ├── images/
│   ├── annotations_hand.txt
│   ├── annotations_palm.txt
│   └── README.md
└── published/                 # prepare_dragon_gold 生成
```

`configs/dragon_gold.yaml` 只保存上述文件名和 ROI 规则，不保存某批数据的路径、批次 ID、预期 SHA 或预期数量。程序会读取每张图的 EXIF 方向和实际尺寸，并把本批实际 SHA256、计数、拒绝原因写入报告。

### 6.2 发布一批

为每批选择一个不会复用的安全 ID：

```bash
make prepare_dragon_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  DRAGON_SOURCE_ROOT=$HAND_GOLD_ROOT/dragon/<dragon-batch-id>/source \
  DRAGON_BATCH_ID=<unique-dragon-batch-id>
```

结果：

```text
$HAND_GOLD_ROOT/dragon/<unique-dragon-batch-id>/published/
├── 02_roi_crops/
├── 03_reviewed/
├── source_images/
├── finetune_source.json
└── qc/gold_source_report.json
```

来了 N 批就执行 N 次，每次使用不同 `DRAGON_BATCH_ID`。`source/` 和 `published/` 都不可覆盖；相同输入需要重发时也应使用新 ID，让旧批次继续可追溯。

## 7. Finetune Gold：新录制与 HLML 选样

### 7.1 新录制来源的 Gold 任务

先建立批次目录，把原图和 00～03 全部放在 `source/` 中：

```text
$HAND_GOLD_ROOT/new_recorded_gold/<source-id>/source/
├── images/
├── 01_palm/
├── 02_roi_crops/
└── qc/
```

以这个 `source/` 作为 `HLMF_SOURCE_ROOT` 跑第 3 节，再由程序确定性抽样；人工不要从原图目录随意挑选：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=<unique-new-recorded-source-id> \
  FINETUNE_SOURCE_MODE=native_existing \
  FINETUNE_RAW_SOURCE_ROOT=$HAND_GOLD_ROOT/new_recorded_gold/<unique-new-recorded-source-id>/source \
  FINETUNE_MAX_ITEMS=<this-task-limit>
```

`FINETUNE_MAX_ITEMS` 是本轮计划决定的上限，必须显式传入。程序按 session、来源图片和稳定哈希分散选择，并生成不可变任务包。

### 7.2 disagreement Gold 任务

这里的 disagreement 指“HLML 当前 student 模型预测的 21 点与 MediaPipe teacher 伪标签差异较大的 ROI”。它只说明值得人工核查，不说明 teacher 一定正确。HLML 完成一轮 `prepare-finetune-round` 后：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=disagreement_gold_<round-id> \
  FINETUNE_SOURCE_MODE=selection_subset
```

HLMF 会在该 finetune 工作区的 `mining/rounds/` 中寻找唯一 request，并验证 request、来源 manifest、ROI 与伪标签 SHA。任何上游变化都会阻止导出。

### 7.3 negative-removed Gold 任务

negative-removed 指 pretrain 负样本人工删除复核时，被人工从“明确背景”候选树中删掉的 ROI。它们往往含手、手指、手腕，或模糊到不能当作可靠负样本；只有再次做 21 点/presence 人工 Gold 标注后，才适合以困难样本身份参与 finetune。

该来源不是 HLMF 自行从 GoldSource 猜出来的。HLML 在一个新的 finetune 数据 ID 下启用 `configs/prepare_finetune_sources.yaml` 的 `selection.negative_removed`、设定本轮数量并执行 `make prepare-finetune-sources`，生成冻结的 selection request。随后在 HLMF 导出：

```bash
make export_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=negative_removed_gold_<round-or-batch-id> \
  FINETUNE_SOURCE_MODE=selection_subset
```

每次重新抽样都使用新的 source ID。程序会把 GoldSource 中历史 published、pending task、当前 request 以及 Val/Test 身份一起排除，因此旧 Gold 不会浪费，也不需要删除。

### 7.4 CVAT 人工工作

四类 Gold 使用统一的批次身份，但目录按生命周期出现，不是三个目录永久并存：

```text
$HAND_GOLD_ROOT/<domain>/<source-id>/
├── source/                    # 新录制/Dragon 原始真源；选样类可无
├── task/                      # 仅“等待人工/等待 import”期间存在
│   ├── images/                # 本任务选中的 ROI；与 source 能硬链接则不复制数据块
│   ├── hand_roi_crops_manifest.jsonl
│   ├── hand_landmarks_autolabel_draft.jsonl
│   ├── cvat_autolabel.xml
│   ├── task_descriptor.json
│   ├── reviewed.xml           # 人工返回
│   └── qc/cvat_job_plan.json
└── published/                 # import/prepare 成功后生成；此时 task/ 自动删除
    ├── finetune_source.json
    ├── 02_roi_crops/
    ├── 03_reviewed/
    ├── audit/                 # reviewed.xml、自动标注 XML、任务描述符等小型审计文件
    └── qc/
```

`<domain>` 只能是 `new_recorded_gold`、`disagreement_gold`、`negative_removed_gold` 或 `dragon`。一个领域可以有任意多个批次；`source-id` 在整个 GoldSource 中必须唯一，并且必须包含批次信息，不能直接使用领域名。例如历史来源可命名为 `disagreement_gold_hlml2.0`，新任务可命名为 `disagreement_gold_r02`。

`source`、`task`、`published` 不能按文件名相似就强行合并：`source` 是原始真源，`task` 是暂存的人工任务快照，`published` 是认证训练产物。程序只消除没有长期价值的 task：发布成功后把必要的小型审计文件移入 `published/audit/`，再删除 task。Dragon 的原始整图/标注与生成的 Hand ROI 数据性质不同，长期保留 `source + published`。

按 `cvat_job_plan.json` 的 job 边界分工，不自行拆散或重命名图片。团队先用少量校准图统一 21 点、左右手和 ignore 规则；正式 job 不交叉。完成后把完整 task 导出的 XML 保存为：

```text
.../<domain>/<source-id>/task/reviewed.xml
```

### 7.5 严格导入与最终聚合

```bash
make import_finetune_gold HAND_FINETUNE_ID=<finetune-data-id>
make finalize_train_finetune HAND_FINETUNE_ID=<finetune-data-id>
```

程序先完整预检所有待导入 task，再事务式发布到同批次的 `published/`。发布成功后 task 自动退休；不能同时看到 task 和 published，若同时存在说明上一次发布后的清理异常，应先检查 `published/finetune_source.json`，不要重复 import。最终聚合仍是当前训练版本的派生产物：

```text
$HAND_WORK_ROOT/finetune/<finetune-data-id>/hmlf_gold_merged/
├── 05_labels/
├── hmlf_gold_aggregate.json
└── qc/finalize_train_finetune_report.json
```

如果只想导入一个已经完成的 task，可显式指定：

```bash
make import_finetune_gold \
  HAND_FINETUNE_ID=<finetune-data-id> \
  FINETUNE_SOURCE_ID=<source-id>
```

不传 `FINETUNE_SOURCE_ID` 时会预检并导入当前所有 pending task。不要把尚未标完的 task 混在 `--all` 中；要么先完成它，要么显式逐批导入已经完成的 source。

## 8. 多轮 Gold 与历史复用

同一领域可以不断增加批次，例如 `new_recorded_gold_r02`、`new_recorded_gold_r03`。每轮必须使用新的 `<round-id>` 和全局唯一 `<source-id>`，作废的 ID 也不复用。HLML 抽样会扫描 GoldSource 内所有 pending task 和 published Gold 的身份、ROI SHA、像素 SHA，再叠加 Val/Test 与当前 mining request 排重。

Gold 不再从旧 finetune 工作区 seed。任意新训练版本都直接发现 `$HAND_GOLD_ROOT/*/*/published/finetune_source.json`；人工 Gold 只有一份真源，训练版本只保存本次选择清单和聚合快照。

### 8.1 Gold 发布不绑定某个 finetune ID

`GoldSource/<domain>/<source-id>/published/` 是长期、不可变、可跨实验复用的认证数据源，不属于某一次训练。命令里出现 `HAND_FINETUNE_ID` 有两个技术原因：

- 对 disagreement/negative-removed 这类 `selection_subset`，HLMF 要到该 finetune 工作区找到 HLML 冻结的 selection request；
- `finalize_train_finetune` 要把“当时仓库中全部 published Gold”的认证聚合快照写入该 finetune 工作区，并在报告里记录数据 ID。

这不意味着 published 批次只能由该 ID 使用。未来的新 finetune ID 仍会从 GoldSource 发现、校验并复用同一批 Gold；不会复制图片，也不需要重新发布。Dragon 和 `native_existing` 新录制 Gold 即使没有 mining request，也沿用相同命令接口，以便目录和审计方式一致。

### 8.2 HLMF 聚合与 HLML 训练选择是两层不同操作

HLMF 的 `make finalize_train_finetune` 会认证并聚合 **GoldSource 中全部 published 批次**，它故意不判断某次模型训练该启用哪些来源。这样可以先发现重复、冲突、文件损坏或描述符变化。

随后 HLML 为本次 `HAND_FINETUNE_ID` 生成 `gold_selection.yaml`，逐个 source ID 明确写 `enabled: true/false`。只有 `true` 的 Gold 才进入训练；replay 由 HLML 另行生成并强制参与。完整顺序是：

```text
HLMF 发布每个 Gold 批次
  -> HLMF 聚合并认证全部 published Gold
  -> HLML 逐 source ID 冻结启用/禁用清单
  -> HLML 将 enabled Gold 与 mandatory replay 合成训练快照
```

因此“发布”不是“自动参加训练”，而“disabled”也不是删除数据。更换来源组合时使用新的 finetune 数据 ID，保留旧快照可追溯。

### 8.3 时间不足时可以不新增困难样本 Gold

disagreement 和 negative-removed 都是可选的人工增量来源，不是构建 finetune 的前置条件。时间不足时：

1. 不创建本轮 selection round，也不导出新的 CVAT task；
2. 保留 HLML 自动生成的 disagreement 分数池，供以后继续使用；
3. HLMF 只导入已经完成的 task，并聚合现有 published Gold；
4. HLML 可显式启用历史 published 困难样本 Gold，或把这些领域全部设为 disabled；
5. 只要至少一个合格 Gold 来源满足门控，且 mandatory replay 存在，就可以继续 finetune。

HLMF 不制作 replay。replay 来自 HLML 的 authenticated pretrain source registry 和 curated multitask 标签，由 HLML 确定性抽取。

## 9. 常见错误

- `Gold batch task or published output already exists`：ID 已使用；换新 round/source ID，不覆盖或复用历史批次。
- `raw source must be archived at .../source`：先按规范建立 GoldSource 批次，不能从临时目录发布。
- `DRAGON_SOURCE_ROOT/DRAGON_BATCH_ID is required`：Dragon 批次身份必须在命令中显式提供。
- `native_existing requires max_items`：新录制任务必须显式冻结数量。
- `SHA mismatch`：descriptor 生成后输入内容变化；恢复原文件，或用新 ID 重新发布。
- CVAT import blocking errors：查看任务 QC，修正缺点、重复点、presence/handedness 冲突。
- `landmarks_out_of_crop`：点确实无法在 ROI 内可靠表达时标 `ignore_for_training`，不要把点硬拉回 ROI，也不要绕过门控。

## 10. 交接 HLML

HLMF 完成 `train_pretrain_merged/`、`val_merged/`、`test_merged/`，并把本轮已经完成的人工任务发布到 GoldSource、生成当前 `finetune/<id>/hmlf_gold_merged/` 后，切换到 `/root/HandLandmarkerLab`。HLML 会为每个 published 子批次生成显式参与/不参与决定。

交接前应检查：

```bash
test -f "$HAND_WORK_ROOT/finetune/<finetune-data-id>/hmlf_gold_merged/hmlf_gold_aggregate.json"
find "$HAND_GOLD_ROOT" -path '*/published/finetune_source.json' -print
```

`hmlf_gold_aggregate.json` 的 source 数量应与当时 GoldSource 中的 published 描述符一致；有未完成 task 不会自动进入聚合或训练。后续 replay、Gold source 选择、curation 和训练命令均在 HLML 执行。
