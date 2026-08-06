# HLMF 常见问题与解答

## 文档使用规则

本文档记录项目成员在与编码助手对话时明确要求沉淀的疑问与解答。只有用户主动提出“将本次问答写入 `HLMF_qa.md`”时才更新本文档；其他任务不需要主动检查、扩写或同步本文档。

以下内容基于当前 HLMF 3.0 实现。命令中的 `HAND_DATASET_ROOT` 示例均为服务器目录 `/root/autodl-tmp/DatesetFab`。

---

## Q1：误启动 Train 批处理并中断，而后续数据属于不同 `DATASET_ID`，应该如何处理？

### 结论

不同 `DATASET_ID` 的数据应直接上传到新的 dataset 目录，逐来源执行 `source-check`，然后使用新的 `DATASET_ID` 启动批处理。它不会自动并入旧的 `FullEnhance0801`，也不需要为了区分 dataset 而强制更换 `PROPOSAL_VARIANT`。

目录格式：

```text
HAND_DATASET_ROOT/PretrainSource/<new_dataset_id>/<capture_source_id>/images/*.tif[f]
```

例如：

```bash
make source-check \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_SCOPE=pretrain \
  DATASET_ID=<new_dataset_id> \
  CAPTURE_SOURCE_ID=<new_capture_source_id> \
  PROPOSAL_VARIANT=eos_1.0-gate
```

所有新来源注册完成后：

```bash
make batch-train-autolabel \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  DATASET_ID=<new_dataset_id> \
  PROPOSAL_VARIANT=eos_1.0-gate \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

批处理脚本只扫描下列位置的 `source.json`：

```text
HAND_DATASET_ROOT/PretrainSource/<DATASET_ID>/<capture_source_id>/source.json
```

因此：

- 对新 `DATASET_ID` 执行批处理时，只会处理这个 dataset 下已经完成 `source-check` 的来源。
- 旧 `FullEnhance0801` 中的 95 个来源不会被新 dataset 的批处理重复处理。
- 仅上传 `images/` 而不执行 `source-check` 时，批处理发现不了该来源。

### 全局唯一性约束

虽然 dataset 目录彼此隔离，但 `Registry/registry.sqlite3` 是整个 `HAND_DATASET_ROOT` 共用的全局 Registry：

- `dataset_id` 不能跨 `pretrain/eval` scope 重用。
- `capture_source_id` 是全局主键，不能把已经属于旧 dataset 的同一个 `capture_source_id` 再归属到新 dataset。
- 新 dataset 中的每个来源必须使用新的、全局未占用的 `capture_source_id`。
- proposal variant 状态以 `(capture_source_id, proposal_variant)` 为主键；只要新 dataset 使用的是新来源 ID，同一个 `PROPOSAL_VARIANT` 名称可以继续使用。

### 本次误操作的实际状态与恢复方式

误执行的命令为：

```bash
make batch-train-autolabel \
  DATASET_ID=FullEnhance0801 \
  PROPOSAL_VARIANT=eos_1.0-gate \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

在输出第一个来源后按 `Ctrl+C`。服务器只读核对结果如下：

- 批处理和 `hlmf.py` 进程已经停止，只剩 screen 会话本身。
- `proposal_variants` 表中没有 `eos_1.0-gate`。
- `rois` 表中没有该变体的 ROI。
- `FullEnhance0801/dataset_manifest.json` 中没有发布该变体。
- 第一个来源仅留下 `01_palm/02_roi_crops/03_reviewed/05_labels/qc` 下的同名空目录；文件数和磁盘占用均为 0。

所以这次中断发生在实际 Palm/ROI 生成及 Registry 注册之前，旧 dataset 中的这个变体名没有被占用。处理方式有两种：

1. 直接保留空目录。它们不影响新 dataset，也不影响以后以相同配置重跑旧 dataset。
2. 如果只想让目录整洁，可以用 `rmdir` 删除精确空目录；`rmdir` 遇到非空目录会拒绝执行：

```bash
SOURCE=/root/autodl-tmp/DatesetFab/PretrainSource/FullEnhance0801/complex-far-bright-random-train-s01-peak
VARIANT=eos_1.0-gate

rmdir -- \
  "$SOURCE/01_palm/$VARIANT" \
  "$SOURCE/02_roi_crops/$VARIANT" \
  "$SOURCE/03_reviewed/$VARIANT" \
  "$SOURCE/05_labels/$VARIANT" \
  "$SOURCE/qc/$VARIANT"
```

此时不要执行 `source-variant-delete`。即使变体尚未注册，只要发现精确变体派生目录，该命令也会创建 retired tombstone，导致该来源以后不能使用这个变体名。

也不要手工修改 `registry.sqlite3`。Registry 会从既有 ROI 自动回填 active 状态；只删除一张表中的记录可能在下次打开 Registry 时重新出现，或者造成 ROI、发布数据、负样本引用与文件状态不一致。

---

## Q2：哪些情况下无法重新用同一个变体名字执行 autolabel？

### 1. 同一来源/变体已经 retired

这是最明确、最直接的禁止条件。`source-variant-delete` 会把以下主键永久标记为 retired：

```text
(capture_source_id, proposal_variant)
```

之后该来源上的 Palm、ROI、Hand landmark、Train/Eval autolabel、CVAT 和发布流程在建立 source context 时都会拒绝继续，典型错误是：

```text
proposal variant is retired and cannot be reused: <capture_source_id>/<proposal_variant>
```

`source-check` 只负责原始来源注册，本身仍可重复执行；但它不能解除 tombstone，后续派生阶段仍会失败。retired 是永久墓碑，不提供“取消 retired”接口。应该为后续新配置选择新的 `PROPOSAL_VARIANT`。

如果批处理中的某一个来源已经 retired，而其他来源没有 retired，脚本会：

- 让该来源失败；
- 继续尝试其余来源；
- 最终因为至少一个来源失败而返回非零。

因此不要在同一个批次中混用“部分来源已 retired、部分来源仍可用”的同名变体。

### 2. 同一个 `capture_source_id` 被放入另一个 dataset

`capture_source_id` 在全局 Registry 中只能归属一个 `dataset_id` 和一个 split。即使目录不同，只要来源 ID 相同，新的 `source-check` 就会因 ownership 冲突失败。此时问题表面上发生在 autolabel 前，根因不是变体名，而是来源身份不合法。

### 3. active 变体重跑时，Palm/ROI 身份输入已经变化

active 变体允许幂等重跑，但“幂等”要求身份和几何输入保持一致，例如：

- 同一个 raw image 仍对应同一来源和相对路径；
- proposal slot 和 ROI contract 没有改变；
- 同一个稳定 `roi_id` 对应的 crop 内容没有改变。

如果仍使用原变体名，但修改了原图、Palm/ROI 几何参数或其他会改变 crop 的配置，系统可能拒绝覆盖已有 ROI：

```text
refusing to overwrite changed ROI: <crop_path>
```

如果稳定 ID 已在 Registry 中指向不同的 raw image、variant 或 slot，还会触发 `roi_id collision`。

这种情况下应把变化视为新的 proposal 版本，使用新的 `PROPOSAL_VARIANT`，不要试图清理 SQLite 后强行复用旧名字。

### 4. 变体名或执行前提不满足契约

以下错误也会让命令无法执行，但不代表名字已经被占用：

- 变体名不符合安全 ID 格式；
- 来源没有先执行 `source-check`，缺少 `raw_images.jsonl/source.json`；
- train/eval 命令与 `capture_source_id` 中的 split 不匹配；
- 模型、Palm 输出、ROI manifest 或 reviewed label 等当前阶段输入缺失；
- 新 dataset 只上传了图片，但没有生成 `source.json`，所以批处理找不到来源。

修复缺失输入后可以继续使用原名，前提是该 source/variant 没有 retired，也没有与既有 ROI 发生内容冲突。

### 5. active 状态下什么时候可以继续使用同名变体？

以下场景允许：

- 上次在真正注册 ROI 前中断，只留下空目录或可被同配置覆盖的中间文件。
- 使用完全相同的来源、Palm/ROI 配置和变体身份进行幂等重跑。
- 同一个 variant 名用于另一个全新 `capture_source_id`；Registry 主键包含来源 ID。
- 新 dataset 使用全新来源 ID，并希望沿用同一套 proposal 配置名称。

虽然当前实现允许 active 变体重写 Hand landmark draft，但为了数据可追溯，如果教师后端、模型版本、Palm/ROI 配置或质量门控发生了有意义的变化，仍建议使用新的变体名。

---

## Q3：目前仓库的完整标注链路是什么？

### 阶段 0：准备数据身份与目录

每个来源 ID 固定为：

```text
<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>
```

train 放入：

```text
HAND_DATASET_ROOT/PretrainSource/<dataset_id>/<capture_source_id>/images/*.tif[f]
```

val/test 放入：

```text
HAND_DATASET_ROOT/EValSource/<dataset_id>/<capture_source_id>/images/*.tif[f]
```

`images/` 必须平铺。split 直接来自 `capture_source_id`，不是发布时随机划分。

### 阶段 1：来源检查与注册

```bash
make source-check \
  DATASET_SCOPE=pretrain|eval \
  DATASET_ID=<dataset_id> \
  CAPTURE_SOURCE_ID=<capture_source_id> \
  PROPOSAL_VARIANT=<variant>
```

系统执行：

1. 校验 dataset scope 与来源 ID 中的 split 是否一致。
2. 校验 TIFF、尺寸和方向，必要时按既有规则规范方向。
3. 为原图生成稳定 `raw_image_id` 和轻量指纹。
4. 注册 dataset、capture source 和 raw image。
5. 来源注册本身不创建 proposal variant 状态；Palm/ROI 等派生阶段开始时才检查 source/variant 是否 retired。

输出：

```text
<source>/source.json
<source>/raw_images.jsonl
<source>/qc/image_validation_report.json
HAND_DATASET_ROOT/Registry/registry.sqlite3
```

### 阶段 2：Eos Palm 推理

Train/Eval 总流程都会调用 Palm 阶段。Eos 在每张原图上输出 bbox、p0、p9、score 和 proposal slot：

- 达到 runtime 阈值的 proposal 标为 `proposal_kind=runtime`。
- Train 可保留 low-score proposal，标为 `proposal_kind=negative_candidate`。
- Eval 强制不保留 low-score candidate。
- Palm 输出不允许人工修改。

输出：

```text
<source>/01_palm/<variant>/palm_detections.jsonl
<source>/qc/<variant>/palm_detection_report.json
```

### 阶段 3：构造固定 Hand ROI

程序根据 Eos 的 bbox、p0、p9 和 `configs/autolabel.yaml` 中的 scale/shift 构造 ROI，投影并裁剪为 `256×256` PNG。

每个 ROI 的稳定 ID 由 raw image、proposal variant、proposal slot 和 ROI contract version 派生。ROI 注册完成时，source/variant 在 Registry 中成为 active。

输出：

```text
<source>/02_roi_crops/<variant>/images/<roi_id>.png
<source>/02_roi_crops/<variant>/hand_roi_crops_manifest.jsonl
<source>/qc/<variant>/roi_build_report.json
```

Palm 几何和 ROI 不进入人工编辑流程。

### 阶段 4：Hand landmark 教师标注

后端由全局 YAML 或单次参数决定：

```bash
HAND_LANDMARK_BACKEND=mediapipe_tasks
HAND_LANDMARK_BACKEND=rtmpose_onnx
```

MediaPipe 路径：

- 对 ROI 运行 MediaPipe Tasks。
- 使用 MediaPipe 的 presence、handedness 和 21 点。
- 保持既有 MediaPipe provenance。

RTMPose 路径：

- 只对 runtime ROI 运行 RTMPose，固定输出 21 点。
- 同一个 runtime ROI 再运行 HCF，输出 Left/Right 和概率。
- CUDA provider 可用时优先 CUDA，否则使用 CPU；实际 provider 写入 QC。
- `hand_presence.present=true` 只是发布路由哨兵，不是真实 presence 标签。
- low-score candidate 不运行 RTMPose/HCF，关键点为空、handedness 为 `unknown/null`，provenance 为 `unresolved/unlabeled_v1`。

输出：

```text
<source>/02_roi_crops/<variant>/hand_landmarks_autolabel_draft.jsonl
<source>/qc/<variant>/mediapipe_report.json
```

`mediapipe_report.json` 是为了兼容既有路径而保留的报告名，RTMPose 运行时也写入该路径。

### 阶段 5A：Train 质量分流与自动发布

```bash
make train-autolabel \
  DATASET_SCOPE=pretrain \
  DATASET_ID=<dataset_id> \
  CAPTURE_SOURCE_ID=<train_source_id> \
  PROPOSAL_VARIANT=<variant> \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

该命令连续执行 source check、Palm、ROI、Hand landmark 和 Train 发布。

发布分流：

- 教师认为 present 且通过质量门控：进入 `hand_training_labels.jsonl`。
- Train low-score candidate：进入 `candidate_negatives.jsonl`，等待负样本人工复核。
- `ignore_for_training=true` 或 positive 未通过质量门控：进入 `ignored.jsonl`。
- HCF handedness 分数低于当前阈值 `0.7` 时，整个 positive 行进入 ignored。
- 仅 RTMPose Train runtime 行统计 42 个 crop x/y 值；精确为 `0.0/255.0` 的值达到 3 个时进入 ignored，1–2 个仍通过。

输出：

```text
<source>/05_labels/<variant>/hand_training_labels.jsonl
<source>/05_labels/<variant>/candidate_negatives.jsonl
<source>/05_labels/<variant>/ignored.jsonl
<source>/qc/<variant>/source_publish_report.json
<dataset>/dataset_manifest.json
```

### 阶段 5B：Eval CVAT 复核与发布

```bash
make eval-autolabel ... DATASET_SCOPE=eval
make hand-cvat-export ... DATASET_SCOPE=eval
```

`eval-autolabel` 只生成教师 draft，不把它直接作为正式评估真值。`hand-cvat-export` 生成：

```text
<source>/03_reviewed/<variant>/cvat_autolabel.xml
```

人工在 CVAT 中：

1. Images 排序选择 `Lexicographical`。
2. 只复核/修改 21 个关键点、Left/Right/unknown、`no_hand`、`ignore_for_training`。
3. 不修改 Palm bbox、p0、p9 或 ROI。
4. 将结果保存为 `<source>/03_reviewed/<variant>/cvat_reviewed.xml`。

然后执行：

```bash
make hand-cvat-import ... DATASET_SCOPE=eval
make source-publish ... DATASET_SCOPE=eval
```

输出：

```text
<source>/03_reviewed/<variant>/hand_landmarks_reviewed.jsonl
<source>/05_labels/<variant>/hand_evaluation_labels.jsonl
<source>/05_labels/<variant>/ignored.jsonl
<dataset>/dataset_manifest.json
```

Eval 发布前会按整个 val/test split 的 prospective manifest 检查 raw image 和 ROI 上限。

### 阶段 6：可视化、审计和生命周期维护

- `autolabel-visualize-roi`：从既有 draft 重建 ROI 审核图；RTMPose 只取 runtime ROI。
- `autolabel-visualize-original`：生成原图关键点 PNG，并默认生成按文件名字典序排列的 MP4。
- `autolabel-visualizations-clean`：只删除可重建可视化，不改变 Registry。
- `registry-check`：查看 dataset/source/raw/variant/ROI/负样本/selection 状态。
- `source-variant-delete`：删除精确来源/变体的派生产物、重建 dataset manifest，并写永久 retired tombstone；保留原图、raw/source 元数据和 ROI Registry 元数据。

### 批处理行为

`batch-train-autolabel` 和 `batch-eval-autolabel` 按指定 `DATASET_ID` 下的 `source.json` 发现来源，按来源顺序调用单来源流程：

- 每个来源单独写日志。
- 一个来源失败不会阻止脚本继续尝试后续来源。
- 只要有一个来源失败，批处理最终返回非零。
- Train 批处理不会自动关机。

---

## Q4：负样本与困难样本如何人工复核？

两条流程都是“程序准备独立图片副本 → 人工只通过删除图片表达拒绝 → 程序发布保留图片”，但人工判断方向相反：

| 流程 | 输入对象 | 人工保留 | 人工删除 |
|---|---|---|---|
| 负样本 | Eos low-score candidate | 明确没有手的纯背景 | 有手、疑似有手、遮挡不清或任何不确定内容 |
| 困难样本 | 外部 mining request 中的 Train hard positive | 教师关键点明确正确、可用于训练的困难正样本 | 教师关键点明显错误的 ROI |

人工阶段不要重命名、移动或新增 review 图片，也不要直接编辑 manifest。当前接口只把“文件是否仍存在”当作保留/删除决定。

### A. 负样本人工复核

#### A1. 前置条件

先完成 Train autolabel。每个来源的候选位于：

```text
<source>/05_labels/<variant>/candidate_negatives.jsonl
```

候选必须：

- `split=train`；
- ROI 已在 Registry 注册；
- source/variant 仍为 active；
- `crop_relpath` 指向仍存在的 ROI PNG。

为本次负样本发布选择一个从未使用过的 `NEGATIVE_DATASET_ID`。该 ID 在 prepare 开始时就会被 Registry 保留；不要复用旧 ID。

#### A2. 生成人工复核目录

单个 candidate JSONL：

```bash
make negative-review \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  NEGATIVE_DATASET_ID=background-neg-20260806 \
  NEGATIVE_CANDIDATE_LABELS=/abs/path/to/candidate_negatives.jsonl
```

需要把多个来源的 candidate JSONL 放入同一个负样本批次时，可以直接使用 CLI 的可重复参数：

```bash
python -B scripts/hlmf.py prepare-negative-review \
  --dataset-root /root/autodl-tmp/DatesetFab \
  --negative-dataset-id background-neg-20260806 \
  --candidate-labels /abs/source-a/candidate_negatives.jsonl \
  --candidate-labels /abs/source-b/candidate_negatives.jsonl
```

程序逐行核对 Registry，然后把 ROI 复制到：

```text
HAND_DATASET_ROOT/GoldSource/NegativeSamples/<negative_dataset_id>/review/
  README.json
  candidate_manifest.jsonl
  images/<capture_source_id>/<roi_id>.png
```

这里的图片是独立副本，不是硬链接。

#### A3. 人工操作

进入 `review/images/`，逐张查看：

- 明确完全没有手、可作为背景负样本：保留图片。
- 出现完整手、局部手、疑似手、镜面/阴影造成无法确认，或任何不确定情况：删除图片。

只删除图片，不修改 `candidate_manifest.jsonl`。不要新增图片；发布时发现未登记文件会报错。

至少保留一张。全部删除时，系统会拒绝发布空负样本集。

#### A4. 发布

```bash
make negative-publish \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  NEGATIVE_DATASET_ID=background-neg-20260806
```

程序把仍存在的 review 图片再次复制到 published，并为它们写入人工负样本标签：

```text
HAND_DATASET_ROOT/GoldSource/NegativeSamples/<negative_dataset_id>/published/
  images/<capture_source_id>/<roi_id>.png
  negative_labels.jsonl
  manifest.json
  review_report.json
```

发布记录具有：

- `label_origin=human`；
- `annotation_style=project_consensus_v1`；
- `human_reviewed=true`；
- `hand_presence.present=false`；
- 空关键点；
- 指向独立发布副本的 `published_relpath`。

发布成功后，Registry 将 negative dataset 标为 published，并删除临时 `review/`。因为 published 是独立副本，之后删除源 proposal variant 不会破坏已发布负样本。

同一个 ROI 不能再次发布到另一个负样本数据集；Registry 的 `published_negatives.roi_id` 是唯一主键。

#### A5. 常见错误

- `negative_dataset_id has already been used`：该 ID 已 reserved/published，换新 ID，不要改 SQLite。
- `review request references retired roi_id`：源变体已 retired，不能再从它准备新复核批次。
- `review tree contains unmanifested files`：人工新增、复制或改名了图片；恢复为 manifest 对应文件集合。
- `cannot publish an empty negative dataset`：所有候选都被删除，必须重新准备一个有至少一个真实负样本的新批次。
- `negative dataset is already published`：发布是一次性的，不可覆盖。

### B. 困难样本人工复核

#### B1. 输入 `MINING_REQUEST`

困难样本不是从 `candidate_negatives.jsonl` 自动产生，而是由后续训练/挖掘流程生成一个 JSONL 请求。每行必须精确引用一个已注册且 active 的 Train ROI，至少保持这些身份字段与 Registry 一致：

```text
roi_id/crop_id
raw_image_id
capture_source_id
proposal_variant
crop_relpath 或 crop_path
split=train
```

通常请求行应直接沿用来源 label/draft 中的完整记录，以同时携带 teacher landmarks、handedness 和 provenance。选择一个从未使用过的 `SELECTION_ID`；selection ID 也是一次性的。

#### B2. 生成人工复核目录

```bash
make hard-review \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  SELECTION_ID=hard-20260806 \
  MINING_REQUEST=/abs/path/to/mining_request.jsonl
```

程序校验每个 ROI 的 Registry 身份和源文件，然后生成：

```text
HAND_DATASET_ROOT/Selections/<selection_id>/review/
  README.json
  request_manifest.jsonl
  images/<capture_source_id>/<roi_id>.png
```

review 图片是独立复制的原始 ROI crop。它本身不把关键点画在图上；需要判断 teacher 关键点时，应结合 `request_manifest.jsonl` 中的关键点，或先用对应 source/variant 的 `autolabel-visualize-roi` 输出按同一 `roi_id` 对照查看。

#### B3. 人工操作

- teacher 关键点明确正确，且该 ROI 确实是有价值的困难正样本：保留 review 图片。
- teacher 关键点明显错误、手部目标不成立或不应进入困难正样本集：删除 review 图片。
- 不在此流程中手工移动关键点或改写标签；该流程是筛选，不是修点。
- 不重命名、移动或新增图片，不编辑 `request_manifest.jsonl`。

至少保留一张，否则系统拒绝发布空 selection。

#### B4. 发布

```bash
make hard-publish \
  HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab \
  SELECTION_ID=hard-20260806
```

程序将保留图片复制到：

```text
HAND_DATASET_ROOT/Selections/<selection_id>/published/
  images/<capture_source_id>/<roi_id>.png
  selection.jsonl
  manifest.json
```

`selection.jsonl` 保留原请求的 teacher label/provenance，并增加：

- `selection_id`；
- `source_crop_relpath`：原始 source ROI 路径；
- `published_relpath`：独立 published 图片路径。

`manifest.json` 记录保留数量和删除的 teacher error 数量。发布成功后 Registry 把 selection 标为 published，并删除临时 `review/`。即使以后删除源 proposal variant，published 图片仍可读取。

#### B5. 常见错误

- `selection_id has already been used`：ID 已 reserved/published，换新 ID。
- `hard-positive selection accepts Train requests only`：请求中混入了 val/test。
- `review request references unregistered/retired roi_id`：ROI 未注册或源变体已 retired。
- `review request disagrees with registry`：请求中的 raw/source/variant/crop path 与 Registry 不一致。
- `selection review contains unmanifested files`：review 中出现人工新增或改名文件。
- `cannot publish an empty hard-positive selection`：人工删除了全部图片。
- `selection is already published`：发布不可覆盖。

### C. 两条人工流程的操作检查表

准备前：

- 确认输入均来自 Train、source/variant 为 active。
- 为 negative dataset 或 selection 选择全新 ID。
- 确认 ROI crop 文件仍存在。

人工复核时：

- 只删除应拒绝的图片。
- 不新增、不改名、不移动图片。
- 不编辑 JSONL manifest。
- 负样本只保留明确无手图；困难样本只保留 teacher 标注正确的困难正样本。

发布后：

- 检查 published `manifest.json` 中的 `records`。
- 负样本检查 `negative_labels.jsonl` 和 `review_report.json`。
- 困难样本检查 `selection.jsonl` 的 `published_relpath`。
- 确认 `review/` 已被程序删除。
- 再执行源变体删除也不会影响已发布副本；但删除前必须先完成 prepare，因为 retired ROI 不能再进入新的复核批次。
