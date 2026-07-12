# HandLandmarkerFab 处理系统修正计划：07A / 07B

> 文档定位：供后续代码实现使用的工程计划，不是最终用户操作手册。  
> 更新时间：2026-07-10。  
> 本轮只写计划，不修改代码。  
> 实施时以本文件和四份数据处理文档为需求基线。

## 1. 修正目标

保留人工复核前已经稳定的数据生成链路：

```text
00_validate_images.py
01_export_palm_detections.py
02_build_hand_roi_crops.py
03_run_mediapipe_on_rois.py
04_export_cvat_xml.py
```

Train/Val/Test 继续只通过各自配置文件区分路径和 Palm candidate 策略。Palm decode、NMS、ROI 几何、MediaPipe 推理与现有 CVAT label 均不改变。

人工复核后保留公共导入和可视化：

```text
05_import_cvat_xml.py
06_visualize_autolabels.py
```

最终处理使用两个独立入口：

```text
07A_finalize_training_labels.py
07B_finalize_evaluation_labels.py
```

核心职责：

- **07A / Train**：接受有噪声 pseudo 与少量 human Gold，完成结构门禁、样本分型、伪标签 QC、同原图重复 ROI 降采样、阶段化权重与训练清单输出。
- **07B / Val-Test**：接受完整人工复核结果，执行严格的一一覆盖和 Gold schema 校验；不运行 pseudo 质量启发式、不降采样、不按 Palm/teacher 分数过滤困难样本。

## 2. 明确不改的内容

### 2.1 00-03 处理逻辑

以下内容不得因 07 拆分而改变：

- 项目正式数据约定继续为 upright `1280x720` 灰度 TIFF；当前 `00` 仍兼容其他图像扩展名和可转灰通道，本次不改变这种兼容行为；
- aethersign ONNX Palm decode；
- score threshold、NMS 和 cross-head suppression；
- bbox + p0/p9 到 rotated ROI 的几何；
- `scale_x=scale_y=1.8`、`shift_y=-0.1`；
- ROI 输出 `256x256`；
- 越界区域黑色 padding；
- Google MediaPipe IMAGE mode 标注；
- `hand_presence` 只包含 `present`，不伪造 score。

### 2.2 CVAT labels

继续使用：

```text
no_hand
Left
Right
ignore_for_training
hand_landmarks
```

不新增 multi-hand、quality-tier、pseudo/gold 等 CVAT label。训练质量、数据来源、重复簇和权重属于 JSONL/QC 元数据，不应污染人工标注 schema。

### 2.3 04 导出流程

`04_export_cvat_xml.py` 已能输出现有 skeleton/tags，且直接使用 `02_roi_crops/images`，无需复制图片。07 拆分不要求修改其核心逻辑。

## 3. 当前实现的关键问题

### 3.1 当前 07 只做了结构规范化的一小部分

当前脚本会：

- 跳过 `ignore_for_training=true`；
- 跳过 presence=true 但 norm landmarks 不是 21 点；
- 跳过 presence=false 但仍携带 norm landmarks；
- 规范 negative 的 landmarks/handedness；
- 写入三个 0/1 head loss weight。

但不会：

- 区分 pseudo 与 human Gold；
- 区分四种 `palm_valid × presence`；
- 检查三个 landmark 数组的 id、有限值和坐标一致性；
- 拒绝 manifest/reviewed 覆盖不完整；
- 检测重复 `crop_id`/basename；
- 拒绝 `needs_review=true` 的 Gold；
- 保留具体 QC flags；
- 识别同一原图重复手实例；
- 限制低分负候选；
- 设置质量/采样权重；
- 生成 Stage 1/Stage 2 不同清单。

### 3.2 当前索引可能静默覆盖

`index_by()` 和按 basename 的索引遇到重复 key 时会以后一个覆盖前一个。两位成员存在同名 TIFF 时，直接合并会产生 `palm_det_id`、`crop_id` 和 PNG basename 冲突。

新 finalizer 必须显式收集全部 key：同一 source 内 local ID/basename 重复和跨 source global ID 重复立即报错，不能复用 last-wins 语义。不同 namespace 下相同原始 basename 本身允许存在，但在物理合并目录或同一 CVAT task 前必须 materialize 为全局唯一 basename。

### 3.3 当前 05 的导入诊断没有完整落到行内

当前 CVAT 导入行为包括：

- 多 skeleton 只 warning，仍取第一个；
- `no_hand + skeleton` 产生 error，并走 negative 分支；
- 无 skeleton 且无 `no_hand` 会按 negative 导入并 warning；
- 缺失于 XML 的图片可能补回 draft，设置 `source=cvat_reviewed_missing_image`；
- 最终行通常只剩 `needs_review`，具体 import warning/error 主要在统计文件中。

07B 不能只看一个 `needs_review` 布尔值判断 Gold 完整性。

### 3.4 当前多手 QC 对 ROI draft 基本无效

`03` 的 ROI MediaPipe 配置为 `num_hands=1`，因此 `mediapipe_num_hands_detected > 1` 基本不会出现。真实图片中有两只手但 teacher 只返回一只时，自动 QC 无法发现。

双手且目标不唯一的 ROI 继续依赖人工 `ignore_for_training`；07B 必须让 ignored 行在严格点校验之前退出。

## 4. 推荐代码边界

新增：

```text
scripts/07A_finalize_training_labels.py
scripts/07B_finalize_evaluation_labels.py
hand_autolabel/finalization.py
```

如果 `finalization.py` 过大，再拆为：

```text
hand_autolabel/training_selection.py
hand_autolabel/evaluation_finalization.py
```

结构校验纯函数放在 `quality_checks.py` 或独立 validation module。CLI 脚本只负责：

- 参数解析；
- 加载配置和文件；
- 调用纯函数；
- 原子写输出；
- 打印摘要。

不得在两个 CLI 中复制一套不同的 landmark schema 检查。

## 5. 05 的最小兼容增强

不改变任何 CVAT label 和现有导入语义，只在每个 reviewed row 增加：

```text
cvat_image_seen
cvat_review_status
cvat_import_warnings
cvat_import_errors
```

推荐 `cvat_review_status`：

```text
reviewed_positive
reviewed_negative
reviewed_ignored
missing_from_xml
import_conflict
```

同时避免当前 `summarize_label_rows()` 结果被 `stats.update(import_stats)` 的同名键覆盖。报告改为固定嵌套结构：

```text
import_integrity:
  warnings: []
  errors: []
  coverage: {...}

label_heuristics:
  warnings: []
  errors: []
  needs_review_count: 0
```

行内 warning/error 字段始终为 list。`cvat_review_status` 只表示 review/coverage 状态，具体冲突仍由独立数组表达；ignored 行可以绕过点位 Gold 校验，但仍必须满足 `cvat_image_seen=true`。

如果暂时不改 05，07B 必须同时读取 `cvat_import_stats.json`，按 crop ID 恢复具体导入诊断；不能只读取 reviewed JSONL。

## 6. 公共结构校验函数

建议新增以下纯函数概念：

```text
validate_manifest_coverage(...)
validate_unique_ids_and_paths(...)
validate_positive_label_structure(...)
validate_negative_label_structure(...)
validate_landmark_coordinate_consistency(...)
validate_manifest_label_identity(...)
```

### 6.1 Positive 硬条件

- 三组 landmark 均为 21 点；
- id 恰为唯一 `0..20`；
- x/y 全部 finite；
- crop norm 与 crop px 在容差内一致；
- crop px 经 `roi_corners_px` 投影后与 image px 在容差内一致；
- handedness 是 Left/Right，或只在 Train pseudo 中允许 unknown 并关闭 handedness loss；
- 目标文件存在且为 `256x256`。

### 6.2 Negative 硬条件

- 三组 landmark 全空；
- handedness 为 unknown，score 为 null；
- hand_id 为 null；
- crop 文件与 manifest 身份合法。

### 6.3 Manifest 权威

以下字段由 manifest 作为权威来源：

```text
image
crop_path
palm_det_id
palm_valid
palm_score
roi_rect
roi_corners_px
```

如果 label 与 manifest 冲突，应记录错误后 fail closed；不能用 `setdefault()` 静默保留冲突 label 元数据。

## 7. 07A：训练集 finalizer 契约

### 7.1 CLI

建议：

```powershell
python scripts/07A_finalize_training_labels.py `
  --config configs/finalize_train.yaml `
  --stage pretrain

python scripts/07A_finalize_training_labels.py `
  --config configs/finalize_train.yaml `
  --stage finetune
```

禁止根据目录名猜 split 或监督来源。所有数据源必须在配置中显式注册。

### 7.2 训练 finalizer 配置

新增：

```text
configs/finalize_train.yaml
```

示意：

```yaml
schema_version: train_finalize_v1

sources:
  - dataset_id: peak_train_v1
    contributor: Peak
    root: path/to/peak/train_data
    autolabel_config: path/to/peak/autolabel_train.yaml
    manifest: path/to/peak/hand_roi_crops_manifest.jsonl
    pseudo_labels: path/to/peak/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: path/to/peak/02_roi_crops/images
    gold_manifest: path/to/peak/review_subset_manifest.jsonl
    gold_labels: path/to/peak/hand_landmarks_reviewed_subset.jsonl
    gold_import_report: path/to/peak/cvat_import_stats.json
    supervision: pseudo_with_optional_gold

  - dataset_id: soar_train_v1
    contributor: Soar
    root: path/to/soar/train_data
    autolabel_config: path/to/soar/autolabel_train.yaml
    manifest: path/to/soar/hand_roi_crops_manifest.jsonl
    pseudo_labels: path/to/soar/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: path/to/soar/02_roi_crops/images
    gold_manifest: null
    gold_labels: null
    gold_import_report: null
    supervision: pseudo

dedup:
  positive_normalized_distance: 0.18
  max_positive_representatives_per_cluster: 1
  positive_cluster_warning_threshold_per_image: 2
  max_hard_negatives_per_image: 1
  max_easy_negatives_per_image: 1

sampling:
  positive: 0.70
  runtime_negative_candidate: 0.25
  low_palm_negative_candidate: 0.05
  gold_batch_fraction: 0.40

profiles:
  pretrain: {}
  finetune: {}
```

具体阈值必须可配置，报告中写回实际使用值。

### 7.3 输入

每个 source：

- 唯一 `dataset_id`；
- 数据 `root` 与对应 upstream autolabel config；
- manifest；
- pseudo label draft；
- 可选 human Gold subset、其 subset manifest 与 CVAT import report；
- 可计算时记录配置和模型 hash；历史数据缺少模型文件/hash 时报告 `unknown`，但不因此把样本判为结构无效。

第一阶段直接读取脚本 `03` 输出的 `02_roi_crops/hand_landmarks_autolabel_draft.jsonl`，不要求把 66000 ROI 全量走 CVAT。第二阶段同时读取 pseudo 和人工复核训练子集。Gold subset manifest 是验证该 CVAT 子任务是否完整覆盖的必需输入，Gold labels 是稀疏 override，不能假设覆盖完整训练 manifest。

不能只根据 `row.source == cvat_reviewed` 推断 Gold，因为一个 XML 即使没有真正人工修改，经过 05 后也可能带该 source。注册表中的 supervision 和明确的 Gold 文件才是权威。

### 7.4 输出

```text
pretrain:
  ../autodl-tmp/train_pretrain_merged/05_labels/hand_train_catalog_pretrain.jsonl
  ../autodl-tmp/train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl
  ../autodl-tmp/train_pretrain_merged/05_labels/hand_training_excluded_pretrain.jsonl
  ../autodl-tmp/train_pretrain_merged/qc/finalize_train_pretrain_report.json

finetune:
  ../autodl-tmp/train_finetune_merged/05_labels/hand_train_catalog_finetune.jsonl
  ../autodl-tmp/train_finetune_merged/05_labels/hand_training_labels_finetune.jsonl
  ../autodl-tmp/train_finetune_merged/05_labels/hand_training_excluded_finetune.jsonl
  ../autodl-tmp/train_finetune_merged/qc/finalize_train_finetune_report.json
```

含义：

- stage-specific `catalog`：保留该次运行看到的全部候选及分类、质量、重复簇信息，不物理删除图片，也不被另一阶段覆盖；
- `hand_training_labels_{stage}`：该阶段实际可用的 canonical 清单；
- stage-specific `excluded`：ID、action 和 reasons，避免第二阶段覆盖第一阶段排除记录；
- report：数量、配置、hash 和错误。

`outputs.pretrain` 和 `outputs.finetune` 是两个强制配置块；两个阶段写入不同根目录。stage-specific 文件是唯一 canonical 输出，训练 loader 必须显式选择 pretrain 或 finetune 文件。

### 7.5 新增字段

保留旧训练字段，并增加：

```text
schema_version
dataset_id
source_crop_id
global_crop_id
source_group_id
annotation_provenance
supervision_tier
sample_type
quality_tier
quality_flags
selection_action
duplicate_cluster_id
duplicate_cluster_size
duplicate_rank
sampling_bucket
sampling_weight
supervision_loss_weight
presence_quality_weight
landmark_quality_weight
handedness_quality_weight
training_stage
```

canonical 输出规定：

```text
crop_id = global_crop_id
source_crop_id = 00-03 产生的原始 local crop_id
```

否则合并后的旧 loader 仍会因 local ID 冲突。三个现有 `*_loss_weight` 在新 schema 中只作为 0/1 head applicability mask；`supervision_loss_weight` 表示 pseudo/Gold 的整体监督可信权重；三个 `*_quality_weight` 表示具体 head 的标签可信度；`sampling_weight` 只用于 sampler，禁止再次乘入 loss。训练时：

```text
effective_head_weight = head_applicability_mask
                      * supervision_loss_weight
                      * head_quality_weight
L_head = sum(effective_head_weight_i * loss_i) / max(sum(effective_head_weight_i), eps)
```

`sampling_bucket` 用于两级分层 sampler。07A 只输出 bucket 与权重，70/25/5 类型比例和 30%～50% Gold 比例由训练 loader 的“先选 supervision，再选 sample type”两级 sampler 保证，不能声称单个标量 `sampling_weight` 同时精确满足两个交叉约束。

枚举维度必须分开：

```text
annotation_provenance: mediapipe_pseudo | human_gold
sample_type: POS_RUNTIME | POS_LOW_PALM | NEG_RUNTIME_CANDIDATE | NEG_LOW_PALM_CANDIDATE
quality_tier: HIGH | MEDIUM | PRESENCE_ONLY | AMBIGUOUS | INVALID
selection_action: include | hold | drop_ignore | drop_invalid | drop_duplicate
```

`human_gold` 不是 quality tier，`drop_duplicate` 也不是 quality tier。

### 7.6 07A 执行顺序

1. 读取所有 source 并建立 namespace；
2. 在每个 source 内检测 local ID/basename 重复、manifest orphan、label orphan和路径缺失；不同 `dataset_id` 的同名 basename 允许由 namespace 隔离，只有物理汇入同一平面目录或同一 CVAT task 时才要求全局 basename 唯一；
3. 分别校验 pseudo、Gold subset manifest、Gold labels 与 Gold import integrity；
4. Gold 通过 global ID 覆盖对应 pseudo，形成 `effective_label`，然后才允许分类、聚类和 profile；
5. 有效 Gold 覆盖了 malformed pseudo 时，malformed pseudo 进入 source warning/quarantine 统计，但不阻断该有效 Gold；Gold 本身无效、重复或与 subset manifest 冲突时必须 fatal；
6. 对 effective label 与主 manifest 的身份、路径和几何进行一致性检查；
7. `ignore_for_training=true` 进入 exclusions；
8. 执行 positive/negative 结构 gate；
9. 按 `palm_valid × presence` 分类；
10. pseudo 才计算 teacher/Palm quality flags；human Gold 仍必须通过 CVAT import integrity 和 Gold schema，但不运行 pseudo 启发式；
11. 同原图 positive 用 `landmarks_image_px` 聚类同一物理手；
12. negative 用 rotated ROI overlap/空间差异聚类与限额；
13. 检查 pseudo negative 是否与同图确认 positive 手重叠；重叠只增加 `possible_false_negative` 并转 hold，绝不自动改成 positive；
14. 应用 `pretrain` 或 `finetune` profile；
15. 稳定排序并原子写该 stage 的 catalog、canonical labels、excluded 和 report；
16. 任一 fatal error 时不覆盖上一次 canonical 输出。

### 7.7 样本分类

最低四类：

```text
POS_RUNTIME
POS_LOW_PALM
NEG_RUNTIME_CANDIDATE
NEG_LOW_PALM_CANDIDATE
```

另外分别表达：

```text
annotation_provenance = human_gold | mediapipe_pseudo
quality_tier = HIGH | MEDIUM | PRESENCE_ONLY | AMBIGUOUS | INVALID
selection_action = include | hold | drop_ignore | drop_invalid | drop_duplicate
```

不得把 `needs_review=false` 命名为绝对 clean 或 Gold，也不得把 provenance、quality 和 action 混成一个枚举。

### 7.8 重复 ROI

按 `source_group_id=dataset_id:image` 分组：

- positive 在 `landmarks_image_px` 中比较。对 A、B，使用两者 landmark-derived bbox 对角线平均值作为尺度 `s`，定义 `D(A,B)=median_j(||pAj-pBj||)/max(s, eps)`；bbox overlap 指 21 点外接框 IoU，不是 Palm bbox；
- handedness 仅作软一致性信号：空间强一致但 handedness 冲突时仍可同簇，但增加 flag，避免一次 teacher handedness 错误拆开同一只手；
- 聚类采用按稳定 crop ID 排序的 deterministic complete-linkage agglomeration：只有候选与簇内所有成员的 `D` 都不超过阈值，且 landmark bbox IoU 满足配置条件时才加入，避免单链桥接把相邻双手合并；
- 每个 cluster 优先 Gold、无 flags、palm_valid、质量更好的代表；
- 同一物理手默认保留 1 个全权重代表，可配置保留第 2 个不同几何 ROI；
- 当前数据通常每图最多两只手；超过 `positive_cluster_warning_threshold_per_image` 时报告人工检查，但聚类算法不把“两簇”硬编码成自动删除上限；
- negative 使用 `roi_corners_px` 的 rotated polygon IoU 聚类，例如 IoU≥0.8 作为起点；每原图默认最多 1 hard + 1 easy；
- catalog 保留全部，stage 文件只选代表；
- 所有选择必须确定性，可配置 seed 并写入 report；
- 当前没有可靠 sequence metadata，不允许根据文件名猜连续帧进行强制时序去重。

距离阈值、bbox IoU、rotated IoU、complete-linkage 规则和 `eps` 必须进入配置与报告，并用人工抽样校准。一个 cluster 有 Gold 后，finetune 默认只使用 Gold 代表并排除同簇 pseudo siblings；若未来选择把 Gold 反投影到 siblings，必须做完整边界/投影校验，不能让 Gold 与冲突 pseudo 同时 replay。

### 7.9 Stage profiles

`pretrain`：

- 默认排除 pseudo_flagged、hold 和 invalid；
- clean-candidate positive 训练三个适用 head；
- low-Palm positive 限额；
- pseudo hard negative 因 teacher 漏手风险进入审计、hold 或降低监督权重；人工确认的 Gold hard negative 使用完整 presence 监督，不得按类别一刀切降权；
- easy negative 低采样；
- batch 比例由 sampler 使用 `sampling_weight` 实现，不随机永久删除大量行。

`finetune`：

- Gold 覆盖 pseudo；
- Gold batch 占 30%～50%；
- 高质量 pseudo replay 占 50%～70%；
- pseudo_flagged 默认排除；
- Gold 与 pseudo 的质量系数从配置读取，不能硬编码。

### 7.10 人工复核候选输出（第二期）

07A 可选读取 student prediction/disagreement 文件，输出：

```text
review_candidates_manifest.jsonl
review_candidates_draft.jsonl
review_candidates.csv
```

再复用当前 04 的 `--manifest`、`--draft-jsonl`、`--output-xml` 生成训练子集 CVAT 任务。05 使用同一 subset manifest/draft 导回 Gold，最后重新运行 07A `--stage finetune`。

首版必须按 `dataset_id` 分别生成 CVAT 子任务，因为当前 04 在 XML 中只写 crop basename，05 也按 basename 匹配；跨来源同名 PNG 放进同一 task 会冲突。后续若要合并任务，07A 必须先 materialize 一个 staging 目录，把文件复制/硬链接为 namespaced basename，并同步重写 subset manifest 的 `crop_path`。不能只增加 JSON 字段却继续上传原同名文件。

首版实现可先支持随机分层 candidates，学生分歧排序放到第二期。

## 8. 07B：验证/测试 finalizer 契约

### 8.1 CLI

```powershell
python scripts/07B_finalize_evaluation_labels.py `
  --config configs/finalize_val.yaml `
  --split val

python scripts/07B_finalize_evaluation_labels.py `
  --config configs/finalize_test.yaml `
  --split test
```

`--split` 必须显式为 `val|test`，并与聚合配置中的 `dataset.split` 一致：

```yaml
dataset:
  id: merged_val_v1
  split: val
```

每条 Val 路线的聚合配置列出 Peak shared、Soar shared 和当前路线 independent 三个 source；Test 聚合配置列出 Peak Test、Soar Test 两个 source。每个 source 都有独立 images、manifest、reviewed JSONL、import report 和可选 review context。

### 8.2 输入

- 一个或多个 source 的 manifest；
- 每个 source 的 `hand_landmarks_reviewed.jsonl`；
- CVAT import report（强制），并优先同时使用 05 写到行内的 import diagnostics；report 必须包含 XML hash、extra/duplicate XML images、missing manifest images 和按 crop ID 的冲突；
- 可选但在双手/ignore 分组报告中必需的 `review_context.csv` sidecar；它保存 review reason/context，不改变 CVAT labels；
- split、config 和版本元数据。

07B 不允许直接用 draft 代替 reviewed Gold。每个 Eval source 必须配置唯一 `dataset_id`；07B 与 07A 一样生成 `dataset_id:source_crop_id` 全局 namespace。跨 source local ID/basename 允许相同，不保留任何依赖原始文件名前缀的旧机制。

### 8.3 输出

Val：

```text
../autodl-tmp/val_merged/05_labels/hand_validation_labels.jsonl
../autodl-tmp/val_merged/05_labels/hand_val_ignored.jsonl
../autodl-tmp/val_merged/qc/finalize_val_report.json
```

Test：

```text
../autodl-tmp/test_merged/05_labels/hand_test_labels.jsonl
../autodl-tmp/test_merged/05_labels/hand_test_ignored.jsonl
../autodl-tmp/test_merged/qc/finalize_test_report.json
```

included 行等权。可以保留三个 head mask 供 Val loss/指标计算，但不得写训练集式 pseudo quality 或降采样权重。

### 8.4 07B 执行顺序

1. 分 source 验证 manifest、CVAT image、reviewed row 一一覆盖且唯一，检查 source 内 local ID/basename 唯一以及跨 source `dataset_id` 唯一；
2. 拒绝 duplicate、orphan、missing XML image 和 `cvat_reviewed_missing_image`；
3. 先识别 `ignore_for_training=true`，写入 ignored 输出；
4. ignored 行不要求修正 skeleton，避免在歧义双手上浪费人工时间；
5. ignored 输出必须标记 `ground_truth_valid=false`，其中保留的自动 presence/handedness/landmarks 不具备 Gold 语义；实现可选择清空这些监督字段，但无论如何训练和评测 loader 都不得读取；
6. 如提供 `review_context.csv`，校验 crop ID 唯一、`palm_det_id` 与 manifest 一致，并要求所有 ignored/双手行都有合法 reason/context；
7. 对非 ignored 行执行严格 positive/negative schema gate；Gold 点必须满足 norm∈[0,1]、crop px∈[0,width-1]×[0,height-1]，无法可靠标到 crop 内的真实点应回 CVAT 标 ignore；Train pseudo 的轻微越界仍只是独立 quality policy，不能复用这条 Gold fatal 规则；
8. 拒绝 multiple skeleton、no_hand conflict、missing handedness、错误点数、非 finite、坐标不一致和 import errors；
9. 正式 Val/Test 默认要求 `palm_valid=true`，否则视为配置/数据漂移；
10. 接受人工确认的 `palm_valid=true, presence=false`，不应用 high-palm-negative heuristic；
11. 不自动更改人工 presence、handedness 或 landmarks；
12. 稳定排序并原子写 canonical、ignored 和 report；
13. 任一 fatal error 时不覆盖旧 canonical 文件。

### 8.5 07B 原则

> 结构严格，人工语义权威。

Val/Test 不需要：

- teacher TTA 筛选；
- student/teacher 一致性筛选；
- 同原图重复 ROI 降采样；
- 低分负样本限额；
- pseudo 质量权重；
- 根据 Palm score 排除困难样本。

同一原图的两个有效 ROI 可以都保留。评测时除 ROI 级指标外，可额外按 source image 聚合，不能在 finalizer 中删除。

### 8.6 双手 ignored 处理

若两只手都可能成为模型合理目标、Palm anchor 不清、交叉后点归属不唯一，则人工添加 `ignore_for_training`。07B 将其排除主评测。

若 Palm anchor 唯一指向一只目标手且可可靠标 21 点，则只标目标手，第二只手作为干扰，进入主评测。

无需新增 CVAT label。07B 报告 ignored 数量和比例；20%～30% ignore 产生显著 warning，但不自动修改标签。可以配置最低 included 样本数，低于门槛时要求定向补录。

当前 CVAT/JSONL 只有 ignore 布尔值，无法自动恢复多手原因。需要细分 `multi_hand_target_clear/ambiguous` 时必须读取 sidecar。ignored 多手没有唯一 landmark Gold，因此 challenge 默认只报告数量、覆盖率、anchor ambiguity/failure rate 和定性案例；定量双手 landmark/end-to-end 指标需要额外原图级双手实例 Gold。

### 8.7 Per-crop Palm anchor reference

为了让人工能按 [验证集目标手规则](hand_landmarker_val_dataset_processing.md) 追溯目标，后续扩展 `06_visualize_autolabels.py` 或新增只读辅助脚本。不得修改 `00-03`、原始 ROI 或 CVAT labels。

输入：

```text
palm_detections.jsonl
hand_roi_crops_manifest.jsonl
hand_landmarks_autolabel_draft.jsonl
原始 TIFF 与 ROI images
```

对每个 manifest row：

1. 通过 `image + palm_det_id` 找到且只找到一个 detection，重复/缺失立即报告；
2. 原图 reference 只高亮当前 detection 的 bbox、p0、p9、p0→p9 箭头和当前 ROI，其他 detections 灰显；
3. 把 p0、p9 和 bbox 四角映射到 crop。对 `C0,C1,C3`：

   ```text
   ex = C1 - C0
   ey = C3 - C0
   u = dot(P-C0, ex) / dot(ex, ex)
   v = dot(P-C0, ey) / dot(ey, ey)
   crop_px = (u*(width-1), v*(height-1))
   ```

   使用点积公式前必须检查 `ex/ey` 非零、`abs(dot(ex,ey))/(||ex||·||ey||)` 小于配置容差，并执行原图→crop→原图 round-trip。当前 `roi_corners_px()` 生成正交矩形，公式与 `cv2.getAffineTransform([C0,C1,C3], ...)` 一致；若 corners 损坏或未来不再正交，应直接求 2×2 仿射逆矩阵或 fail closed，不能生成看似合理的错误 reference。

4. 生成不参与训练的 crop reference，绘制映射后的 bbox 四边形、p0、p9、箭头及 `crop_id/palm_det_id`；
5. 输出：

   ```text
   04_visualization/target_anchor/global/<safe_crop_basename>.png
   04_visualization/target_anchor/crop/<safe_crop_basename>.png
   04_visualization/target_anchor_review_index.csv
   ```

CSV 至少包含：

```text
crop_id,image,palm_det_id,palm_score
p0_image_x,p0_image_y,p9_image_x,p9_image_y
p0_crop_x,p0_crop_y,p9_crop_x,p9_crop_y
bbox_crop_polygon,global_reference_path,crop_reference_path
```

输出文件名必须使用 manifest 中已有 crop basename 或复用 `_safe_name()` 规则；不能直接把包含 `:` 的原始 `crop_id` 当 Windows 文件名。

测试必须覆盖旋转 ROI 的原图→crop→原图 round-trip，并验证 reference 不覆盖原始 ROI 文件。该工具只显示证据，不自动决定目标手；最终 `multi_hand_target_clear/ambiguous` 仍由人工写入 sidecar。

## 9. QC 函数拆分

建议将当前 `label_issues()` 拆为：

```text
label_structure_issues()
pseudo_quality_flags()
evaluation_integrity_issues()
```

保留旧 `label_issues()` wrapper 一版以兼容其他脚本。

原因：

- `negative_with_high_palm_score` 对 Train pseudo 是有用风险 flag；
- 对人工确认的 Val/Test hard negative，它是合法数据，不能成为拒绝理由；
- handedness teacher score 低对 pseudo 有意义，人工 Gold 导入后 score 本来就是 null；
- Train 需要启发式质量分层，Gold Eval 只需要结构与导入完整性。

## 10. Makefile 计划

新增正式 targets：

```text
finalize_train_pretrain
finalize_train_finetune
finalize_val
finalize_test
```

映射：

- `finalize_train_pretrain/finetune` → 07A；
- `finalize_val/test` → 07B；
- help 同步说明 canonical 输出。

- 新增 `FINALIZE_TRAIN_CONFIG ?= configs/finalize_train.yaml`；

- `finalize_val`/`finalize_test` 直接指向 07B；
- 不提供含义模糊的 `finalize_train` 别名，必须显式选择 pretrain 或 finetune。

另外新增轻量 `audit_split_leakage` 纯函数或 target：输出 Train/Val/Test 的 `dataset_id + source_image_uid` inventory，并在原图 hash 可用时检测“改名但内容相同”的泄漏。同一原图及其所有 ROI 必须只属于一个 split；缺少原图文件时明确报告 hash coverage，不能把“未发现”写成“确认无泄漏”。

## 12. 报告与可复现性

07A report：

- 每个 source 输入、合法、保留、排除数量；
- 四象限类型；
- quality tier/flags；
- duplicate clusters 和降采样；
- negative 限额；
- sampling bucket、归一化方式和期望抽样分布；只有明确定义估计方法时才报告 sampling-weight effective sample size；
- stage profile；
- 配置、输入、输出 hash；
- seed 和代码版本。

07B report：

- manifest/XML/reviewed 覆盖率；
- included/ignored/invalid；
- positive/negative/Left/Right/hard-negative；
- ignore 比例；
- import/structure errors；
- config、Gold、eligible、ignored 和输出 hash；
- split 和评测 schema 版本。

canonical JSONL 和一个排除生成时间、绝对路径、dirty git 状态等 volatile metadata 的 `data_fingerprint` 必须跨相同输入重复运行保持稳定。完整 report 可以包含时间与运行环境，因此不要求整份 report 文件 SHA256 跨机器一致；发布前应核对 report schema、内部输入/输出 hash 和 `data_fingerprint`，而不是比较整个 report 的字节 hash。

## 13. 稳定接口约束

1. 07A/07B 是唯一正式 finalizer 入口；
2. stage-specific JSONL 是唯一 canonical 训练输出；
3. `06_visualize_autolabels.py --labels-jsonl` 对原始 per-source local-ID 数据继续不变；canonical merged 07A 输出使用 global `crop_id`，现有 06 无法直接与 local manifest join；
4. 不自动删除、移动或重写现有大规模数据；
5. 合并两位成员数据前先生成 namespace catalog 和冲突报告。

## 14. 实施顺序

### Phase 1：公共结构与 07B

1. 增加 CVAT 行级 import diagnostics；
2. 实现公共唯一性、coverage 和 landmark schema 校验；
3. 实现 per-crop Palm anchor reference 与 review sidecar 校验；
4. 实现严格 07B；
5. 先冻结可信 Val/Test。

理由：没有可信 Val/Test，无法判断 07A 的任何筛选或权重是否有效。

### Phase 2：07A 最小版本

1. source registry 与 namespacing；
2. 硬门禁；
3. 四象限分类；
4. 对 draft 重新计算 flags，并在 baseline 中只排除 pseudo `needs_review=true`；human Gold 不应用 teacher/Palm heuristic；
5. 同原图 positive 聚类；
6. negative 限额；
7. Stage 1 catalog/labels/report；
8. 训练第一版基座。

### Phase 3：Gold 与 Stage 2

1. 随机分层 review candidates；
2. current 04/05 子集 CVAT round-trip；
3. Gold override；
4. finetune profile；
5. Gold + pseudo replay；
6. Val early stopping；
7. Test 冻结评测。

### Phase 4：增强质量分析

时间允许再增加：

- teacher 多变换一致性；
- student/teacher disagreement；
- 骨架几何异常；
- 图像黑边、模糊和亮度特征；
- 明确 session metadata 后的跨帧近重复降采样；
- challenge set 自动索引。

## 15. 非目标

本修正计划不包括：

- 重训或修改 Palm Detector；
- 修改 Palm threshold/NMS/ROI 几何；
- 改变 Google MediaPipe 模型；
- 增加 CVAT label；
- 支持一个 ROI 同时输出两套手关键点；
- 自动证明人工 Gold 100% 正确；
- 自动删除用户已有数据；
- 使用 Test 调参。

## 16. 实现验收标准

- [ ] 00-03 代码和结果不因 finalizer 拆分改变；
- [ ] CVAT labels 完全不变；
- [ ] 两位成员同名数据不会静默覆盖；
- [ ] 07A 能直接使用 draft 完成 Stage 1；
- [ ] 07A 能用 Gold 覆盖 pseudo 并生成 Stage 2；
- [ ] 07A 输出样本类型、具体 flags、重复簇和独立 sampling weight；
- [ ] 07B 不接受 draft 或缺失人工覆盖；
- [ ] 07B 接受人工确认的高分 hard negative；
- [ ] 07B 对非 ignored CVAT 冲突 fail closed；
- [ ] ignored 双手不要求浪费时间修点，也不进入主指标；
- [ ] 任一 fatal error 不覆盖 canonical 文件；
- [ ] Val/Test 输出、ignored 清单和 report 可 hash 冻结；
- [ ] Makefile 正式与 smoke targets 清晰分流；
- [ ] 单元测试覆盖成功、冲突、ignore、去重和迁移路径；
- [ ] README 和训练接口文档在实现后同步更新。
