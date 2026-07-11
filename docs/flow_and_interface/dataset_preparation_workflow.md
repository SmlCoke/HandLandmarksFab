# Hand Landmarker 数据集制作操作手册

> 适用范围：训练集、共享/独立验证集和共享测试集的自动标注、CVAT 复核、07A/07B 冻结以及训练接口。  
> 环境：所有命令均在仓库根目录执行，并使用 README 指定的 `anfab` 环境。  
> namespace：Train 由 07A 自动添加；Val/Test 已在录制文件名中使用 `peak_` / `soar_`，07B 只校验、不重命名。

## 1. 最终数据组织

| 数据集 | 原始数据 | 是否完整人工复核 | 最终处理 | 用途 |
|---|---|---:|---|---|
| Train pseudo | Peak 约 2.2 万 ROI + Soar 约 4.6 万 ROI | 否 | 07A `pretrain` | 第一阶段教师—学生伪标签训练 |
| Train Gold 子集 | 从训练候选中挑选的数百到数千 ROI | 是 | 07A `finetune`，Gold 覆盖 pseudo | 第二阶段精调 |
| Val shared | `vals_data` | 是 | 与 independent Val 一起交给 07B | 两条训练路线的共同比较基准 |
| Val independent | `vali_data` | 是 | 与 shared Val 一起交给 07B | 保留训练路线独立性 |
| Test shared | `test_data` | 是 | 07B `test` | 两人 100% 共享的最终冻结评测 |

`vals_data` 和 `vali_data` 的“80%/40%”只描述采集计划，不作为脚本采样参数。07B 按实际输入和实际 included ROI 合并，并在 `finalize_val_report.json` 中报告每个 source/partition 的真实数量。

推荐目录：

```text
../autodl-tmp/
├─ train_data/
├─ soar_train_data/          # 示例：Soar 的训练数据，可自行修改
├─ vals_data/                # 共享 Val
├─ vali_data/                # 当前路线独立 Val
├─ val_merged/               # 07B 生成的最终 Val
├─ test_data/                # 100% 共享 Test
└─ test_merged/              # 07B 生成的最终 Test
```

每个尚未合并的数据目录内部保持：

```text
images/
01_palm/
02_roi_crops/
03_reviewed/
04_visualization/
05_labels/
qc/
```

## 2. 环境与执行原则

激活环境：

```powershell
conda activate anfab
```

确认当前目录：

```powershell
Get-Location
```

应位于 `HandLandmarkerFab` 仓库根目录。

必须遵守：

1. Train、Val、Test 的 ROI 参数、Palm threshold、NMS 必须与板端一致。
2. Val/Test 不导出低分 `negative_candidates`。
3. `peak_` / `soar_` 前缀必须在运行 00 之前已经存在于原始文件名中。
4. 不得把 Val/Test ROI 加入训练集。
5. Test 在模型、阈值、量化方案冻结前不得用于调参。

## 3. 配置文件职责

| 配置 | 作用 |
|---|---|
| `configs/autolabel_train.yaml` | 单个 Train 来源运行 00–06 |
| `configs/finalize_train.yaml` | 07A 合并多个训练来源、自动 namespace、分型和降采样 |
| `configs/autolabel_val.yaml` | shared Val，即 `vals_data`，运行 00–06 |
| `configs/autolabel_vali.yaml` | independent Val，即 `vali_data`，运行 00–06 |
| `configs/finalize_val.yaml` | 07B 合并 `vals_data + vali_data` 并冻结最终 Val |
| `configs/autolabel_test.yaml` | shared Test 运行 00–06 |
| `configs/finalize_test.yaml` | 07B 汇总并冻结最终 Test |

### 3.1 修改自动标注路径

例如 shared Val：

```yaml
dataset:
  id: vals_shared_v1
  split: val
  partition: shared

paths:
  images_dir: ../autodl-tmp/vals_data/images
  palm_outputs_dir: ../autodl-tmp/vals_data/01_palm
  roi_crops_dir: ../autodl-tmp/vals_data/02_roi_crops
  reviewed_dir: ../autodl-tmp/vals_data/03_reviewed
  labels_dir: ../autodl-tmp/vals_data/05_labels
  qc_dir: ../autodl-tmp/vals_data/qc
```

如果服务器目录不同，只修改 `paths`；不要为了减少标注量修改 Palm、NMS 或 ROI 参数。

## 4. Train 第一阶段：生成 pseudo 训练集

### 4.1 每个训练来源分别运行 00–03

Peak 数据：

```powershell
make validate_images_train
make palm_detection_train
make build_roi_train
make run_mediapipe_train
```

Soar 数据必须使用指向 Soar 目录的独立自动标注配置运行相同四个脚本。可以复制 `configs/autolabel_train.yaml` 为个人配置，只修改 `paths`，不要修改模型和 ROI 参数。

第一阶段需要的核心输入是：

```text
02_roi_crops/hand_roi_crops_manifest.jsonl
02_roi_crops/hand_landmarks_autolabel_draft.jsonl
02_roi_crops/images/*.png
```

不需要对全部训练集执行 CVAT 复核。

### 4.2 配置多个 Train 来源

编辑 `configs/finalize_train.yaml`：

```yaml
sources:
  - dataset_id: peak_train_v1
    contributor: Peak
    root: .
    autolabel_config: configs/autolabel_train.yaml
    manifest: ../autodl-tmp/train_data/02_roi_crops/hand_roi_crops_manifest.jsonl
    pseudo_labels: ../autodl-tmp/train_data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: ../autodl-tmp/train_data/02_roi_crops/images

  - dataset_id: soar_train_v1
    contributor: Soar
    root: .
    autolabel_config: configs/autolabel_train_soar.yaml
    manifest: ../autodl-tmp/soar_train_data/02_roi_crops/hand_roi_crops_manifest.jsonl
    pseudo_labels: ../autodl-tmp/soar_train_data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl
    crop_images_dir: ../autodl-tmp/soar_train_data/02_roi_crops/images
```

要求：

- 每个 `dataset_id` 全局唯一且版本固定；
- 两个 `- dataset_id` 必须在 YAML 中保持同一级缩进；其余 source 字段比 `-` 多缩进两个空格；
- Train 原文件可以同名，07A 会生成 `dataset_id:source_crop_id` 形式的 `global_crop_id`；
- 不要手工把 `peak_` / `soar_` 再加到 Train JSONL，避免双重 namespace；
- `autolabel_config` 必须与该来源实际运行 00–03 时使用的配置一致。
- 如果数据搬迁后 manifest 内的 `crop_path` 已过期，必须设置 `crop_images_dir`；07A 会按 basename 定位真实图片，并把旧值保存在 `source_crop_path` 中。

### 4.3 生成 pretrain 清单

```powershell
make finalize_train_pretrain
```

主要输出：

```text
hand_train_catalog_pretrain.jsonl
hand_training_labels_pretrain.jsonl
hand_training_excluded_pretrain.jsonl
qc/finalize_train_pretrain_report.json
```

训练 loader 读取 `hand_training_labels_pretrain.jsonl`。07A 已完成：

- 多来源 namespace；
- manifest/label/图片结构门禁；
- 四种 `sample_type` 分类；
- pseudo 质量 flags；
- 同原图正样本聚类；
- 负样本去重和限额；
- sampling bucket、sampling weight 和各 head loss weight 输出。

有效 head loss 权重：

```text
head_loss_weight
× supervision_loss_weight
× 对应的 head_quality_weight
```

`sampling_weight` 只用于 sampler，不乘入 loss。

## 5. Train 第二阶段：制作 Gold 子集并 finetune

### 5.1 选择人工复核候选

优先选择：

- 不同手型、距离、旋转、光照和背景；
- `POS_LOW_PALM`；
- `NEG_RUNTIME_CANDIDATE`；
- teacher/student 分歧或 `quality_flags` 较多的样本；
- 少量普通高质量样本作为稳定基线。

同一原图或连续视频帧不要大量重复进入 Gold。

### 5.2 导出并完成 CVAT 复核

对选中的 subset manifest/draft 运行：

```powershell
python scripts/04_export_cvat_xml.py `
  --config <该来源的 autolabel 配置> `
  --manifest <subset_manifest.jsonl> `
  --draft-jsonl <subset_draft.jsonl> `
  --output-xml <subset_autolabel.xml>
```

上传：

1. subset 对应的 ROI 图片；
2. `subset_autolabel.xml`；
3. 保持现有 labels：`no_hand`、`Left`、`Right`、`ignore_for_training`、`hand_landmarks`。

人工规则：

- 有手：只保留一个目标手 skeleton，修正完整 21 点，并标 Left/Right；
- 无手：删除 skeleton，添加 `no_hand`；
- 双手但 Palm anchor 唯一：只标 anchor 指向的目标手；
- 双手目标不唯一、点归属不可靠：添加 `ignore_for_training`；
- 不要用“更大、居中或 Google 首先检测到”代替 Palm anchor。

导出 reviewed XML 后运行 05：

```powershell
python scripts/05_import_cvat_xml.py `
  --config <该来源的 autolabel 配置> `
  --reviewed-xml <subset_reviewed.xml> `
  --manifest <subset_manifest.jsonl> `
  --draft-jsonl <subset_draft.jsonl> `
  --output-jsonl <subset_gold.jsonl>
```

检查对应 `qc/cvat_import_stats.json`，必须解决所有非 ignored import errors。

### 5.3 把 Gold 接入 07A

在对应 source 下增加：

```yaml
gold_manifest: ../path/to/subset_manifest.jsonl
gold_labels: ../path/to/subset_gold.jsonl
gold_import_report: ../path/to/cvat_import_stats.json
```

然后运行：

```powershell
make finalize_train_finetune
```

07A 会先按原始 `crop_id` 用 Gold 覆盖 pseudo，再做分类和去重。训练 loader 改读：

```text
hand_training_labels_finetune.jsonl
```

## 6. shared Val：制作 `vals_data`

### 6.1 自动标注

确认 `configs/autolabel_val.yaml` 指向 `../autodl-tmp/vals_data`，然后运行：

```powershell
make validate_images_vals
make palm_detection_vals
make build_roi_vals
make run_mediapipe_vals
make export_cvat_vals
```

### 6.2 CVAT 复核与导入

上传：

```text
../autodl-tmp/vals_data/02_roi_crops/images/
../autodl-tmp/vals_data/02_roi_crops/cvat_autolabel.xml
```

完整人工复核全部 ROI，导出至：

```text
../autodl-tmp/vals_data/03_reviewed/cvat_reviewed.xml
```

然后运行：

```powershell
make import_cvat_vals
make visualize_vals
```

不要在这一步运行 07B；必须等 independent Val 也完成后统一合并。

## 7. independent Val：制作 `vali_data`

确认 `configs/autolabel_vali.yaml` 指向当前路线自己的 `vali_data`：

```powershell
make validate_images_vali
make palm_detection_vali
make build_roi_vali
make run_mediapipe_vali
make export_cvat_vali
```

在 CVAT 完整复核后，将 XML 放到：

```text
../autodl-tmp/vali_data/03_reviewed/cvat_reviewed.xml
```

再运行：

```powershell
make import_cvat_vali
make visualize_vali
```

Peak 和 Soar 若各自维护不同的 independent Val，应各自使用自己的 `vali_data` 路径和 `finalize_val.yaml`；shared `vals_data` 保持完全相同。

## 8. 合并并冻结最终 Val

`configs/finalize_val.yaml` 的关键结构：

```yaml
dataset:
  id: merged_val_v1
  split: val

evaluation:
  allowed_namespace_prefixes: [peak_, soar_]

sources:
  - source_id: vals_shared
    partition: shared
    root: ../autodl-tmp/vals_data
    autolabel_config: configs/autolabel_val.yaml

  - source_id: vali_independent
    partition: independent
    root: ../autodl-tmp/vali_data
    autolabel_config: configs/autolabel_vali.yaml

outputs:
  labels_dir: ../autodl-tmp/val_merged/05_labels
  qc_dir: ../autodl-tmp/val_merged/qc
```

运行：

```powershell
make finalize_val
```

07B 会：

1. 分别检查 shared/independent 的 manifest、reviewed、XML import report；
2. 检查跨来源 `crop_id` 和 crop basename 不重复；
3. 检查原图和 `crop_id` 已以 `peak_` / `soar_` 开头，但不修改它们；
4. 分离 ignored；
5. 对 included Gold 执行严格 21 点、handedness、坐标和图片校验；
6. 合并并原子写入最终 Val；
7. 按 source 和 partition 报告实际 included 数量。

最终文件：

```text
../autodl-tmp/val_merged/05_labels/hand_validation_labels.jsonl
../autodl-tmp/val_merged/05_labels/hand_val_ignored.jsonl
../autodl-tmp/val_merged/qc/finalize_val_report.json
```

## 9. 制作并冻结 100% 共享 Test

Test 只运行一份共享流程：

```powershell
make validate_images_test
make palm_detection_test
make build_roi_test
make run_mediapipe_test
make export_cvat_test
```

在 CVAT 完整复核后，将 XML 放到：

```text
../autodl-tmp/test_data/03_reviewed/cvat_reviewed.xml
```

运行：

```powershell
make import_cvat_test
make visualize_test
make finalize_test
```

`configs/finalize_test.yaml` 只包含 `test_shared` source。07B 同样检查 `peak_` / `soar_` 前缀和跨文件唯一性，但不会自动添加 namespace。

最终文件：

```text
../autodl-tmp/test_merged/05_labels/hand_test_labels.jsonl
../autodl-tmp/test_merged/05_labels/hand_test_ignored.jsonl
../autodl-tmp/test_merged/qc/finalize_test_report.json
```

两人必须使用完全相同的 Test canonical JSONL 和 SHA-256。Test 只在最终 checkpoint、阈值和量化方案冻结后运行。

## 10. 07B 常见失败及处理

| 错误 | 含义 | 处理 |
|---|---|---|
| `manifest_file_missing` | source 路径配置错误或 02 未完成 | 检查 `root` 和 manifest 路径 |
| `reviewed_file_missing` | 05 尚未完成 | 先运行对应 `import_cvat_*` |
| `missing_cvat_import_report` | 缺少 05 的 QC 报告 | 重新运行 05，不要手工伪造 |
| `cross_source_duplicate_crop_id` | shared/independent ID 冲突 | 检查预置 `peak_` / `soar_` 前缀和数据重复 |
| `cross_source_duplicate_crop_basename` | 合并后文件名冲突 | 在 00 之前修正原始命名并重新生成 |
| `crop_id_missing_declared_namespace_prefix` | ID 没有合法前缀 | 不让 07B 自动改名；回到源数据修正 |
| `manifest_without_reviewed` | CVAT/05 覆盖不完整 | 补齐 CVAT XML 后重新导入 |
| `invalid_reviewed_rows` | 非 ignored Gold 有结构或语义冲突 | 根据 report 中 crop ID 回 CVAT 修正 |
| `evaluation_requires_palm_valid` | Eval 混入低分 negative candidate | 检查 Val/Test 配置是否错误开启候选导出 |

07B 出现 fatal 时不会覆盖上一版 canonical JSONL。不要通过删除困难 ROI、降低校验标准或修改 Palm threshold 来“修复”报告。

## 11. 最终执行清单

Train：

- [ ] 每个来源 00–03 完成；
- [ ] `finalize_train.yaml` 中每个 `dataset_id` 唯一；
- [ ] `make finalize_train_pretrain` 成功；
- [ ] 人工 Gold subset 完成 CVAT 和 05；
- [ ] Gold 三个路径写入对应 source；
- [ ] `make finalize_train_finetune` 成功。

Val：

- [ ] `vals_data` 完成 00–05 和完整人工复核；
- [ ] `vali_data` 完成 00–05 和完整人工复核；
- [ ] 两部分使用预置 `peak_` / `soar_` 前缀；
- [ ] `make finalize_val` 成功；
- [ ] 查看 shared/independent 实际数量、ignored 比例和 SHA-256。

Test：

- [ ] 共享 `test_data` 完成 00–05 和完整人工复核；
- [ ] 最终模型与阈值已冻结；
- [ ] `make finalize_test` 成功；
- [ ] Peak/Soar 使用相同 canonical 文件和 SHA-256。
