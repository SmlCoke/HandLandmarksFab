# HLMF 3.0 Quick Start

## 1. 环境检查

输入：仓库、现有 `anfab` 环境和已部署的 Eos/Hand landmark ONNX；RTMPose 模式还需要 `models/handedness-handpresence-0807/model.onnx` 双头 HCF。

```bash
cd /root/HandLandmarksFab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
make compile
make test
```

输出：代码和环境检查结果。

## 2. 注册来源

输入：`<source>/images/*.tif[f]`。

```bash
make source-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0
```

输出：`source.json`、`raw_images.jsonl`、图像 QC 和 Registry 记录。

## 3. Train 自动标注

输入：已注册 train 来源、Eos、RTMPose 和双头 HCF；`configs/autolabel.yaml` 当前使用 presence 阈值 `0.5`，并默认开启 RTMPose 连接对长度门控。将 `rtmpose_train_connection_length_gate_enabled` 设为 `false` 可只关闭第四条门控。

```bash
make train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输出：Palm、单通道 `uint8 256×256` 无损 PNG ROI、含 HCF presence/handedness 的 draft、`hand_training_labels.jsonl`、`candidate_negatives.jsonl`、`ignored.jsonl` 和 QC；任一 Train 质量门控失败的行进入 `ignored.jsonl`。连接长度门控关闭时，presence、handedness 和边界坐标门控仍正常工作。

批量处理全部含 `images/` 的 train 来源；无需预先运行 `source-check`：

```bash
make batch-train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_ID=FullEnhance0801 PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

## 4. Eval 自动标注

输入：已注册 val/test 来源、Eos、RTMPose 和双头 HCF。

```bash
make eval-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输出：Palm、单通道 `uint8 256×256` 无损 PNG ROI、含 HCF presence/handedness 的 draft 和 QC；Train presence 阈值不作用于 Eval，正式评估标签仍需 CVAT 复核。

批量处理全部含 `images/` 的 Eval 来源并导出各来源 CVAT XML；无需预先运行 `source-check`：

```bash
make batch-eval-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_ID=FullEnhanceVal0801 PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

## 5. 可视化

输入：既有 ROI images 和 draft。

```bash
make autolabel-visualize-roi HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0

make autolabel-visualize-original HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0
```

输出：ROI PNG、原图 PNG 和默认 MP4。只生成 PNG 时追加 `ORIGINAL_VIDEO=false`。

清理可重建可视化：

```bash
make autolabel-visualizations-clean HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0
```

批量清理同一数据集全部来源的该变体可视化：

```bash
make batch-autolabel-visualizations-clean HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 PROPOSAL_VARIANT=eos-2.0
```

输出：删除 ROI/原图可视化 PNG、MP4 和 visualization QC；标签、manifest 与 Registry 不变。

## 6. Eval CVAT 与发布

输入：Eval ROI/draft；导入阶段还需 `03_reviewed/<variant>/cvat_reviewed.xml`。

```bash
make hand-cvat-export HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0

make hand-cvat-import HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0

make source-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0
```

输出：reviewed JSONL、evaluation labels、ignored 和 dataset manifest。未完成 CVAT 导入与 `source-publish` 的来源不进入 manifest，下游 HLML 会忽略。

## 7. 负样本与困难样本

```bash
make negative-review HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS=/abs/candidate_negatives.jsonl
make negative-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  NEGATIVE_DATASET_ID=background-neg-0801

make hard-review HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  SELECTION_ID=hard-0801 MINING_REQUEST=/abs/request.jsonl
make hard-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  SELECTION_ID=hard-0801
```

输出：review/published 独立图片副本和对应 manifest。

## 8. 永久删除一个来源变体

输入：精确 scope、dataset、source、variant，以及完全相同的确认值。

```bash
make source-variant-delete HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 CONFIRM_DELETE=eos-2.0
```

输出：派生产物被删除、dataset manifest 被重建、Registry 写入 retired tombstone；原图和 raw/source 元数据保留。同名变体不能再次使用。

批量永久删除数据集全部已注册来源的同名变体：

```bash
make batch-source-variant-delete HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  PROPOSAL_VARIANT=eos-2.0 CONFIRM_DELETE=eos-2.0
```

输出：逐来源写入 retired tombstone、删除精确变体产物，最后重建 dataset manifest。确认值必须与变体名完全一致。

## 9. 最终检查

```bash
make registry-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT"
make compile
make test
make help
```
