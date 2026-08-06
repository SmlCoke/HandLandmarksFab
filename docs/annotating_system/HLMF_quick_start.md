# HLMF 3.0 Quick Start

## 1. 环境检查

输入：仓库、现有 `anfab` 环境和已部署的 Eos/Hand landmark ONNX。

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

输入：已注册 train 来源、Eos、RTMPose 和 HCF。

```bash
make train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输出：Palm、ROI、draft、`hand_training_labels.jsonl`、`candidate_negatives.jsonl`、`ignored.jsonl` 和 QC。

批量处理全部已注册 train 来源：

```bash
make batch-train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_ID=FullEnhance0801 PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

## 4. Eval 自动标注

输入：已注册 val/test 来源和教师模型。

```bash
make eval-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-val-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输出：Palm、ROI、draft 和 QC；正式评估标签仍需 CVAT 复核。

批量处理并导出各来源 CVAT XML：

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

输出：reviewed JSONL、evaluation labels、ignored 和 dataset manifest。

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

## 9. 最终检查

```bash
make registry-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT"
make compile
make test
make help
```
