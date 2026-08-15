# HLMF 3.0 Quick Start

## 1. 环境检查

输入：仓库、现有 `anfab` 环境、`models/palm_detector/eos-2.0/model_384x224_opt.onnx` 和 Hand landmark ONNX；RTMPose 模式还需要双头 HCF，以及默认开启的 MediaPipe TFLite 补救模型和独立环境。

```bash
cd /root/HandLandmarksFab
source /root/miniconda3/etc/profile.d/conda.sh
conda activate anfab
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
/root/miniconda3/envs/anfab/bin/python -m pip uninstall -y onnxruntime
/root/miniconda3/envs/anfab/bin/python -m pip install -r requirements.txt
make compile
make test
```

首次部署 TFLite 补救：

```bash
/root/miniconda3/envs/anfab/bin/python -m venv \
  /root/miniconda3/envs/hlmf-mp-tflite
/root/miniconda3/envs/hlmf-mp-tflite/bin/python -m pip install \
  -r requirements-mediapipe-tflite.txt
mkdir -p models/mediapipe/hand_landmarker_tflite
# 将 Eos-2.0 部署到 models/palm_detector/eos-2.0/model_384x224_opt.onnx。
# 将 hand_landmark_full.tflite 部署到上述目录。
# 将双头 HCF 部署到 models/hand_classifier/handedness-handpresence-0813/model.onnx。
```

输出：代码和环境检查结果。默认链路为 RTMPose + Hand Classifier + 质量门控 + MediaPipe Hand Landmarker TFLite rescue。Eos-2.0 参数为 score `0.25`、全局 NMS `0.10`、ROI scale `1.8/1.8`，只支持 near/mid，默认 proposal variant 为 `eos-2.0`；设备为 Palm/HCF GPU（不可用时回退 CPU）、RTMPose CPU，RTMPose/HCF batch 为 64。

## 2. 注册来源

输入：`<source>/images/*.tif[f]`。

```bash
make source-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0

make palm-distance-check \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak
```

输出：`source.json`、`raw_images.jsonl`、图像 QC 和 Registry 记录。

## 3. Train 自动标注

输入：已注册 train 来源、Eos、RTMPose、双头 HCF 和 TFLite 补救资产；当前 RTMPose Train presence 阈值为 `0.025`，连接长度门控及 `rtmpose_train_mediapipe_tflite_rescue_enabled` 均默认开启。分别设为 `false` 可独立关闭连接长度门控或补救。

```bash
make train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-mid-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-2.0 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输出：Palm、单通道 `uint8 256×256` 无损 PNG ROI、含 HCF presence/handedness 与可选 TFLite 补救记录的 draft、三类发布 JSONL 和 QC；source report 记录四条门控互斥淘汰数，dataset manifest 记录 dataset 总计及各 capture source 明细。补救关闭时保持原 RTMPose 分流，连接长度门控关闭时其余三条门控仍正常工作。

批量处理全部含 `images/` 的 train 来源；无需预先运行 `source-check`：

```bash
make batch-train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_ID=FullEnhance0801 PROPOSAL_VARIANT=eos-2.0 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

输出：批处理显式跳过 far 并列出 `SKIPPED_UNSUPPORTED_DISTANCE`；near/mid 正常执行。若全部来源均被跳过则返回非零。

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

输出：同样只处理 near/mid，far 不生成或更新 proposal、ROI、CVAT 和发布资产。

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

## 7. 新录制 Gold、负样本与困难样本

新录制 Gold 来源输入 train 原图，执行与 Eval 相同的自动标注、CVAT 导出/导入和发布；输出 `GoldSource/ReviewedDatasets/<dataset_id>/`，同时含人工确认 positive/negative：

```bash
make gold-autolabel DATASET_SCOPE=gold DATASET_ID=gold-national-r1 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-train-s06-peak PROPOSAL_VARIANT=eos-2.0
make hand-cvat-export DATASET_SCOPE=gold DATASET_ID=gold-national-r1 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-train-s06-peak PROPOSAL_VARIANT=eos-2.0
# 放入 03_reviewed/eos-2.0/cvat_reviewed.xml
make hand-cvat-import DATASET_SCOPE=gold DATASET_ID=gold-national-r1 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-train-s06-peak PROPOSAL_VARIANT=eos-2.0
make source-publish DATASET_SCOPE=gold DATASET_ID=gold-national-r1 \
  CAPTURE_SOURCE_ID=complex-mid-bright-random-train-s06-peak PROPOSAL_VARIANT=eos-2.0
```

```bash
make negative-review HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS=/abs/candidate_negatives.jsonl
make negative-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  NEGATIVE_DATASET_ID=background-neg-0801

make hard-review HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  HARD_DATASET_ID=hard-hands-r1 MINING_REQUEST=/abs/request.jsonl
make hard-import HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  HARD_DATASET_ID=hard-hands-r1
make hard-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  HARD_DATASET_ID=hard-hands-r1
```

困难样本在 `hard-review` 后上传 `GoldSource/HardSamples/<id>/review/images/` 和 `cvat_autolabel.xml`，精修并放回 `cvat_reviewed.xml` 后再执行 import/publish。输出：负样本 HCF 预审清单，以及 CVAT 精修后的通用 `GoldSource/HardSamples/<id>/published/` 独立副本和 manifest。

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

如果要把未发布的 `eos_2.0-rtmpose-gate` 整轮清理后重跑，使用新名称：

```bash
make batch-source-variant-delete HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval DATASET_ID=FullEnhanceVal0801 \
  PROPOSAL_VARIANT=eos_2.0-rtmpose-gate \
  CONFIRM_DELETE=eos_2.0-rtmpose-gate
make batch-eval-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_ID=FullEnhanceVal0801 \
  PROPOSAL_VARIANT=eos_2.0-rtmpose-gate-r2 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

清理命令只在明确决定放弃旧草稿时执行；retired 名称不能复用。若直接继续现有 near/mid 草稿的人工复核，则继续使用原 variant，不要重新自动标注。

## 9. 最终检查

```bash
make registry-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT"
make compile
make test
make help
```
