# HLMF 3.0 Quick Start

本页给出完整操作顺序和每个阶段的最小输入/输出说明。原理与参数调整见 `HLMF_annotating_workflow.md`。

## 环境依赖（首次部署）

输入：仓库根目录的 `requirements.txt`。处理：创建 Python 3.11 环境并安装 HLMF 依赖。输出：Conda 环境 `anfab` 及代码检查结果。

```bash
cd /path/to/HandLandmarkerFab
conda create -n anfab python=3.11 pip -y
conda activate anfab
python -m pip install -r requirements.txt
python -m pip check
make compile
make test
```

已有环境时只执行 `conda activate anfab`。新服务器若缺 OpenCV/MediaPipe 系统动态库，按 workflow 的环境依赖部分一次性安装。

## 0. 设置目录和来源身份

输入图片必须位于以下二者之一，且 `images/` 中只放平铺 TIFF：

```text
HAND_DATASET_ROOT/PretrainSource/<dataset_id>/<capture_source_id>/images/
HAND_DATASET_ROOT/EValSource/<dataset_id>/<capture_source_id>/images/
```

来源名固定为 `<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>`。

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
cd /path/to/HandLandmarkerFab
```

四个公共配置各自只有一种职责：`configs/autolabel.yaml` 管 Palm/ROI/MediaPipe，`configs/review.yaml` 管 Hand ROI CVAT 规则，`configs/datasets.yaml` 管发布集合和数据上限，`configs/cvat_label.json` 是 CVAT label schema。

## 1. 来源检查（Source Check）

输入：Train 或 Eval 来源的 TIFF 原图。处理：校验/幂等旋转、建立稳定 raw ID 和 registry。输出：来源下的 `raw_images.jsonl`、`source.json` 和 `qc/image_validation_report.json`。

```bash
make source-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT" DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 CAPTURE_SOURCE_ID=white-far-bright-fist-train-s01-peak PROPOSAL_VARIANT=palm-v1
```

## 2A. Train 自动标注与发布（Train Autolabel）

输入：已放好的 Train TIFF、Palm ONNX、MediaPipe task 和 `configs/autolabel.yaml`。处理：Palm → 程序化 Hand ROI → MediaPipe → 质量分流 → 发布。输出位于来源的 `01_palm/palm-v1/`、`02_roi_crops/palm-v1/`、`05_labels/palm-v1/` 和 `qc/palm-v1/`。

```bash
make train-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" DATASET_SCOPE=pretrain DATASET_ID=FullEnhance0801 CAPTURE_SOURCE_ID=white-far-bright-fist-train-s01-peak PROPOSAL_VARIANT=palm-v1
```

`hand_training_labels.jsonl` 是通过门控的 positive；`candidate_negatives.jsonl` 只能进入后续删除式复核，不能直接训练。

## 2B. Val/Test 自动标注（Eval Autolabel）

输入：Eval TIFF 和两个模型。处理：只对 Palm 实际检测到的 proposal 自动生成 ROI 和 MediaPipe draft；不保留低分候选负样本。输出位于来源的 `01_palm/`、`02_roi_crops/` 和 `qc/`，此时尚未发布评估标签。

```bash
make eval-autolabel HAND_DATASET_ROOT="$HAND_DATASET_ROOT" DATASET_SCOPE=eval DATASET_ID=national-eval-0801 CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
```

## 3. 导出 Hand ROI CVAT（Hand CVAT Export）

输入：`02_roi_crops/palm-v1/images/`、ROI manifest 和 MediaPipe draft。输出：`03_reviewed/palm-v1/cvat_autolabel.xml`。

```bash
make hand-cvat-export HAND_DATASET_ROOT="$HAND_DATASET_ROOT" DATASET_SCOPE=eval DATASET_ID=national-eval-0801 CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
```

在 CVAT 中只复核程序生成的 Hand ROI。完整标签为 `no_hand`、`Left`、`Right`、`unknown_handedness`、`ignore_for_training` 和 21 点 `hand_landmarks` skeleton。不要绘制或调整 ROI，也不要修改 Palm bbox/p0/p9。

## 4. 导入 CVAT（Hand CVAT Import）

从 CVAT 导出 Images 1.1 XML，放到 `03_reviewed/palm-v1/cvat_reviewed.xml`。输入为该 XML、原 draft 和 ROI manifest；输出为 `hand_landmarks_reviewed.jsonl` 与 `qc/palm-v1/cvat_import_report.json`。

```bash
make hand-cvat-import HAND_DATASET_ROOT="$HAND_DATASET_ROOT" DATASET_SCOPE=eval DATASET_ID=national-eval-0801 CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
```

## 5. 发布 Val/Test 来源（Source Publish）

输入：已通过检查的 reviewed JSONL。处理：发布固定 ROI 标签并更新 dataset manifest。输出：`05_labels/palm-v1/hand_evaluation_labels.jsonl`、发布报告和 `EValSource/<dataset_id>/dataset_manifest.json`。

```bash
make source-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" DATASET_SCOPE=eval DATASET_ID=national-eval-0801 CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice PROPOSAL_VARIANT=palm-v1
```

## 6. 真负样本复核与发布（Negative Review）

输入：Train 的 `candidate_negatives.jsonl`。输出：`GoldSource/NegativeSamples/<id>/review/images/` 删除式审核树。

```bash
make negative-review HAND_DATASET_ROOT="$HAND_DATASET_ROOT" NEGATIVE_DATASET_ID=background-neg-0801 NEGATIVE_CANDIDATE_LABELS="$HAND_DATASET_ROOT/PretrainSource/FullEnhance0801/<capture_source_id>/05_labels/palm-v1/candidate_negatives.jsonl"
```

人工删除所有含手、模糊或无法确认的 ROI，再发布剩余真背景。输出到 `GoldSource/NegativeSamples/<id>/published/`。审核树初始为硬链接；允许只把 `review/images/` 经压缩包和网盘带到本地复核，再用保持原相对路径和文件名的普通文件替换服务器上的 `review/images/`。保留 manifest/README，不删除整个 `review/`。

```bash
make negative-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" NEGATIVE_DATASET_ID=background-neg-0801
```

## 7. 困难正样本复核与发布（Hard Positive Review）

输入：HLML 的 `mining/<snapshot_id>/hlmf_review_request.jsonl`。输出：`Selections/<id>/review/images/` 审核树。

```bash
make hard-review HAND_DATASET_ROOT="$HAND_DATASET_ROOT" SELECTION_ID=hard-positive-0801 MINING_REQUEST=/root/autodl-tmp/TrainFab/HLML-4.0/mining/v4-r1/hlmf_review_request.jsonl
```

人工只删除 MediaPipe 点明显错误的 ROI，不重标点。允许采用与负样本相同的压缩包/网盘/本地复核方式，只替换 `review/images/` 并保留 request manifest。发布输出为 `Selections/<id>/published/selection.jsonl` 和 `manifest.json`，继续零拷贝引用原 ROI。

```bash
make hard-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" SELECTION_ID=hard-positive-0801
```

## 8. 完整性与代码检查

输入为 registry、配置和代码；输出为终端检查结果，不改数据。

```bash
make registry-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT"
make compile
make test
make help
```

Val/Test 只评估 Palm 已生成的固定 Hand ROI；当前流程不统计 Palm 漏检或原图级联性能。
