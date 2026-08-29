<div align="center">

<img src="./docs/assets/aethersign-logo-minimal.svg" alt="AetherSign minimal logo" width="116" />

<h1>HandLandmarkerFab（HLMF 3.0）</h1>

**AetherSign Iris 数据制作、自动标注与人工复核系统**

[![Archive](https://img.shields.io/badge/Status-Competition_Final-8B5CF6?style=flat-square)](#-i-项目定位与归档状态) [![Tag](https://img.shields.io/badge/Tag-HLMF_3.0_final-0891B2?style=flat-square)](https://github.com/SmlCoke/HandLandmarksFab/tree/HLMF-3.0-final) [![Tests](https://img.shields.io/badge/Tests-79_Passed-059669?style=flat-square)](#-viii-依赖管理) [![ROI](https://img.shields.io/badge/Hand_ROI-256%C3%97256-2563EB?style=flat-square)](#411-默认-eos-21-配置)

[项目定位](#-i-项目定位与归档状态) · [复现环境](#-ii-全国总决赛阶段复现环境) · [模型配置](#-iv-模型与推理配置) · [常用操作](#-v-常用操作) · [质量边界](#-vii-train-质量边界)

</div>

---

## ✦ I. 项目定位与归档状态

### 1.1 项目定位

HandLandmarkerFab 是 Hand Landmarker(**即 Iris 模型**) 训练系统的上游数据制作仓库。系统默认从 Eos-2.1 Palm Detector 的 bbox、p0、p9 构造固定 `256×256` Hand ROI，再执行 RTMPose-m Hand5、双头 HCF、质量门控和 MediaPipe Hand Landmarker TFLite rescue；MediaPipe Tasks 与 HaMeR 是两条独立的显式覆盖链路。

#### 1.1.1 仓库边界

本仓库只制作 Hand Landmarker 数据，不负责 Palm Detector 训练或下游 Iris/Muse 训练。Palm 几何不能人工修改，人工复核对象是程序生成的 Hand ROI、21 个关键点、handedness 以及 `no_hand/ignore_for_training` 状态。

#### 1.1.2 Hand ROI 输入契约

Hand ROI 的模型输入契约是解码后的单通道 `uint8 256×256` 像素。仓库使用无损 PNG 保存 ROI；PNG/TIFF 是存储容器，不改变相同灰度数组的像素域。板端从 `SSNE_Y_8` 摄像头内存直接构造 ROI，并不把 TIFF 文件送入 Hand Landmarker。

### 1.2 比赛归档状态

> [!IMPORTANT]
> AetherSign 已于 2026-08-25 完成全国总决赛答辩并获得全国一等奖。本仓库针对本届比赛的数据制作使命已经完成，最终可复现代码状态由 annotated tag `HLMF-3.0-final` 固定。

#### 1.2.1 Git 归档范围

tag 保存代码、配置、测试、文档和校准报告。模型权重、正式 `HAND_DATASET_ROOT` 数据仓、HaMeR checkpoint/MANO 及其他外部模型资产仍按既有策略独立保存，不包含在 Git tag 中。

#### 1.2.2 正式提交模型

AetherSign 正式提交的 Hand Landmarker 为 Iris-2.0-Lite (multitask) 与 Iris-2.0-Max (multi-finetune)，不包含 Pro 版本。

---

## ◇ II. 全国总决赛阶段复现环境

> [!NOTE]
> 以下是比赛阶段使用的服务器基础实例配置，方便后续复现和其他开发者参考。它描述基础服务器，不替代各运行环境的 requirements 契约。

### 2.1 软件镜像

#### 2.1.1 基础镜像配置

| 项目 | 配置 |
| --- | --- |
| 镜像 | TensorFlow 2.9.0 |
| Python | 3.8 |
| 操作系统 | Ubuntu 20.04 |
| CUDA | 11.2 |

#### 2.1.2 HLMF 运行环境说明

基础镜像的 Python 版本不等同于 HLMF 每个实际运行环境。主 HLMF 环境、MediaPipe TFLite venv 与 HaMeR 独立环境的安装方法和版本约束，以[完整工作流](docs/annotating_system/HLMF_annotating_workflow.md)及仓库 requirements 文件为准。

### 2.2 硬件资源

#### 2.2.1 计算资源

| 项目 | 配置 |
| --- | --- |
| GPU | RTX 3090（24GB）× 1 |
| CPU | 14 vCPU，Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz |
| 内存 | 90GB |

#### 2.2.2 存储资源

| 类型 | 容量 |
| --- | ---: |
| 系统盘 | 30GB |
| 免费数据盘 | 50GB |
| 付费数据盘 | 176GB |

---

## 🧭 III. 文档与审计入口

### 3.1 核心入口文档

| 文档 | 作用 |
| :-- | :-- |
| [完整工作流](docs/annotating_system/HLMF_annotating_workflow.md) | 各阶段输入、命令、输出与配置原理 |
| [快速开始](docs/annotating_system/HLMF_quick_start.md) | 端到端操作命令速查 |
| [数据契约](docs/annotating_system/HLMF_data_contract.md) | 目录、manifest、标签和 Registry 接口 |
| [当前状态](docs/annotating_system/HLMF_current_status.md) | 比赛归档状态与最后服务器历史快照 |
| [常见问题与解答](docs/annotating_system/HLMF_qa.md) | 已记录的仓库问题与答案 |

### 3.2 模型与系统审计

- [Eos-2.1 审计与校准报告](assets/palm_detector/eos_2_1_adaptation.md)
- [HCF v1 MobileNetV3-Large 接入与校准报告](assets/hand_classifier/v1_mobilenet_v3_large.md)
- [HaMeR 标注后端接入报告](assets/hand_landmark/hamer_integration.md)
- [Iris Eval 就绪度评估](assets/evaluation/iris_eval_readiness.md)
- [ONNX CPU/GPU 性能报告](assets/device_perf/onnx_cpu_gpu_benchmark.md)

---

## ⬡ IV. 模型与推理配置

| 阶段 | 默认 | 显式替代 | 主要输出 |
| :-- | :-- | :-- | :-- |
| Palm | Eos-2.1 | — | bbox、p0、p9、proposal score |
| Hand Landmark | RTMPose-m Hand5 | MediaPipe Tasks、HaMeR | 21 点 Hand ROI 坐标 |
| ROI 分类 | HCF v1 MobileNetV3-Large | — | hand presence、handedness |
| 几何补救 | MediaPipe Hand Landmarker TFLite | 可关闭 | 重新预测的 21 点坐标 |

### 4.1 Palm Detector

#### 4.1.1 默认 Eos-2.1 配置

Eos-2.1 输入灰度 `[1,1,224,384]`，使用 score `0.25`、全局 NMS IoU `0.10`、最多 2 个检测及 840 anchors。Hand ROI 几何为 `scale=1.8/1.8、shift=0/-0.1`，默认 proposal variant 为 `eos-2.1`。

#### 4.1.2 拍摄距离能力边界

`supported_capture_distances=[near,mid]`。历史 far 回放召回不足，因此 HLMF、后续 Iris/Muse 和端侧演示均不支持 far。far 原始来源应保留，但不得进入 Eos-2.1 的 Palm→ROI→Landmark→复核→发布链路；单来源命令硬拒绝，批处理显式跳过。

#### 4.1.3 Low-score candidate

Eos low-score candidate 不运行 RTMPose/HCF，presence 与 handedness 保持 `unknown/null`，进入独立人工候选链路。

### 4.2 Hand Landmark 教师后端

#### 4.2.1 RTMPose-m Hand5

默认配置为 `hand_landmark.backend: rtmpose_onnx`。原始 SimCC logits 直接 argmax，再除以固定 `2.0`；每个 runtime ROI 固定输出 21 个关键点。

#### 4.2.2 MediaPipe Tasks

通过 `HAND_LANDMARK_BACKEND=mediapipe_tasks` 对单次命令进行显式覆盖。它不是默认后端，也不与 RTMPose/HaMeR 输出混合。

#### 4.2.3 HaMeR

通过 `HAND_LANDMARK_BACKEND=hamer` 显式启用。HaMeR 在独立 `.hamer` 环境加载 official CVPR24 checkpoint，使用 `rescale=0.75` 和 CUDA，仅负责预测 21 点。输出按 Hand ROI 像素域裁到 `[0,255]`，再应用与 RTMPose 相同的四项 Train 门控和 TFLite 几何补救。

HaMeR 内部 ViTPose/亮度 handedness fallback 不进入 HLMF；外部 HCF 直接决定 HaMeR 左右手翻转及最终 presence/handedness。

比赛最终提交的 Iris-2.0 系列模型的 Hand Landmarker 关键点教师正是 HaMeR，因为其在**SC132GS 域下各种简单或复杂手势下的精度都较高，鲁棒性不错**。

### 4.3 双头 Hand Classifier（HCF）

#### 4.3.1 Runtime ROI 分类

模型位于 `models/hand_classifier/v1-mobilenet_v3_large/model.onnx`，输入为灰度 `[N,1,256,256]`，输出 `handedness` 与 `hand_presence` 两个 `[N,2]` logits。模型 ID 为 `hand-classifier-v1-mobilenet_v3_large`，用于 RTMPose/HaMeR runtime ROI；旧模型仅作归档。

#### 4.3.2 HaMeR 外部 HCF

HaMeR 默认使用 `HLMF-Enhance/hand_classifier/v1-mobilenet_v3_large/model.onnx`。该副本应与 HLMF 主环境中的 HCF 保持同一版本。

#### 4.3.3 负样本预审

`negative-review` 复用 `hand_classifier.model_onnx_path` 批量计算 `P(has_hand)`，并在 selected/excluded 清单和 README 中记录实际 HCF 模型 ID。预审不能替代人工负样本复核。

### 4.4 MediaPipe TFLite 几何补救

#### 4.4.1 触发条件

仅当 RTMPose/HaMeR Train runtime 未通过边界门控或已开启的连接长度门控时，纯 Hand Landmarker TFLite 才重新预测 21 点。

#### 4.4.2 权威字段

TFLite presence/handedness 输出被丢弃，HCF 始终是这两个字段的唯一权威来源。补救结果必须重新通过边界和连接长度检查后才能替换原关键点。

### 4.5 ONNX Runtime 与设备选择

#### 4.5.1 Provider 配置

Eos-2.1 Palm 与 HCF 默认 `auto`：CUDA 可用时使用 GPU，否则回退 CPU。RTMPose 固定 CPU，因为 GPU 虽更快，但在人工复核 Eval 上轻微降低了关键点精度。可在 `onnx_runtime.model_providers` 中配置 `auto|cuda|cpu`。

#### 4.5.2 Batch 配置

RTMPose/HCF 动态 batch 默认为 `64`；Palm 模型输入固定为 batch 1。实测依据见[性能报告](assets/device_perf/onnx_cpu_gpu_benchmark.md)。

### 4.6 模型资产分发边界

ONNX/TFLite 模型遵循仓库忽略策略，不纳入 Git。代码和配置通过 Git 同步，模型需要在执行环境中按配置路径和 model ID 单独部署。

---

## 🚀 V. 常用操作

### 5.1 基础检查

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
make compile
make test
make help
make registry-check
```

### 5.2 来源注册与距离检查

```bash
make source-check DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice \
  PROPOSAL_VARIANT=eos-2.1

make palm-distance-check \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice
```

### 5.3 自动标注

#### 5.3.1 Train 来源

```bash
make train-autolabel DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice \
  PROPOSAL_VARIANT=eos-2.1 HAND_LANDMARK_BACKEND=rtmpose_onnx

make train-autolabel DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s01-alice \
  PROPOSAL_VARIANT=eos-2.1-hamer HAND_LANDMARK_BACKEND=hamer
```

#### 5.3.2 Eval 来源

```bash
make eval-autolabel DATASET_SCOPE=eval DATASET_ID=demo-eval \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice \
  PROPOSAL_VARIANT=eos-2.1 HAND_LANDMARK_BACKEND=rtmpose_onnx
```

#### 5.3.3 Gold 来源

```bash
make gold-autolabel DATASET_SCOPE=gold DATASET_ID=gold-demo \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-train-s06-alice \
  PROPOSAL_VARIANT=eos-2.1
```

### 5.4 可视化

#### 5.4.1 生成可视化

```bash
make autolabel-visualize-roi DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.1
make autolabel-visualize-original DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.1
```

`autolabel-visualize-original` 默认在 PNG 完成后生成 `visualizations/original_image_landmarks/<variant>.mp4`；传 `ORIGINAL_VIDEO=false` 可只生成 PNG。

#### 5.4.2 清理可视化

```bash
make autolabel-visualizations-clean DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.1
```

该命令只删除可重建可视化，不写入 retired tombstone。

### 5.5 批量处理与变体生命周期

#### 5.5.1 批量自动标注

```bash
make batch-train-autolabel DATASET_ID=demo PROPOSAL_VARIANT=eos-2.1 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
make batch-eval-autolabel DATASET_ID=demo-eval PROPOSAL_VARIANT=eos-2.1 \
  HAND_LANDMARK_BACKEND=rtmpose_onnx
```

#### 5.5.2 删除精确变体

```bash
make source-variant-delete DATASET_SCOPE=pretrain DATASET_ID=demo \
  CAPTURE_SOURCE_ID=... PROPOSAL_VARIANT=eos-2.1 CONFIRM_DELETE=eos-2.1
```

该命令删除精确来源/变体的派生产物，保留 `images/`、`raw_images.jsonl` 与 `source.json`，并写入永久 retired tombstone。同一来源不能再次使用已 retired 的变体名。

### 5.6 人工复核数据集

#### 5.6.1 负样本复核

```bash
make negative-review NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS=/abs/candidate_negatives.jsonl
```

#### 5.6.2 困难样本复核

```bash
make hard-review HARD_DATASET_ID=hard-hands-r1 \
  MINING_REQUEST=/abs/hlmf_review_request.jsonl
# CVAT 精修并放回 review/cvat_reviewed.xml 后：
make hard-import HARD_DATASET_ID=hard-hands-r1
make hard-publish HARD_DATASET_ID=hard-hands-r1
```

---

## 🔒 VI. 操作不变量

### 6.1 距离能力门控

批量自动标注先按 Palm 能力预检全部来源：near/mid 正常执行，far 输出 `SKIPPED_UNSUPPORTED_DISTANCE` 并计入 skipped，不计作运行失败；若没有任何兼容来源则返回非零。`source-check`、历史可视化和精确变体清理仍允许处理 far，以保留原始资产和诊断能力。

### 6.2 Proposal variant 隔离

现有 near/mid 草稿可以继续沿用原 variant 完成后续复核。若要从头重跑，必须使用从未写入过的新 variant；不能复用 retired 名称，也不能在已有 ROI/CVAT 资产上原地重跑。

### 6.3 数据持久化边界

所有正式数据写入 `HAND_DATASET_ROOT`，不得绑定训练 run ID。HLMF 发布的数据集应可被多个训练仓库和多个训练实验复用。

---

## ◈ VII. Train 质量边界

### 7.1 四项质量门控

RTMPose 或 HaMeR Train runtime 行满足以下任一条件时，整行进入 `ignored.jsonl`。

| 门控 | 当前配置 | 失败结果 |
| :-- | :-- | :-- |
| Hand presence | `P(has_hand) < 0.5` 或值无效 | `ignored.jsonl` |
| Handedness | HCF score `< 0.8` | `ignored.jsonl` |
| 边界坐标 | 42 个坐标值中边界值数量 `>= 2` | `ignored.jsonl` |
| 连接长度 | 任一连接严格超过距离专属阈值 | `ignored.jsonl` |

#### 7.1.1 Hand presence

`hand_presence.score=P(has_hand)` 缺失、非有限或低于 `quality.rtmpose_train_hand_presence_threshold` 时拒绝。v1 MobileNetV3-Large 校准值为 `0.5`。

#### 7.1.2 Handedness

HCF handedness 分数低于 `quality.handedness_review_threshold` 时拒绝，当前阈值为 `0.8`。

#### 7.1.3 边界坐标

42 个 crop 坐标值中，精确边界值达到 `quality.rtmpose_train_boundary_coordinate_reject_threshold` 时拒绝，当前阈值为 `2`。

#### 7.1.4 连接长度

开启连接长度门控时，任一指定连接的 crop 像素长度严格超过当前距离对应阈值时拒绝。

### 7.2 连接长度校准

连接长度门控由 `quality.rtmpose_train_connection_length_gate_enabled` 独立控制并默认开启。near/mid 阈值使用 9,237 条人工复核 Gold hand 的 image-space 点重投影到 Eos-2.1 ROI，并按 `ceil(P99.95 × 1.05)` 重算；Gold 保留率为 99.632%。

Eos-2.1 不支持 far，因此 far 只保留不可达的历史占位值，不宣称为新统计结论。完整结果见[连接长度分布报告](assets/quality_gate/rtmpose_connection_length_distribution.md)。

### 7.3 TFLite 补救语义

`quality.rtmpose_train_mediapipe_tflite_rescue_enabled` 默认开启。补救点只有重新通过边界与连接长度检查后才替换原关键点，否则保留原教师点并继续拒绝。该补救不是第五条门控，也不使用 TFLite 的 presence/handedness 输出。

### 7.4 门控适用范围

Presence、边界和连接长度门控作用于 RTMPose/HaMeR Train runtime；handedness 门控还适用于 MediaPipe Train positive。四条门控不改变 Eval 发布，也不作用于 Eos low-score candidate。Eval 正式真值始终以 CVAT 人工复核为准。

### 7.5 报告与互斥计数

发布报告按既有优先级对四条门控进行互斥计数。单来源结果写入 `source_publish_report.json`，dataset 总计及 `capture_source_id` 明细写入 `dataset_manifest.json`。

---

## 🧩 VIII. 依赖管理

### 8.1 主环境

主环境由 `requirements.txt` 管理，使用 `onnxruntime-gpu==1.18.0`，并为其 ABI 约束 NumPy `<2`、OpenCV `<4.11`。既有 `anfab` 环境需先移除 CPU 包 `onnxruntime`，再安装仓库依赖。

### 8.2 MediaPipe TFLite 环境

TFLite 补救使用独立 Python 3.11 venv 和 `requirements-mediapipe-tflite.txt`，不向主 `anfab` 环境引入 TensorFlow。

### 8.3 HaMeR 环境

HaMeR 使用独立 `.hamer` 环境及外部 checkpoint/MANO 资产，PyTorch、Detectron2 等依赖不进入 HLMF 主环境。

如果复现，需要额外在仓库外部克隆 [HaMeR 源码仓库](https://github.com/geopavlakos/hamer.git) 单独进行环境依赖配置。

---

<div align="center">

<img src="./docs/assets/aethersign-logo-minimal.svg" alt="AetherSign" width="52" />

<sub>AetherSign · Eos → Iris → Muse · 全国总决赛一等奖归档</sub>

</div>
