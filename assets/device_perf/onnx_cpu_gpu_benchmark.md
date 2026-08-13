# ONNX CPU/GPU 性能与一致性测试

## 结论

Eos-2.0 Palm Detector 与 0813 Hand Classifier（HCF）在 GPU 上明显更快，且未改变本次设备一致性回放的分类或门控判断，默认使用 `auto`（CUDA 优先、不可用时回退 CPU）。RTMPose GPU 虽快约 27 倍，但对全部 9,868 条人工复核 Eval 回放的平均关键点误差比 CPU 高 `0.0145 px`，因此按精度优先原则固定使用 CPU。RTMPose/HCF 默认 batch 为 `64`；Palm 模型输入固定为 batch 1。

## 测试环境与模型

| 项目 | 内容 |
|---|---|
| CPU | 2× Intel Xeon Gold 6330，56 核/112 线程 |
| GPU | NVIDIA GeForce RTX 3090 24 GB |
| 驱动 / CUDA / cuDNN | 580.105.08 / 11.2 / 8.1 |
| Python / NumPy / OpenCV / ONNX Runtime | 3.11.15 / 1.26.4 / 4.10.0 / `onnxruntime-gpu==1.18.0` |
| Palm | Eos 2.0，`models/palm_detector/eos-2.0/model_384x224_opt.onnx` |
| RTMPose | RTMPose-m Hand5 256×256，`models/rtmpose/rtmpose-m_hand5_256x256.onnx` |
| HCF | handedness-handpresence-0813，`models/hand_classifier/handedness-handpresence-0813/model.onnx`，模型 ID `hand-classifier-handedness-handpresence-0813` |

测试原因是原 `anfab` 仅安装 CPU 版 `onnxruntime 1.28.0`，provider 列表没有 CUDA；同时 Palm 代码曾显式锁定 CPU，RTMPose/HCF 又逐张推理，导致 CPU 高负载而 3090 利用率低。

## 方法

- 吞吐：固定相同 float32 输入，分别用 CPU/CUDA provider 预热后记录纯模型推理耗时中位数。Eos-2.0 Palm 重复 100 次；RTMPose 沿用 batch `1/8/32/64` 扫描；0813 HCF 使用 batch 64。
- Palm 一致性：使用 1 组固定随机输入和 16 张真实 TIFF，比较 TensorFlow、ONNX CPU、ONNX GPU 的四个原始输出以及正式阈值解码结果。
- HCF 结构：0809/0813 均执行 ONNX full checker，并比较节点拓扑、initializer 名称/形状/类型、参数元素数及动态 batch 输入输出契约。
- RTMPose 精度只读回放 `FullEnhanceVal0801:eos-1.0` 与 `FullEnhanceVal0808:eos_1.0-gate_r2` 的全部 9,868 条人工复核 hand，并比较 21 点与人工点的 crop 像素误差。0813 HCF 只读回放 10,773 条 near/mid 人工 Gold ROI（10,675 hand、98 no_hand），比较 CPU/GPU 类别、人工 handedness、presence 分布及门控跨阈值情况。
- 正式数据仓库全程只读；环境与流水线集成另在独立临时 `HAND_DATASET_ROOT` 验证。

## 性能结果

| 模型 | batch | CPU images/s | GPU images/s | GPU 加速比 | 默认设备 |
|---|---:|---:|---:|---:|---|
| Eos-2.0 Palm Detector | 1（模型固定） | 175.0 | 426.2 | 2.43× | GPU，CPU fallback |
| RTMPose | 64 | 64.1 | 1,734.0 | 27.07× | **CPU（精度约束）** |
| HCF 0813 | 64 | 324.9 | 8,443.3 | 25.99× | GPU，CPU fallback |

RTMPose GPU 吞吐随 batch `1/8/32/64` 分别为 `317.5/1,363.6/1,684.2/1,734.0 images/s`。0813 HCF 在 batch 64 下 CPU/GPU 中位耗时为 `196.97/7.58 ms`，吞吐为 `324.9/8,443.3 images/s`，因此继续采用 batch 64。

0809/0813 HCF 均有 142 个节点、110 个 initializer 和 1,515,612 个参数元素；节点及 initializer schema 完全一致，输入均为动态 batch `[N,1,256,256]`，输出均为 `handedness/hand_presence [N,2]`。两者权重输出不同，符合重新训练预期。

## 输出与精度结果

| 模型 | 回放结果 | 判定 |
|---|---|---|
| Eos-2.0 Palm | TF↔ONNX raw 最大绝对差 `5.4e-7`，CPU↔GPU 为 `5.6e-7`；score `0.15/0.25` 的 decoded 数量一致，几何/分数最大差 `2.7e-7` | 结构、转换和设备一致性通过，采用 GPU |
| HCF 0813 | 10,773 条回放中 CPU/GPU presence 与 handedness argmax 全部一致，最大 logits 差 `0.005410`、最大概率差 `0.000786`，presence `0.025/0.5` 与 handedness `0.7` 均 0 次跨阈值；人工 handedness raw accuracy 为 `98.1601%` | 设备间未观察到分类或门控变化，采用 GPU |
| RTMPose | CPU/GPU 平均点误差分别为 `1.4585/1.4729 px`；GPU 增加 `0.0145 px`；207,228 个点中 6,044 个点解码不同 | GPU 有轻微精度下降，保留 CPU |

0813 在 10,675 条人工 hand 上以 Train 阈值 `0.025` 保留 10,669 条（99.944%），并拒绝 88/98 条 no-hand。全局使用 `0.5` 虽可保留 99.550% hand，但独立 s05 只保留 86.538%，因此 Train 仍用 `0.025`；`negative_review.hand_presence_threshold` 继续使用独立的 `0.5` 预审核分界。handedness `0.7` 覆盖 98.733%，覆盖内准确率 98.640%。详细校准见 `assets/hand_classifier/handedness_handpresence_0813.md`。

## 何时重测

更换服务器 GPU/CUDA/ONNX Runtime、更新任一模型，或新增代表性人工复核 Eval 后，应在正式数据只读条件下重新测试。只有当候选设备不降低人工 gold 精度、不改变门控跨阈值结果，并且吞吐更高时，才更新 `configs/autolabel.yaml` 的逐模型 provider 与 batch。
