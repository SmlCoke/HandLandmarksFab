# ONNX CPU/GPU 性能与一致性测试

## 结论

Eos-2.0 Palm Detector 与 Hand Classifier（HCF）在 GPU 上明显更快，且未改变本次回放的发布判断，默认使用 `auto`（CUDA 优先、不可用时回退 CPU）。RTMPose GPU 虽快约 27 倍，但对全部 9,868 条人工复核 Eval 回放的平均关键点误差比 CPU 高 `0.0145 px`，因此按精度优先原则固定使用 CPU。RTMPose/HCF 默认 batch 为 `64`；Palm 模型输入固定为 batch 1。

## 测试环境与模型

| 项目 | 内容 |
|---|---|
| CPU | 2× Intel Xeon Gold 6330，56 核/112 线程 |
| GPU | NVIDIA GeForce RTX 3090 24 GB |
| 驱动 / CUDA / cuDNN | 580.105.08 / 11.2 / 8.1 |
| Python / NumPy / OpenCV / ONNX Runtime | 3.11.15 / 1.26.4 / 4.10.0 / `onnxruntime-gpu==1.18.0` |
| Palm | Eos 2.0，`models/palm_detector/eos-2.0/model_384x224_opt.onnx` |
| RTMPose | RTMPose-m Hand5 256×256，`models/rtmpose/rtmpose-m_hand5_256x256.onnx` |
| HCF | handedness-handpresence-0807，`models/hand_classifier/handedness-handpresence-0807/model.onnx`，模型 ID `hand-classifier-handedness-handpresence-0807` |

测试原因是原 `anfab` 仅安装 CPU 版 `onnxruntime 1.28.0`，provider 列表没有 CUDA；同时 Palm 代码曾显式锁定 CPU，RTMPose/HCF 又逐张推理，导致 CPU 高负载而 3090 利用率低。

## 方法

- 吞吐：固定随机种子生成同一 float32 张量，分别用 CPU/CUDA provider 预热 5 次；Eos-2.0 Palm 重复 100 次，HCF 重复 80 次，RTMPose 重复 20 次，记录纯模型推理耗时中位数。动态模型比较 batch `1/8/32/64`，下表展示典型大量 ROI 场景的最快已测配置。
- Palm 一致性：使用 1 组固定随机输入和 16 张真实 TIFF，比较 TensorFlow、ONNX CPU、ONNX GPU 的四个原始输出以及正式阈值解码结果。
- RTMPose/HCF 精度：只读回放 `FullEnhanceVal0801:eos-1.0` 与 `FullEnhanceVal0808:eos_1.0-gate_r2` 的全部 9,868 条人工复核 hand；RTMPose 比较 21 点与人工点的 crop 像素误差，HCF 比较类别、人工 handedness 准确率及 `0.5/0.7` 门控跨阈值情况。
- 正式数据仓库全程只读；环境与流水线集成另在独立临时 `HAND_DATASET_ROOT` 验证。

## 性能结果

| 模型 | batch | CPU images/s | GPU images/s | GPU 加速比 | 默认设备 |
|---|---:|---:|---:|---:|---|
| Eos-2.0 Palm Detector | 1（模型固定） | 175.0 | 426.2 | 2.43× | GPU，CPU fallback |
| RTMPose | 64 | 64.1 | 1,734.0 | 27.07× | **CPU（精度约束）** |
| HCF | 64 | 567.8 | 8,810.4 | 15.52× | GPU，CPU fallback |

RTMPose GPU 吞吐随 batch `1/8/32/64` 分别为 `317.5/1,363.6/1,684.2/1,734.0 images/s`；HCF 为 `801.4/5,697.3/8,475.5/8,810.4 images/s`，因此默认采用 64。

## 输出与精度结果

| 模型 | 回放结果 | 判定 |
|---|---|---|
| Eos-2.0 Palm | TF↔ONNX raw 最大绝对差 `5.4e-7`，CPU↔GPU 为 `5.6e-7`；score `0.15/0.25` 的 decoded 数量一致，几何/分数最大差 `2.7e-7` | 结构、转换和设备一致性通过，采用 GPU |
| HCF | 9,868 条回放全部 argmax 一致；CPU/GPU handedness accuracy 均为 `96.2406%`；最大概率差 `0.000986`；presence `0.5` 与 handedness `0.7` 均 0 次跨阈值 | 未观察到精度或门控变化，采用 GPU |
| RTMPose | CPU/GPU 平均点误差分别为 `1.4585/1.4729 px`；GPU 增加 `0.0145 px`；207,228 个点中 6,044 个点解码不同 | GPU 有轻微精度下降，保留 CPU |

## 何时重测

更换服务器 GPU/CUDA/ONNX Runtime、更新任一模型，或新增代表性人工复核 Eval 后，应在正式数据只读条件下重新测试。只有当候选设备不降低人工 gold 精度、不改变门控跨阈值结果，并且吞吐更高时，才更新 `configs/autolabel.yaml` 的逐模型 provider 与 batch。
