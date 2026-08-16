# ONNX CPU/GPU 性能与一致性测试

## 结论

Eos-2.1 Palm Detector 与 0814 HCF 在 RTX 3090 上均显著快于 CPU，且人工 Gold 回放未发生分类、正式解码或门控跨阈值变化，因此继续使用 `auto`（CUDA 优先、CPU fallback）。RTMPose 模型未更新，继续沿用上一轮人工 Gold 设备结论并固定 CPU。RTMPose/HCF batch 为 64；Palm 输入固定 batch 1。

## 环境与模型

| 项目 | 内容 |
|---|---|
| 服务器 | 14 vCPU Intel Xeon Gold 6330 / RTX 3090 24 GB |
| 主环境 | Python 3.11、NumPy 1.26.4、OpenCV 4.10.0、`onnxruntime-gpu==1.18.0` |
| Palm | `models/palm_detector/eos-2.1/model_384x224_opt.onnx` |
| RTMPose | `models/rtmpose/rtmpose-m_hand5_256x256.onnx`（未更新） |
| HCF | `models/hand_classifier/handedness-handpresence-0814/model.onnx` |

## 性能

纯模型推理先预热 10 次，再取重复运行中位耗时；Palm 重复 100 次，HCF batch 64 重复 80 次。

| 模型 | batch | CPU images/s | GPU images/s | 加速比 | 默认设备 |
|---|---:|---:|---:|---:|---|
| Eos-2.1 Palm | 1 | 199.3 | 573.2 | 2.88× | GPU，CPU fallback |
| RTMPose | 64 | 64.1 | 1,734.0 | 27.07× | **CPU（既有精度约束）** |
| HCF 0814 | 64 | 333.9 | 8478.6 | 25.39× | GPU，CPU fallback |

## 输出一致性

| 模型 | 回放 | 结果 |
|---|---|---|
| Eos-2.1 | 128 张真实输入；raw 最大差 `7.15e-07` | score 0.25 / NMS 0.10 / max 2 解码 0 次不一致 |
| HCF 0814 | 9,279 条 Gold ROI；logits 最大差 `0.005184`、概率最大差 `0.000707` | presence/handedness argmax 与 0.025/0.5/0.8 门控均 0 次变化 |
| RTMPose | 沿用上一轮 9,868 条人工 Gold 回放 | GPU 平均点误差比 CPU 高 0.0145 px，继续固定 CPU |

正式 `HAND_DATASET_ROOT` 只读，性能与一致性分析使用隔离输出。更换 GPU/CUDA/ONNX Runtime 或任一模型后必须重测；只有在不降低人工 Gold 精度、不改变门控且吞吐更高时才更新 provider/batch。
