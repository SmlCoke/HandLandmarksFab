# Hand Classifier 0813 接入与校准

## 结论

`handedness-handpresence-0813` 已通过接口、Gold 回放及设备一致性检查，可作为 HLMF 默认 HCF。默认路径改为 `models/hand_classifier/handedness-handpresence-0813/model.onnx`，模型 ID 为 `hand-classifier-handedness-handpresence-0813`。Train presence `0.025`、负样本预审核 `0.5`、handedness review `0.7` 均保持不变；HCF 继续使用 `auto`（CUDA 优先、CPU fallback）。

## 模型与自带指标

0813 在 Eos-2.0 与历史 Eos-1.0 数据上混合训练。ONNX full checker 通过；模型有 142 个节点、110 个 initializer、1,515,612 个参数元素，与 0809 的拓扑和 initializer schema 相同。输入为动态 batch `input [N,1,256,256]`，输出为 `handedness/hand_presence [N,2]`。

| 0813 自带 validation 指标 | 结果 |
|---|---:|
| Presence accuracy / ROC AUC | 0.960505 / 0.992157 |
| Presence confusion matrix | `[[355,54],[43,2004]]` |
| Handedness accuracy / ROC AUC | 0.963036 / 0.992225 |
| Handedness confusion matrix | `[[1118,33],[42,836]]` |

自带指标不能与 0809 直接横比：0813 validation 包含新的 Eos-2.0/s05 及更多 no-hand 数据，分布更难。是否接入以统一人工 Gold 回放为准。

## 只读回放

正式 `HAND_DATASET_ROOT` 只读。共回放 10,773 条 near/mid 人工 Gold ROI，其中 10,675 hand、98 no-hand：

| 数据 | 来源 | Gold ROI |
|---|---:|---:|
| FullEnhanceVal0801 / Eos-2.0 | 4 | 5,795 |
| RTMPose-Finetune-Test-0812 / Eos-2.0 s05 | 1 | 396 |
| FullEnhanceVal0801 / Eos-1.0 非重叠来源 | 4 | 3,416 |
| FullEnhanceVal0808 / Eos-1.0 非 far 来源 | 2 | 1,166 |

| 检查项 | 0809 | 0813 | 采用 |
|---|---:|---:|---|
| Presence：Train `0.025` hand 保留率 | 98.511% | 99.944% | `0.025` |
| Presence：`0.5` hand 保留率 | 95.026% | 99.550% | 仅负候选预审核 |
| Presence：`0.5` no-hand 拒绝率 | 97.959% | 96.939% | 人工复核兜底 |
| Handedness raw accuracy | 92.612% | 98.160% | 0813 |
| Handedness `0.7` 覆盖内 accuracy | 93.992% | 98.640% | `0.7` |

0813 在 Train `0.025` 下保留 10,669/10,675 条 hand，并拒绝 88/98 条 no-hand。独立 s05 的 hand 保留率为 98.077%；若 Train 改用 `0.5`，该来源只保留 86.538%，因此不能把负候选预审核阈值用于 Train 门控。handedness `0.7` 覆盖 98.733% 的有效人工标签，覆盖内准确率 98.640%；提高阈值会把较多正确样本送入 ignored，当前无必要。

## 设备结果

服务器 RTX 3090、`onnxruntime-gpu==1.18.0`、batch 64：CPU 324.9 images/s，GPU 8,443.3 images/s，GPU 为 25.99×。10,773 条回放的最大概率差为 `0.000786`，presence argmax、handedness argmax，以及 `0.025/0.5/0.7` 三个门控均为 0 次设备间变化。因此默认保留 GPU 优先。

模型资产按仓库策略不进入 Git；本地、服务器和板端应单独部署同一 0813 ONNX。每次 HCF 重训都必须重新执行上述阈值、准确率、设备一致性与吞吐检查，不能继承本报告结论。
