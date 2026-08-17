# Hand Classifier 0814 接入与校准

## 结论（历史）

本报告保留 0814 模型当时的接入与校准结果。自 2026-08-18 起，默认 HCF 已由 `v1-mobilenet_v3_large` 取代；当前路径、阈值和设备结论见 `assets/hand_classifier/v1_mobilenet_v3_large.md`，不得从本报告继承。

`handedness-handpresence-0814` 当时已通过 ONNX 接口、人工 Gold 回放、Eos-2.1 重建 ROI 回放和设备一致性检查，并曾作为 HLMF 默认 HCF。模型路径为 `models/hand_classifier/handedness-handpresence-0814/model.onnx`，模型 ID 为 `hand-classifier-handedness-handpresence-0814`。

0814 当时的 RTMPose Train presence 阈值为 `0.025`；负候选人工预审独立为 `0.5`；handedness review 阈值为 `0.8`。这些都是 0814 专属历史结论。

## 模型与自带指标

ONNX full checker 通过；142 个节点、110 个 initializer、1,515,612 个参数元素，initializer schema 与 0813 相同。接口为动态 batch `input [N,1,256,256]`，输出 `handedness/hand_presence [N,2]`。

| 0814 自带 validation 指标 | 结果 |
|---|---:|
| Presence accuracy / ROC AUC | 0.984676 / 0.997041 |
| Presence confusion matrix | `[[369, 40], [7, 2651]]` |
| Handedness accuracy / ROC AUC | 0.971783 / 0.996871 |
| Handedness confusion matrix | `[[1345, 54], [21, 1238]]` |

0814 的 train/validation split 覆盖了本次 7 个 Gold 来源：5 个进入 train，2 个进入 validation。因此下面的统一 Gold 回放是配置校准证据，不是独立盲测，结果可能乐观；新 HCF 版本仍需另建未参与训练的人工 Gold。

## 正式发布 Gold ROI 回放

| 数据 | ROI | hand / no-hand |
|---|---:|---:|
| `eos_2.0-rtmpose-hcf0813-gate`（3 来源） | 3,484 | 3,454 / 30 |
| `eos_2.0-rtmpose-gate`（4 来源） | 5,795 | 5,783 / 12 |
| **合计** | **9,279** | **9,237 / 42** |

### Presence

| 阈值 | hand 保留 | no-hand 拒绝 |
|---:|---:|---:|
| 0.01 | 9,237/9,237 (100.000%) | 41/42 (97.619%) |
| 0.025 | 9,237/9,237 (100.000%) | 41/42 (97.619%) |
| 0.1 | 9,237/9,237 (100.000%) | 41/42 (97.619%) |
| 0.5 | 9,233/9,237 (99.957%) | 42/42 (100.000%) |
| 0.7 | 9,230/9,237 (99.924%) | 42/42 (100.000%) |
| 0.9 | 9,222/9,237 (99.838%) | 42/42 (100.000%) |

0.01–0.10 的结果完全相同；提高到 0.5 才多拒绝最后 1 个 no-hand，但开始漏掉 4 只手。0.025 在发布 Gold 与 9,237 个 Eos-2.1 重建 hand ROI 上均保留全部 hand，因此保持 0.025，不与负候选预审 0.5 合并。

### Handedness

raw accuracy 为 `98.679%`。

| 阈值 | 覆盖 | 覆盖率 | 覆盖内正确 | 覆盖内准确率 | 错误 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 9,237/9,237 | 100.000% | 9,115 | 98.679% | 122 |
| 0.7 | 9,170/9,237 | 99.275% | 9,083 | 99.051% | 87 |
| 0.8 | 9,099/9,237 | 98.506% | 9,033 | 99.275% | 66 |
| 0.9 | 8,943/9,237 | 96.817% | 8,902 | 99.542% | 41 |
| 0.95 | 8,736/9,237 | 94.576% | 8,707 | 99.668% | 29 |

0.8 是本次覆盖/质量折点：相对 0.7 额外忽略 71 条，其中消除 21 个错误；仍覆盖 98.506%，覆盖内准确率提高到 99.275%。提高到 0.9 会再忽略 156 条，只再消除 25 个错误。逐来源 0.8 覆盖率均不低于 97.027%，因此采用 0.8。

## Eos-2.1 重建 ROI 正样本回放

该回放用 Eos-2.1 detection 构造 ROI，再投影同一人工 Gold；它包含 9,237 个匹配 hand，没有可靠的重建 no-hand。

| 检查 | 结果 |
|---|---:|
| Presence 0.025 hand 保留 | 9,237/9,237 (100.000%) |
| Presence 0.5 hand 保留 | 9,226/9,237 (99.881%) |
| Handedness raw accuracy | 98.051% |
| Handedness 0.8 coverage / accuracy | 97.813% / 98.783% |

重建 ROI 再次证明 Train presence 不应提高到 0.5；handedness 0.8 仍位于 0.7 与 0.9 的覆盖/质量中间点。

## 负候选预审回放

Eos-2.1 在 0.15–0.25 之间每图最多保留 10 个原始 Anchor candidate，共 37,937 个；36,460 个与 Gold hand 几何关联，说明大部分是同一手的低分重复 Anchor。HCF 0.5 预审选择 584 个供人工复核，其中 543 个未关联、41 个与 hand 关联；提高到 0.75 仅多选 33 个未关联候选，却多带入 21 个 hand 关联候选。0.5 仍是明确的 presence argmax 分界和人工兜底折点。

候选关联不是独立人工 negative 真值，因此不能把 543/1,477 解释为正式负样本召回；负候选发布仍必须人工删除有手和不确定图片。

## 设备一致性与吞吐

| 项目 | 结果 |
|---|---:|
| CPU / GPU batch-64 吞吐 | 333.9 / 8478.6 images/s |
| GPU 加速比 | 25.39× |
| 9,279 ROI 最大 logits / probability 差 | 0.005184 / 0.000707 |
| presence/handedness argmax 跨设备变化 | 0 / 0 |
| presence 0.025、0.5；handedness 0.8 跨阈值 | 0 / 0 / 0 |

模型资产不进入 Git；服务器已部署 0814 ONNX。本地和其他执行环境需单独部署同一版本。每次 HCF 重训都必须重新执行接口、Gold、负候选、设备一致性与吞吐检查。
