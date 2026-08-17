# Hand Classifier v1 MobileNetV3-Large 接入与校准

## 结论

`v1-mobilenet_v3_large` 已替换 0814，作为 HLMF 默认双头 Hand Classifier。HLMF 路径为 `models/hand_classifier/v1-mobilenet_v3_large/model.onnx`，HaMeR 外部副本为 `/root/autodl-tmp/HLMF-Enhance/hand_classifier/v1-mobilenet_v3_large/model.onnx`，模型 ID 为 `hand-classifier-v1-mobilenet_v3_large`。

RTMPose/HaMeR Train presence 阈值从 `0.025` 调整为 `0.5`；negative-candidate review 阈值独立保持 `0.5`；handedness quality 阈值保持 `0.8`。三个用途继续使用三个独立配置结论，不因两个 presence 阈值数值相同而合并。HCF 不改变 Palm、ROI 或 landmark geometry，因此 boundary `2` 与 near/mid connection-length thresholds 不重算。

## 模型 provenance 与接口

| 项目 | 结果 |
|---|---|
| 最终 ONNX bytes | `16,837,557` |
| SHA-256 | `36deea0520a0bba13ce6557906fd94ffd4a65cc290c1a9d9440857f1f830847b` |
| ONNX full checker | 通过 |
| 图 | 162 nodes、130 initializers、4,194,668 parameter elements |
| 输入 | `input [N,1,256,256]` float |
| 输出 | `handedness [N,2]`、`hand_presence [N,2]` float |

HLMF 与 HaMeR 部署副本的最终 ONNX 哈希一致。HaMeR 仓库提交 `b29f1b397ed5ef36eba8f9498dd719949615fe09` 将独立入口的默认路径、`(gray/255-0.485)/0.229` 预处理和 handedness `0.8` 阈值同步到当前模型；同一张 ROI 在 HLMF 与 HaMeR 入口上的 `has_hand/right` 概率差均小于 `1e-6`。

模型自带 validation 指标为：

| 指标 | 结果 |
|---|---:|
| Presence accuracy / ROC AUC | `0.999299 / 0.999964` |
| Presence confusion | `[[1204,2],[1,3073]]` |
| Handedness accuracy / ROC AUC | `0.994470 / 0.999723` |
| Handedness confusion | `[[1589,10],[7,1468]]` |

## 校准数据边界

只读使用 prompt 指定的 7 个 `FullEnhanceVal0801` 已发布人工 Gold 来源，共 9,279 条 ROI：9,237 hand、42 no_hand。三个 `eos_2.0-rtmpose-hcf0813-gate` 来源贡献 3,484 条，四个 `eos_2.0-rtmpose-gate` 来源贡献 5,795 条。

新模型 split 已覆盖全部 7 个来源，其中 5 个进入 train、2 个进入 validation。因此本报告用于模型版本特定的配置校准和兼容回放，不是独立盲测；结果可能乐观，后续仍需补充未参与 HCF 训练的代表性新 Gold。

## Train hand-presence 阈值

| 阈值 | hand 保留 | no-hand 拒绝 |
|---:|---:|---:|
| 0.025 | 9,237/9,237 | 42/42 |
| 0.5 | 9,237/9,237 | 42/42 |
| 0.8 | 9,237/9,237 | 42/42 |
| 0.9 | 9,237/9,237 | 42/42 |
| 0.95 | 9,234/9,237 | 42/42 |

Gold hand 的最小 `P(has_hand)` 为 `0.914486`，no_hand 的最大值为 `2.38e-09`。虽然 0.9 在这批数据上仍全通过，但校准集不是独立盲测；采用自然二分类边界 `0.5`，既严格禁止模型判为 no_hand 的 runtime ROI，也避免依据非盲回放把质量门限推高到 0.9。

## Handedness quality 阈值

raw handedness accuracy 为 `98.993%`。

| 阈值 | 覆盖 | 覆盖率 | 覆盖内准确率 | 覆盖内错误 |
|---:|---:|---:|---:|---:|
| 0.5 | 9,237 | 100.000% | 98.993% | 93 |
| 0.7 | 9,152 | 99.080% | 99.301% | 64 |
| 0.8 | 9,101 | 98.528% | 99.451% | 50 |
| 0.9 | 9,022 | 97.672% | 99.612% | 35 |
| 0.95 | 8,935 | 96.731% | 99.709% | 26 |

0.8 继续作为覆盖/质量折点。相对 0.7，它额外过滤 51 条并移除其中 14 个错误；提高到 0.9 还会过滤 79 条，仅再移除 15 个错误。逐来源 0.8 覆盖率最低为 `95.048%`，覆盖内准确率最低为 `98.455%`。

## Negative-candidate review 阈值

使用当前 Eos-2.1 在 4,777 张受审图像上重新产生 score `[0.15,0.25)`、每图最多 10 个的 37,937 个 raw Anchor candidate。36,460 个与 Gold hand 几何关联，1,477 个未关联；关联只是重复 Anchor/背景诊断，不是人工 negative 真值。

| `P(has_hand)<threshold` | 送审总数 | 未关联 | 与 hand 关联 |
|---:|---:|---:|---:|
| 0.25 | 546 | 543 | 3 |
| 0.5 | 553 | 547 | 6 |
| 0.75 | 555 | 549 | 6 |
| 0.8 | 562 | 552 | 10 |
| 0.9 | 571 | 555 | 16 |

0.5 保持自然 presence argmax 分界；提高到 0.75 只增加 2 个未关联候选，收益不足以采用非自然门限。该阈值只控制人工预审工作集，候选仍必须人工删除有手和不确定图片后才能发布。

## CPU/GPU 一致性与吞吐

| 项目 | 结果 |
|---|---:|
| CPU / GPU batch-64 吞吐 | `207.4 / 4,265.5 images/s` |
| GPU 加速比 | `20.56×` |
| 9,279 ROI logits / probability 最大差 | `0.009796 / 0.000623` |
| presence / handedness argmax 变化 | `0 / 0` |
| presence 0.5 / handedness 0.8 跨阈值 | `0 / 0` |

handedness 0.7 有 1 条 CPU/GPU 跨阈值，采用的 0.8 没有变化。HCF 继续使用 `auto`（CUDA 优先、CPU fallback），batch 保持 64。

## 数据与依赖影响

正式 `HAND_DATASET_ROOT` 在分析期间只读，校准 JSON 写到 `/root/autodl-tmp` 隔离位置；没有运行正式自动标注、发布或 manifest 重建。ONNX 输入输出契约不变，HLMF 依赖未变化，不需要更新 `requirements.txt` 或 `anfab`。模型资产按仓库策略不进入 Git；本地、服务器和板端需要单独部署同一最终 ONNX。
