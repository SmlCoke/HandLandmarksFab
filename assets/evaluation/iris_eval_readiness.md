# Iris Eval 就绪度评估（2026-08-18）

## 结论

当前 7 个已发布来源、9,279 条人工 Gold 是 Eos-2.1/HCF 配置校准及后续 Iris near/mid 评估的固定标准集。它覆盖 val/test、near/mid、complex/white、bright/dark 和 peak/soar，但人员与会话仍集中在 s01，不能据此宣称跨人员、跨会话或 far 泛化。

这些标签由历史 Eos-2.0 变体发布；Eos-2.1 的 Palm、ROI 与连接阈值校准从原图重新检测并投影人工 image-space 点，没有把旧 crop 几何冒充新模型输出。Eos-1.0 只能单独作为 legacy/stress 回放，不能与 Eos-2.1 合并为 headline 指标。

## 固定 Gold 清单

| 角色 | 发布变体 | 数据来源 | Gold ROI | hand / no-hand |
|---|---|---|---:|---:|
| Val | `eos_2.0-rtmpose-hcf0813-gate` | `complex-mid-bright-random-val-s01-peak` | 1,170 | 1,170 / 0 |
| Val | `eos_2.0-rtmpose-hcf0813-gate` | `complex-mid-bright-random-val-s01-soar` | 1,145 | 1,115 / 30 |
| Val | `eos_2.0-rtmpose-hcf0813-gate` | `complex-near-bright-random-val-s01-peak` | 1,169 | 1,169 / 0 |
| Val | `eos_2.0-rtmpose-gate` | `complex-mid-dark-random-val-s01-peak` | 2,302 | 2,302 / 0 |
| Val | `eos_2.0-rtmpose-gate` | `white-mid-bright-random-val-s01-soar` | 1,151 | 1,151 / 0 |
| Test | `eos_2.0-rtmpose-gate` | `complex-mid-bright-random-test-s01-peak` | 1,166 | 1,154 / 12 |
| Test | `eos_2.0-rtmpose-gate` | `complex-near-bright-random-test-s01-peak` | 1,176 | 1,176 / 0 |
| **合计** | 2 个历史发布变体 | 7 个来源 | **9,279** | **9,237 / 42** |

Val 共 6,937 条，Test 共 2,342 条。训练和模型选择只使用 Val 指标；Test 必须冻结到最终 checkpoint 后再评估。每次报告同时给出关键点误差/PCK、presence、handedness，并按来源和距离拆分，不能只给合并平均值。

## 独立性边界

当前 v1 MobileNetV3-Large HCF 的 train/validation split 已覆盖上述全部 7 个来源，其中 5 个进入 train、2 个进入 validation。因此这批 Gold 可以校准 HLMF 的 HCF 门控，却不是该 HCF 的独立盲测，相关结果可能乐观。下一次录制应优先增加未参与训练的新人员/新会话、white-dark near/mid，并冻结为新的 HCF/Iris blind test。

Eos-2.1 在这批 near/mid Gold 上完成 Palm 与 ROI 回放；历史 far 兼容回放召回不足，当前能力契约继续拒绝 far。不得用删除 far 原图、复用 Eos-1.0 far 结果或只报告已检出 ROI 的方式绕过 Palm 能力边界。

## Eos-1.0 legacy 使用边界

推荐单独使用以下不与主 Gold 重复的来源做兼容性回放：

- `FullEnhanceVal0808:eos_1.0-gate_r2` 的 `white-mid-dark-random-val-s03-soar` 与 `white-near-dark-random-val-s03-soar`，补充 white-dark 与 s03；
- 时间允许时，再加入 `FullEnhanceVal0801:eos-1.0` 中不与主集复用原图的 near/mid 来源。

这些 ROI 来自旧 Palm 几何，只能回答“新 Iris 是否退化旧链路兼容性”。结果必须分别标记为 `Eos-2.1 primary` 与 `Eos-1.0 legacy`，并排除所有 far。
