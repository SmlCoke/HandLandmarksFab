# Iris Eval 就绪度评估（2026-08-13）

## 结论

当前 Gold 足够启动 near/mid 范围内的 HLMF 训练集自动标注和本轮 Iris 训练，但只足以支撑当前比赛/演示边界，不能用于宣称跨背景、跨人员、跨距离的广泛泛化。Eos-2.0 应作为主评测；Eos-1.0 可作为单独的 legacy/stress 回放，不能与 Eos-2.0 合并成一个 headline 指标，也不能包含 far。

## 当前 Eos-2.0 Gold

| 角色 | 数据来源 | Gold ROI | hand / no-hand |
|---|---|---:|---:|
| Dev | `complex-mid-dark-random-val-s01-peak` | 2,302 | 2,302 / 0 |
| Dev | `white-mid-bright-random-val-s01-soar` | 1,151 | 1,151 / 0 |
| Test | `complex-mid-bright-random-test-s01-peak` | 1,166 | 1,154 / 12 |
| Test | `complex-near-bright-random-test-s01-peak` | 1,176 | 1,176 / 0 |
| Frozen Iris test | `complex-mid-bright-random-test-s05-peak` | 396 | 312 / 84 |
| **合计** | 3,200 张原图、5 个来源 | **6,191** | **6,095 / 96** |

建议只用两条 Val 调参与选择 checkpoint；两条 s01 Test 只做主测试；s05 最后只运行一次。s05 未进入 RTMPose/Iris 训练，但曾出现在 HCF 0813 validation，并用于连接长度门控的尺度校准；因此它是 Iris 关键点模型层面的独立测试，不是整条 Palm→Gate→HCF→Iris 链路完全未接触的盲测。

现有覆盖包含 near/mid、complex/white、bright/dark，但缺少 Eos-2.0 的 white-dark、near-dark、更多人员/会话及非 random 动作。若还有极少量标注预算，优先补新人员/新会话的 white-dark near/mid，而不是继续扩充相同 s01 random 条件。

## Eos-1.0 的使用边界

推荐作为独立 legacy/stress 集使用以下不与 Eos-2.0 主评测重复的来源：

- `FullEnhanceVal0808:eos_1.0-gate_r2` 的 `white-mid-dark-random-val-s03-soar` 与 `white-near-dark-random-val-s03-soar`：最小补充集，用于填补 white-dark 与 s03。
- 时间允许时再加入 `FullEnhanceVal0801:eos-1.0` 的 `complex-mid-bright-random-val-s01-peak`、`complex-mid-bright-random-val-s01-soar`、`complex-mid-dark-random-test-s01-peak`、`complex-near-bright-random-val-s01-peak`。

这些 ROI 来自旧 Palm 几何，只能回答“新 Iris 是否退化旧链路兼容性”，不能替代 Eos-2.0 端到端指标。排除所有 far、排除与 Eos-2.0 主集使用同一原图的旧 variant，并分别报告 `Eos-2.0 primary`、`s05 Iris test`、`Eos-1.0 legacy` 三组结果。

## 启动条件

- 可以启动 `DatesetFab/PretrainSource` 的 near/mid HLMF 自动标注；far 继续由 Eos-2.0 距离硬门控拒绝。
- 在训练 Iris 前冻结上述 Dev/Test/Blind 清单，避免后续把 Test 或 s05 用于调参、早停或模型选择。
- 本轮训练后同时给出关键点误差/PCK、presence、handedness，并按来源拆分；不要只给合并平均值。
- 若主测试与 s05 明显分化，以 s05 作为泛化风险信号，不回头用 s05 调参；除非明确将其降级为 Dev 并另建新盲测。
