# HLMF 当前数据状态

更新时间：2026-07-18。本文只记录会变化的服务器事实；通用方法见 [完整标注流程](HLMF_annotating_workflow.md)，后续操作见 [当前下一步计划](HLMF_next_step_plan.md)。

## 1. 根目录

```text
数据仓库: /root/autodl-tmp/DatesetFab
Gold 真源: /root/autodl-tmp/DatesetFab/GoldSource
Pretrain 真源: /root/autodl-tmp/DatesetFab/PretrainSource
Val/Test 真源: /root/autodl-tmp/DatesetFab/eval_sources
当前派生工作区: /root/autodl-tmp/TrainFab/HLML-3.0
```

GoldSource 使用 `领域/source-id/生命周期目录`。`task/` 只表示等待人工或等待 import；成功发布后 task 自动删除，批次改由 `published/` 表示。`source/` 只保存原始真源，不能与 task 或 published 混为同一概念。

## 2. 当前有效 Gold

| 批次 | 状态 | 内容 |
|---|---|---|
| `disagreement_gold/disagreement_gold_hlml2.0` | published | HLML-2.0 人工精标 300 ROI，原报告 ignored 63 |
| `negative_removed_gold/negative_removed_gold_hlml2.0` | published | HLML-2.0 困难样本人工精标 300 ROI，原报告 ignored 40 |
| `dragon/dragon_gold_0716_v1` | source + published | 5,191 ROI，5,189 trainable、2 ignored；保留但最终 TIFF 域训练建议禁用 |

历史两个来源已在批次 ID 中加入 `_hlml2.0`。领域目录只表示数据性质，子目录 source ID 才表示具体批次；以后同一领域可继续加入 `_r02`、日期或其他唯一批次后缀。

## 3. 已作废批次

`new_recorded_gold_r01` 在人工标注时发现大量 Hand ROI 同时包含双手，不符合单手 Hand Landmarker 的训练输入假设。该批次的 GoldSource source/task、旧 HLML-3.0 CVAT task 和对应 `HandViolenceHard0718/peak` 来源均已删除；没有 published 标签进入训练。此 ID 永久作废，不再复用。

当前没有待完成的 Gold task。下一批从 `new_recorded_gold_r02` 开始。

## 4. 存储状态

历史迁移和 PretrainSource 镜像使用同文件系统硬链接；相同 ROI 即使从不同语义入口可见，也不重复占用图片数据块。新的 HLMF import 会在发布成功后删除 task，进一步减少重复目录项。

Dragon 例外地长期保留 `source + published`：source 是 Dragon 原始整图与标注，published 是 HLMF 生成并认证的 Hand ROI，两者不可互换。

## 5. 当前训练状态

HLML `v3-pretrain-r1` 的正式 geometry 仍在运行，继续读取原冻结路径。GoldSource 整理不改 geometry 输入、run 目录或权重文件。
