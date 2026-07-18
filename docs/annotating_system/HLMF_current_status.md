# HLMF 当前数据状态

更新时间：2026-07-18。本文只记录会变化的服务器事实；通用方法见 [完整标注流程](HLMF_annotating_workflow.md)。

## 1. 根目录

```text
数据仓库: /root/autodl-tmp/DatesetFab
Gold 真源: /root/autodl-tmp/DatesetFab/GoldSource
Pretrain 真源: /root/autodl-tmp/DatesetFab/PretrainSource
Val/Test 真源: /root/autodl-tmp/DatesetFab/eval_sources
当前派生工作区: /root/autodl-tmp/TrainFab/HLML-3.0
```

GoldSource 使用 `领域/source-id/{source,task,published}`。训练版本不再拥有 Gold 真源。

## 2. 已归档 Gold

- `disagreement_gold/disagreement_gold`：HLML-2.0 人工任务 300 ROI，published 标签 300，原报告 ignored 63。
- `negative_removed_gold/negative_removed_gold`：HLML-2.0 人工任务 300 ROI，published 标签 300，原报告 ignored 40。
- `dragon/dragon_gold_0716_v1`：5,191 ROI，5,189 trainable、2 ignored；数据保留但当前不建议训练使用。
- `new_recorded_gold/new_recorded_gold_r01`：无损 TIFF 图片流来源，任务 300 ROI；CVAT 人工标注进行中，尚未 published。

迁移使用同文件系统硬链接归档，源文件内容 SHA256 不变；HLML-3.0 现有目录未移动或删除。

## 3. 当前人工工作

`new_recorded_gold_r01` 的 `reviewed.xml` 尚待返回。团队还可以建立多个新的 `new_recorded_gold_r*` 批次，但所有当前人工任务合计应控制在可完成范围内。

## 4. Pretrain 目录迁移状态

规范 `PretrainSource` 已建立。由于当前 geometry 正在从原路径持续读取 ROI，旧 Pretrain 源目录暂不删除；规范目录可使用硬链接镜像建立，不改变当前训练。geometry 结束后再只做目录入口清理，不修改标签或权重文件。
