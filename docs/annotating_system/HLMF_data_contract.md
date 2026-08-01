# HLMF 3.0 数据契约

## 长期目录

```text
HAND_DATASET_ROOT/
  PretrainSource/<dataset_id>/<capture_source_id>/
  EValSource/<dataset_id>/<capture_source_id>/
  GoldSource/NegativeSamples/<negative_dataset_id>/published/
  Selections/<selection_id>/published/
  Registry/registry.sqlite3
```

每个来源包含平铺 `images/`，派生产物按 proposal 变体隔离：

```text
01_palm/<proposal_variant>/
02_roi_crops/<proposal_variant>/
03_reviewed/<proposal_variant>/
05_labels/<proposal_variant>/
qc/<proposal_variant>/
```

## 稳定身份

- `capture_source_id`：`<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>`。
- `raw_image_id`：首次验证时生成并持久化；改用另一 Palm 模型仍共享该 ID。
- `roi_id`：由 `raw_image_id + proposal_variant + proposal kind/slot + ROI contract version` 确定；同变体重跑稳定，不同变体互不覆盖。
- SQLite 对 capture source、raw、ROI、negative dataset 和 selection 执行唯一性约束。

## 标签字段

每条发布记录至少包含：schema、dataset/source/split、`raw_image_id`、`roi_id`、`proposal_variant`、proposal slot/kind、仓库相对 crop 路径、Palm score、presence、handedness、21 点、`label_origin`、`annotation_style`、`human_reviewed` 和 `human_modified_landmark_ids`。

Val/Test 的 `no_hand` 是固定 ROI 上的真值 negative，不是 Palm 漏检；`ignore_for_training` 不进入训练或评估标签。

## 发布契约

- Train：positive 与 candidate negative 分文件；candidate negative 未经删除式审核不得训练。
- Val/Test：只发布实际生成且经过 CVAT 决策的固定 ROI，不保留 candidate negative。
- 真负样本：published 图片必须与审核树中保留文件一一对应，并带 manifest 与审核报告。服务器内直接复核时使用硬链接节省空间；审核树经过压缩包/网盘/本地复核后可以是普通文件，允许发布这一份人工确认后的图片副本。
- 困难正样本：selection 只保存零拷贝引用，不生成图片副本。

## 完整性策略

常规验证使用稳定 ID、SQLite、文件大小、尺寸与 TIFF/ROI 解码。图片本来就被读取时缓存像素 CRC32 和 dHash64。仅疑似冲突项逐字节比较；流水线不对图片重复计算 SHA-256。

HLMF 3.0 不迁移或删除旧 schema 数据，HLML 4.0 只选择新 manifest。
