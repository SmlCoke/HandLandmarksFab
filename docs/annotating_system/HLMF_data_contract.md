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
  images/<roi_id>.png
  hand_roi_crops_manifest.jsonl
  hand_landmarks_autolabel_draft.jsonl
  hand_landmarks_roi_visualization/<roi_id>.png  # 可选 Hand ROI 审核图
03_reviewed/<proposal_variant>/
05_labels/<proposal_variant>/
qc/<proposal_variant>/
visualizations/original_image_landmarks/<proposal_variant>/
  <original_image_stem>.png
```

`hand_landmarks_roi_visualization/` 是可删除、可重建的 Hand ROI 自动标注审核派生物，不是训练或评估输入。启用时 Train 只包含按 manifest 顺序等距抽取的最多 `visualization.train_max_samples` 张 ROI；Val/Test 包含该来源的全部实际 ROI。对应的 resolved 开关、触发方式、选择策略、可用/选择/保存数量和输出路径记录在 `qc/<proposal_variant>/roi_visualization_report.json`。已有 draft 可通过 `make autolabel-visualize-roi ...` 重建该目录；该操作只读取 ROI `images/` 和 `hand_landmarks_autolabel_draft.jsonl`。CVAT 导出与导入不写入该目录。

`visualizations/original_image_landmarks/<proposal_variant>/` 是按变体隔离、可删除重建的原图审核派生物。每个变体目录必须为来源平铺 `images/` 中的每一张 TIFF 输出一张同 stem PNG，例如 `frame.tiff → frame.png`；同一原图关联的所有 positive ROI 使用 draft 的 `landmarks_image_px` 一并绘制，没有 positive 时输出 `hands=0`。同一来源不得存在会映射到相同 PNG 名称的 `.tif/.tiff` stem。PNG 压缩级别固定为 3，并在报告中记录 `output_format=png`、`png_compression=3`。该目录不进入训练、评估或 CVAT。全局开关为 `visualization.original_image_enabled`，Train/Eval 单次覆盖为 `ORIGINAL_VISUALIZATION=true|false`，已有 draft 可用 `make autolabel-visualize-original ...` 重建；执行记录写入 `qc/<proposal_variant>/original_image_visualization_report.json`。

## 模型版本契约

Palm Detector 的产品名为 **Eos**。当前冻结模型为 `eos-1.0`，仓库相对路径是 `models/palm_detector/eos-1.0/model_opt.onnx`，对应的 `proposal_variant` 为 `eos-1.0`。后续模型文件放入 `models/palm_detector/eos-*/model_opt.onnx`；更换 Eos 版本或修改会改变 proposal/ROI 的参数时，必须使用新的唯一 `proposal_variant`，不得覆盖已有派生产物。

## 稳定身份

- `capture_source_id`：`<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>`。
- `raw_image_id`：首次验证时生成并持久化；改用另一 Palm 模型仍共享该 ID。
- `roi_id`：由 `raw_image_id + proposal_variant + proposal kind/slot + ROI contract version` 确定；同变体重跑稳定，不同变体互不覆盖。
- SQLite 对 capture source、raw、ROI、negative dataset 和 selection 执行唯一性约束。

## 标签字段

每条发布记录至少包含：schema、dataset/source/split、`raw_image_id`、`roi_id`、`proposal_variant`、proposal slot/kind、仓库相对 crop 路径、Palm score、presence、handedness、21 点、`label_origin`、`annotation_style`、`human_reviewed` 和 `human_modified_landmark_ids`。

Val/Test 的 `no_hand` 是固定 ROI 上的真值 negative，不是 Palm 漏检；`ignore_for_training` 不进入训练或评估标签。

## CVAT frame 对齐契约

`cvat_autolabel.xml` 的 `<image id>` 从 0 开始，严格按照 `02_roi_crops/<proposal_variant>/images/` 的 crop 文件名字典序排列；CVAT Images 任务必须使用 `Lexicographical` Sorting method。每个 manifest `crop_id` 必须恰好对应一条 draft，positive 必须完整包含 landmark ID `0..20`，并按 ID 映射为 CVAT skeleton 子点 `1..21`。`cvat_export_report.json` 记录 `image_order=crop_filename_lexicographic` 和相对 manifest 输入顺序发生重排的数量。

## 发布契约

- Train：positive 与 candidate negative 分文件；candidate negative 未经删除式审核不得训练。
- Val/Test：只发布实际生成且经过 CVAT 决策的固定 ROI，不保留 candidate negative。
- 真负样本：published 图片必须与审核树中保留文件一一对应，并带 manifest 与审核报告。服务器内直接复核时使用硬链接节省空间；审核树经过压缩包/网盘/本地复核后可以是普通文件，允许发布这一份人工确认后的图片副本。
- 困难正样本：selection 只保存零拷贝引用，不生成图片副本。

## 完整性策略

常规验证使用稳定 ID、SQLite、文件大小、尺寸与 TIFF/ROI 解码。图片本来就被读取时缓存像素 CRC32 和 dHash64。仅疑似冲突项逐字节比较；流水线不对图片重复计算 SHA-256。

HLMF 3.0 不迁移或删除旧 schema 数据，HLML 4.0 只选择新 manifest。
