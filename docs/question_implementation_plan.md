# question.md 修改建议实现计划

本文面向后续实现参考，不是用户说明文档。本轮不修改代码，只记录计划、取舍和验证标准。

## 总体目标

围绕 `question.md`，后续代码修改应聚焦四件事：

1. 简化重复输出和 ID。
2. 去掉 CVAT 图片复制，节省磁盘。
3. 调整 `data/` 输出目录结构。
4. 增强 CVAT 负样本语义、可视化和 QC。

不能改变核心标注内容 schema，尤其是：

- bbox、ROI、原图坐标仍属于 `1280x720` upright 图坐标系。
- `hand_presence` 继续只写 `{"present": true/false}`，不写 score。
- ROI 几何继续与板端 `hand_landmarker.cpp` 保持一致。
- 超出图像的 ROI crop 区域继续使用黑色 padding，并参与训练/推理。

## 建议合理性评估

### 建议 4：只保留一份 MediaPipe draft

合理。

当前 `hand_landmarks_mediapipe_raw.jsonl` 与 `hand_landmarks_autolabel_draft.jsonl` 完全相同，属于冗余。建议后续只保留：

```text
hand_landmarks_autolabel_draft.jsonl
```

如果保留 raw，则必须让 raw 成为真正的 MediaPipe 原始输出，不应与 draft 重复。为了简洁，推荐删除 raw 文件输出。

### 建议 5：ID 后缀改为 `:crop` 和 `:hand`

合理。

当前系统是一 palm 一 crop，一 crop 最多一 hand。因此：

```text
crop_id = palm_det_id:crop
hand_id = crop_id:hand
```

比 `:crop0`、`:hand0` 更清晰。

潜在风险：如果未来做 multi-scale ROI augmentation 或一 palm 多 crop，需要恢复 index 后缀。当前任务不需要。

### 建议 6：不复制 CVAT 上传图片

合理。

`cvat_upload_images/` 与 `roi_crops/images/` 图片内容一致，复制浪费空间。CVAT XML 只需要图片文件名匹配，因此可以直接上传 ROI crop 图片目录。

### 建议 7/8：负样本处理

合理，但需要更严格 QC。

后续应明确：

- `palm_valid` 只表示 Palm 来源，不决定训练正负。
- `hand_presence.present` 才决定 Hand Landmarker 训练正负。
- 如果 CVAT XML 中同一 image 同时存在 `no_hand` tag 和 `hand_landmarks` points，应写入 QC conflict。

### 建议 9：调整输出目录

合理，但需修正目录拼写并补齐 `05_labels`。

建议采用：

```text
data/
  images/
  01_palm/
  02_roi_crops/
  03_reviewed/
  04_visualization/
  05_labels/
  qc/
```

`data/review/` 不再作为主输出目录。若考虑兼容旧 README，可在文档中标为 legacy，而不是继续写入。

### 建议 10：解释并增强可视化

合理。

当前已有 global overlay。建议新增 crop overlay：

```text
data/04_visualization/crop_images/
```

用于直接检查 `landmarks_crop_px`。

## 具体实现计划

### Step 1：更新配置结构

修改 `configs/autolabel.yaml`，新增更细粒度输出路径，减少一个 `labels_dir` 承担多个阶段的歧义。

建议配置：

```yaml
paths:
  images_dir: data/images
  palm_model_onnx: materials/preminilary/palm/model_opt.onnx
  palm_outputs_dir: data/01_palm
  roi_crops_dir: data/02_roi_crops
  reviewed_dir: data/03_reviewed
  visualization_dir: data/04_visualization
  labels_dir: data/05_labels
  qc_dir: data/qc
```

为了降低一次性改动风险，可以让 `formats.cfg_path()` 继续工作，并在脚本里按需读取新 key：

- `reviewed_dir` 不存在时 fallback 到旧 `review_dir`。
- `visualization_dir` 不存在时 fallback 到旧 `review_dir/overlay_images`。

但如果希望彻底清爽，也可以直接一次性改配置和脚本，不做旧 key fallback。

### Step 2：调整输出路径

逐脚本修改默认路径：

`01_export_palm_detections.py`

- 输出：`data/01_palm/palm_detections.jsonl`
- QC 不变：`data/qc/palm_detection_stats.json`

`02_build_hand_roi_crops.py`

- 输入：`data/01_palm/palm_detections.jsonl`
- 输出：
  - `data/02_roi_crops/images/*.png`
  - `data/02_roi_crops/hand_roi_crops_manifest.jsonl`

`03_run_mediapipe_on_rois.py`

- 输入：`data/02_roi_crops/images/`
- 输出：
  - 推荐只输出 `data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl`
  - 如果保留 raw，则也放在 `data/02_roi_crops/hand_landmarks_mediapipe_raw.jsonl`

`04_export_cvat_xml.py`

- 输入：
  - `data/02_roi_crops/hand_roi_crops_manifest.jsonl`
  - `data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl`
- 输出：
  - `data/02_roi_crops/cvat_autolabel.xml`
- 删除 `data/review/cvat_upload_images/` 复制逻辑。

`05_import_cvat_xml.py`

- 输入：
  - `data/03_reviewed/cvat_reviewed.xml`
  - `data/02_roi_crops/hand_roi_crops_manifest.jsonl`
  - `data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl`
- 输出：
  - `data/03_reviewed/hand_landmarks_reviewed.jsonl`

`06_visualize_autolabels.py`

- 输入：
  - `data/01_palm/palm_detections.jsonl`
  - `data/02_roi_crops/hand_roi_crops_manifest.jsonl`
  - `data/03_reviewed/hand_landmarks_reviewed.jsonl`
- 输出：
  - `data/04_visualization/global_images/*.png`
  - `data/04_visualization/crop_images/*.png`
  - `data/04_visualization/review_index.csv`

`07A_finalize_training_labels.py`

- 输入：
  - `data/03_reviewed/hand_landmarks_reviewed.jsonl`
  - `data/02_roi_crops/hand_roi_crops_manifest.jsonl`
- 输出：
  - `data/05_labels/hand_training_labels_{stage}.jsonl`

### Step 3：删除 CVAT 图片复制逻辑

修改 `hand_autolabel/cvat_io.py`：

- 删除或停用 `prepare_cvat_upload_images()`。
- `export_cvat_xml()` 不再接收 `upload_dir`，或者保留参数但不复制。
- `cvat_export_stats.json` 改写：

```json
{
  "images": 226,
  "positive_shapes": 178,
  "negative_tags": 48,
  "upload_images_dir": "data/02_roi_crops/images",
  "copied_images": 0,
  "copy_policy": "disabled_use_roi_crops_images_directly"
}
```

修改 `04_export_cvat_xml.py`：

- 不创建 `cvat_upload_images/`。
- 打印提示：直接上传 `data/02_roi_crops/images/`。

修改 README：

- CVAT 上传步骤改为上传 `data/02_roi_crops/images/` 和 `data/02_roi_crops/cvat_autolabel.xml`。

### Step 4：简化 ID 生成

修改 `hand_autolabel/formats.py`：

当前：

```python
def make_crop_id(palm_det_id, index=0):
    return f"{palm_det_id}:crop{index}"
```

改为：

```python
def make_crop_id(palm_det_id):
    return f"{palm_det_id}:crop"

def make_hand_id(crop_id):
    return f"{crop_id}:hand"
```

同步修改：

- `02_build_hand_roi_crops.py`
- `mediapipe_roi_labeler.py`
- `cvat_io.py`
- `07A_finalize_training_labels.py` 中所有 fallback hand_id。

验证：

- `crop_id` 不包含 `crop0`。
- `hand_id` 不包含 `hand0`。
- 文件名安全替换函数仍可处理冒号。

### Step 5：只保留 draft 或让 raw 真正 raw

推荐方案：只保留 draft。

修改 `03_run_mediapipe_on_rois.py`：

- 删除 `raw_path = labels_dir / "hand_landmarks_mediapipe_raw.jsonl"`。
- 只写 `hand_landmarks_autolabel_draft.jsonl`。
- `mediapipe_roi_stats.json` 删除 `raw_jsonl` 或设为 `null`。

修改 README 和 docs：

- 删除 raw 文件作为必需输出的表述。

如果担心 prompt 原始要求，可以折中：

- 默认只写 draft。
- 增加 CLI 参数 `--write-raw` 才输出 raw。

但用户明确希望简洁，优先删除 raw。

### Step 6：增强 CVAT 导入冲突 QC

修改 `hand_autolabel/cvat_io.py` 的 `import_cvat_xml()`：

当前逻辑：

```text
no_hand OR no points -> present=false
```

建议改成：

1. 统计 `no_hand` tag。
2. 统计 `hand_landmarks` points。
3. 决策：

```text
points_count == 1 且 point_num == 21 且 no_hand == false
  -> present=true

points_count == 0 且 no_hand == true
  -> present=false

points_count == 0 且 no_hand == false
  -> present=false + warning missing_no_hand_tag

points_count >= 1 且 no_hand == true
  -> conflict error, needs_review=true
     保守策略：present=false 或跳过，需明确选择
```

推荐保守策略：

- 导入为 `present=false`。
- 清空 landmarks。
- `needs_review=true`。
- QC error: `conflicting_no_hand_and_points`。

这样不会把冲突标注误作为正样本训练。

### Step 7：新增 crop 小图可视化

修改 `hand_autolabel/visualization.py`：

新增函数：

```python
render_crop_overlays(manifest_rows, label_rows, root, crop_output_dir, cfg)
```

逻辑：

1. 读取 `crop_path`。
2. 转 BGR。
3. 如果 `hand_presence.present=true` 且 `landmarks_crop_px` 有 21 点：
   - 画 21 点骨架。
   - 写 handedness/source/palm_score。
4. 如果 false：
   - 写 `no_hand` 和 palm score。
5. 输出到 `data/04_visualization/crop_images/`。

修改 `06_visualize_autolabels.py`：

- `global_images/` 调用原有原图 overlay。
- `crop_images/` 调用新增 crop overlay。

`review_index.csv` 增加列：

```text
crop_overlay_path
global_overlay_path
needs_review
palm_valid
palm_score
```

### Step 8：更新 README

重点更新：

- 新目录结构。
- 新运行顺序。
- `.task` 已放在 `models/mediapipe/hand_landmarker.task` 的相对路径约定。
- CVAT 不再复制上传图片。
- 只保留 draft 文件。
- ID 格式说明。
- 负样本参与训练规则。

### Step 9：清理旧输出

代码改完后，不要自动删除用户数据。提供一个可选脚本或 README 指令：

```powershell
# 手动迁移或清理旧目录
data/palm
data/roi_crops
data/labels
data/review
```

如果需要自动迁移，另写一次性迁移脚本，例如：

```text
scripts/migrate_data_layout.py
```

但默认不建议自动迁移，以免误删人工复核结果。

## 验证计划

### 编译与单元自检

```powershell
python -m compileall hand_autolabel scripts
python -m hand_autolabel.roi_geometry
python -m hand_autolabel.projection
```

### 全流程烟测

```powershell
python scripts/00_validate_images.py --config configs/autolabel.yaml
python scripts/01_export_palm_detections.py --config configs/autolabel.yaml --backend aethersign_onnx
python scripts/02_build_hand_roi_crops.py --config configs/autolabel.yaml
D:\Anaconda\envs\anfab\python.exe scripts/03_run_mediapipe_on_rois.py --config configs/autolabel.yaml
python scripts/04_export_cvat_xml.py --config configs/autolabel.yaml
Copy-Item data/02_roi_crops/cvat_autolabel.xml data/03_reviewed/cvat_reviewed.xml -Force
python scripts/05_import_cvat_xml.py --config configs/autolabel.yaml
python scripts/06_visualize_autolabels.py --config configs/autolabel.yaml
# 正式 Train 使用 configs/finalize_train.yaml，并显式选择 pretrain 或 finetune
python scripts/07A_finalize_training_labels.py --config configs/finalize_train.yaml --stage pretrain
```

### 验证项

- `data/01_palm/palm_detections.jsonl` 存在。
- `data/02_roi_crops/images/*.png` 存在。
- `data/02_roi_crops/hand_roi_crops_manifest.jsonl` 存在。
- `data/02_roi_crops/hand_landmarks_autolabel_draft.jsonl` 存在。
- `data/02_roi_crops/cvat_autolabel.xml` 存在。
- 不存在新生成的 `data/review/cvat_upload_images/`。
- `data/03_reviewed/hand_landmarks_reviewed.jsonl` 存在。
- `data/04_visualization/global_images/*.png` 存在。
- `data/04_visualization/crop_images/*.png` 存在。
- `../autodl-tmp/train_pretrain_merged/05_labels/hand_training_labels_pretrain.jsonl` 存在。
- `data/qc/*.json` 仍按原结构输出。
- JSONL 中不再出现 `:crop0` 和 `:hand0`。
- `hand_presence` 中不出现 `score` 字段。

## 风险与注意事项

- 改目录结构会导致旧数据和 README 中的路径过期。需要一次性全局替换。
- `05_import_cvat_xml.py` 依赖 draft 恢复元数据，删除 raw 可以，但不能删除 draft。
- 删除 CVAT 图片复制后，CVAT XML 中 `<image name>` 必须继续只写 basename，确保上传 ROI images 时能匹配。
- crop ID 改变会让旧 JSONL 和新 JSONL 不兼容。建议改完后清空并重跑流水线，或提供迁移脚本。
- global overlay 上同时画很多 ROI 时会很拥挤，这是数据真实情况。crop overlay 能缓解人工检查压力。
