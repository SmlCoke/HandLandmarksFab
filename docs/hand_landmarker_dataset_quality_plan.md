# Hand Landmarker 验证/测试集构建与 7 万伪标注训练集自动质量分级方案

> 适用项目：AetherSign / PeakDragonSoar 端侧 Hand Landmarker 重训练  
> 适用阶段：使用 Google MediaPipe Hand Landmarker 生成伪标签，对约 7 万个 ROI 进行第一阶段基座训练  
> 不包含：后续少量人工精标数据的微调方案  
> 文档目标：在不全量人工复核训练集的前提下，提高伪标签训练基座的可靠性，并建立可信、规模可控的验证集与测试集

---

## 0. 核心结论

本项目建议采用以下总路线：

1. **训练集约 7 万个 ROI 不做逐张人工复核**，但不能将原始 `hand_training_labels.jsonl` 中的全部样本直接等权训练。
2. 新增一个自动质量分析阶段，例如：

   ```text
   08_score_and_filter_training_samples.py
   ```

   该脚本读取：

   ```text
   data/05_labels/hand_training_labels.jsonl
   data/02_roi_crops/images/*.png
   原始 1280x720 图片（推荐，用于全图交叉验证）
   ```

   输出一份**增强后的训练标注文件**：

   ```text
   data/05_labels/hand_training_labels_scored.jsonl
   ```

3. 自动质量分析不能只依赖 `palm_score`、`needs_review` 和单次 MediaPipe 输出。必须同时使用：

   - JSON 结构与坐标合法性；
   - MediaPipe 多变换重复推理的一致性；
   - 骨架几何异常检测；
   - 图像极端异常检测；
   - 同一原图内重复 ROI/重复手实例聚类；
   - Palm 分数与 Hand Landmarker 结果之间的冲突关系。

4. 第一阶段训练只使用：

   - 高质量正样本；
   - 中质量正样本，降低 landmark loss 权重；
   - 高置信负样本；
   - 少量经自动交叉确认的 hard negative。

   其余样本保留在文件中，但设置为不参与当前基座训练。

5. 新录制的 Val/Test 各约 1000 张原图非常适合独立构建验证集和测试集。**主验证集/测试集应只包含板端运行时真正会送给 Hand Landmarker 的 Palm detections，不应包含海量低分 `negative_candidates`。**

6. 当前 YAML 中：

   ```yaml
   max_detections: 2
   keep_low_score_candidates_for_negatives: true
   negative_candidate_threshold: 0.15
   ```

   `max_detections` 通常只限制有效 detections，不限制所有低分负候选。低阈值负候选是每张原图产生 8～10 个 ROI 的主要原因。对于主 Val/Test，应关闭该功能。

---

# 第一部分：验证集与测试集构建方案

## 1. Val/Test 的目标必须先区分清楚

建议将评测数据分成三种不同用途，不能混在同一个指标里。

### 1.1 主验证集和主测试集：板端运行分布

用途：

- 选择训练 epoch；
- 调整 Hand Landmarker 训练超参数；
- 对比 FP32、量化仿真、板端 INT8 精度；
- 衡量完整 Palm Detector → ROI → Hand Landmarker 链路中，Hand Landmarker 实际接收到的数据分布。

它必须满足：

- Palm 阈值与最终板端配置一致；
- NMS、跨 head 抑制、`max_detections` 与板端一致；
- ROI 构造参数与板端一致；
- 不导出运行时不会送给 Hand Landmarker 的低分候选。

这是正式报告结果时应使用的主数据集。

### 1.2 Teacher-clean 自动诊断集：可选

用途：

- 快速检查学生模型是否基本学会 MediaPipe 教师；
- 在尚未完成人工复核时做早期 smoke test；
- 验证训练代码和模型结构是否能收敛。

该集合可以使用更严格的 Palm 和 MediaPipe 阈值，追求高精度自动伪标签，但**不能替代人工复核后的主测试集**。

### 1.3 Hard-negative 压力集：可选且单独统计

用途：

- 测试 hand presence head 是否会把疑似手掌纹理误判为手；
- 测试 Palm Detector 误检传入 Hand Landmarker 后，后级是否能拒绝。

该集合专门从临近 Palm 阈值的低分候选中抽取，不能与主测试集混在一起计算总体准确率，否则大量负样本会掩盖 landmark 误差。

---

## 2. 主 Val/Test 推荐 YAML

建议分别复制为：

```text
configs/autolabel_val_runtime.yaml
configs/autolabel_test_runtime.yaml
```

除路径外，两者配置完全一致。

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

image:
  width: 1280
  height: 720
  channels: 1
  orientation: upright

palm:
  backend: aethersign_onnx
  input_size: 224

  # 必须与最终板端使用的 Palm threshold 一致。
  # 当前板端若使用 0.50，主 Val/Test 就保持 0.50。
  score_threshold: 0.50

  # 保持与板端后处理一致，不为减少数据量而单独修改。
  nms_iou_threshold: 0.30
  cross_head_suppress_iou: 0.35
  max_detections: 2

  # 主 Val/Test 的关键修改：关闭低分负候选导出。
  keep_low_score_candidates_for_negatives: false

  # 关闭低分候选后不再实际生效，保留字段即可。
  negative_candidate_threshold: 0.15

  compatible_bbox_expand: 0.25
  mediapipe_official_full_image_first: false
  mediapipe_official_tile_sizes: [512]
  mediapipe_official_tile_overlap: 0.5

hand_roi:
  output_width: 256
  output_height: 256

  # 必须与训练集和板端完全一致。
  scale_x: 1.8
  scale_y: 1.8
  shift_x: 0.0
  shift_y: -0.1

mediapipe:
  model_asset_path: models/mediapipe/hand_landmarker.task
  num_hands: 1

  # 用于生成初始自动标注，不建议在主评测集上为了“纯化”而提高。
  # 最终标签应由人工复核确认。
  min_hand_detection_confidence: 0.5
  min_hand_presence_confidence: 0.5
  min_tracking_confidence: 0.5
```

### 2.1 为什么主 Val/Test 不建议直接把阈值提高到 0.7

提高 Palm threshold 确实会：

- 减少 ROI 数量；
- 提高保留下来的 Palm detections 的平均置信度；
- 降低人工复核工作量。

但同时也会系统性删除：

- 模糊手；
- 边缘手；
- 暗光手；
- 特殊角度；
- 遮挡手；
- Palm Detector 较难、但板端实际可能遇到的样本。

结果是测试集只剩容易样本，测试精度虚高。

因此，正式主 Val/Test 的正确原则是：

> **阈值与部署一致，通过关闭低分候选和去重控制规模，而不是通过提高部署阈值美化测试集。**

---

## 3. 预计 ROI 数量

主配置满足：

```yaml
max_detections: 2
keep_low_score_candidates_for_negatives: false
```

因此每张原图最多产生 2 个主评测 ROI。

对于每个 split 约 1000 张原图：

```text
理论上限：约 2000 ROI
```

实际数量通常低于上限，取决于：

- 图片中是否存在手；
- 是否有双手；
- Palm Detector 是否超过阈值；
- NMS 和跨 head 抑制结果。

合理预期通常是数百到约 1500～2000 个 ROI，而不是 8000～10000 个。

这个规模已经足够作为 Hand Landmarker 的 Val/Test，并且比 8000 个高度重复、包含大量低分候选的 ROI 更可信。

---

## 4. Val/Test 是否需要人工复核

### 4.1 主测试集必须人工复核

训练集可以使用伪标签，但测试集如果仍完全使用 Google MediaPipe 的自动输出作为标签，只能回答：

> 学生模型与 Google MediaPipe 输出有多相似？

不能可靠回答：

> 学生模型的关键点是否真的正确？

因此建议：

- 主 Val：人工复核所有保留 ROI；
- 主 Test：人工复核所有保留 ROI；
- 严重遮挡、无法定义可靠 2D 关键点的样本标记 `ignore_for_training` 或评测 ignore；
- 无手 ROI 标记 `no_hand`；
- 有手 ROI 修正 presence、handedness 和 21 点。

因为关闭低分候选后每个 split 最多约 2000 ROI，工作量远小于原计划的 8000～10000 ROI。

### 4.2 时间仍然不足时的最低方案

优先级如下：

1. Test 全量人工复核；
2. Val 全量复核 presence，关键点对可疑样本重点复核；
3. Teacher-clean 集保留为自动诊断，不对外作为最终精度。

---

## 5. 可选 Teacher-clean YAML

仅用于快速自动诊断，不作为最终主测试集。

```yaml
palm:
  backend: aethersign_onnx
  input_size: 224
  score_threshold: 0.60
  nms_iou_threshold: 0.30
  cross_head_suppress_iou: 0.35
  max_detections: 2
  keep_low_score_candidates_for_negatives: false
  negative_candidate_threshold: 0.15

mediapipe:
  model_asset_path: models/mediapipe/hand_landmarker.task
  num_hands: 1
  min_hand_detection_confidence: 0.65
  min_hand_presence_confidence: 0.65
  min_tracking_confidence: 0.5
```

推荐范围：

```text
Palm score threshold：0.55～0.65
MediaPipe detection/presence threshold：0.60～0.70
```

不建议未经验证直接超过 0.70，因为可能大量删除困难姿态。

---

## 6. 可选 Hard-negative YAML 与脚本扩展

当前配置会保留从 `negative_candidate_threshold=0.15` 到 `score_threshold=0.50` 的大量低分候选。建议 hard-negative 集只选择临近运行阈值的候选，例如：

```yaml
palm:
  score_threshold: 0.50
  keep_low_score_candidates_for_negatives: true
  negative_candidate_threshold: 0.35
```

同时建议给 `01_export_palm_detections.py` 增加：

```yaml
palm:
  max_negative_candidates_per_image: 2
```

选择策略：

```text
每张原图在 [0.35, 0.50) 的候选中，按 palm_score 从高到低最多保留 2 个。
```

不能保留每张图所有低分 anchor，否则：

- 数据量膨胀；
- 同一背景重复出现；
- 负样本严重压倒正样本；
- 评测结果被大量容易负样本主导。

Hard-negative 集应单独报告：

```text
hand presence false-positive rate
hard-negative rejection accuracy
```

不要与 landmark PCK/NME 混合为一个总体 accuracy。

---

## 7. 阈值调整实验流程

### 7.1 不得使用 Test 调阈值

推荐流程：

1. 使用训练集或 Val 原图缓存 Palm 原始输出；
2. 在 Val 上离线扫描多个阈值；
3. 确定最终板端阈值；
4. 冻结阈值；
5. Test 只运行一次最终配置。

建议扫描：

```text
0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65
```

每个阈值记录：

- 每张图平均有效 detections 数量；
- 含手原图的 Palm recall；
- 无手原图的 false detections/image；
- ROI 中真实有手比例；
- 两只手场景的双手召回率；
- 后续 Hand Landmarker 的 presence recall；
- 完整链路 landmark 指标。

阈值选择原则不是“ROI 越少越好”，而是：

> 在维持足够 Palm recall 的前提下，降低错误 ROI 数量和端侧计算开销。

---

## 8. Val/Test 推荐指标

### 8.1 Hand presence

报告：

- Precision；
- Recall；
- F1；
- False Positive Rate；
- 按 Palm score 分桶后的结果。

不要只报告 accuracy，避免负样本比例影响判断。

### 8.2 Landmarks

只在人工确认 `hand_presence=true` 的样本上计算。

推荐：

#### NME

以人工关键点手尺度归一化：

\[
\mathrm{NME}
=
\frac{1}{21}
\sum_{j=0}^{20}
\frac{\|\hat p_j-p_j\|_2}{s}
\]

归一化尺度可选择：

```text
s = ||p0 - p9||
```

或者人工关键点外接框对角线。整个项目必须固定一种定义。

#### PCK

报告：

```text
PCK@0.05
PCK@0.10
PCK@0.15
```

阈值同样相对手尺度定义。

### 8.3 Handedness

在人工确认有手且左右手可判定的样本上报告 accuracy。

### 8.4 分场景指标

推荐至少保留标签或元数据：

- 正常光照；
- 暗光/红外；
- 运动模糊；
- 边缘手；
- 部分遮挡；
- 双手；
- ROI 偏移；
- 特殊手势。

最终模型可能总体 NME 不错，但在暗光或边缘手上彻底失效。分场景结果更能指导工程决策。

---

# 第二部分：约 7 万训练 ROI 的自动质量分级

## 9. 仅靠最终 JSONL 中已有字段为什么不够

当前最终标注文件包含：

- `hand_presence.present`；
- `handedness`；
- 21 点坐标；
- `palm_valid` 与 `palm_score`；
- `needs_review`；
- ROI 几何；
- loss weights。

这些字段可以发现结构错误和明显异常，但不能自动判断：

- MediaPipe 的 21 点是否整体偏到了错误位置；
- 某几个手指关键点是否稳定；
- `presence=false` 是否是假负样本；
- 同一原图产生的多个 ROI 是否实际上是同一只手的重复副本。

因此建议把最终 JSONL 当成：

> 样本索引、初始伪标签和元数据入口。

自动质量判定脚本必须重新读取 crop 图像，并进行额外计算。

---

## 10. 新增输出字段设计

不要覆盖原始 `hand_training_labels.jsonl`。输出新文件：

```text
data/05_labels/hand_training_labels_scored.jsonl
```

建议为每行新增：

```json
{
  "quality": {
    "tier": "POS_HIGH",
    "score": 0.93,
    "flags": [],
    "presence_vote_ratio": 1.0,
    "landmark_jitter_median": 0.021,
    "landmark_jitter_p90": 0.047,
    "initial_to_consensus_nme": 0.032,
    "handedness_vote_ratio": 1.0,
    "geometry_outlier_score": 1.4,
    "black_pixel_ratio": 0.12,
    "duplicate_cluster_id": "image_x:hand0",
    "duplicate_rank": 0
  },
  "training_control": {
    "use_for_base_training": true,
    "sample_weight": 1.0,
    "hand_presence_loss_weight": 1.0,
    "landmark_loss_weight": 1.0,
    "handedness_loss_weight": 1.0
  },
  "pseudo_label_source": "mediapipe_tta_consensus"
}
```

建议保留原始字段：

```text
original_hand_presence
original_landmarks_crop_norm
original_handedness
```

如自动共识改变标签，不应丢失初始版本，便于审计。

---

## 11. 自动质量处理总体流程

```text
输入最终 JSONL
    ↓
A. Schema 与数值硬校验
    ↓
B. 图像和 ROI 极端异常检测
    ↓
C. MediaPipe TTA 多次推理与共识标签
    ↓
D. 骨架几何与统计异常检测
    ↓
E. 同一原图内重复手实例聚类/去重
    ↓
F. 样本类型判定与 loss/sample weight 赋值
    ↓
输出 scored JSONL + QC 统计报告
```

---

## 12. 阶段 A：Schema 与数值硬校验

### 12.1 所有样本必须检查

- `crop_id` 唯一；
- `crop_path` 存在且可读；
- 图片为 256×256；
- 所有数值有限，不允许 NaN/Inf；
- `palm_score` 在合理范围；
- `roi_rect.width/height > 0`；
- `roi_corners_px` 数量和形状正确；
- `source_image_width/height` 与数据集配置一致。

### 12.2 正样本必须检查

当：

```text
hand_presence.present=true
```

必须满足：

- `landmarks_crop_norm` 恰好 21 点；
- 点 ID 唯一且完整为 0～20；
- `landmarks_crop_px` 恰好 21 点；
- `landmarks_image_px` 恰好 21 点；
- norm 与 px 坐标近似一致：

  ```text
  x_px ≈ x_norm × 255 或 x_norm × 256
  y_px ≈ y_norm × 255 或 y_norm × 256
  ```

  具体采用 255 还是 256，应与现有投影实现保持一致，并设置小容差。

### 12.3 负样本必须检查

当：

```text
hand_presence.present=false
```

必须满足：

- landmark 列表为空；
- handedness 为 `unknown` 或对应 loss weight 为 0；
- landmark/handedness loss weight 不得为正。

### 12.4 硬拒绝

满足以下任一条件，直接：

```text
quality.tier = DROP_INVALID
use_for_base_training = false
```

包括：

- 图片不可读；
- 点数不为 21；
- ID 缺失或重复；
- 坐标 NaN/Inf；
- norm/px 严重不一致；
- 正负样本字段自相矛盾；
- ROI 几何无法解析。

---

## 13. 阶段 B：图像与边界异常检测

### 13.1 计算基础图像指标

对每个 256×256 灰度 crop 计算：

```text
mean_intensity
std_intensity
p01 / p99 intensity
black_pixel_ratio
white_pixel_ratio
laplacian_variance
```

其中：

- `black_pixel_ratio` 可以反映 ROI 越界后的黑色填充比例；
- `std_intensity` 可以发现近乎纯黑/纯灰帧；
- `laplacian_variance` 可作为模糊程度的辅助指标。

### 13.2 不要把暗光和模糊直接全部删除

暗光、红外、运动模糊本来就是目标部署场景。图像质量指标只用于：

- 发现几乎完全无信息的 crop；
- 降低极端异常样本权重；
- 生成质量 flag；
- 结合教师一致性共同判断。

不能使用：

```text
“模糊度低于某值 → 全部删除”
```

否则模型会失去项目最需要的困难场景能力。

### 13.3 初始极端阈值

建议先用非常宽松的硬阈值：

```text
black_pixel_ratio >= 0.85
或 std_intensity <= 2（uint8 标度）
```

标记：

```text
IMAGE_ALMOST_EMPTY
```

这类样本通常不参与 landmark 训练。阈值应根据全数据直方图再校准。

---

## 14. 阶段 C：MediaPipe 多变换一致性（最重要）

### 14.1 原理

如果一张 ROI 的 MediaPipe 伪标签可靠，那么对图像施加轻微、可逆、不会改变手势语义的扰动后：

- hand presence 应保持稳定；
- 逆变换后的 21 点应接近；
- handedness 应保持一致。

如果输出在轻微扰动下剧烈变化，则该伪标签不应作为高权重回归监督。

### 14.2 推荐 TTA 变换

建议第一版使用 5 次推理：

```text
T0: identity
T1: rotate -5°
T2: rotate +5°
T3: scale 0.96，保持中心
T4: scale 1.04，保持中心
```

也可以将缩放替换成 ±4 px 小平移。

第一版不建议使用水平翻转，原因是：

- handedness 标签需要同步反转；
- 摄像头镜像语义可能与现有系统定义不一致；
- 容易引入额外实现错误。

待基础脚本验证后再增加 flip。

### 14.3 逆变换与共识标签

每个 TTA 结果中的关键点先使用对应逆仿射矩阵映射回原始 crop 坐标。

对成功检测到手的结果，逐点取中位数：

\[
\tilde p_j = \operatorname{median}_t(p_{t,j})
\]

最终共识 landmarks 使用 21 个中位点。

handedness 使用多数投票；若得票比例不足阈值，则设为 `unknown`。

### 14.4 一致性指标定义

设成功检测到手的 TTA 数量为：

```text
n_present
```

总 TTA 次数为 5：

\[
r_{presence}=n_{present}/5
\]

定义手尺度：

\[
s=\max(\|\tilde p_0-\tilde p_9\|_2, 0.08)
\]

其中坐标已归一化到 ROI；`0.08` 用于防止异常小尺度导致除零或过度放大。

每次推理的关键点抖动：

\[
d_{t,j}=\frac{\|p_{t,j}-\tilde p_j\|_2}{s}
\]

统计：

```text
landmark_jitter_median = median(d_tj)
landmark_jitter_p90    = percentile90(d_tj)
```

初始标签与共识标签差异：

\[
D_{initial}=\frac{1}{21}\sum_j
\frac{\|p^{initial}_j-\tilde p_j\|_2}{s}
\]

### 14.5 初始阈值建议

这些阈值是起始值，最终应观察全数据分布并人工抽查少量样本校准。

#### 高稳定正样本

```text
n_present = 5
landmark_jitter_median <= 0.04
landmark_jitter_p90 <= 0.08
initial_to_consensus_nme <= 0.08
```

#### 中等稳定正样本

```text
n_present >= 4
landmark_jitter_median <= 0.08
landmark_jitter_p90 <= 0.15
initial_to_consensus_nme <= 0.15
```

#### 不稳定样本

```text
n_present <= 3
或 landmark_jitter_p90 > 0.15～0.20
```

第一阶段基座训练中，默认不使用其 landmark 监督。

### 14.6 自动纠正单次伪标签

#### 原始为负，但 TTA 共识为正

```text
original present=false
n_present >= 4
```

自动改为正样本：

- `hand_presence=true`；
- landmarks 使用 TTA 中位数；
- handedness 使用共识；
- 标记 `RELABEL_NEG_TO_POS_BY_TTA`。

#### 原始为正，但 TTA 几乎均为负

```text
original present=true
n_present <= 1
```

不要立刻自动改为可靠负样本。先：

- 标记 `POSITIVE_NOT_REPRODUCIBLE`；
- 若全图交叉验证也没有重叠手，才可改为负样本；
- 否则设为 `DROP_AMBIGUOUS`。

这是因为“把真实有手样本错误地训练成负样本”比少使用一个样本危害更大。

---

## 15. 可选：原图 MediaPipe 交叉验证

如果保留了原始 1280×720 图片，建议每张原图只运行一次官方 MediaPipe，缓存检测到的手和 21 点。

对每个 crop：

1. 将全图 MediaPipe landmarks 与 crop 的 `roi_rect` 做几何关系判断；
2. 检查是否有全图手实例落在该 ROI 内；
3. 对负样本检查全图教师是否在 ROI 对应区域检测到手。

建议新增：

```text
full_image_teacher_overlap
full_image_teacher_num_hands
full_image_teacher_match
```

用途：

- crop teacher 为负、全图 teacher 明确有重叠手：疑似假负，丢弃或重新标正；
- crop teacher 与全图 teacher 都为负：负标签可信度提高；
- 同一原图多个 crop 对应同一全图手：用于重复聚类。

如果运行成本有限，该步骤可以只对：

- 所有负样本；
- `n_present` 不稳定样本；
- 高 Palm score 但 Hand presence=false 的样本；

执行，而不必对全部 7 万样本执行。

---

## 16. 阶段 D：关键点边界与骨架几何检查

### 16.1 坐标边界分级

对正样本统计：

```text
num_points_outside_01
max_outside_distance
```

建议：

#### 正常

```text
所有点位于 [0, 1]
```

#### 轻微越界

```text
最多 2 个点位于 [-0.05, 1.05]
```

可能是边缘手或指尖略超 ROI，不应直接删除，但最多归为中质量。

#### 严重越界

```text
任一点超出 [-0.10, 1.10]
或超过 4 个点位于 [0,1] 外
```

标记 `LANDMARKS_SEVERELY_OUT_OF_BOUNDS`，第一阶段不使用 landmark loss。

### 16.2 手部包围框

由 21 点计算：

```text
bbox_width
bbox_height
bbox_area
bbox_center
```

过小可能表示：

- 关键点缩成一团；
- ROI 中手太小；
- 教师输出错误。

过大或大面积越界可能表示：

- ROI 构造错误；
- 手严重裁切；
- 标注整体偏移。

不建议一开始设非常严格的固定阈值。可先将以下作为宽松警告：

```text
bbox_width < 0.10
bbox_height < 0.10
bbox_width > 1.10
bbox_height > 1.10
```

### 16.3 骨架连接关系

使用 MediaPipe 21 点拓扑计算骨长：

```text
0-1-2-3-4
0-5-6-7-8
5-9-10-11-12
9-13-14-15-16
13-17-18-19-20
0-17
```

不能使用过于严格的“手指一定单调伸长”等规则，因为弯曲、投影和遮挡会使其失效。

更稳妥的方法是采用**数据驱动的鲁棒异常检测**。

### 16.4 数据驱动的几何异常分数

先从满足以下条件的 provisional clean pool 中学习分布：

```text
n_present = 5
所有点在 [0,1]
needs_review = false
landmark_jitter_p90 较低
```

为每个样本提取特征：

- bbox width/height/area；
- `||p0-p9||`、`||p0-p5||`、`||p0-p17||`；
- 每条骨长除以 `||p0-p9||`；
- 五根手指总长度除以 palm scale；
- 掌部几个关键点间距离比。

对每个特征使用 median 和 MAD：

\[
z_{robust} = \frac{|x-\operatorname{median}(x)|}
{1.4826\cdot MAD(x)+\epsilon}
\]

定义：

```text
geometry_outlier_score = 特征 robust z 的高分位数或最大值
```

初始建议：

```text
<= 4.5：正常
4.5～6.0：轻度异常
> 6.0：严重异常
```

分 Left/Right 统计通常不是必须的，因为距离比例对镜像基本不变；如果加入有方向角度特征，则应分左右手或先统一镜像到同一侧。

---

## 17. 阶段 E：同一原图内重复 ROI 聚类（必做）

### 17.1 为什么必须去重

一张原图可能产生多个 Palm candidate。即使它们的 crop 坐标不同，Google MediaPipe 仍可能在多个 ROI 中找到同一只手。

`example.jsonl` 就呈现了这种情况：同一原图包含多个低分 Palm candidate，其中多个正样本反投影回原图后的 21 点几乎重合。

如果这些样本全部等权参与训练，会导致：

- 同一只手、同一帧被重复训练多次；
- 某些原图被放大 5～10 倍；
- 数据集有效多样性远低于表面上的 7 万；
- 训练/采样分布被候选数量而不是场景数量决定。

### 17.2 使用 `landmarks_image_px` 聚类

对相同 `image` 的所有正样本，计算两两距离。

设两个样本 A、B 的原图关键点为：

\[
p^A_j, p^B_j
\]

定义原图手尺度：

\[
s_{AB}=\max((s_A+s_B)/2, 10\text{ pixels})
\]

其中 `s_A`、`s_B` 可取 `||p0-p9||`。

定义：

\[
D(A,B)=\operatorname{median}_j
\frac{\|p^A_j-p^B_j\|_2}{s_{AB}}
\]

初始聚类阈值：

```text
D(A,B) <= 0.15～0.20
且 handedness 一致
```

可视为同一原图中的同一只手实例。

也可以结合：

- 关键点 bbox IoU；
- ROI 中心距离；
- 全图 teacher hand ID。

### 17.3 每个聚类如何保留

对同一真实手的重复 crop：

- 默认只保留质量分最高的 1 个作为全权重样本；
- 可额外保留第 2 个 ROI，前提是它与第一名的 crop 几何差异明显，可提供 ROI 偏移鲁棒性；
- 第 2 个样本 `sample_weight` 建议为 0.25～0.5；
- 其余重复样本设置：

  ```text
  quality.tier = DROP_DUPLICATE
  use_for_base_training = false
  ```

这样不是删除磁盘文件，只是不让同一帧被不合理地重复放大。

### 17.4 负样本去重

每张原图建议最多保留：

```text
1 个可靠普通负样本
1 个可靠 hard negative
```

其余低分负候选不参与基座训练。

---

## 18. 阶段 F：最终样本类型与训练权重

建议不要只分“高/低质量”，而是使用语义明确的类型。

## 18.1 `POS_HIGH`：高质量正样本

条件建议：

```text
最终共识 present=true
n_present = 5/5
schema 合法
无严重图像异常
所有点位于 [0,1]，或没有明显越界
landmark_jitter_median <= 0.04
landmark_jitter_p90 <= 0.08
initial_to_consensus_nme <= 0.08
geometry_outlier_score <= 4.5
同一原图去重后被保留
```

训练控制：

```json
{
  "sample_weight": 1.0,
  "hand_presence_loss_weight": 1.0,
  "landmark_loss_weight": 1.0,
  "handedness_loss_weight": 1.0
}
```

handedness 若无稳定共识，则其权重设为 0，不影响 presence 和 landmark。

---

## 18.2 `POS_MEDIUM`：可用的中质量正样本

条件建议：

```text
最终共识 present=true
n_present >= 4/5
无硬错误
landmark_jitter_median <= 0.08
landmark_jitter_p90 <= 0.15
initial_to_consensus_nme <= 0.15
geometry_outlier_score <= 6.0
最多轻微越界
```

训练控制：

```json
{
  "sample_weight": 0.5,
  "hand_presence_loss_weight": 1.0,
  "landmark_loss_weight": 0.5,
  "handedness_loss_weight": 1.0
}
```

如果 handedness 投票比例不足 0.8：

```text
handedness_loss_weight = 0
```

---

## 18.3 `POS_PRESENCE_ONLY`：确认有手，但关键点不稳定

可能条件：

```text
n_present >= 4/5
但 landmark jitter 或几何异常超过中质量阈值
```

对于当前“提高关键点精度”的第一阶段基座，建议默认：

```json
{
  "sample_weight": 0.25,
  "hand_presence_loss_weight": 0.5,
  "landmark_loss_weight": 0.0,
  "handedness_loss_weight": 0.0
}
```

如果 presence head 已经容易训练，也可以完全排除该类，避免增加复杂度。

---

## 18.4 `NEG_RELIABLE`：可靠负样本

条件建议：

```text
原始或共识 present=false
TTA 5/5 均无手
palm_valid=false
palm_score 较低，例如 <= 0.35
若执行全图交叉验证，则无重叠手
图像不是损坏/纯黑异常
每张原图去重与限额后被保留
```

训练控制：

```json
{
  "sample_weight": 0.5,
  "hand_presence_loss_weight": 1.0,
  "landmark_loss_weight": 0.0,
  "handedness_loss_weight": 0.0
}
```

不需要让容易负样本与正样本数量相同。

---

## 18.5 `NEG_HARD`：自动交叉确认的困难负样本

条件建议：

```text
TTA 5/5 均无手
且 palm_valid=true，或 palm_score 接近/超过阈值
全图 teacher 未发现与 ROI 重叠的手
图像有效
```

训练控制：

```json
{
  "sample_weight": 0.5,
  "hand_presence_loss_weight": 0.5,
  "landmark_loss_weight": 0.0,
  "handedness_loss_weight": 0.0
}
```

为什么 presence weight 不一定给 1.0：

- 高 Palm score 但 teacher 无手可能是真 hard negative；
- 也可能是 teacher 对困难真实手的稳定假负；
- 无人工复核时应保守使用。

建议每个 batch 中 `NEG_HARD` 不超过全部样本的 5%～10%。

---

## 18.6 `DROP_AMBIGUOUS`：不确定样本

包括：

- TTA presence 在 2/5～3/5 间摇摆；
- 原始正样本无法被重复推理复现；
- 高 Palm score、crop 中疑似有手，但 teacher 多次失败；
- 关键点严重越界；
- 骨架几何极端异常；
- 多手冲突；
- `needs_review=true` 且无法区分具体原因；
- full-image 与 crop teacher 强冲突。

训练控制：

```json
{
  "sample_weight": 0.0,
  "hand_presence_loss_weight": 0.0,
  "landmark_loss_weight": 0.0,
  "handedness_loss_weight": 0.0
}
```

在 7 万样本中少使用一部分模糊样本，通常比错误监督更安全。

---

## 18.7 `DROP_DUPLICATE` 和 `DROP_INVALID`

均不参与基座训练：

```text
sample_weight = 0
所有 head loss weight = 0
```

但必须保留记录和统计，便于知道表面 7 万样本的真实有效样本量。

---

## 19. `needs_review` 的处理方式

当前最终文件只有布尔值：

```text
needs_review=true/false
```

但不同原因的严重程度不同：

- handedness 分数略低；
- 一个点轻微越界；
- 多手；
- 高分 Palm 被标无手；
- 数据结构 warning。

建议修改 `07_finalize_training_labels.py`，除 `needs_review` 外保留：

```json
"qc_flags": [
  "LANDMARK_OUT_OF_BOUNDS",
  "HIGH_PALM_SCORE_BUT_NO_HAND"
]
```

在完成这一修改前，第一版自动策略可以保守设置：

```text
needs_review=true → 至多进入 POS_MEDIUM / NEG_HARD
不能进入 POS_HIGH / NEG_RELIABLE
```

如果 `needs_review=true` 且 TTA/几何也异常，则直接 `DROP_AMBIGUOUS`。

---

## 20. 推荐自动质量配置文件

建议新增：

```text
configs/training_sample_quality.yaml
```

示例：

```yaml
quality_scoring:
  input_labels: data/05_labels/hand_training_labels.jsonl
  output_labels: data/05_labels/hand_training_labels_scored.jsonl
  output_stats: data/qc/training_sample_quality_stats.json

  tta:
    enabled: true
    rotations_deg: [0, -5, 5]
    scales: [0.96, 1.04]
    use_horizontal_flip: false
    min_present_high: 5
    min_present_medium: 4

  consistency:
    palm_scale_min: 0.08
    high_jitter_median_max: 0.04
    high_jitter_p90_max: 0.08
    high_initial_consensus_nme_max: 0.08
    medium_jitter_median_max: 0.08
    medium_jitter_p90_max: 0.15
    medium_initial_consensus_nme_max: 0.15

  coordinate_qc:
    mild_outside_margin: 0.05
    severe_outside_margin: 0.10
    max_mild_outside_points: 2
    max_severe_outside_points: 4

  geometry_qc:
    robust_z_high_max: 4.5
    robust_z_medium_max: 6.0
    provisional_clean_only: true

  image_qc:
    almost_empty_black_ratio: 0.85
    almost_empty_std_uint8: 2.0

  dedup:
    group_key: image
    landmark_image_distance_threshold: 0.18
    max_positive_samples_per_hand_cluster: 2
    second_duplicate_sample_weight: 0.35
    max_reliable_negatives_per_image: 1
    max_hard_negatives_per_image: 1

  negative_rules:
    reliable_negative_palm_score_max: 0.35
    require_tta_all_negative: true
    require_full_image_no_overlap_for_hard_negative: true

  weights:
    POS_HIGH:
      sample: 1.0
      presence: 1.0
      landmark: 1.0
      handedness: 1.0
    POS_MEDIUM:
      sample: 0.5
      presence: 1.0
      landmark: 0.5
      handedness: 1.0
    POS_PRESENCE_ONLY:
      sample: 0.25
      presence: 0.5
      landmark: 0.0
      handedness: 0.0
    NEG_RELIABLE:
      sample: 0.5
      presence: 1.0
      landmark: 0.0
      handedness: 0.0
    NEG_HARD:
      sample: 0.5
      presence: 0.5
      landmark: 0.0
      handedness: 0.0
    DROP_AMBIGUOUS:
      sample: 0.0
      presence: 0.0
      landmark: 0.0
      handedness: 0.0
```

---

## 21. 分类伪代码

```python
for row in labels:
    # A. 硬校验
    if not schema_is_valid(row):
        assign(row, "DROP_INVALID")
        continue

    image = load_crop(row["crop_path"])
    image_metrics = compute_image_metrics(image)

    # C. 教师多变换推理
    tta_results = run_mediapipe_tta(image)
    consensus = build_consensus(tta_results)

    # 原始负、TTA 稳定为正：允许纠正
    if not row["hand_presence"]["present"] and consensus.num_present >= 4:
        replace_with_consensus_positive_label(row, consensus)
        add_flag(row, "RELABEL_NEG_TO_POS_BY_TTA")

    # 原始正、TTA 基本无法复现：先视为模糊，不直接变负
    if row["hand_presence"]["present"] and consensus.num_present <= 1:
        if full_image_teacher_confirms_no_hand(row):
            replace_with_negative_label(row)
        else:
            assign(row, "DROP_AMBIGUOUS")
            continue

    if consensus.is_positive:
        geometry = compute_geometry_features(consensus.landmarks)

        if severe_coordinate_or_geometry_error(consensus, geometry):
            assign(row, "DROP_AMBIGUOUS")
        elif is_high_quality_positive(row, consensus, geometry, image_metrics):
            assign(row, "POS_HIGH")
        elif is_medium_quality_positive(row, consensus, geometry, image_metrics):
            assign(row, "POS_MEDIUM")
        else:
            assign(row, "POS_PRESENCE_ONLY")
    else:
        if not consensus.all_negative:
            assign(row, "DROP_AMBIGUOUS")
        elif is_reliable_easy_negative(row, image_metrics):
            assign(row, "NEG_RELIABLE")
        elif full_image_teacher_confirms_no_hand(row):
            assign(row, "NEG_HARD")
        else:
            assign(row, "DROP_AMBIGUOUS")

# E. 必须在初次分级后按原图去重
cluster_duplicate_hands_by_image(labels)
cap_negative_samples_per_image(labels)

write_scored_jsonl(labels)
write_quality_report(labels)
```

---

## 22. 训练采样比例

即使自动分级后某一类数量很多，也不要完全按文件中的自然比例随机训练。

推荐每个 batch 或每个 epoch 的有效采样比例：

```text
正样本总计：75%～85%
负样本总计：15%～25%
```

正样本内部：

```text
POS_HIGH：约 70%～85% 的正样本配额
POS_MEDIUM：约 15%～30%
POS_PRESENCE_ONLY：0%～5%，也可以不用
```

负样本内部：

```text
NEG_RELIABLE：大多数
NEG_HARD：总 batch 的 5%～10% 以内
```

原因：Palm Detector 已经是前级筛选器，Hand Landmarker 的真实运行输入并不是随机背景图。过多负样本会让 hand presence head 很容易，但削弱 landmark 回归训练。

---

## 23. 第一阶段训练的 curriculum

虽然当前只讨论约 7 万数据的基座训练，仍建议分两个子阶段：

### 23.1 基座 warm-up

前 30%～40% epoch：

- POS_HIGH；
- NEG_RELIABLE；
- 少量 NEG_HARD。

目标：先学到稳定的关键点映射。

### 23.2 扩展训练

后 60%～70% epoch：

- 保留全部 POS_HIGH；
- 加入 POS_MEDIUM，landmark weight=0.5；
- 保持负样本比例受控。

默认不加入 `DROP_AMBIGUOUS`、`DROP_DUPLICATE`、`DROP_INVALID`。

这样仍属于同一次 7 万伪标注基座训练，不涉及后续人工微调。

---

## 24. 自动 QC 报告必须包含的统计

`training_sample_quality_stats.json` 或 Markdown 报告至少输出：

### 24.1 总体数量

```text
输入样本数
实际可读样本数
POS_HIGH
POS_MEDIUM
POS_PRESENCE_ONLY
NEG_RELIABLE
NEG_HARD
DROP_AMBIGUOUS
DROP_DUPLICATE
DROP_INVALID
```

### 24.2 正负与来源分布

按以下字段交叉统计：

- `palm_valid`；
- Palm score 分桶；
- `source`；
- `needs_review`；
- handedness；
- 原始图片；
- TTA presence 投票数。

### 24.3 去重统计

```text
原始正样本 ROI 数
聚类出的真实手实例近似数量
平均每个手实例重复 ROI 数
最大重复数
被 DROP_DUPLICATE 的数量
```

这能揭示“7 万 ROI”究竟对应多少独立手实例。

### 24.4 阈值分布图所需数据

保存直方图或 CSV：

- palm_score；
- jitter median/P90；
- initial-to-consensus NME；
- geometry robust z；
- black pixel ratio；
- duplicate cluster size。

正式锁定阈值前，应随机可视化每个区间的几十张样本。

这不是全量人工复核，而是对自动规则做一次小规模校准。

---

## 25. 阈值校准的最低人工抽查

虽然 7 万训练样本不人工逐张复核，但自动判定规则仍建议抽查：

```text
POS_HIGH：随机 100 张
POS_MEDIUM：随机 100 张
NEG_RELIABLE：随机 100 张
NEG_HARD：随机 100 张
DROP_AMBIGUOUS：随机 100 张
每个主要阈值边界附近：各 50 张
```

总计约 500～800 张快速浏览即可。

目的不是修正这些样本，而是回答：

- POS_HIGH 是否真的大多正确；
- NEG_RELIABLE 中是否存在明显真实手；
- TTA jitter 阈值是否过严/过松；
- 几何异常规则是否误杀弯曲手势；
- 黑边规则是否误杀边缘手。

这是自动系统上线前的规则验证，不等同于 7 万张人工标注。

---

## 26. 推荐实施顺序

### Step 1：构建 Val/Test

- 使用独立录制的 Val/Test 原图；
- 关闭低分负候选；
- 保持部署阈值和 ROI 参数；
- 对最终保留的约数百～2000 ROI/split 做人工复核；
- Test 冻结。

### Step 2：实现训练集质量脚本的最小版本

第一版先完成：

1. schema 检查；
2. TTA 5 次共识；
3. 点越界检查；
4. 同原图 `landmarks_image_px` 去重；
5. POS_HIGH / POS_MEDIUM / NEG_RELIABLE / DROP 分类；
6. 输出新 JSONL 和统计。

### Step 3：加入增强规则

随后增加：

- 骨架 robust z；
- full-image teacher 交叉验证；
- NEG_HARD；
- 图像极端异常；
- curriculum 和分层采样。

### Step 4：训练与消融

至少对比：

```text
A. 原始 7 万全部等权
B. 只用 needs_review=false
C. 本文自动分级 + 去重 + 加权
```

在同一人工 Val/Test 上比较：

- hand presence F1；
- landmark NME/PCK；
- handedness accuracy；
- FP32 到 INT8 的退化。

如果 C 明显优于 A/B，说明自动质量规则有效。

---

# 第三部分：最终推荐配置摘要

## 27. Val/Test

主评测：

```yaml
palm:
  score_threshold: 0.50        # 与板端一致
  nms_iou_threshold: 0.30
  cross_head_suppress_iou: 0.35
  max_detections: 2
  keep_low_score_candidates_for_negatives: false

mediapipe:
  min_hand_detection_confidence: 0.5
  min_hand_presence_confidence: 0.5
```

预期每 1000 张原图最多约 2000 个主 ROI，而不是 8000～10000 个。

## 28. 7 万训练集

必须增加：

```text
TTA 一致性
同原图去重
结构/坐标检查
几何异常检测
负样本保守筛选
分层 loss 权重和 sample weight
```

第一版基座训练实际使用：

```text
POS_HIGH
POS_MEDIUM
NEG_RELIABLE
少量 NEG_HARD
```

默认排除：

```text
DROP_AMBIGUOUS
DROP_DUPLICATE
DROP_INVALID
```

## 29. 最重要的工程原则

1. **不要用海量低分 Palm candidates 构成主 Val/Test。**
2. **不要通过提高测试集阈值删除困难样本来换取更高测试精度。**
3. **不要将同一原图、同一只手的多个重叠 ROI 全部等权训练。**
4. **不要把 TTA 不稳定的 `presence=false` 直接当作可靠负样本。**
5. **landmark 标签不稳定时，可以降低或关闭 landmark loss，而不是必须整张样本二选一。**
6. **训练集可以是伪标签，但 Test 必须尽可能是人工金标准。**
7. **所有自动规则都保留原始标签、flags 和统计，保证可审计、可做消融。**

