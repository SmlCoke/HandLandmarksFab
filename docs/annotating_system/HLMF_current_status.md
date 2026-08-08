# HLMF 当前状态（2026-08-08）

## 代码与配置

HLMF 3.0 统一入口为 `scripts/hlmf.py` 和 Makefile；公开配置为 `configs/autolabel.yaml`、`configs/review.yaml`、`configs/datasets.yaml`、`configs/cvat_label.json`。MediaPipe Tasks 仍是全局默认 Hand landmark 后端，RTMPose-m Hand5 可通过单次参数启用。

RTMPose runtime ROI 已接入新的 MobileNetV3-Small 双头 HCF。模型路径为 `models/handedness-handpresence-0807/model.onnx`，模型 ID 为 `hand-classifier-handedness-handpresence-0807`；旧 handedness-only 资产保存在 `models/handedness-0806/`，不再参与推理。新模型输入为 `[N,1,256,256]`，输出 `handedness` 与 `hand_presence` 两个 `[N,2]` logits。服务器 ONNX Runtime 当前只激活 CPU；真实双模型冒烟中 RTMPose 与 HCF 均使用 `CPUExecutionProvider`，RTMPose 输出 21 个有限坐标，HCF 两个分类头均输出有限概率。Eos low-score candidate 不运行两模型，保持未解析状态。

新 HCF 自带验证集指标：presence accuracy `0.991279`、presence ROC AUC `0.999312`；handedness accuracy `0.992248`、handedness ROC AUC `0.996689`。presence 混淆矩阵为 `[[322,3],[9,1042]]`（0=no_hand，1=has_hand）。

当前 Train 质量配置：handedness review threshold 为 `0.7`；RTMPose Train runtime 的 `P(has_hand)` 阈值为 `0.5`；42 个 crop 坐标值中精确边界值达到 2 个时拒绝。presence 分数缺失、非有限或低于阈值时，以 `rtmpose_hand_presence_gate` 进入 `ignored.jsonl`。等于阈值通过。Eval、MediaPipe 和 Eos low-score candidate 不应用 RTMPose Train presence/边界门控。

Presence 阈值分析放在仓库外的 `/root/hcf_presence_threshold_0807/`。分析使用复制出的 `FullEnhanceVal0801` 人工复核 ROI，没有写入现有数据集。有效样本共 7,907 条：7,892 条 hand、15 条 no_hand。正样本 `P(has_hand)` 均值为 `0.993917`，no_hand 最大值为 `0.0192493`。正式阈值 `0.5` 与模型 argmax 决策边界一致，拒绝全部 15 条 no_hand，并保留 7,856/7,892 条 hand（99.5438%）。全局 recall 约束产生的候选阈值 `0.843369` 虽仍保留 99.0117% 的全局 hand，却把一个来源的 hand recall 降至 88.24%，因此未采用；`0.5` 在该来源的已知 recall 为 94.75%，这部分低置信 ROI 在 Train 中会被拒绝，Eval 仍由人工复核纠正。

原图可视化默认生成按 PNG 文件名字典序排列的 30 FPS MP4；RTMPose ROI 可视化只抽样 runtime ROI。可视化可以独立清理。Registry 使用 source/variant active/retired 状态表，删除变体会留下永久 tombstone。

自动标注批处理现从数据集直接子目录的 `images/` 发现来源，不再要求首次运行前已有 `source.json`。新增数据集级 `batch-autolabel-visualizations-clean` 与 `batch-source-variant-delete`；后者保留精确确认和 retired tombstone，并在批处理末尾重建 dataset manifest。未执行 `source-publish` 的来源仍不会进入 manifest。

服务器 Hand ROI 像素域审查覆盖 Pretrain/Eval 的 71,779 张原图 TIFF，全部为单通道 8-bit、`1280×720`；`FullEnhanceVal0801/eos-1.0` 的 8,248 张 ROI 全部为单通道 8-bit、`256×256` PNG。跨 10 个来源抽查 50 个 ROI，无损 PNG 与无损 TIFF 往返后逐像素一致。Python OpenCV 与板端 C++ 双线性采样的对比中，3,276,800 个像素仅 242 个相差 1 灰度级（0.0074%），差异来自插值舍入而非文件格式；现有 ROI、人工复核结果和发布资产均不重建。

负样本和困难样本的 review/published 图片均使用独立复制；困难样本 published 记录包含 `published_relpath`，不依赖源变体存活。当前服务器存在 `background-neg-0801` review 工作区，以及已发布 1,543 条记录/图片的 `background-neg-0801-full`；本轮均未修改。

## 服务器数据仓库

Eval 数据集 `FullEnhanceVal0801` 当前有 10 个来源：6 个 val、4 个 test，均已发布 `eos-1.0`：

- val：4,200 张原图、5,288 个 ROI、5,091 条发布标签；
- test：2,800 张原图、2,960 个 ROI、2,816 条发布标签。

Pretrain 数据集 `FullEnhance0801` 当前有 95 个 train 来源：

- `eos-1.0`：95 个来源的变体派生产物已全部删除，Registry 保留 95 个 retired tombstone；原始 `images/`、`raw_images.jsonl`、`source.json` 和既有 ROI Registry 元数据均保留；
- `eos_1.0-gate`：95 个来源均完整发布，共 564,243 个 ROI、65,089 条 positive、492,017 条 candidate、7,137 条 ignored。

Eval 数据集 `FullEnhanceVal0808` 当前有 3 个 val 来源，每个来源 600 张原图，均已使用 `eos_1.0-gate_r2 + rtmpose_onnx` 完成自动标注和 Hand ROI CVAT XML 导出：

- `white-far-dark-random-val-s03-soar`：880 个 ROI/draft；
- `white-mid-dark-random-val-s03-soar`：727 个 ROI/draft；
- `white-near-dark-random-val-s03-soar`：441 个 ROI/draft；
- 合计 1,800 张原图、2,048 个 ROI/draft、3 份 `cvat_autolabel.xml`；RTMPose 与 HCF 均使用 `CPUExecutionProvider`。

该 Eval 数据集尚未导入人工复核 XML，也未执行 `source-publish`，因此当前没有 `dataset_manifest.json`，不会进入下游 HLML 的正式评估数据。

本轮永久删除只作用于 `FullEnhance0801/eos-1.0`。`FullEnhance0801/eos_1.0-gate` 的 95 个 Palm/ROI/labels/发布报告以及所有既有 EValSource 变体均未被该删除操作修改。

Registry 仍保留历史残留 `white-far-bright-random-val-s01-dragon/eos-1.0`（109 ROI），对应来源目录不存在且不在当前 Eval manifest 中；本轮按约定保留。Registry 当前计数为 4 个 dataset、116 个 capture source、72,379 张 raw image、116 个 active proposal variant、95 个 retired proposal variant、1,142,002 个 ROI。active variant 由 11 个 `eos-1.0`、95 个 `eos_1.0-gate` 和 10 个 `eos_1.0-gate_r2` 组成。

## 验收状态

- `make compile`：28 个 Python 文件语法检查通过；
- `make test`：44 项测试通过；
- `make help`：通过；
- 4 个 Bash 批处理脚本均通过 `bash -n`；未注册 Eval 来源发现测试识别 3/3 个仅含 `images/` 的来源；
- 数据集级可视化清理与永久变体删除均在隔离临时数据仓库通过，永久删除后 dataset manifest 已重建；
- 新 HCF ONNX 接口验证通过：动态 batch、`input`、`handedness`、`hand_presence`、float32 与双 `[N,2]` 输出均符合契约；
- 真实 RTMPose+双头 HCF runtime ROI 冒烟通过：21 个有限坐标、有效 handedness 与 `P(has_hand)`；
- `FullEnhance0801/eos-1.0` 批量删除 95/95 成功，7 类目标路径均清零，dataset manifest 已重建且 Registry 为 `retired:95`；
- `FullEnhance0801/eos_1.0-gate` 的 95 个发布来源完整保留；
- `FullEnhanceVal0808/eos_1.0-gate_r2` 自动标注与 CVAT 导出 3/3 成功，日志无 `ERROR` 或 `FAILED`；
- 阈值扫描程序、配置、复制 ROI 和结果全部位于仓库外；
- `requirements.txt` 未改变，无需重建 `anfab` 环境。
