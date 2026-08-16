# HLMF 当前状态（2026-08-16）

## 代码与配置

HLMF 3.0 统一入口为 `scripts/hlmf.py` 和 Makefile；公开配置为 `configs/autolabel.yaml`、`configs/review.yaml`、`configs/datasets.yaml`、`configs/cvat_label.json`。默认标注链路已切换为 RTMPose-m Hand5 + 双头 Hand Classifier + 四项质量门控 + MediaPipe Hand Landmarker TFLite 几何补救；MediaPipe Tasks 仍可作为单次显式覆盖。

2026-08-16 新增两个与训练 run 解耦的 Gold 发布合同：困难 ROI 经 CVAT 1.1 精修后发布到 `GoldSource/HardSamples/<hard_dataset_id>`；新录制 train 来源复用 Eval 的自动标注与人工 CVAT 链路，positive/negative 一并发布到 `GoldSource/ReviewedDatasets/<dataset_id>`。旧 `Selections` 已发布资产仍可被 HLML 读取，新困难复核不再使用删除式流程。Registry 通过新增表扩展，不迁移或改写既有 EValSource/PretrainSource 资产。

默认 Palm Detector 已升级为 Eos-2.1：模型路径 `models/palm_detector/eos-2.1/model_384x224_opt.onnx`，默认 proposal variant `eos-2.1`，输入 `[1,1,224,384]`，score `0.25`、全局 NMS IoU `0.10`、最多 2 手、negative candidate 下限 `0.15`。两个矩形 feature level 为 `14×24/7×12`，共 840 anchors。7 个最新 Gold 来源的只读回放关联 9,237/9,237 个 hand；ROI 保持 `scale_x=scale_y=1.8、shift_x=0、shift_y=-0.1`，单点入框率为 99.999%。

Eos-2.1 的能力契约仍为 `supported_capture_distances: [near, mid]`。历史 far 人工标签兼容回放在 score 0.25 下的召回分别只有 55.115% 和 37.745%，不足以启用 far。单来源 Palm→ROI→Landmark→CVAT→发布命令在写入前硬拒绝 far；Train/Eval 批处理显式跳过并汇总 far。far 原始来源和历史已发布资产保留，不做追溯删除。

Eos-2.1 ONNX full checker 通过，接口为 153 个节点、160 个 initializer、1,368,068 个 initializer element。该模型没有配套 TensorFlow/H5 参考，因此不宣称 TF↔ONNX 一致性；128 张真实输入的 CPU↔GPU raw 最大差为 `7.15e-07`，正式解码 0 次不一致。CPU/GPU 吞吐为 `199.3/573.2 images/s`，保持 GPU 优先、CPU fallback。完整审计位于 `assets/palm_detector/eos_2_1_adaptation.md`。

RTMPose runtime ROI 当前使用 `models/hand_classifier/handedness-handpresence-0814/model.onnx`，模型 ID 为 `hand-classifier-handedness-handpresence-0814`；旧 HCF 不再参与推理。服务器主环境使用 `onnxruntime-gpu==1.18.0`。Palm/HCF 默认 CUDA 且不可用时回退 CPU，RTMPose 模型未变并沿用既有人工 Gold 精度结论固定 CPU；RTMPose/HCF batch 为 64。完整 HCF 校准与设备报告位于 `assets/hand_classifier/handedness_handpresence_0814.md` 和 `assets/device_perf/onnx_cpu_gpu_benchmark.md`。Eos low-score candidate 仍不在自动标注阶段运行 RTMPose/HCF。

0814 HCF 自带验证集指标：presence accuracy `0.984676`、presence ROC AUC `0.997041`、混淆矩阵 `[[369,40],[7,2651]]`；handedness accuracy `0.971783`、ROC AUC `0.996871`、混淆矩阵 `[[1345,54],[21,1238]]`。0814 的 train/validation split 已覆盖本次 7 个 Gold 来源，因此统一回放用于配置校准，不是独立盲测，结果可能乐观。

当前 Train 质量配置：handedness review threshold 为 `0.8`；RTMPose Train runtime 的 `P(has_hand)` 阈值为 `0.025`；42 个 crop 坐标值中精确边界值达到 2 个时拒绝；连接对长度门控默认开启，并按 `near/mid/far` 使用独立阈值。连接长度严格超过阈值时以 `rtmpose_connection_length_gate` 进入 `ignored.jsonl`，等于阈值及长度 0 通过。关闭该开关不影响其余三条门控。Eval、MediaPipe 和 Eos low-score candidate 不应用 RTMPose Train presence/边界/连接长度门控。

发布时已新增四门控互斥统计：每个 `source_publish_report.json` 记录当前 capture source/variant 的四项淘汰数，`dataset_manifest.json` 记录 dataset 总计和每个 `capture_source_id` 合计；多门控同时失败时按既有 `presence → boundary → connection length → handedness/其他` 优先级只归因一次。

RTMPose Train 几何补救现已默认开启：RTMPose 点触发边界或已开启的连接长度门控时，独立 `hlmf-mp-tflite` 环境批量运行 `hand_landmark_full.tflite`；补救点通过两项几何复检才替换原点。TFLite 的 presence/handedness 被丢弃，HCF 仍是唯一置信度来源。补救失败保留原 RTMPose 点。该能力有独立开关，不改变四条门控及拒绝优先级。

服务器已部署 Python 3.11 `venv` 与 `tflite-runtime 2.14.0`。真实只读 ROI 冒烟输出 21 个有限坐标；端到端补救冒烟中，伪造 RTMPose 行同时触发边界和两条连接长度错误，TFLite 复检错误清零，HCF presence/handedness 保持不变，最终以 1 条 positive、0 条 ignored 完成分流。

0814 阈值复核只读使用 9,279 条 near/mid 人工 Gold ROI：9,237 hand、42 no_hand。Train presence `0.025` 保留全部 hand，并拒绝 41/42 no_hand（97.619%）；提高到 `0.5` 会漏掉 4 条 hand，因此保持 `0.025`。负样本预审仍使用独立 `0.5` 并由人工兜底。handedness `0.8` 覆盖 9,099/9,237（98.506%），覆盖内准确率为 99.275%，较 0.7 减少 21 个覆盖内错误；逐来源覆盖率最低为 97.027%。

连接长度门控已针对 Eos-2.1 ROI 正式重算：程序在 7 个指定 `FullEnhanceVal0801` Gold 来源的原图上重新检测，把 9,237 条人工 image-space 关键点投影到新 crop，其中 near 2,345、mid 6,892。阈值继续采用各连接 `ceil(P99.95 × 1.05)`；Gold 保留 9,203/9,237（99.632%）。同一 ROI 上的 RTMPose CPU 回放命中 454 条，其中 243 条平均点误差至少 10 px。far 无正式 2.1 Gold，仅保留距离硬门控后不可达的历史阈值。可复现工具位于 `tools/analyze_eos_hcf_recalibration.py`，完整结果位于 `assets/quality_gate/rtmpose_connection_length_distribution.md`。

原图可视化默认生成按 PNG 文件名字典序排列的 30 FPS MP4；RTMPose ROI 可视化只抽样 runtime ROI。可视化可以独立清理。Registry 使用 source/variant active/retired 状态表，删除变体会留下永久 tombstone。

自动标注批处理现从数据集直接子目录的 `images/` 发现来源，不再要求首次运行前已有 `source.json`。批处理先执行统一 Palm 距离预检，near/mid 正常处理，far 显示 `SKIPPED_UNSUPPORTED_DISTANCE`；若至少存在一个兼容来源且无真实失败，跳过 far 不影响成功状态。数据集级 `batch-autolabel-visualizations-clean` 与 `batch-source-variant-delete` 保持独立；后者保留精确确认和 retired tombstone，并在批处理末尾重建 dataset manifest。未执行 `source-publish` 的来源仍不会进入 manifest。

服务器 Hand ROI 像素域审查覆盖 Pretrain/Eval 的 71,779 张原图 TIFF，全部为单通道 8-bit、`1280×720`；`FullEnhanceVal0801/eos-1.0` 的 8,248 张 ROI 全部为单通道 8-bit、`256×256` PNG。跨 10 个来源抽查 50 个 ROI，无损 PNG 与无损 TIFF 往返后逐像素一致。Python OpenCV 与板端 C++ 双线性采样的对比中，3,276,800 个像素仅 242 个相差 1 灰度级（0.0074%），差异来自插值舍入而非文件格式；现有 ROI、人工复核结果和发布资产均不重建。

`negative-review` 现先用 HCF 批量预审并显示进度，只复制 `P(has_hand)<0.5` 的候选，同时保存 selected/excluded 清单和分数；人工复核及 `negative-publish` 契约不变。负样本和困难样本的 review/published 图片均使用独立复制；困难样本 published 记录包含 `published_relpath`，不依赖源变体存活。当前服务器既有 `background-neg-0801` review 工作区，以及已发布 1,543 条记录/图片的 `background-neg-0801-full`；本轮均未修改。

2026-08-14 已发布负样本集 `neg-eos_2.0-hcf0813-hp0.5`：HCF 0.5 预审从 573,786 个输入候选选择 167,760 个供人工复核，最终保留 16,910 个真负样本并复制到 published，删除 150,850 个有手或不确定候选。`complex-mid-bright-flat-train-s01-soar` 的 1,136 个候选全部被人工删除；该来源零保留不会阻止整个数据集发布，最终 manifest 含 91 个非空来源、16,910 条 label/图片，Registry 状态为 `published`。

## 服务器数据仓库

Eval 数据集 `FullEnhanceVal0801` 当前有 10 个来源：6 个 val、4 个 test，均已发布 `eos-1.0`：

- val：4,200 张原图、5,288 个 ROI、5,091 条发布标签；
- test：2,800 张原图、2,960 个 ROI、2,816 条发布标签。

同一数据集当前作为配置校准 Gold 标准的 7 个 near/mid 来源均已发布，共 9,279 条 Gold 标签（9,237 hand、42 no-hand）。其中 `eos_2.0-rtmpose-hcf0813-gate` 有 3 个来源、3,484 条：

- `complex-mid-bright-random-val-s01-peak`：1,170 条 Gold；
- `complex-mid-bright-random-val-s01-soar`：1,145 条 Gold（1,115 hand、30 no-hand）；
- `complex-near-bright-random-val-s01-peak`：1,169 条 Gold。

`eos_2.0-rtmpose-gate` 有 4 个来源、5,795 条：

- `complex-mid-bright-random-test-s01-peak`：1,166 条 Gold；
- `complex-mid-dark-random-val-s01-peak`：2,302 条 Gold；
- `complex-near-bright-random-test-s01-peak`：1,176 条 Gold；
- `white-mid-bright-random-val-s01-soar`：1,151 条 Gold。

这 7 组标签仍属于历史 Eos-2.0 发布变体；本轮 Eos-2.1 校准只读使用原图和人工 image-space 点，没有创建或发布 Eos-2.1 数据变体。其余来源仍未发布。两个 far 草稿分别为 `complex-far-bright-random-test-s01-peak`（1,005 ROI）和 `complex-far-bright-random-val-s01-peak`（985 ROI），已知含大量无手 ROI，不得复核或发布。未发布 near/mid 草稿可沿用原 variant 继续复核；若从头重跑，必须退役旧变体并使用新名称，不能混合新旧结果。

独立 Eval `RTMPose-Finetune-Test-0812` 已发布 `complex-mid-bright-random-test-s05-peak:rtmpose-finetune-test`：200 张原图、396 条 Gold（312 hand、84 no-hand）。它未进入 RTMPose/Iris 训练，可作为 Iris 关键点 frozen test；但它出现在 HCF 0813 validation 与连接长度门控尺度校准中，因此不是整条端到端链路完全未接触的盲测。

Pretrain 数据集 `FullEnhance0801` 当前有 95 个 train 来源：

- `eos-1.0`：95 个来源的变体派生产物已全部删除，Registry 保留 95 个 retired tombstone；原始 `images/`、`raw_images.jsonl`、`source.json` 和既有 ROI Registry 元数据均保留；
- `eos_1.0-gate`：95 个来源均完整发布，共 564,243 个 ROI、65,089 条 positive、492,017 条 candidate、7,137 条 ignored。

Eval 数据集 `FullEnhanceVal0808` 当前有 3 个 val 来源，每个来源 600 张原图，均已使用 `eos_1.0-gate_r2 + rtmpose_onnx` 完成人工复核和发布：

- `white-far-dark-random-val-s03-soar`：880 个 ROI，823 条发布标签、57 条 ignored；
- `white-mid-dark-random-val-s03-soar`：727 个 ROI，726 条发布标签、1 条 ignored；
- `white-near-dark-random-val-s03-soar`：441 个 ROI，440 条发布标签、1 条 ignored；
- 合计 1,800 张原图、2,048 个 ROI、1,989 条发布标签和 59 条 ignored。

该数据集已有 `dataset_manifest.json`，下游 HLML 可读取其正式评估标签。本次连接长度统计从其中排除 13 条 no_hand，使用 1,976 条有效 gold hand。

本轮永久删除只作用于 `FullEnhance0801/eos-1.0`。`FullEnhance0801/eos_1.0-gate` 的 95 个 Palm/ROI/labels/发布报告以及所有既有 EValSource 变体均未被该删除操作修改。

Registry 仍保留历史残留 `white-far-bright-random-val-s01-dragon/eos-1.0`（109 ROI），对应来源目录不存在且不在当前 Eval manifest 中；本轮按约定保留。Registry 当前计数为 7 个 dataset、138 个 capture source、85,779 张 raw image、222 个 active proposal variant、95 个 retired proposal variant、1,825,896 个 ROI；另有 4 个 negative dataset（3 个 published、1 个 reserved）、18,734 条已发布 negative，尚无 published selection。

## 验收状态

- `make compile`：39 个 Python 文件通过语法检查，新增统一重标定工具已纳入；
- `make test`：73 项测试通过；
- `make help`：通过；
- 4 个 Bash 批处理脚本均通过 `bash -n`；Palm 距离门控的 near/mid、far/unknown、非法配置、写入前拒绝和批处理接口测试通过；
- 服务器隔离临时数据根的真实混合批次通过：Eval discovered/supported/skipped 为 `3/2/1`，Train 为 `2/1/1`；far 未产生 proposal 目录，单来源自动标注和发布均被拒绝，near/mid 的 Palm/发布 QC 均记录完整能力契约；临时目录已删除，正式仓库只读；
- 数据集级可视化清理与永久变体删除均在隔离临时数据仓库通过，永久删除后 dataset manifest 已重建；
- 新 HCF ONNX 接口验证通过：动态 batch、`input`、`handedness`、`hand_presence`、float32 与双 `[N,2]` 输出均符合契约；
- 真实 RTMPose+双头 HCF runtime ROI 冒烟通过：21 个有限坐标、有效 handedness 与 `P(has_hand)`；
- 真实 MediaPipe TFLite worker 与 RTMPose 几何补救冒烟通过：21 个有限坐标、补救复检通过、HCF 字段保持且发布为 positive；
- `FullEnhance0801/eos-1.0` 批量删除 95/95 成功，7 类目标路径均清零，dataset manifest 已重建且 Registry 为 `retired:95`；
- `FullEnhance0801/eos_1.0-gate` 的 95 个发布来源完整保留；
- `FullEnhanceVal0808/eos_1.0-gate_r2` 的 CVAT 导入、发布和 dataset manifest 均已完成；
- 统一重标定工具对 7 个指定 Gold 来源只读运行；Eos-2.1 重建 ROI 的 near/mid 报告与 YAML 的 40 个新阈值一致，far 的 20 个历史值被明确标为无 2.1 正式样本；
- 三个 ONNX 模型的设备结论已同步；Eos-2.1 Palm CPU/GPU 为 `199.3/573.2 images/s`，HCF0814 为 `333.9/8478.6 images/s`，RTMPose 沿用上一轮精度回放并固定 CPU；
- HCF0814 的 ONNX full checker、142 节点、110 个 initializer、1,515,612 个参数元素及输入输出契约通过；9,279 条人工 Gold 的 CPU/GPU argmax 与 presence 0.025/0.5、handedness 0.8 门控均 0 次跨设备变化；
- HCF0814 在 9,237 个 Eos-2.1 重建 positive ROI 上保持 presence 0.025 全保留；采用的 handedness 0.8 在正式 Gold 上覆盖 98.506%、覆盖内准确率 99.275%；
- Eos-2.1 ONNX full checker、Anchor/score/NMS/max/ROI、历史 far 兼容性与 CPU/GPU 解码一致性回放通过；score `0.25` 在 near/mid Gold 上关联 9,237/9,237 个 hand；
- 隔离临时仓库真实流水线验证 provider 为 Palm CUDA / RTMPose CPU / HCF CUDA，10 条 runtime ROI 使用 batch 64；负样本 HCF 预审、进度条和两级门控统计字段均通过，临时目录已删除；
- `requirements.txt` 已从 CPU Runtime 改为 `onnxruntime-gpu==1.18.0`，并固定兼容的 NumPy `<2`、OpenCV `<4.11`；服务器 `anfab` 已更新且 `pip check` 通过。TFLite 继续使用独立的 `requirements-mediapipe-tflite.txt` 和 Python 3.11 环境。
