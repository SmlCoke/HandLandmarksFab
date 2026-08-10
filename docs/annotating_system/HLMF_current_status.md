# HLMF 当前状态（2026-08-10）

## 代码与配置

HLMF 3.0 统一入口为 `scripts/hlmf.py` 和 Makefile；公开配置为 `configs/autolabel.yaml`、`configs/review.yaml`、`configs/datasets.yaml`、`configs/cvat_label.json`。MediaPipe Tasks 仍是全局默认 Hand landmark 后端，RTMPose-m Hand5 可通过单次参数启用。

默认 Palm Detector 已升级为 Eos-2.0：模型路径 `models/palm_detector/eos-2.0/model_384x224_opt.onnx`，默认 proposal variant `eos-2.0`，输入 `[1,1,224,384]`，score `0.25`、全局 NMS IoU `0.10`、最多 2 手。两个矩形 feature level 为 `14×24/7×12`，共 840 anchors。ROI 兼容性优先，继续使用 `scale_x=scale_y=1.8、shift_x=0、shift_y=-0.1`。

Eos-2.0 TensorFlow 参考模型参数量为 `1,367,620`，ONNX full checker 通过；17 组随机/真实输入的 TF↔ONNX raw 最大差 `5.4e-7`，CPU↔GPU 为 `5.6e-7`，score `0.15/0.25` 解码数量一致。Eos-2.0 Palm CPU/GPU 吞吐为 `175.0/426.2 images/s`，因此保持 GPU 优先、CPU fallback。完整审计位于 `assets/palm_detector/eos_2_0_adaptation.md`。

RTMPose runtime ROI 已接入 MobileNetV3-Small 双头 HCF。当前模型为 `models/hand_classifier/handedness-handpresence-0809/model.onnx`，模型 ID 从版本目录自动生成，当前为 `hand-classifier-handedness-handpresence-0809`；旧 handedness-only 资产保存在 `models/handedness-0806/`，不再参与推理。服务器主环境现使用 `onnxruntime-gpu==1.18.0`。逐模型实测后，Palm/HCF 默认 CUDA 且不可用时回退 CPU，RTMPose 因 GPU 在 9,868 条人工 gold 上平均点误差增加 `0.0145 px` 而固定 CPU；RTMPose/HCF batch 为 64。完整设备性能与一致性报告位于 `assets/device_perf/onnx_cpu_gpu_benchmark.md`。Eos low-score candidate 仍不运行 RTMPose/HCF。

0809 HCF 自带验证集指标：presence accuracy `0.970930`、presence ROC AUC `0.996065`；handedness accuracy `0.991279`、handedness ROC AUC `0.993745`。presence 混淆矩阵为 `[[288,37],[3,1048]]`（0=no_hand，1=has_hand）。

当前 Train 质量配置：handedness review threshold 为 `0.7`；RTMPose Train runtime 的 `P(has_hand)` 阈值为 `0.025`；42 个 crop 坐标值中精确边界值达到 2 个时拒绝；连接对长度门控默认开启，并按 `near/mid/far` 使用独立阈值。连接长度严格超过阈值时以 `rtmpose_connection_length_gate` 进入 `ignored.jsonl`，等于阈值及长度 0 通过。关闭该开关不影响其余三条门控。Eval、MediaPipe 和 Eos low-score candidate 不应用 RTMPose Train presence/边界/连接长度门控。

发布时已新增四门控互斥统计：每个 `source_publish_report.json` 记录当前 capture source/variant 的四项淘汰数，`dataset_manifest.json` 记录 dataset 总计和每个 `capture_source_id` 合计；多门控同时失败时按既有 `presence → boundary → connection length → handedness/其他` 优先级只归因一次。

RTMPose Train 几何补救现已默认开启：RTMPose 点触发边界或已开启的连接长度门控时，独立 `hlmf-mp-tflite` 环境批量运行 `hand_landmark_full.tflite`；补救点通过两项几何复检才替换原点。TFLite 的 presence/handedness 被丢弃，HCF 仍是唯一置信度来源。补救失败保留原 RTMPose 点。该能力有独立开关，不改变四条门控及拒绝优先级。

服务器已部署 Python 3.11 `venv` 与 `tflite-runtime 2.14.0`。真实只读 ROI 冒烟输出 21 个有限坐标；端到端补救冒烟中，伪造 RTMPose 行同时触发边界和两条连接长度错误，TFLite 复检错误清零，HCF presence/handedness 保持不变，最终以 1 条 positive、0 条 ignored 完成分流。

0809 Presence 阈值复核复用仓库外 `/root/hcf_presence_threshold_0807/data` 中复制出的 `FullEnhanceVal0801` 人工 ROI，没有写入现有数据集。有效样本仍为 7,907 条：7,892 条 hand、15 条 no_hand。0809 的正样本 `P(has_hand)` 均值为 `0.968126`，no_hand 最大值为 `0.00002914`。若沿用 `0.5`，仅保留 7,667/7,892 条 hand（97.149%），最弱来源为 76.97%，因此不可沿用旧模型校准。正式 Train 门控阈值改为 `0.025`：保留 7,859/7,892（99.582%），拒绝全部 15 条 no_hand，最弱来源保留率 96.58%。负样本预审仍使用独立的 `negative_review.hand_presence_threshold=0.5` 作为 argmax 分界。

连接长度统计使用 `FullEnhanceVal0801:eos-1.0` 与 `FullEnhanceVal0808:eos_1.0-gate_r2` 的 13 个已发布来源，共 9,868 条有效 gold hand。阈值为各距离/连接的 `ceil(P99.95 × 1.05)`；gold 回放保留 9,832/9,868（99.635%），RTMPose 草标命中 366 条，其中 352 条后来确有人工修点。Eos-2.0 以旧 gold 做只读兼容回放：9,176 条关联 hand 在 scale `1.8` 下整手入框率为 99.902%，旧门控命中 52 条（0.567%），因此暂留旧阈值并保持门控开启；首个 Eos-2.0 人工复核 Eval 发布后必须正式重算。可复现工具位于 `tools/analyze_rtmpose_connection_lengths.py` 与 `tools/analyze_eos2_adaptation.py`。

原图可视化默认生成按 PNG 文件名字典序排列的 30 FPS MP4；RTMPose ROI 可视化只抽样 runtime ROI。可视化可以独立清理。Registry 使用 source/variant active/retired 状态表，删除变体会留下永久 tombstone。

自动标注批处理现从数据集直接子目录的 `images/` 发现来源，不再要求首次运行前已有 `source.json`。新增数据集级 `batch-autolabel-visualizations-clean` 与 `batch-source-variant-delete`；后者保留精确确认和 retired tombstone，并在批处理末尾重建 dataset manifest。未执行 `source-publish` 的来源仍不会进入 manifest。

服务器 Hand ROI 像素域审查覆盖 Pretrain/Eval 的 71,779 张原图 TIFF，全部为单通道 8-bit、`1280×720`；`FullEnhanceVal0801/eos-1.0` 的 8,248 张 ROI 全部为单通道 8-bit、`256×256` PNG。跨 10 个来源抽查 50 个 ROI，无损 PNG 与无损 TIFF 往返后逐像素一致。Python OpenCV 与板端 C++ 双线性采样的对比中，3,276,800 个像素仅 242 个相差 1 灰度级（0.0074%），差异来自插值舍入而非文件格式；现有 ROI、人工复核结果和发布资产均不重建。

`negative-review` 现先用 HCF 批量预审并显示进度，只复制 `P(has_hand)<0.5` 的候选，同时保存 selected/excluded 清单和分数；人工复核及 `negative-publish` 契约不变。负样本和困难样本的 review/published 图片均使用独立复制；困难样本 published 记录包含 `published_relpath`，不依赖源变体存活。当前服务器既有 `background-neg-0801` review 工作区，以及已发布 1,543 条记录/图片的 `background-neg-0801-full`；本轮均未修改。

## 服务器数据仓库

Eval 数据集 `FullEnhanceVal0801` 当前有 10 个来源：6 个 val、4 个 test，均已发布 `eos-1.0`：

- val：4,200 张原图、5,288 个 ROI、5,091 条发布标签；
- test：2,800 张原图、2,960 个 ROI、2,816 条发布标签。

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

Registry 仍保留历史残留 `white-far-bright-random-val-s01-dragon/eos-1.0`（109 ROI），对应来源目录不存在且不在当前 Eval manifest 中；本轮按约定保留。Registry 当前计数为 4 个 dataset、116 个 capture source、72,379 张 raw image、116 个 active proposal variant、95 个 retired proposal variant、1,142,002 个 ROI。active variant 由 11 个 `eos-1.0`、95 个 `eos_1.0-gate` 和 10 个 `eos_1.0-gate_r2` 组成。

## 验收状态

- `make compile`：33 个 Python 文件语法检查通过；
- `make test`：59 项测试通过；
- `make help`：通过；
- 4 个 Bash 批处理脚本均通过 `bash -n`；未注册 Eval 来源发现测试识别 3/3 个仅含 `images/` 的来源；
- 数据集级可视化清理与永久变体删除均在隔离临时数据仓库通过，永久删除后 dataset manifest 已重建；
- 新 HCF ONNX 接口验证通过：动态 batch、`input`、`handedness`、`hand_presence`、float32 与双 `[N,2]` 输出均符合契约；
- 真实 RTMPose+双头 HCF runtime ROI 冒烟通过：21 个有限坐标、有效 handedness 与 `P(has_hand)`；
- 真实 MediaPipe TFLite worker 与 RTMPose 几何补救冒烟通过：21 个有限坐标、补救复检通过、HCF 字段保持且发布为 positive；
- `FullEnhance0801/eos-1.0` 批量删除 95/95 成功，7 类目标路径均清零，dataset manifest 已重建且 Registry 为 `retired:95`；
- `FullEnhance0801/eos_1.0-gate` 的 95 个发布来源完整保留；
- `FullEnhanceVal0808/eos_1.0-gate_r2` 的 CVAT 导入、发布和 dataset manifest 均已完成；
- 连接长度统计工具对正式 Eval 只读运行，报告与 YAML 的 60 个阈值一致；
- 三个 ONNX 模型完成 CPU/GPU 吞吐对比；Eos-2.0 Palm 2.43×、RTMPose 27.07×、0809 HCF 25.52×，并按精度回放结果选择逐模型默认设备；
- 0809 HCF 的 ONNX full checker、142 节点拓扑、110 个 initializer schema、1,515,612 个参数元素及输入输出契约均与 0807 一致；7,907 条人工 ROI 的阈值与 CPU/GPU 一致性回放通过；
- 0809 HCF 在隔离数据根完成 6 张真实 TIFF 全链路验证：9 条 runtime ROI 使用 CUDA 且 provenance 均为 0809，`0.025` presence 门控拒绝 3 条极低分行，39 条 Eos candidate 未运行 HCF；
- 负样本 `negative-review` 预审核隔离实测加载同一 0809 HCF 并使用 CUDA，返回结果、所选/排除清单及 `README.json` 均记录实际 0809 模型 ID，筛选阈值保持独立的 `0.5`；
- Eos-2.0 模型结构、H5/ONNX 数值和 CPU/GPU 解码一致性通过；307 张测试 TIFF 在 score `0.25`、NMS `0.10` 下得到 543 个 detection、8 张零检测，旧 Eval 兼容回放全程只读；
- 隔离临时仓库真实流水线验证 provider 为 Palm CUDA / RTMPose CPU / HCF CUDA，10 条 runtime ROI 使用 batch 64；负样本 HCF 预审、进度条和两级门控统计字段均通过，临时目录已删除；
- `requirements.txt` 已从 CPU Runtime 改为 `onnxruntime-gpu==1.18.0`，并固定兼容的 NumPy `<2`、OpenCV `<4.11`；服务器 `anfab` 已更新且 `pip check` 通过。TFLite 继续使用独立的 `requirements-mediapipe-tflite.txt` 和 Python 3.11 环境。
