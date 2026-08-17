# HaMeR 标注后端接入报告

## 结论

HaMeR 已作为 `HAND_LANDMARK_BACKEND=hamer` 的独立第三种 Hand landmark 后端接入 HLMF。默认后端仍为 `rtmpose_onnx`，没有自动融合或替换既有 RTMPose/MediaPipe 资产。

该后端只负责 21 个关键点。HLMF 主环境中的双头 Hand Classifier 仍是 presence 与 handedness 的唯一教师，HCF Left/Right 直接决定 HaMeR 的左右手翻转；HaMeR 工具自带的 ViTPose 与亮度 handedness fallback 不进入 HLMF。这使关键点采用的手性与标签/provenance 保持一致。

## 模型与环境

- HaMeR repository：`/root/autodl-tmp/HLMF-Enhance/hamer`
- repository HEAD/config lock：`603105f586337aa704483c4525f5934047f02f2b`
- HLMF model ID：`hamer-cvpr24-official-603105f`
- checkpoint：`_DATA/hamer_ckpts/checkpoints/hamer.ckpt`
- MANO：`_DATA/data/mano/MANO_RIGHT.pkl`
- Python：`hamer/.hamer/bin/python`（Python 3.8.10）
- runtime：CUDA，`rescale=0.75`
- HCF：`/root/autodl-tmp/HLMF-Enhance/hand_classifier/handedness-handpresence-0814/model.onnx`

HaMeR 的 PyTorch、Detectron2、ViTPose、MANO 等依赖保持在 `.hamer`，未并入 `anfab`；HLMF `requirements.txt` 因此不变。HLMF 通过 JSONL request/response worker 一次加载 HaMeR 模型并处理一个来源的全部 runtime ROI。

## 输入、输出与 provenance

输入仍是 Eos-2.1 生成的单通道 `uint8 256×256` Hand ROI。worker 复用 HaMeR 仓库 `predict_hand_keypoints.py` 的 full-ROI ViTDet 预处理、MANO 回归和逆仿射投影，输出 OpenPose/COCO-wholebody 顺序的 21 个 2D ROI 像素点。

HaMeR 2D 输出裁到 `[0,255]` 后写入：

- `source=hamer_official_cvpr24`
- `label_origin=hamer`
- `annotation_style=hamer_openpose21_v1`
- `teacher_model_id=hamer-cvpr24-official-603105f`
- `hamer_inference={model_id,device,rescale,flipped,bbox_size,clipped_coordinate_values,handedness_source}`

HCF 的 `handedness_teacher_model_id` 与 `hand_presence_teacher_model_id` 由外部模型版本目录产生。low-score Eos candidate 仍不运行 HaMeR/HCF，保持 unresolved candidate 契约。

## 质量门控与补救

HaMeR Train runtime 使用与 RTMPose 相同的四项门控：

1. HCF `P(has_hand)`；
2. HCF handedness score；
3. 42 个 crop 坐标的精确边界值数量；
4. 按 capture distance 选择的 20 个连接长度阈值。

HaMeR 错误和 `ignore_reason` 使用 `hamer_*` 前缀，来源与 dataset 报告仍按 `hand_presence/boundary_coordinate/connection_length/handedness` 四项互斥聚合，拒绝优先级保持 presence → boundary → connection → handedness/other。

边界或连接门控失败时继续使用既有 MediaPipe Hand Landmarker TFLite rescue。成功行切换为 `mediapipe_tflite_rescue_v1` provenance，但保留 HCF 输出和 `hamer_geometry_rescue`；失败行保留 HaMeR 点。

## 阈值状态

本轮没有更新 Eval 或 Hand Classifier，按任务要求不修改现有阈值：presence `0.025`、handedness `0.8`、boundary `2` 及 Eos-2.1 ROI 的 near/mid connection thresholds。

Presence/handedness 仍有同一 HCF0814 的校准依据，geometry thresholds 仍绑定同一 Eos-2.1 ROI；但是 HaMeR 权重的输出分布尚未使用新的人审代表性 Eval 正式重校准。因此本次只确认门控已正确接入，不能把现有阈值表述为 HaMeR 模型版本的最终校准结论。批量生产前应先建立并人工复核代表性 HaMeR Eval/Gold，再做只读正式复核。

## 融合策略

本次不实现“RTMPose 与 HaMeR 自动二选一”。没有人工真值时，模型间分歧不能可靠判断哪一个正确，直接按置信度或骨架长度选择可能静默引入错误标签。

推荐先把 HaMeR 用于独立 proposal variant，覆盖人工或既有困难挖掘确认的 RTMPose 困难来源。后续若要做自动补充，安全的第一步是仅在 RTMPose 明确触发 geometry gate 时尝试 HaMeR，并要求 HaMeR 通过全部四门控；对两者都通过但差异较大的样本进入人工复核，而不是自动选边。

## 验证

- 新增 worker 协议、HCF 权威手性、21 点/provenance、candidate skip、四门控互斥分流和 TFLite rescue 单元测试。
- 本地合成完整测试：79 项通过。
- 服务器真实 ROI 模块冒烟：`.hamer` CUDA + HCF CUDA 成功输出 21 个有限点；HaMeR model ID、repository commit、HCF0814 teacher ID、Left flip 与 `rescale=0.75` 均按配置写入。
- 服务器隔离完整流水线：从正式仓库只读复制 1 张 TIFF 到临时 `HAND_DATASET_ROOT`，执行 Eos-2.1 → 12 ROI（2 runtime、10 low-score candidate）→ HaMeR/HCF → 发布；2 条 runtime 均为 21 个有限点并发布，四门控拒绝均为 0，candidate 保持 unresolved 且不运行 HaMeR/HCF。临时根在核对报告后删除。
- 正式 `HAND_DATASET_ROOT` 未执行自动标注、发布或 manifest 重建，既有数据资产保持只读。
