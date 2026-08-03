# HLMF 当前状态（2026-08-03）

当前代码已切换到 HLMF 3.0 破坏性数据契约：公共配置按单一职责保留自动标注 `autolabel.yaml`、Hand ROI 复核 `review.yaml`、数据目录 `datasets.yaml` 和 CVAT 标签 schema `cvat_label.json`；公共入口为统一 `scripts/hlmf.py` 与高层 Make 目标。

已实现并测试：TIFF 幂等旋转、稳定 raw/ROI ID、proposal variant 隔离、Train positive/candidate negative 分流、Hand ROI CVAT 来源记录、Val/Test 无候选负样本、真负样本审核树与发布、selection 零拷贝发布与 registry 唯一性。真负样本和困难正样本审核树默认使用服务器硬链接，也允许经压缩包/网盘往返后以保持原相对路径和文件名的普通文件完成发布。

Train 与 Val/Test 的高层 autolabel 现已在来源检查、Palm 推理、ROI 裁剪和 Hand landmark 推理阶段显示 tqdm 进度。默认教师仍为 MediaPipe Tasks；已正式集成可选 `rtmpose_onnx`，支持 YAML 全局选择及 CLI/Make 单次覆盖，未知后端直接报错。Hand ROI 可视化接口已统一为 `visualization.roi_enabled`、`ROI_VISUALIZATION` 与 `make autolabel-visualize-roi`；Train 使用最多 200 张的确定性等距抽样审核图，Val/Test 输出全部实际 ROI 审核图，CVAT 导入导出不重复可视化。

已完成自动标注但未生成 Hand ROI 审核图的来源，可通过 `make autolabel-visualize-roi ...` 直接读取现有 Hand landmark draft 补生成，不重跑自动标注链路；该入口明确启用绘图，不受全局开关当前值影响。

原图可视化分支已接入 Train/Eval autolabel：`visualization.original_image_enabled` 默认为 `false`，单次可用 `ORIGINAL_VISUALIZATION=true|false` 覆盖。启用后使用 draft 的 `landmarks_image_px` 在来源全部原图上绘制关键点，输出到 `visualizations/original_image_landmarks/<variant>/`；输出统一为与原图同 stem、压缩级别 3 的 PNG，便于逐图比较不同阈值变体并减小存储和传输体积。已有 draft 可通过 `make autolabel-visualize-original ...` 直接补生成。

CVAT Images 1.1 导出已修复 frame ID 与 ROI 上传顺序不一致的问题：XML 现按 crop 文件名字典序编号，并强制 manifest/draft 一一对应及 positive landmark ID 完整。创建 CVAT 任务时必须使用 `Lexicographical` 排序，防止 skeleton 或 `no_hand` 被绑定到其他 ROI。

系统当前不存在 Palm CVAT、Palm 标注导入、人工 bbox/p0/p9 修改或人工 ROI 绘制入口。Palm Detector 只负责产生 proposal；CVAT 只复核程序生成的 Hand ROI 内信息。

Palm Detector 的产品名已统一为 **Eos**。当前自动标注链路固定使用 `eos-1.0`，模型路径为 `models/palm_detector/eos-1.0/model_opt.onnx`，默认 `PROPOSAL_VARIANT` 也为 `eos-1.0`。该文件与旧路径 `materials/preminilary/palm/model_opt.onnx` 已执行一次性校验：文件大小均为 5,520,144 字节，SHA-256 均为 `521246FD7CA7F1A10DFB2288683C053852C42C950AAB340DE03CCB6618000E96`；自动标注链路不再引用旧路径。

RTMPose-m Hand5 转换已通过审查并正式接入：仓库部署路径为 `models/rtmpose/rtmpose-m_hand5_256x256.onnx`，独立文件大小 55,119,819 字节，SHA-256 为 `45206307fe9fca886f9a9e3b6d335370b43083924ae78042b77ee771132cbaa3`；输入为 `[N,3,256,256]`，输出为两个 `[N,21,512]` SimCC 张量。服务器旧 `.onnx.data` 是无效残留，不属于部署资产。运行时按 CUDA→CPU 顺序选择可用 provider，并记录到现有 QC 报告路径。

RTMPose 当前只强制标注 Eos runtime ROI，低分候选不推理并以 `unresolved/unlabeled_v1` 继续人工负样本审核。其 `hand_presence.present=true` 仅是发布路由值，`handedness=unknown/null`；当前训练边界是 Iris geometry pretrain 必须忽略 presence/handedness，后续 multitask 和正式评估仍需独立分类器或人工真值。

截至本状态文档更新时，本地 HLMF 单元测试为 30 项通过，覆盖 RTMPose 预处理、SimCC 解码、形状/有限值检查、后端选择、候选分流和 provenance。服务器 `HAND_DATASET_ROOT` 中的旧数据未迁移、未删除；只有按 3.0 schema 新发布的数据可供 HLML 4.0 选择。
