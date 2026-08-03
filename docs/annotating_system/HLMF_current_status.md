# HLMF 当前状态（2026-08-03）

当前代码已切换到 HLMF 3.0 破坏性数据契约：公共配置按单一职责保留自动标注 `autolabel.yaml`、Hand ROI 复核 `review.yaml`、数据目录 `datasets.yaml` 和 CVAT 标签 schema `cvat_label.json`；公共入口为统一 `scripts/hlmf.py` 与高层 Make 目标。

已实现并测试：TIFF 幂等旋转、稳定 raw/ROI ID、proposal variant 隔离、Train positive/candidate negative 分流、Hand ROI CVAT 来源记录、Val/Test 无候选负样本、真负样本审核树与发布、selection 零拷贝发布与 registry 唯一性。真负样本和困难正样本审核树默认使用服务器硬链接，也允许经压缩包/网盘往返后以保持原相对路径和文件名的普通文件完成发布。

Train 与 Val/Test 的高层 autolabel 现已在来源检查、Palm 推理、ROI 裁剪和 MediaPipe Hand Landmarker 推理阶段显示 tqdm 进度。Hand ROI 可视化接口已统一为 `visualization.roi_enabled`、`ROI_VISUALIZATION` 与 `make autolabel-visualize-roi`；Train 使用最多 200 张的确定性等距抽样审核图，Val/Test 输出全部实际 ROI 审核图，CVAT 导入导出不重复可视化。

已完成自动标注但未生成 Hand ROI 审核图的来源，可通过 `make autolabel-visualize-roi ...` 直接读取现有 MediaPipe draft 补生成，不重跑自动标注链路；该入口明确启用绘图，不受全局开关当前值影响。

原图可视化分支已接入 Train/Eval autolabel：`visualization.original_image_enabled` 默认为 `false`，单次可用 `ORIGINAL_VISUALIZATION=true|false` 覆盖。启用后使用 draft 的 `landmarks_image_px` 在来源全部原图上绘制关键点，输出到 `visualizations/original_image_landmarks/<variant>/`；输出统一为与原图同 stem、压缩级别 3 的 PNG，便于逐图比较不同阈值变体并减小存储和传输体积。已有 draft 可通过 `make autolabel-visualize-original ...` 直接补生成。

CVAT Images 1.1 导出已修复 frame ID 与 ROI 上传顺序不一致的问题：XML 现按 crop 文件名字典序编号，并强制 manifest/draft 一一对应及 positive landmark ID 完整。创建 CVAT 任务时必须使用 `Lexicographical` 排序，防止 skeleton 或 `no_hand` 被绑定到其他 ROI。

系统当前不存在 Palm CVAT、Palm 标注导入、人工 bbox/p0/p9 修改或人工 ROI 绘制入口。Palm Detector 只负责产生 proposal；CVAT 只复核程序生成的 Hand ROI 内信息。

Palm Detector 的产品名已统一为 **Eos**。当前自动标注链路固定使用 `eos-1.0`，模型路径为 `models/palm_detector/eos-1.0/model_opt.onnx`，默认 `PROPOSAL_VARIANT` 也为 `eos-1.0`。该文件与旧路径 `materials/preminilary/palm/model_opt.onnx` 已执行一次性校验：文件大小均为 5,520,144 字节，SHA-256 均为 `521246FD7CA7F1A10DFB2288683C053852C42C950AAB340DE03CCB6618000E96`；自动标注链路不再引用旧路径。

截至本状态文档更新时，本地 HLMF 单元测试为 22 项通过。服务器 `HAND_DATASET_ROOT` 中的旧数据未迁移、未删除；只有按 3.0 schema 新发布的数据可供 HLML 4.0 选择。
