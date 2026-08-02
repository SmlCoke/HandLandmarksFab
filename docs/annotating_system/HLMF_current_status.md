# HLMF 当前状态（2026-08-01）

当前代码已切换到 HLMF 3.0 破坏性数据契约：公共配置按单一职责保留自动标注 `autolabel.yaml`、Hand ROI 复核 `review.yaml`、数据目录 `datasets.yaml` 和 CVAT 标签 schema `cvat_label.json`；公共入口为统一 `scripts/hlmf.py` 与高层 Make 目标。

已实现并测试：TIFF 幂等旋转、稳定 raw/ROI ID、proposal variant 隔离、Train positive/candidate negative 分流、Hand ROI CVAT 来源记录、Val/Test 无候选负样本、真负样本审核树与发布、selection 零拷贝发布与 registry 唯一性。真负样本和困难正样本审核树默认使用服务器硬链接，也允许经压缩包/网盘往返后以保持原相对路径和文件名的普通文件完成发布。

系统当前不存在 Palm CVAT、Palm 标注导入、人工 bbox/p0/p9 修改或人工 ROI 绘制入口。Palm Detector 只负责产生 proposal；CVAT 只复核程序生成的 Hand ROI 内信息。

Palm Detector 的产品名已统一为 **Eos**。当前自动标注链路固定使用 `eos-1.0`，模型路径为 `models/palm_detector/eos-1.0/model_opt.onnx`，默认 `PROPOSAL_VARIANT` 也为 `eos-1.0`。该文件与旧路径 `materials/preminilary/palm/model_opt.onnx` 已执行一次性校验：文件大小均为 5,520,144 字节，SHA-256 均为 `521246FD7CA7F1A10DFB2288683C053852C42C950AAB340DE03CCB6618000E96`；自动标注链路不再引用旧路径。

截至本状态文档更新时，本地 HLMF 单元测试为 14 项通过。服务器 `HAND_DATASET_ROOT` 中的旧数据未迁移、未删除；只有按 3.0 schema 新发布的数据可供 HLML 4.0 选择。
