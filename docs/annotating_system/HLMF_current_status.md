# HLMF 当前状态（2026-08-01）

当前代码已切换到 HLMF 3.0 破坏性数据契约：公共配置按单一职责保留自动标注 `autolabel.yaml`、Hand ROI 复核 `review.yaml`、数据目录 `datasets.yaml` 和 CVAT 标签 schema `cvat_label.json`；公共入口为统一 `scripts/hlmf.py` 与高层 Make 目标。

已实现并测试：TIFF 幂等旋转、稳定 raw/ROI ID、proposal variant 隔离、Train positive/candidate negative 分流、Hand ROI CVAT 来源记录、Val/Test 无候选负样本、真负样本审核树与发布、selection 零拷贝发布与 registry 唯一性。真负样本和困难正样本审核树默认使用服务器硬链接，也允许经压缩包/网盘往返后以保持原相对路径和文件名的普通文件完成发布。

系统当前不存在 Palm CVAT、Palm 标注导入、人工 bbox/p0/p9 修改或人工 ROI 绘制入口。Palm Detector 只负责产生 proposal；CVAT 只复核程序生成的 Hand ROI 内信息。

截至本状态文档更新时，本地 HLMF 单元测试为 14 项通过。服务器 `HAND_DATASET_ROOT` 中的旧数据未迁移、未删除；只有按 3.0 schema 新发布的数据可供 HLML 4.0 选择。
