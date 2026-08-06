# HLMF 当前状态（2026-08-06）

## 代码与配置

HLMF 3.0 统一入口为 `scripts/hlmf.py` 和 Makefile；公开配置为 `configs/autolabel.yaml`、`configs/review.yaml`、`configs/datasets.yaml`、`configs/cvat_label.json`。MediaPipe Tasks 仍是全局默认 Hand landmark 后端，RTMPose-m Hand5 可通过单次参数启用。

RTMPose runtime ROI 已接入 MobileNetV3-Small HCF：模型路径 `models/hand_classifier/model.onnx`，模型 ID `hand-classifier-mobilenetv3-small-v1`。服务器现有 ONNX Runtime 为 CPU 构建，真实双模型冒烟中 RTMPose 与 HCF 的实际 provider 均为 `CPUExecutionProvider`。low-score candidate 不运行两模型，保持未解析状态。

当前 Train 质量配置：handedness review threshold 为 `0.7`；RTMPose Train runtime 行的 42 个 crop 坐标值中，精确边界值达到 3 个时进入 `ignored.jsonl`。Eval、MediaPipe 和 candidate 不应用边界门控。

原图可视化默认生成按 PNG 文件名字典序排列的 30 FPS MP4；RTMPose ROI 可视化只抽样 runtime ROI。可视化可以独立清理。

Registry 已增加 source/variant 的 active/retired 状态表。已有 106 个 ROI 来源/变体已自动回填为 active。删除变体会留下永久 tombstone，禁止同一来源复用同名变体。

负样本和困难样本的 review/published 图片均使用独立复制；困难样本 published 记录包含 `published_relpath`，不依赖源变体存活。当前服务器没有既有 `GoldSource/NegativeSamples` 或 `Selections` 数据需要迁移。

批处理脚本已移入 `scripts/`，由环境变量接收数据根、数据集、变体、后端、Python、仓库和日志目录；任一来源失败时最终返回非零，Train 脚本没有自动关机。

## 服务器数据仓库

Eval 数据集 `FullEnhanceVal0801` 当前有 10 个来源：6 个 val、4 个 test，均已发布 `eos-1.0`：

- val：4200 张原图、5288 个 ROI、5091 条发布标签；
- test：2800 张原图、2960 个 ROI、2816 条发布标签。

Pretrain 数据集 `FullEnhance0801` 当前有 95 个 train 来源，均已发布 `eos-1.0`。合计 105 个 manifest 中已发布来源变体的必要文件全部存在。本轮没有删除、重跑或改写这些发布资产。

Registry 仍保留历史残留 `white-far-bright-random-val-s01-dragon/eos-1.0`（109 ROI）。对应来源目录不存在，且该记录不在当前 Eval manifest 中；本轮按约定保留，没有自动清理。

Registry 当前计数：2 个 dataset、106 个 capture source、68479 张 raw image、106 个 active proposal variant、572600 个 ROI。

## 验收状态

- `make compile`：27 个 Python 文件语法检查通过；
- `make test`：37 项测试通过；
- `make help`：通过；
- 两个批处理脚本 `bash -n` 通过，缺失数据集时返回非零；
- 真实 RTMPose+HCF runtime ROI 冒烟通过：21 个有限坐标、有效 Left/Right 概率；
- 临时 PNG→MP4 冒烟通过：`mp4v`、30 FPS、帧数正确；
- 实施前后 Eval 计数与 105 个已发布变体必要文件完整性保持不变；
- `requirements.txt` 未改变，无需重建 `anfab` 环境。
