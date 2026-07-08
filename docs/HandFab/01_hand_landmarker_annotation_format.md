# Hand Landmarker 标注文件建议

本文回答问题 (1)：训练 Hand Landmarker 需要什么格式的标注文件。

结论：只有一堆未标注的 `.tiff` 灰度图，不能直接监督训练 Hand Landmarker。至少需要每只手的 21 个 2D 关键点；如果要让训练分布贴近最终端到端 pipeline，还需要一份 Palm ROI 辅助文件，用来复现训练和推理时的 Hand crop 几何。

当前统一约定：进入本轮标注、训练、评估和 PC 端推理的数据图片已经是 `1280x720` 的正向灰度图。所有 bbox、landmark、ROI 坐标都以这张 `1280x720` 正向图为准。训练工程中不再编写或依赖原始图片旋转脚本。

## 1. 必须标什么

每张图可以有 0、1、2 只手。对每一只可见手，建议标注：

- `image`：图像文件名，例如 `seq001_f000123.tiff`。
- `width`、`height`：固定记录为 `1280`、`720`。
- `handedness`：左手或右手。训练时可映射为 `Left=0`、`Right=1`；若暂时不确定，可填 `unknown`，并在 handedness loss 中降权或跳过。
- `landmarks`：MediaPipe 手部 21 点，全部标在 `1280x720` 正向图坐标系中。
- `visible` 或 `valid`：每个点是否可见/可信。遮挡点或低置信点建议保留字段，训练时可降低权重。

注意：Palm Detector 的输出不能提供左手/右手标签。Palm 只判断候选区域是否为手掌，并回归 palm bbox 与少量关键点；左右手监督必须来自人工标注，或者来自 Hand Landmarker 的 handedness head 训练结果。

21 点顺序必须固定：

| id | 含义 |
|---:|---|
| 0 | wrist |
| 1-4 | thumb CMC/MCP/IP/tip |
| 5-8 | index MCP/PIP/DIP/tip |
| 9-12 | middle MCP/PIP/DIP/tip |
| 13-16 | ring MCP/PIP/DIP/tip |
| 17-20 | pinky MCP/PIP/DIP/tip |

灰度图本身没有问题，Hand Landmarker 的输入也是灰度 ROI。关键是坐标系必须统一：人工标注、Palm ROI、Hand ROI 裁剪、训练目标、可视化和评估都使用同一套 `1280x720` 正向图坐标。

## 2. 推荐主标注格式

建议主标注保存为 JSONL，一行一张图。JSONL 比纯 txt 更不容易产生歧义，后续再写 converter 转成任意训练脚本需要的格式。

```json
{"image":"seq001_f000123.tiff","width":1280,"height":720,"orientation":"upright_1280x720","hands":[{"hand_id":"h0","handedness":"Right","landmarks":[{"id":0,"x":519.9,"y":335.2,"visible":1},{"id":1,"x":549.6,"y":319.8,"visible":1}]}]}
```

实际文件中，每只手必须包含 `id=0..20` 共 21 个点。建议保存正向图上的像素坐标 `x/y`，因为像素坐标最方便人工审查；训练前再统一归一化成 `x / 1280`、`y / 720`。

如果使用 CVAT、Label Studio、Roboflow 等工具，也可以先导出 COCO/JSON/XML。核心要求不变：必须能恢复到“图像名 + 每只手 21 点 + 左右手 + 可见性”。

## 3. 最小训练派生格式

本轮目标是从零重写训练工程，不继承旧训练脚本。下面的 txt 只是推荐的最小派生格式：它足够表达 Hand Landmarker 训练所需信息，也方便三个人保持数据接口一致。

第一类是 Hand GT：

```text
annotation_id image_name hand_flag handedness x0 y0 x1 y1 ... x20 y20 [第二只手继续 44 个值]
```

说明：

- `annotation_id` 可以使用 `source:frame_id` 或任意唯一字符串。
- `image_name` 要能在图像目录中按文件名或 stem 找到。
- `hand_flag=1` 表示有手。
- `handedness` 建议 `0=Left`、`1=Right`；不训练左右手时可填 `0.5` 并在脚本中降权。
- `x0 y0 ... x20 y20` 是 `1280x720` 正向图坐标归一化到 `[0,1]` 后的 21 点。
- 一张图多只手时，可以在同一行追加第二只手的 `44` 个值。

示例：

```text
manual:000001 seq001_f000123.tiff 1 1 0.4653 0.5938 ... 0.5120 0.4211
```

第二类是 Palm ROI 辅助文件：

```text
image_name xmin ymin xmax ymax p0_x p0_y p9_x p9_y [第二个 ROI 继续 8 个值]
```

说明：

- 坐标同样是 `1280x720` 正向图的归一化坐标。
- `xmin ymin xmax ymax` 是 Palm 上游检测框。
- `p0` 通常对应 wrist，`p9` 通常对应 middle MCP，用来估计 Hand ROI 旋转方向。
- 这份文件不是人工 landmark 标注，而是训练时构造 Hand crop 的辅助输入。

Palm bbox 通常只覆盖手掌附近，不保证包含完整手指。因此这份 ROI 文件不能被理解为“直接 resize 成 Hand 输入的框”。训练脚本应使用 Palm bbox 加 `p0/p9` 构造更大的 rotated Hand ROI，再裁剪成 `256x256` 送入 Hand Landmarker。

Palm ROI 可以有两种来源：

1. 使用初赛冻结的 `preminilary/palm/model_opt.onnx` 对全部训练图推理生成。
2. 使用 MediaPipe 官方 Palm/Hand 检测模型推理生成，前提是输入尺寸、输出含义、坐标定义可以被规范化到同一份 ROI txt。

无论来源是哪一种，导出的 `points_pred_palm.txt` 都必须使用相同坐标系和相同字段定义，避免后续 Hand 训练代码感知上游差异。

## 4. 推荐数据流

建议 AetherSign 保留一个“干净主标注”加两个“训练派生文件”：

1. `hand_landmarks.jsonl`：人工维护的主标注，保存 `1280x720` 正向图像素坐标，含 visibility。
2. `annotations_hand.txt`：由 converter 生成，供当前 hand 训练使用。
3. `points_pred_palm.txt`：由 Palm ROI 生成脚本得到，供 hand 训练裁 ROI 使用。

这样做的好处是：标注格式不被某一次训练脚本绑定；后续如果更换训练策略，只要重写 converter，不需要重标数据。

## 5. 质量要求

标注质量会直接决定 Hand Landmarker 上限，建议一开始就做这些检查：

- 抽样可视化：把 21 点和骨架画回 `1280x720` 正向图，人工看 100-300 张。
- 坐标合法性：绝大多数点应在图内；少量遮挡/出界要用 `visible=0` 标识。
- 左右手一致性：左右手标签不要随镜像或相机视角混乱。
- 困难姿态覆盖：弯曲、遮挡、两手重叠、底部边缘、强暗光/强反光都要有足够样本。
- 训练/验证隔离：连续帧不要随机打散到 train 和 val，至少按片段/block 划分，否则验证集会虚高。

## 6. 当前采用边界

- 只把 `hand/root/hand/model_2d_NHWC.py` 视为必须继承的 Hand Landmarker 模型定义。
- 新训练脚本、新数据转换脚本、新评估脚本都由 AetherSign 重新编写。
- `hand/root/hand/model_2d_NHWC.py` 的输出是 42 维 landmark、1 维 hand flag、1 维 handedness，因此标注文件至少要能监督这三类输出。
- Palm 侧仅以 `preminilary/palm` 下的冻结 ONNX、结构定义和必要后处理函数为准。
