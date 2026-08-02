# HLMF 3.0 标注与发布工作流

## 0. 环境依赖

本轮 HLMF 3.0 更新没有改变标注环境：继续使用 Conda 环境 `anfab`，Python 版本保持为 3.11，Python 依赖以仓库根目录的 `requirements.txt` 为准。新代码使用的 SQLite、压缩包处理和文件操作均来自 Python 标准库，不需要新增第三方包。

首次创建环境：

```bash
cd /path/to/HandLandmarkerFab
conda create -n anfab python=3.11 pip -y
conda activate anfab
python -m pip install -r requirements.txt
python -m pip check
```

**Ubuntu 服务器缺少 OpenCV/MediaPipe 所需系统动态库时**，一次性安装：

```bash
apt-get update
apt-get install -y libglvnd0 libgles2 libegl1 libgl1 libglib2.0-0
ldconfig
```

输入：仓库根目录的 `requirements.txt`。

处理：在 `anfab` 中安装 NumPy、OpenCV、Pillow、PyYAML、MediaPipe、ONNX Runtime 和 tqdm。系统动态库只需在新服务器首次部署时安装，不要在每次标注任务中重复执行。

输出：名为 `anfab` 的 Conda 环境；`python -m pip check` 应无依赖冲突。创建完成后运行 `make compile` 和 `make test` 验证代码。已有环境时只需执行 `conda activate anfab`。

## 1. 系统边界

项目中的三级模型功能名与产品名固定对应如下：

- Palm Detector：**Eos**。如第一缕微光划破黑暗，模型首先从灰度画面中发现并定位手掌，为后续链路指明方向。
- Hand Landmarker：**Iris**。模型连接离散关键点，将像素编织成完整、可解释的手部几何结构。
- Gloss Translator：**Muse**。模型为物理动作赋予语言与语义，将骨骼序列转化为人类可读的 Gloss。

HLMF 从既有 Palm Detector 模型 Eos 开始工作。当前冻结版本为 `eos-1.0`，文件位于 `models/palm_detector/eos-1.0/model_opt.onnx`。程序在原图上运行 Eos，原样使用模型给出的 bbox、p0 和 p9，随后自动生成 `256×256` canonical Hand ROI，再在 ROI 内运行 MediaPipe Hand Landmarker。这里的 MediaPipe 模型是 HLMF 的自动标注工具，不是产品链路中的 Iris 部署模型。

后续 Eos 模型统一放入 `models/palm_detector/eos-*/model_opt.onnx`。切换模型版本时必须同步修改 `configs/autolabel.yaml` 中的模型路径，并使用新的 `PROPOSAL_VARIANT` 隔离派生产物。

HLMF 不制作 Palm Detector 训练数据，不导出或导入 Palm CVAT 标注，也不允许人工修改 bbox、p0、p9 或手工划分 Hand ROI。唯一的人工复核对象是程序已经生成的 Hand ROI，复核内容仅包括 21 个关键点、handedness 和 Hand ROI 内的状态标签。

长期数据写入 `HAND_DATASET_ROOT`；仓库代码、配置和模型文件位于 HLMF 仓库。本文示例使用服务器默认目录：

```bash
export HAND_DATASET_ROOT=/root/autodl-tmp/DatesetFab
cd /path/to/HandLandmarkerFab
```

**每条来源命令都需要四个身份参数**：

- `DATASET_SCOPE`：`pretrain` 或 `eval`。
- `DATASET_ID`：一次数据发布的逻辑 ID。
- `CAPTURE_SOURCE_ID`：一次拍摄来源的固定七段 ID。
- `PROPOSAL_VARIANT`：Palm 模型和 proposal 配置的版本 ID；当前版本使用 `eos-1.0`。

## 2. 来源命名与目录

`capture_source_id` 的顺序固定为：

```text
<background>-<distance>-<lighting>-<condition>-<split>-<session>-<performer>
```

例如：

```text
white-far-bright-fist-train-s01-peak
room-near-daylight-normal-val-s02-alice
```

七段都使用小写字母、数字或下划线；`condition` 不能包含连字符；`session` 必须为 `s` 加数字；`split` 只能是 `train`、`val` 或 `test`。一个来源目录只能属于一个 split。Train 来源放在：

```text
HAND_DATASET_ROOT/PretrainSource/<dataset_id>/<capture_source_id>/images/
```

Val/Test 来源放在：

```text
HAND_DATASET_ROOT/EValSource/<dataset_id>/<capture_source_id>/images/
```

`images/` **必须平铺**，只能放 `.tif` 或 `.tiff`。不要在里面再建立子目录。

每个来源的派生产物按 proposal 变体隔离：

```text
<capture_source_id>/
  images/
  01_palm/<proposal_variant>/
  02_roi_crops/<proposal_variant>/
    hand_landmarks_visualization/  # 可选自动标注审核图
  03_reviewed/<proposal_variant>/
  05_labels/<proposal_variant>/
  qc/<proposal_variant>/
```

同一原图在不同 `proposal_variant` 下共享 `raw_image_id`，但生成不同 `roi_id` 和不同派生目录；同一变体重跑时 ID 保持稳定。

## 3. 可选阶段：本地固定间隔抽帧

阶段名：本地采样，和服务器 HLMF 主流水线隔离。

命令：

```bash
python tools/downsample.py <input_dir> <interval> <output_dir>
```

输入：`input_dir` 是本地相机导出的平铺 TIFF 全帧目录；`interval` 为保留间隔，例如 `5` 表示每 5 帧保留一帧；`output_dir` 必须不存在或为空。

处理：按文件名排序后选择第 0、N、2N……帧，通过文件复制保留 TIFF，不做重编码。输入存在非 TIFF、子目录或目标目录非空时立即拒绝。

输出：筛选后的 TIFF 被平铺写入 `output_dir`。只把这个输出目录中的最终保留帧上传到来源的 `images/`；HLMF 不保存未上传的全帧。

## 4. 阶段一：来源检查与稳定身份建立

阶段名：Source Check。

Train 命令示例：

```bash
make source-check \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain \
  DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-far-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-1.0
```

Val/Test 将 `DATASET_SCOPE` 改为 `eval`，并在 ID 的第 5 段使用 `val` 或 `test`。

输入：

```text
PretrainSource/<dataset_id>/<capture_source_id>/images/*.tif[f]
```

或：

```text
EValSource/<dataset_id>/<capture_source_id>/images/*.tif[f]
```

处理：

1. 检查目录、来源 ID、split 和 TIFF 解码。
2. `720×1280` **TIFF 顺时针无损旋转一次**为 `1280×720`；已经是 `1280×720` 时直接通过，重复运行不会再次旋转。
3. **拒绝其他尺寸**、非 TIFF 和解码失败文件。
4. 首次验证时建立并持久化 `raw_image_id`。
5. 记录文件大小、尺寸、像素 CRC32 和 dHash64 等轻量指纹，并写入 SQLite registry。不会反复计算图片 SHA-256。

输出：

```text
<source>/raw_images.jsonl
<source>/source.json
<source>/qc/image_validation_report.json
HAND_DATASET_ROOT/Registry/registry.sqlite3
```

出现错误时先查看 `image_validation_report.json`；修复输入后可以安全重跑同一命令。

## 5. 阶段二：Train 自动标注与发布

阶段名：Train Autolabel。

命令：

```bash
make train-autolabel \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain \
  DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-far-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-1.0
```

仅本次临时启用可视化时，在同一命令末尾增加：

```bash
VISUALIZATION=true
```

例如 `make train-autolabel ... VISUALIZATION=true`。`VISUALIZATION=false` 可在全局配置已启用时仅关闭本次生成；命令行临时值优先于 `autolabel.yaml`。

输入：来源 `images/`、`configs/autolabel.yaml`、`paths.palm_model_onnx` 指向的 Palm ONNX 模型，以及 `mediapipe.model_asset_path` 指向的 MediaPipe task 文件。

处理：该高层命令依次执行来源检查、Palm 推理、稳定 proposal slot 分配、canonical ROI 裁剪、MediaPipe ROI 推理、质量门控和 Train 来源发布。来源检查、Palm 推理、ROI 裁剪和 MediaPipe 推理都会显示 tqdm 进度、处理速度与预计剩余时间。Palm 结果从产生到发布都不经过人工修改。

启用可视化时，程序按稳定的 ROI manifest 顺序做等距索引抽样，覆盖首尾并尽量均匀地分布在整份来源中；最多输出 `visualization.train_max_samples` 张，默认 200 张。每张审核图直接以 canonical Hand ROI 为底图，叠加 MediaPipe 21 点、骨架连线、presence 和 handedness。该目录只用于快速人工抽查，不替代标签 JSONL，也不进入 CVAT。

输出按步骤写入：

```text
<source>/01_palm/<variant>/palm_detections.jsonl
<source>/02_roi_crops/<variant>/images/<roi_id>.png
<source>/02_roi_crops/<variant>/hand_roi_crops_manifest.jsonl
<source>/02_roi_crops/<variant>/hand_landmarks_autolabel_draft.jsonl
<source>/02_roi_crops/<variant>/hand_landmarks_visualization/<roi_id>.png  # 启用时，Train 抽样
<source>/05_labels/<variant>/hand_training_labels.jsonl
<source>/05_labels/<variant>/candidate_negatives.jsonl
<source>/05_labels/<variant>/ignored.jsonl
<source>/qc/<variant>/*_report.json
<source>/qc/<variant>/autolabel_visualization_report.json
<dataset>/dataset_manifest.json
```

分流规则：

- MediaPipe 确认有手且通过质量门控：发布为 positive，`label_origin=mediapipe`。
- MediaPipe 未确认有手：只进入 `candidate_negatives.jsonl`，不能直接参与训练。
- MediaPipe positive 未通过质量门控：进入 `ignored.jsonl`。
- 常规 Train positive 不逐张进入 CVAT。

## 6. 阶段三：Val/Test 自动标注

阶段名：Eval Autolabel。

命令：

```bash
make eval-autolabel \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval \
  DATASET_ID=national-eval-0801 \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice \
  PROPOSAL_VARIANT=eos-1.0
```

仅本次临时启用或关闭可视化时，同样使用 `VISUALIZATION=true` 或 `VISUALIZATION=false`。

输入：Val/Test 来源的 `images/`、Palm 模型、MediaPipe 模型和 `autolabel.yaml`。

处理：与 Train 一样运行来源检查、Palm、程序化 ROI 和 MediaPipe，并在四个耗时环节显示 tqdm 进度；但只保留 Palm Detector 实际产生的 runtime ROI，并**强制关闭低分候选负样本**。这里**不会补 Palm 漏检**，也**不会从原图人工补 ROI**。启用可视化时，对该来源的全部实际 Hand ROI 生成关键点叠加图，不做抽样。

输出：Palm、ROI、MediaPipe draft 和 QC 文件与 Train 的路径相同；启用时额外生成完整的 `<source>/02_roi_crops/<variant>/hand_landmarks_visualization/`。此时**不发布最终评估标签**。命令返回的下一步是 CVAT 导出；后续 CVAT 导出和导入不重复生成可视化。

限制：每个 Val/Test split 最多 2000 张原图、3000 个实际生成 ROI；最终在来源发布阶段根据 dataset manifest 统一检查。

### 6.1 自动标注后补生成可视化

若执行 Train 或 Eval Autolabel 时没有启用可视化，后续可直接运行：

```bash
make autolabel-visualize \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=pretrain \
  DATASET_ID=FullEnhance0801 \
  CAPTURE_SOURCE_ID=white-far-bright-fist-train-s01-peak \
  PROPOSAL_VARIANT=eos-1.0
```

Val/Test 将 `DATASET_SCOPE` 改为 `eval`，并使用对应的来源 ID。

输入：已经存在的 `<source>/02_roi_crops/<variant>/images/` 与 `hand_landmarks_autolabel_draft.jsonl`。

处理：只读取已有 MediaPipe 自动标注结果并绘制审核图；不重新执行来源检查、Palm 推理、ROI 裁剪、MediaPipe 推理、质量门控或发布，也不受 `visualization.enabled` 当前值影响。Train 使用 `visualization.train_max_samples` 进行等距抽样，Val/Test 绘制全部已有 ROI。

输出：`<source>/02_roi_crops/<variant>/hand_landmarks_visualization/` 和更新后的 `<source>/qc/<variant>/autolabel_visualization_report.json`。

## 7. 阶段四：导出 Hand ROI CVAT 任务

阶段名：Hand CVAT Export。

命令：

```bash
make hand-cvat-export \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval \
  DATASET_ID=national-eval-0801 \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice \
  PROPOSAL_VARIANT=eos-1.0
```

输入：

```text
<source>/02_roi_crops/<variant>/hand_roi_crops_manifest.jsonl
<source>/02_roi_crops/<variant>/hand_landmarks_autolabel_draft.jsonl
<source>/02_roi_crops/<variant>/images/*.png
configs/cvat_label.json
```

处理：为该来源全部实际 ROI 生成 CVAT for images 1.1 XML。Train split 会被拒绝。导出内容只描述 Hand ROI 内的 skeleton 和 tag，不存在 Palm shape。

输出：

```text
<source>/03_reviewed/<variant>/cvat_autolabel.xml
<source>/qc/<variant>/cvat_export_report.json
```

在 CVAT 中创建 Images 任务时，上传 `02_roi_crops/<variant>/images/` 下的 ROI 图片并导入 `cvat_autolabel.xml`。标签契约的实际名称保持为：

- `hand_landmarks`：21 点 skeleton，子点名为 `1` 到 `21`，对应模型 landmark ID `0` 到 `20`。
- `Left`、`Right`：目标手 handedness；有可靠 skeleton 时二选一。
- `unknown_handedness`：确认有手和关键点，但无法可靠判定左右。
- `no_hand`：该固定 Hand ROI 内没有手。
- `ignore_for_training`：目标手或关键点无法可靠判定，本条不进入训练或评估。

复核规则：
1. 教师点正确则不动；
2. **明确错误时只修正错误点**；
3. teacher abstain 且**能可靠判断时删除** `no_hand`、补齐完整 skeleton 和 handedness；
4. 无手时只保留 `no_hand`；
5. 无法可靠决定时使用 `ignore_for_training`。不得绘制、调整或替换 ROI。

## 8. 阶段五：导入 CVAT 复核结果

阶段名：Hand CVAT Import。

先从 CVAT 导出 CVAT for images 1.1 XML，并将文件放到固定位置且改名为：

```text
<source>/03_reviewed/<variant>/cvat_reviewed.xml
```

然后执行：

```bash
make hand-cvat-import \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval \
  DATASET_ID=national-eval-0801 \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice \
  PROPOSAL_VARIANT=eos-1.0
```

输入：reviewed XML、原始 MediaPipe draft 和 ROI manifest。

处理：检查每个 ROI 的 presence、handedness、skeleton 完整性和冲突 tag，并比较教师点与复核点，记录人工实际修改的 landmark ID。

输出：

```text
<source>/03_reviewed/<variant>/hand_landmarks_reviewed.jsonl
<source>/qc/<variant>/cvat_import_report.json
```

provenance 规则：
- 未修改教师点为 `mediapipe/mediapipe_v1`；
- 人工修正教师点为 `mediapipe_human_corrected/project_consensus_v1`；
- teacher abstain 后人工完整补标为 `human/project_consensus_v1`。

所有复核记录同时保存 `human_reviewed` 和 `human_modified_landmark_ids`。存在阻断错误时不会生成可发布结果，应根据导入报告修复 CVAT XML 后重试。

## 9. 阶段六：发布 Val/Test 来源

阶段名：Source Publish。

命令：

```bash
make source-publish \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  DATASET_SCOPE=eval \
  DATASET_ID=national-eval-0801 \
  CAPTURE_SOURCE_ID=room-near-daylight-normal-val-s02-alice \
  PROPOSAL_VARIANT=eos-1.0
```

输入：`hand_landmarks_reviewed.jsonl`、ROI manifest、raw manifest 和 registry。

处理：排除 `ignore_for_training`，验证来源和 proposal variant，发布已复核的 fixed Hand ROI 标签，更新 dataset manifest，并检查 Val/Test 的原图数和 ROI 数上限。

输出：

```text
<source>/05_labels/<variant>/hand_evaluation_labels.jsonl
<source>/05_labels/<variant>/candidate_negatives.jsonl  # 必须为空
<source>/05_labels/<variant>/ignored.jsonl
<source>/qc/<variant>/source_publish_report.json
EValSource/<dataset_id>/dataset_manifest.json
```

这些标签只衡量给定固定 Hand ROI 上的 Hand Landmarker。未产生 ROI 的原图不计入指标，当前系统不报告 Palm 漏检率或原图级联性能。

## 10. 阶段七：真负样本删除式复核与发布

阶段名：Negative Review / Negative Publish。

准备审核树：

```bash
make negative-review \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  NEGATIVE_DATASET_ID=background-neg-0801 \
  NEGATIVE_CANDIDATE_LABELS="$HAND_DATASET_ROOT/PretrainSource/FullEnhance0801/<capture_source_id>/05_labels/eos-1.0/candidate_negatives.jsonl"
```

输入：一个 Train 来源的 `candidate_negatives.jsonl`。需要合并多个来源时可直接调用 CLI 并重复传入 `--candidate-labels`；公共 Make 入口一次接收一个文件。

处理与人工动作：程序按 `capture_source_id` 建立硬链接审核树。人工只在以下目录删除含手、模糊或无法确认的图片，剩下的必须全部是真背景：

```text
HAND_DATASET_ROOT/GoldSource/NegativeSamples/<negative_dataset_id>/review/images/<capture_source_id>/
```

硬链接只用于服务器同一文件系统内节省空间：**不同路径指向同一份文件数据**，**删除审核树中的一个硬链接不会删除 PretrainSource 原始 ROI**。普通 zip/7z 压缩、网盘上传和下载不会保留硬链接关系，但**本流程允许离线复核产生普通文件副本**。推荐操作如下：

1. 只压缩并下载本批次的 `review/images/`；服务器上的 `candidate_manifest.jsonl` 和 `README.json` 保持不动。
2. 在**本地只删除不合格图片，不重编码、不改名、不移动相对路径，也不新增图片**。
3. 复核完成后压缩 `images/`，上传网盘并传回服务器。
4. 核对目标确实是当前 `negative_dataset_id` 后，只删除服务器上的该批次 `review/images/`，再原路径解压复核后的 `images/`。不得删除整个 `review/`、PretrainSource 原始 ROI、Registry 或任何 `published/`。
5. 重新上传的普通文件与硬链接文件均可执行 `negative-publish`；该阶段允许发生一次性数据拷贝。

发布命令：

```bash
make negative-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" NEGATIVE_DATASET_ID=background-neg-0801
```

输出：

```text
GoldSource/NegativeSamples/<id>/published/images/<capture_source_id>/*.png
GoldSource/NegativeSamples/<id>/published/negative_labels.jsonl
GoldSource/NegativeSamples/<id>/published/manifest.json
GoldSource/NegativeSamples/<id>/published/review_report.json
```

如果全程在服务器内复核，发布图片继续通过同文件系统硬链接生成，不重复占用图片数据块；如果审核图片经过网盘往返成为普通文件，**发布结果保留这一份人工确认后的文件副本**，这是**人工复核阶段允许的数据拷贝例外**。**成功后临时 `review/` 被移除**；SQLite 将 `negative_dataset_id` 和 `roi_id` 锁定，**已使用或作废的 ID 不可复用**。

## 11. 阶段八：困难正样本复核与零拷贝发布

阶段名：Hard Positive Review / Publish。

准备审核树：

```bash
make hard-review \
  HAND_DATASET_ROOT="$HAND_DATASET_ROOT" \
  SELECTION_ID=hard-positive-0801 \
  MINING_REQUEST=/root/autodl-tmp/TrainFab/HLML-4.0/mining/v4-r1/hlmf_review_request.jsonl
```

输入：HLML multitask 阶段产生的 Train-only mining request；每条记录必须引用 registry 中已存在的 PretrainSource ROI。

处理与人工动作：审核图片位于：

```text
HAND_DATASET_ROOT/Selections/<selection_id>/review/images/<capture_source_id>/
```

**人工只删除 MediaPipe 21 点明显错误的 ROI，不重新标点，也不修改 Palm/ROI。**

困难正样本支持与真负样本相同的压缩包/网盘/本地删除式复核。只替换当前 `selection_id` 的 `review/images/`，保留服务器上的 `request_manifest.jsonl` 和 `README.json`；图片相对路径与文件名必须保持不变。重新上传的图片可以是普通文件，不要求保留硬链接。发布时这些审核图片**只用于判断哪些 ROI 被保留**，最终 `selection.jsonl` 仍零拷贝引用 PretrainSource 原始 ROI。

发布命令：

```bash
make hard-publish HAND_DATASET_ROOT="$HAND_DATASET_ROOT" SELECTION_ID=hard-positive-0801
```

输出：

```text
Selections/<selection_id>/published/selection.jsonl
Selections/<selection_id>/published/manifest.json
```

**selection 继续引用 PretrainSource 原 ROI，不发布图片副本**；`manifest.json` 记录保留量、删除量和 `zero_copy_reference_pretrain_roi` 策略。

## 12. 阶段九：Registry、配置和代码检查

查看 registry 计数与状态：

```bash
make registry-check HAND_DATASET_ROOT="$HAND_DATASET_ROOT"
```

输入为 `HAND_DATASET_ROOT/Registry/registry.sqlite3`，输出为终端 JSON 报告，不修改数据。

检查 Python 语法并运行完整单元测试：

```bash
make compile
make test
```

查看所有公共入口：

```bash
make help
```

## 13. `configs/autolabel.yaml` 参数说明

### 13.1 路径与图像契约

- `paths.palm_model_onnx`：Eos（Palm Detector）ONNX，相对路径按 HLMF 仓库根目录解析；当前值为 `models/palm_detector/eos-1.0/model_opt.onnx`。后续模型放入对应的 `models/palm_detector/eos-*/` 目录。**更换模型或改变会影响 proposal/ROI 的配置时，必须同时使用新的 `PROPOSAL_VARIANT`，防止不同结果写入同一版本目录**。
- `image.width/height/channels`：规范化原图契约，当前固定为 `1280/720/1`，不可在已有数据中随意修改。
- `image.accepted_extensions`：当前只允许 TIFF；增加有损格式会破坏来源质量假设，不建议修改。

### 13.2 Palm proposal

- `palm.backend`：`aethersign_onnx` 使用 `palm_model_onnx`；`mediapipe_official` 用于受支持的官方 Palm 后端。切换后端应使用新 proposal variant。
- `input_size`：Palm 网络输入尺寸，必须与模型一致。
- `score_threshold`：**runtime proposal 的最低分。提高会减少实际 ROI**，降低会增加 ROI；这不是人工修框入口。
- `nms_iou_threshold`、`cross_head_suppress_iou`：**控制同一检测头和跨检测头的重复 proposal 抑制**。修改会改变 proposal slot 和 ROI 集合，应发布为新 variant。
- `max_detections`：每张原图最多保留的 runtime proposal 数，当前最多双手。
- `keep_low_score_candidates_for_negatives` 和 `negative_candidate_threshold`：**仅影响 Train 候选负样本。Val/Test 会由程序强制关闭候选负样本。**
- `compatible_bbox_expand`、官方 tile 参数：只在对应 Palm backend 中生效；任何几何/搜索范围变化都需要新 variant。

### 13.3 Hand ROI

- `output_width/output_height`：当前固定 `256×256`，与 HLML v2 输入契约一致。
- `scale_x/scale_y`：相对 Palm anchor 的 ROI 扩张比例。
- `shift_x/shift_y`：沿 canonical ROI 坐标轴移动中心，负 `shift_y` 通常向腕部方向补充上下文。

这些参数改变程序生成的 ROI 几何，因此修改时必须新建 `PROPOSAL_VARIANT`；不能在 CVAT 中手工补偿。

### 13.4 MediaPipe

- `model_asset_path`：Hand Landmarker `.task` 文件路径。
- `num_hands`：每个 Palm ROI 只标目标手，当前为 1。
- `min_hand_detection_confidence`、`min_hand_presence_confidence`、`min_tracking_confidence`：教师确认门限。**提高会增加 teacher abstain，降低会增加自动 positive 及潜在误标**。调整前应通过 QC 报告抽查，且不得把 abstain 自动当真负样本。

### 13.5 质量门控

- `quality.handedness_review_threshold`：低于此分数时进入质量复核/忽略判断；调高会减少自动可用 positive。
- `quality.high_palm_score_review_threshold`：帮助报告高 Palm 分但无手的可疑项，不会授权修改 Palm 输出。

### 13.6 自动标注可视化

- `visualization.enabled`：全局开关，默认 `false`。只控制 Train/Eval 自动标注后的 Hand ROI 审核图，不改变 Palm、ROI、标签、质量门控或发布结果。
- `visualization.train_max_samples`：Train 单来源最多生成的等距抽样审核图数量，默认 200，必须至少为 1；来源 ROI 不超过该值时全部生成。Val/Test 始终生成全部实际 ROI，该参数对其无效。
- 临时覆盖：`make train-autolabel ... VISUALIZATION=true|false` 或 `make eval-autolabel ... VISUALIZATION=true|false`。临时值优先于 `autolabel.yaml`；未传时使用全局值。
- 补生成：已有自动标注 draft 时执行 `make autolabel-visualize ...`。该命令本身即表示启用，不读取 `visualization.enabled`，但 Train 仍使用 `train_max_samples`。
- 输出目录固定为 `<source>/02_roi_crops/<variant>/hand_landmarks_visualization/`，执行报告固定为 `<source>/qc/<variant>/autolabel_visualization_report.json`。同一来源和变体再次启用可视化时，程序会清除不属于当前选择结果的旧 PNG，避免抽样配置变化后残留过期审核图。

## 14. `configs/review.yaml` 与 `configs/cvat_label.json`

`review.yaml` 只包含 Hand ROI CVAT **导入导出的语义映射和人工复核门控**；不包含 Palm、ROI 几何或 MediaPipe 参数。`cvat_label.json` 是创建 CVAT 项目/任务时使用的标签 schema。

### 14.1 CVAT 与人工复核参数

- `cvat.label_name`：必须与 `cvat_label.json` 的 `hand_landmarks` 一致。
- `*_label_name`：必须与五个 tag 的大小写完全一致；当前分别是 `no_hand`、`Left`、`Right`、`unknown_handedness`、`ignore_for_training`。
- `skeleton_point_labels`：固定为字符串 `1..21`。
- `review.require_explicit_presence_decision`：**要求每个 ROI 明确为 skeleton 或 `no_hand`**。
- `review.require_explicit_handedness_decision`：**有 skeleton 时要求 Left/Right/unknown 决策**。
- `review.manual_roi_editing`：必须保持 `false`。

## 15. `configs/datasets.yaml`

`datasets.yaml` 是 operator-owned 发布集合目录：`pretrain.dataset_ids`、`evaluation.val_dataset_ids/test_dataset_ids` **记录采用的数据集 ID**，`proposal_variants` 记录各数据集**选择的 Palm 变体**。单来源命令仍通过 Make 参数接收这些 ID；该文件不用于手工拼路径或存图片。

`policies.capture_source_split` 和 `one_proposal_variant_per_capture_source` 应保持 `fail`；`performer_cross_split` 默认 `warn`，需要严格人员隔离时可升级为 `fail`。`evaluation_limits.max_raw_images_per_split` 和 `max_rois_per_split` 是 Val/Test 发布硬上限，默认分别为 2000 和 3000。

`cvat_label.json` 完整包含五个 tag 和 `hand_landmarks` 21 点 skeleton。修改任何实际 label 名称时必须同步修改 `review.yaml` 并运行测试。

## 16. 评估边界

HLMF 只发布 Palm 已经生成的固定 Hand ROI。HLML 的 Val/Test 直接读取这些 ROI，不重新运行 Palm，不把没有 ROI 的原图计入指标，也不报告 Palm 漏检率、部分双手召回率或原图级联准确率。
