# HLMF 3.0 Final：模型资产下载与部署

`HLMF-3.0-final` Release 固定了 AetherSign 全国总决赛阶段的 HLMF 3.0 代码、配置、测试、文档和校准报告。本目录说明运行该 tag 所需的模型资产获取与放置方法；模型不纳入 Git 源码归档。

## 1. Release 附件

从 [HLMF 3.0 Final Release](https://github.com/SmlCoke/HandLandmarksFab/releases/tag/HLMF-3.0-final) 下载以下附件，并在**仓库根目录**解压。压缩包内已保留 `models/` 路径，因此解压后会直接落到正确位置。

| 附件 | 内容 | 运行时是否需要 |
| --- | --- | --- |
| `HLMF-3.0-final-hand-classifier.zip` | 全部本项目自研 HCF（Hand Classifier）版本及其归档训练资产；默认 runtime 使用 `models/hand_classifier/v1-mobilenet_v3_large/model.onnx`。 | RTMPose、HaMeR、negative-review |
| `HLMF-3.0-final-palm-detector.zip` | 本项目自研的 Eos-1.0、Eos-2.0 与最终 Eos-2.1 ONNX Palm Detector。 | 必需；正式配置使用 Eos-2.1 |
| `HLMF-3.0-final-rtmpose-m-hand5-256x256.zip` | RTMPose-m Hand5 ONNX 与 Apache-2.0 许可证副本。 | 默认 Hand Landmark 后端 |
| `HLMF-3.0-final-SHA256SUMS.txt` | 上述 Release 附件的 SHA-256 校验值。 | 建议下载后校验 |

在 Git Bash 中可执行：

```bash
gh release download HLMF-3.0-final --repo SmlCoke/HandLandmarksFab \
  --pattern 'HLMF-3.0-final-*.zip' \
  --pattern 'HLMF-3.0-final-SHA256SUMS.txt'
sha256sum -c HLMF-3.0-final-SHA256SUMS.txt
for archive in HLMF-3.0-final-*.zip; do unzip -o "$archive"; done
```

在 Windows PowerShell 中可执行：

```powershell
gh release download HLMF-3.0-final --repo SmlCoke/HandLandmarksFab `
  --pattern 'HLMF-3.0-final-*.zip' `
  --pattern 'HLMF-3.0-final-SHA256SUMS.txt'
Get-FileHash HLMF-3.0-final-*.zip -Algorithm SHA256
Get-ChildItem HLMF-3.0-final-*.zip | ForEach-Object {
  Expand-Archive -LiteralPath $_.FullName -DestinationPath . -Force
}
```

`HLMF-3.0-final-SHA256SUMS.txt` 使用 GNU `sha256sum` 格式；PowerShell 的 `Get-FileHash` 输出应与其中相应条目的 digest 一致。

## 2. MediaPipe 模型：从 Google 官方位置下载

Google MediaPipe 模型不作为本项目的 Release 附件分发。请直接从 Google 提供的地址下载，并保存到以下**精确路径**：

| 目标文件 | 官方下载地址 |
| --- | --- |
| `models/mediapipe/hand_landmarker.task` | <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task> |
| `models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite` | <https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite> |

Git Bash：

```bash
mkdir -p models/mediapipe/hand_landmarker_tflite
curl -fL --retry 3 \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
  -o models/mediapipe/hand_landmarker.task
curl -fL --retry 3 \
  https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite \
  -o models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
```

PowerShell：

```powershell
New-Item -ItemType Directory -Force models/mediapipe/hand_landmarker_tflite | Out-Null
curl.exe -fL --retry 3 `
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task `
  -o models/mediapipe/hand_landmarker.task
curl.exe -fL --retry 3 `
  https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite `
  -o models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
```

`hand_landmarker.task` 用于显式的 MediaPipe Tasks 后端；`hand_landmark_full.tflite` 用于默认开启的 MediaPipe TFLite 几何补救。即使默认使用 RTMPose，两者也应按上述位置部署。

## 3. RTMPose 来源与许可证

Release 中的 `models/rtmpose/rtmpose-m_hand5_256x256.onnx` 源自 OpenMMLab [MMPose](https://github.com/open-mmlab/mmpose) 的 RTMPose-m Hand5（模型别名 `rtmpose-m_8xb256-210e_hand5-256x256`），并由本项目转换为 ONNX。转换脚本未随归档保留；Release 附件是比赛阶段实际使用的固定 ONNX 工件。

该附件随附 MMPose 的 [Apache License 2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE) 副本和来源声明。使用或再分发时，请保留这些第三方声明并自行评估适用场景的合规要求。

## 4. 最小可运行资产集

按照默认配置运行 HLMF 3.0 时，至少需要以下本地文件：

```text
models/palm_detector/eos-2.1/model_384x224_opt.onnx
models/hand_classifier/v1-mobilenet_v3_large/model.onnx
models/rtmpose/rtmpose-m_hand5_256x256.onnx
models/mediapipe/hand_landmarker.task
models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
```

此外仍需按 [完整工作流](../annotating_system/HLMF_annotating_workflow.md) 配置运行依赖和环境变量。Eos-2.1 的能力边界仅为 `near` 与 `mid`，不得将该 Release 用于 `far` 拍摄距离的正式标注链路。
