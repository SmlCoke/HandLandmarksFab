# HLMF 3.0 Final: model assets

The `HLMF-3.0-final` Release archives the code, configuration, tests, documentation, and calibration reports used by HLMF 3.0 during the AetherSign national final. Model assets are distributed separately from the Git source archive.

## Release assets

Download these assets from the [HLMF 3.0 Final Release](https://github.com/SmlCoke/HandLandmarksFab/releases/tag/HLMF-3.0-final), then extract them at the **repository root**. Each archive retains its `models/` path.

| Asset | Contents | Required by |
| --- | --- | --- |
| `HLMF-3.0-final-hand-classifier.zip` | All in-house HCF versions and their archived training assets. The default runtime model is `models/hand_classifier/v1-mobilenet_v3_large/model.onnx`. | RTMPose, HaMeR, and negative review |
| `HLMF-3.0-final-palm-detector.zip` | In-house Eos-1.0, Eos-2.0, and final Eos-2.1 ONNX Palm Detector models. | Required; production configuration uses Eos-2.1 |
| `HLMF-3.0-final-rtmpose-m-hand5-256x256.zip` | RTMPose-m Hand5 ONNX and an Apache-2.0 license copy. | Default hand-landmark backend |
| `HLMF-3.0-final-SHA256SUMS.txt` | SHA-256 checksums for the model archives. | Recommended integrity check |

In Git Bash:

```bash
gh release download HLMF-3.0-final --repo SmlCoke/HandLandmarksFab \
  --pattern 'HLMF-3.0-final-*.zip' \
  --pattern 'HLMF-3.0-final-SHA256SUMS.txt'
sha256sum -c HLMF-3.0-final-SHA256SUMS.txt
for archive in HLMF-3.0-final-*.zip; do unzip -o "$archive"; done
```

## MediaPipe assets: download from Google

MediaPipe assets are not redistributed by this project. Download them directly from Google and save them at these exact paths under the repository root:

| Destination | Official URL |
| --- | --- |
| `models/mediapipe/hand_landmarker.task` | <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task> |
| `models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite` | <https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite> |

```bash
mkdir -p models/mediapipe/hand_landmarker_tflite
curl -fL --retry 3 \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task \
  -o models/mediapipe/hand_landmarker.task
curl -fL --retry 3 \
  https://storage.googleapis.com/mediapipe-assets/hand_landmark_full.tflite \
  -o models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
```

`hand_landmarker.task` is used by the explicit MediaPipe Tasks backend. `hand_landmark_full.tflite` is used by the default MediaPipe TFLite geometry rescue, so deploy both even when RTMPose is the default backend.

## RTMPose provenance and license

`models/rtmpose/rtmpose-m_hand5_256x256.onnx` is derived from the OpenMMLab [MMPose](https://github.com/open-mmlab/mmpose) RTMPose-m Hand5 model (alias `rtmpose-m_8xb256-210e_hand5-256x256`) and was converted to ONNX by this project. The conversion script is not retained; the Release asset is the fixed ONNX artifact used during the competition.

The archive contains a source notice and a copy of the MMPose [Apache License 2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE). Preserve the third-party notices when using or redistributing the asset and assess compliance for your own deployment.

## Minimum default asset set

```text
models/palm_detector/eos-2.1/model_384x224_opt.onnx
models/hand_classifier/v1-mobilenet_v3_large/model.onnx
models/rtmpose/rtmpose-m_hand5_256x256.onnx
models/mediapipe/hand_landmarker.task
models/mediapipe/hand_landmarker_tflite/hand_landmark_full.tflite
```

Follow the [full workflow](../annotating_system/HLMF_annotating_workflow.md) for environment setup. The final Eos-2.1 contract supports `near` and `mid` capture distances only; it must not be used for the formal `far` labeling pipeline.
