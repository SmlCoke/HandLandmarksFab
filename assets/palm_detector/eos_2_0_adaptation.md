# Eos-2.0 模型审计与 HLMF 适配

## 结论

Eos-2.0 的 TensorFlow 结构、H5 权重与优化 ONNX 数值一致，已作为 HLMF 默认 Palm Detector。正式参数为 score `0.25`、全局 NMS IoU `0.10`、最多 2 手；输入为灰度 `[1,1,224,384]`。兼容性回放不支持把 ROI scale 降到 `1.5`，因此暂保留 `scale_x=scale_y=1.8`。

GPU 吞吐为 `426.2 images/s`，CPU 为 `175.0 images/s`；真实输入解码一致，Palm 继续使用 `auto`（CUDA 优先，CPU fallback）。

## 环境与模型

| 项目 | 内容 |
|---|---|
| CPU / GPU | 2× Intel Xeon Gold 6330 / RTX 3090 24 GB |
| 主环境 | Python 3.11.15、NumPy 1.26.4、OpenCV 4.10.0、ONNX Runtime GPU 1.18.0 |
| 隔离审计环境 | Python 3.8.20、TensorFlow 2.9.0、ONNX 1.16.1；任务结束后删除 |
| TensorFlow 参考 | `/root/Test/Eos-2.0/model.py` + `best_model.weights.h5` |
| ONNX | `models/palm_detector/eos-2.0/model_384x224_opt.onnx` |
| 输入 / 输出 | `[1,1,224,384]`；`[1,16,14,24]`、`[1,2,14,24]`、`[1,16,7,12]`、`[1,2,7,12]` |
| anchors | `14×24` 与 `7×12` 各 2 个，共 840 个 |

模型文件按仓库策略不进入 Git，本地、服务器和板端需单独部署同一资产。

## 审计方法与结果

- 构建参考 TensorFlow 模型并原生加载 H5；参数量为 `1,367,620`，输入和四个 detection head 与模型说明一致。
- ONNX full checker 通过：153 个节点、160 个 initializer。
- 用 1 组固定随机输入和 16 张真实 TIFF 比较 TensorFlow、ONNX CPU、ONNX GPU 四个原始输出；TF↔ONNX 最大绝对差 `5.4e-7`，CPU↔GPU 最大绝对差 `5.6e-7`。
- 在 score `0.15` 和 `0.25` 下，三种运行方式的 detection 数完全一致，解码分数/几何最大差 `2.7e-7`。
- 预处理固定为 `uint8 gray → INTER_AREA 384×224 → float32 / 255 → NCHW`；HLMF 仍只接受实际 `1280×720` upright 原图。

## score 与 NMS

测试集为 `/root/Test/Eos-2.0/testcase/inputs` 的 307 张顶层 TIFF。下表固定全局 NMS IoU `0.10`；本次 NMS `0.10/0.20/0.30` 的 detection 计数相同。

| score | 零/一/二检测图片 | detection 总数 | 旧 gold 关联数/9,868 |
|---:|---:|---:|---:|
| 0.20 | 2 / 27 / 278 | 583 | 9,169 |
| **0.25** | **8 / 55 / 244** | **543** | **9,098** |
| 0.30 | 31 / 101 / 175 | 451 | 8,945 |
| 0.50 | 267 / 40 / 0 | 40 | 3,997 |

旧 gold 关联使用 Eos p0/p9 与人工 0/9 点的尺度归一化距离做一对一匹配，只用于新旧几何比较，不是 Palm 官方 recall。`0.25` 是本次确认的召回/候选量折中值；negative candidate 下限继续为 `0.15`。

## ROI scale 兼容回放

数据来自已发布且人工复核的 `FullEnhanceVal0801:eos-1.0` 与 `FullEnhanceVal0808:eos_1.0-gate_r2`，共 13 个来源、9,868 条 gold hand；Eos-2.0 成功关联 9,176 条。仓库全程只读。

| scale | 整手入框率 | 单点入框率 | 手部占比 P50/P95 | 边缘余量 P01 (px) | 旧连接门控命中 |
|---:|---:|---:|---:|---:|---:|
| 1.5 | 98.300% | 99.918% | 0.614 / 0.729 | -3.70 | 529（5.765%） |
| 1.6 | 99.477% | 99.975% | 0.575 / 0.684 | 4.50 | 203（2.212%） |
| 1.7 | 99.760% | 99.989% | 0.541 / 0.643 | 11.74 | 104（1.133%） |
| **1.8** | **99.902%** | **99.995%** | **0.511 / 0.608** | **18.17** | **52（0.567%）** |

`1.5` 会明显增加裁切和旧门控误拒，故 HLMF 与板端都应暂用 `1.8/1.8、shift_x=0、shift_y=-0.1`。旧阈值回放只是过渡兼容证据，不是 Eos-2.0 的正式阈值重算。

## Eval 使用建议与重跑

旧 Eos-1.x Eval 可继续用于 Iris 模型级历史回归，不需要立即重新人工修点，但它不能永久替代 Eos-2.0 端到端评估。当前连接长度门控保持默认开启并暂留旧值；首个代表性 Eos-2.0 Eval 完成人工复核和发布后，必须正式重算阈值并回放 gold/draft。

```bash
python -B tools/analyze_eos2_adaptation.py \
  --dataset-root /root/autodl-tmp/DatesetFab \
  --dataset FullEnhanceVal0801:eos-1.0 \
  --dataset FullEnhanceVal0808:eos_1.0-gate_r2 \
  --test-images /root/Test/Eos-2.0/testcase/inputs \
  --config configs/autolabel.yaml \
  --output /tmp/eos2_compatibility_replay.md
```

工具只读取测试图片、published manifest 和 gold labels；输出路径不得位于 `HAND_DATASET_ROOT`。
