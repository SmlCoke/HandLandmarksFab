# 小工具脚本

本文件夹下是与半自动化标注工具链流程无关的辅助工具脚本


## 1. 数据集降采样

downsample.py: 用于从原始数据中随机采样一部分数据，生成一个小数据集，便于快速调试和测试

使用方法：

```
python downsample.py /path/to/images N /path/to/output
```

脚本将从 `/path/to/images` 中按照降采样因子 N 进行采样，并将采样结果保存到 `/path/to/output` 中。