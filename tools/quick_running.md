# Hand Landmarks Fab 半自动标注系统快速运行指南

注意记得替换以下文档中的 `<member>` 和 `<source_name>` 为实际的成员名和数据源名。

## I. 数据集准备

在 `autodl-tmp` 目录下新建文件夹：

```bash
cd autodl-tmp/
mkdir -p DatesetFab/PretrainSource/HandViolencePro0719/<member>/
```

然后把数据集压缩包全部移动到：`DatesetFab/PretrainSource/HandViolencePro0719/<member>/`，并且解压，确保格式均满足：

`autodl-tmp/DatesetFab/PretrainSource/HandViolencePro0719/<member>/<source_name>/images/`

## II. 运行环境准备

将脚本 `run_hlmf_autolabel.sh` 保存到 `/root/` 目录下，然后赋予执行权限：

```bash
chmod +x /root/run_hlmf_autolabel.sh
```

创建环境：

```bash
# 仓库
git clone git@github.com:SmlCoke/HandLandmarksFab.git
cd HandLandmarksFab

# Python 虚拟环境
conda create -n anfab python=3.11 -y
conda init
source ~/.bashrc
conda activate anfab

# 动态链接库补齐
apt-get update
apt-get install -y libglvnd0 libgles2 libegl1 libgl1 libglib2.0-0
ldconfig
```

每台服务器第一次运行任务前，单独执行一次代码检查。不要在每个并行任务中重复执行：

```bash
cd /root/HandLandmarksFab
conda activate anfab

make compile
make test
```

## III. 单个 screen 中运行

例如处理：

```text
autodl-tmp/DatesetFab/PretrainSource/HandViolencePro0719/<member>/<source_name>/
```

建立 screen：

```bash
screen -S h_<source_name>
```

在 screen 中设置环境变量并运行：

```bash
export HLMF_MEMBER=<member>
export HLMF_SOURCE_NAME=<source_name>

/root/run_hlmf_autolabel.sh
```

脱离 screen：

```text
Ctrl+A，然后按 D
```

重新进入：

```bash
screen -r h_fist
```

## IV. 一次启动多个后台 screen

不同任务必须使用不同的 `HLMF_SOURCE_NAME`。

例如新开一个 screen ：

```bash
screen -S h_<source_name2>
```

然后：

```bash
export HLMF_MEMBER=<member>
export HLMF_SOURCE_NAME=<source_name2>

/root/run_hlmf_autolabel.sh
```

## V. 完成后的目录

把完整 source 目录下的"01_palm", "02_roi_crops", "qc"压缩后，上传到夸克网盘的对应位置：

```text
/root/autodl-tmp/DatesetFab/PretrainSource/HandViolencePro0719/<member>/<source_name>/
├── images/
├── 01_palm/
├── 02_roi_crops/
└── qc/
```
