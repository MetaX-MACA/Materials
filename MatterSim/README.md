# 沐曦GPU平台 MatterSim 用户指南

[MatterSim](https://github.com/microsoft/mattersim)项目由第三方以开源许可证发布，我方未对其源代码作任何修改。您可通过以下链接 https://github.com/microsoft/mattersim 查阅该项目的源代码、许可证全文、版权与归属声明及其他声明文件。

## 1. 概述

MatterSim 是微软开发的一个深度学习原子模拟模型，可应用于多种元素、温度和压力条件下的材料模拟。研究表明，MatterSim 在吸附能评估等任务中能实现约4000倍于DFT的加速，并保持85.4%的成功率。本指南将介绍如何在沐曦GPU计算平台上配置和运行 MatterSim。

**官方仓库**：https://github.com/microsoft/mattersim

**版本信息**：MatterSim-v1.2.1

## 2. 环境准备

### 2.1 基础镜像

在沐曦GPU平台上，推荐使用沐曦开发者社区提供的[MACA torch2.8镜像]([4](https://developer.metax-tech.com/softnova/docker?chip_name=%E6%9B%A6%E4%BA%91C500%E7%B3%BB%E5%88%97&package_name=maca-pytorch:3.8.2.6-torch2.8-py312-ubuntu24.04-amd64))，该镜像已包含沐曦GPU所需的驱动、PyTorch 和 MXMACA 生态组件，能够保证与沐曦硬件的兼容性。


### 2.2 启动容器

使用以下命令启动容器，并映射必要的设备：
```bash
docker run -it --name test-mattersim \
  --device=/dev/mxcd \
  --device=/dev/dri \
  --group-add video \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /host_dir:/workspace \
  cr.metax-tech.com/public-library/maca-pytorch:3.8.2.6-torch2.8-py312-ubuntu24.04-amd64 \
  /bin/bash
```

## 3. 安装 MatterSim

### 3.1 克隆仓库并进入目录

```bash
git clone https://github.com/microsoft/mattersim.git
cd mattersim
git checkout v1.2.1
```

### 3.2 修改 `pyproject.toml` 文件

由于maca-pytorch基础镜像已包含兼容的PyTorch环境，需要修改 `pyproject.toml` 中的 `torchvision` 版本，避免版本冲突。具体操作如下：

1. 使用 `vi` 或 `nano` 打开 `pyproject.toml`：
   ```bash
   vi pyproject.toml
   ```

2. 找到 `dependencies` 部分中的 `"torchvision>=0.17.0"` 行，将版本降到0.15.0：
   ```toml
   dependencies = [
       # ... 其他依赖 ...
       "torchvision>=0.15.0",  # 降低版本，以适配沐曦镜像环境
       # ... 其他依赖 ...
   ]
   ```

3. 保存并退出。

### 3.3 执行安装

在项目根目录下运行以下命令完成安装：

```bash
pip install -e .
```

`-e` 参数表示以可编辑模式安装，方便后续代码调试和更新。

### 3.4 验证安装

启动Python交互环境，尝试导入MatterSim，检查是否成功：

```python
import mattersim
print("MatterSim imported successfully!")
```

## 4. 模型使用

### 4.1 基础使用示例

以下是一个最小测试用例，用于验证MatterSim在沐曦GPU上的运行：

```python
import torch
from loguru import logger
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import MatterSimCalculator

# 检测可用设备，优先使用GPU (cuda)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Running MatterSim on {device}")

# 创建一个硅晶体结构
si = bulk("Si", "diamond", a=5.43)

# 设置计算器，可选择加载不同模型
# 默认加载1M参数版本
si.calc = MatterSimCalculator(device=device)

# 执行计算并输出结果
logger.info(f"Energy (eV)                 = {si.get_potential_energy()}")
logger.info(f"Energy per atom (eV/atom)   = {si.get_potential_energy()/len(si)}")
logger.info(f"Forces of first atom (eV/A) = {si.get_forces()[0]}")
logger.info(f"Stress[0][0] (eV/A^3)       = {si.get_stress(voigt=False)[0][0]}")
logger.info(f"Stress[0][0] (GPa)          = {si.get_stress(voigt=False)[0][0] / GPa}")
```

### 4.2 切换预训练模型

MatterSim 提供两个预训练模型（基于M3GNet架构），可通过设置 `load_path` 进行切换：

1. **MatterSim-v1.0.0-1M** (默认)：参数较少，计算速度更快。
2. **MatterSim-v1.0.0-5M**：参数更多，精度更高，但需要更多GPU内存。

切换到5M模型的示例：
```python
si.calc = MatterSimCalculator(
    load_path="MatterSim-v1.0.0-5M.pth", 
    device=device
)
```

> **注意**：模型文件会在首次使用时自动下载，请确保网络连接正常。如果遇到内存不足问题，建议使用1M模型。


## 5. 常见问题排查

### 5.1 PyTorch 版本冲突
- **问题**：`torchvision` 版本不兼容。
- **解决**：确保已按第3.2节方法注释 `pyproject.toml` 中的 `torchvision` 依赖。

### 5.2 沐曦GPU不可用
- **问题**：运行 `torch.cuda.is_available()` 返回 `False`。
- **解决**：
  1. 确认容器启动时是否正确映射了 `/dev/dri` 和 `/dev/mxcd` 设备。
  2. 检查沐曦GPU驱动和maca-pytorch镜像版本是否匹配。

### 5.3 模型下载失败
- **问题**：首次运行报错，无法下载预训练模型。
- **解决**：检查网络连接，或手动从GitHub仓库的 `pretrained_models` 目录下载 `.pth` 文件并放置在正确路径。

### 5.4 内存不足
- **问题**：使用5M模型时GPU内存溢出。
- **解决**：更换为1M模型，或适当减小 `batch_size`。

## 6. 引用说明

如在研究中使用MatterSim模型，请务必引用其预印本论文，**并明确注明所使用的模型版本**（如：MatterSim-v1.0.0-1M），以确保结果的可复现性。

```
@article{yang2024mattersim,
      title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
      author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
      year={2024},
      eprint={2405.04967},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2405.04967},
      journal={arXiv preprint arXiv:2405.04967}
}
```

## 7. 参考资源

- MatterSim官方文档：[https://microsoft.github.io/mattersim/](https://microsoft.github.io/mattersim/)
- GitHub仓库：[https://github.com/microsoft/mattersim](https://github.com/microsoft/mattersim)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
