# 使用沐曦GPU运行MatGL的使用手册

本手册指导用户在沐曦（MetaX）GPU上安装并运行 [MatGL](https://github.com/materialsvirtuallab/matgl)（Materials Graph Library，材料图深度学习库），完成 M3GNet 等图神经网络势函数的训练与推理。

[MatGL](https://github.com/materialsvirtuallab/matgl) 项目由 Materials Virtual Lab 以 BSD-3-Clause 许可证开源发布，本指南使用其 v1.1.3 版本。上游源码在沐曦 GPU 上需做少量适配（详见「源码适配说明」）。您可通过 https://github.com/materialsvirtuallab/matgl 查阅项目源代码与许可证全文。

推荐环境：MACA PyTorch 镜像（torch2.4 + Python 3.10 + Ubuntu 22.04）。

---

## 1. 环境准备

### 1.1 启动容器

```bash
docker run -it --name matgl \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --group-add video \
  --shm-size=4G \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  <maca-pytorch镜像地址> \
  /bin/bash
```

### 1.2 验证 GPU 可用

```python
import torch
print(torch.cuda.is_available())      # 应输出 True
print(torch.cuda.device_count())      # 可用 GPU 数量
print(torch.cuda.get_device_name(0))  # GPU 型号
```

---

## 2. 安装 MatGL

### 2.1 获取源码

```bash
git clone https://github.com/materialsvirtuallab/matgl
cd matgl
git checkout v1.1.3
```

### 2.2 源码适配说明

MatGL 基于 DGL + PyTorch，上游代码将 GPU 设备上的张量直接转 numpy（或传给 `np.repeat`）。GPU 设备上的张量需先转为 CPU 张量才能转换为 numpy 数组，因此做了以下适配：

| 文件 | 修改内容 |
|---|---|
| `src/matgl/layers/_atom_ref.py` | `features.numpy()` → 返回张量，拟合时改用 `.cpu().numpy()` |
| `src/matgl/layers/_three_body.py` | `torch.arange(max_l)` → `torch.arange(max_l, device='cpu')`（`np.repeat` 需将张量转换为 numpy 数组，而 numpy 无法直接读取 GPU 设备上的张量） |
| `src/matgl/utils/so3.py` | `lidx.numpy()`/`midx.numpy()` → `.cpu().numpy()` |

除上述改动外，其余源码无需修改。

### 2.3 安装

```bash
pip3 install --upgrade pip
pip3 install matplotlib lightning pymatgen
pip3 install -e .
pip3 install sympy==1.12
pip3 install ase==3.24.0
pip3 install transformers==4.57.5
```

---

## 3. 快速验证（GPU 单元测试）

运行仓库提供的 GPU 测试脚本（覆盖 M3GNet、CHGNet、MEGNet、TensorNet 等模型的测试用例）：

```bash
cd test
python3 test-gpu.py
```

> 脚本说明：`test-gpu.py` 与 `tests-gpu/` 为仓库测试脚本与用例（非上游源码），其调用的 `pytest` 用例覆盖上游模型的图构建、层与模型推理。
> 运行前请设置环境变量 `MACA_VISIBLE_DEVICES=0`。

---

## 4. 使用指南

### 4.1 M3GNet 势能面推理

```python
import matgl
from pymatgen.core import Lattice, Structure

# 加载预训练 M3GNet 势
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
```

结合 ASE 计算器（`M3GNetCalculator`）可用于结构弛豫与分子动力学，详见上游文档 <https://matgl.ai>。

### 4.2 预训练模型

MatGL 提供 M3GNet、CHGNet、MEGNet、TensorNet 等预训练模型，通过 `matgl.load_model()` 按名称加载。

---

## 5. 常见问题

- **GPU 不可用**：确认容器已映射 `--device=/dev/dri --device=/dev/mxcd --group-add video`，宿主机 MXMACA 驱动正常
- **`numpy` 转换报错**：确认已应用第 2.2 节的 `.cpu()` / `device='cpu'` 适配
- **DGL 版本冲突**：确认使用镜像自带的 DGL 版本，勿自行升级
- **显存不足（OOM）**：减小批次或结构规模

---

## 6. 参考资源

- MatGL 上游仓库：[https://github.com/materialsvirtuallab/matgl](https://github.com/materialsvirtuallab/matgl)
- MatGL 官方文档：[https://matgl.ai](https://matgl.ai)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码。本文档所述源码适配修改（第 2.2 节）随本仓库一并提供，修改与再分发应遵守 MatGL 的 BSD-3-Clause 许可证条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
