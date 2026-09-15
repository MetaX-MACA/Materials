# 使用沐曦GPU运行Aviary的使用手册

本手册指导用户在沐曦（MetaX）GPU上安装并运行 [Aviary](https://github.com/CompRhys/aviary)（材料发现多模型统一接口库），完成基于成分的深度学习模型（Roost、Wren、CGCNN 等）的训练与推理。

[Aviary](https://github.com/CompRhys/aviary) 项目以 MIT 许可证开源发布，本指南使用其 1.0.0 版本。上游核心源码在沐曦 MXMACA 软件栈（原生兼容 CUDA API）上**无需修改**即可运行（仅单元测试数据路径有调整，详见「源码适配说明」）。您可通过 https://github.com/CompRhys/aviary 查阅项目源代码与许可证全文。

推荐环境：MACA PyTorch 镜像（torch2.4 + Python 3.10 + Ubuntu 22.04）。

---

## 1. 环境准备

### 1.1 启动容器

```bash
docker run -it --name aviary \
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

## 2. 安装 Aviary

### 2.1 获取源码

```bash
git clone https://github.com/CompRhys/aviary
cd aviary
git checkout 1.0.0
```

### 2.2 源码适配说明

上游核心源码**零修改**，直接运行于 MXMACA 软件栈。仅 `tests/conftest.py` 中 Matbench 单元测试数据集的 `data_home` 路径改为容器内路径（原为 `/tmp`），用于单元测试数据挂载，不影响模型代码。

### 2.3 安装

Aviary 依赖 `torch-scatter`，需从源码编译安装（使用沐曦 MACA 工具链的 CUDA 兼容接口）：

```bash
pip3 install --upgrade pip
pip3 install --upgrade setuptools==69.5.1
FORCE_CUDA=1 pip3 install --no-build-isolation torch-scatter
pip3 install matminer pyxtal
pip3 install ./
```

---

## 3. 快速验证

### 3.1 单元测试

```bash
cd aviary
pytest -vvv tests/
```

> 说明：`tests/` 为上游源码自带测试（其中 `conftest.py` 数据路径按 2.2 节调整）。

### 3.2 Roost 示例训练与评估

使用上游源码自带的示例脚本与示例数据：

```bash
python examples/roost-example.py \
    --train --evaluate \
    --data-path examples/inputs/examples.csv \
    --targets E_f \
    --tasks regression \
    --losses L1 \
    --robust \
    --epoch 10
```

训练与评估正常完成即表示沐曦 GPU 环境运行成功。

---

## 4. 使用指南

Aviary 通过统一接口提供多种模型（`--model` 可选 roost、wren、cgcnn 等），示例脚本支持训练、评估与预测模式（`--train` / `--evaluate` / `--predict`）。详细参数见上游 README 与 `examples/` 目录。

---

## 5. 常见问题

- **GPU 不可用**：确认容器已映射 `--device=/dev/dri --device=/dev/mxcd --group-add video`，宿主机 MXMACA 驱动正常
- **`torch-scatter` 安装失败**：确认使用 `FORCE_CUDA=1 pip3 install --no-build-isolation torch-scatter`，且镜像 MACA 工具链已就绪
- **显存不足（OOM）**：减小 `--batch-size`

---

## 6. 参考资源

- Aviary 上游仓库：[https://github.com/CompRhys/aviary](https://github.com/CompRhys/aviary)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其核心源码的适配修改（第 2.2 节所述测试数据路径调整随本仓库一并提供）。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
