# 使用沐曦GPU运行MACE的使用手册

本手册指导用户在沐曦（MetaX）GPU上安装并运行 [MACE](https://github.com/ACEsuit/mace)（Multi Atomic Cluster Expansion，机器学习原子间势模型），完成训练与推理。

[MACE](https://github.com/ACEsuit/mace) 项目由 ACEsuit 社区以 MIT 许可证开源发布，本指南使用其 v0.3.15 版本。MACE 默认使用 PyTorch + e3nn 实现，上游源码**无需任何修改**即可在沐曦 GPU 上运行。您可通过 https://github.com/ACEsuit/mace 查阅项目源代码与许可证全文。

推荐环境：MACA PyTorch 镜像（torch2.4 + Python 3.10 + Ubuntu 22.04）。

---

## 1. 环境准备

### 1.1 启动容器

```bash
docker run -it --name mace \
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

## 2. 安装 MACE

### 2.1 获取源码

```bash
git clone https://github.com/ACEsuit/mace
cd mace
git checkout v0.3.15
```

### 2.2 源码适配说明

MACE v0.3.15 基于 PyTorch + e3nn 实现，上游源码**无需修改任何内容**，可直接安装运行。

### 2.3 安装

```bash
pip3 install -e .
pip3 install numpy==1.26 transformers==5.6.0
```

如无需从源码安装，也可直接安装固定版本：

```bash
pip3 install mace-torch==0.3.15
pip3 install numpy==1.26 transformers==5.6.0
```

---

## 3. 快速验证（液态水训练）

以仓库测试数据 `test/liquid_water/dataset_1593.xyz`（1593 个液态水构型）为例，复制为训练/测试集后直接运行 `mace_run_train`：

```bash
cd test/liquid_water
cp dataset_1593.xyz train.xyz
cp dataset_1593.xyz test.xyz

mace_run_train \
    --name="water_1k_small" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=64 \
    --max_L=0 \
    --correlation=3 \
    --r_max=6.0 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --energy_key="TotEnergy" \
    --forces_key="force" \
    --batch_size=2 \
    --valid_batch_size=1 \
    --max_num_epochs=32 \
    --ema \
    --swa \
    --error_table='PerAtomMAE' \
    --default_dtype="float64" \
    --device=cuda \
    --seed=123 \
    --save_cpu
```

训练日志正常输出（loss 逐轮下降并完成 32 轮训练）即表示沐曦 GPU 环境运行成功。

> 脚本说明：`mace_run_train` 为上游源码自带的训练入口（`mace/cli/run_train.py`），未做任何修改；`dataset_1593.xyz` 为仓库提供的测试数据，非上游源码。

---

## 4. 使用指南

本章使用的 `mace_run_train`、`mace_eval_configs` 均为上游源码自带的命令行工具，未做修改；ASE 推理代码为官方 README 示例。

### 4.1 训练

```bash
mace_run_train \
    --name="MACE_model" \
    --train_file="train.xyz" \
    --valid_fraction=0.05 \
    --test_file="test.xyz" \
    --E0s="average" \
    --model="MACE" \
    --num_channels=128 \
    --max_L=1 \
    --r_max=5.0 \
    --batch_size=10 \
    --max_num_epochs=100 \
    --ema \
    --amsgrad \
    --restart_latest \
    --device=cuda
```

常用参数：`--hidden_irreps` 控制模型尺寸；`--foundation_model="small|medium|large"` 可在 MACE-MP 基础模型上微调。

### 4.2 评估

```bash
mace_eval_configs \
    --configs="your_configs.xyz" \
    --model="your_model.model" \
    --output="./your_output.xyz"
```

### 4.3 预训练基础模型推理（ASE 接口）

```python
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule('H2O')
calc = mace_mp(model="medium", default_dtype="float32", device='cuda')
atoms.calc = calc
print(atoms.get_potential_energy())
```

### 4.4 使用提示

- 多卡训练加 `--distributed`；大数据集可先用 `preprocess_data.py` 预处理为 HDF5

---

## 5. 常见问题

- **GPU 不可用**：确认容器已映射 `--device=/dev/dri --device=/dev/mxcd --group-add video`，宿主机 MXMACA 驱动正常
- **显存不足（OOM）**：减小 `--batch_size`，或改用 `--default_dtype="float32"`
- **命令不存在**：确认已 `pip3 install -e .`（或 `pip3 install mace-torch==0.3.15`），`pip3 list | grep mace-torch` 应显示 `mace-torch 0.3.15`
- **W&B 报错**：`export WANDB_MODE=offline` 或去掉 `--wandb` 参数

---

## 6. 参考资源

- MACE 上游仓库：[https://github.com/ACEsuit/mace](https://github.com/ACEsuit/mace)
- MACE 官方文档：[https://mace-docs.readthedocs.io](https://mace-docs.readthedocs.io)
- 预训练基础模型：[https://github.com/ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

若使用 MACE 进行科研工作，请引用其 NeurIPS 2022 论文（引用格式见上游仓库 README）。

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
