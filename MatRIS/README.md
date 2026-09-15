# 使用沐曦GPU运行MatRIS的使用手册

本手册指导用户在沐曦（MetaX）GPU上安装并运行 [MatRIS](https://github.com/HPC-AI-Team/MatRIS)（**Mat**erials **R**epresentation and **I**nteraction **S**imulation，材料表示与相互作用模拟基础模型），完成能量/力/应力/磁矩计算、结构优化与分子动力学模拟。

[MatRIS](https://github.com/HPC-AI-Team/MatRIS) 项目以 BSD-3-Clause 许可证开源发布，本指南使用其 V0.9 版本。由于沐曦 MXMACA 软件栈不支持 CUDA JIT 迭代器，上游源码需做少量算子适配（详见「源码适配说明」）。您可通过 https://github.com/HPC-AI-Team/MatRIS 查阅项目源代码与许可证全文。

推荐环境：MACA PyTorch 镜像（torch2.10 + Python 3.12 + Ubuntu 24.04）。

---

## 1. 环境准备

### 1.1 启动容器

```bash
docker run -it --name matris \
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

## 2. 安装 MatRIS

### 2.1 获取源码

```bash
git clone https://github.com/HPC-AI-Team/MatRIS
cd MatRIS
git checkout v0.9
```

### 2.2 源码适配说明

上游源码中 3 个文件使用了 `torch.cuda.jiterator._create_jit_fn`（CUDA JIT 迭代器），MXMACA 软件栈不支持该接口，需将其改为 `torch.jit.script` 实现（功能等价）：

| 文件 | 修改内容 |
|---|---|
| `matris/model/op/fuse_basis_func.py` | `torch.cuda.jiterator._create_jit_fn` → `torch.jit.script` |
| `matris/model/op/fuse_sigmoid_op.py` | 同上 |
| `matris/model/op/fused_silu_op.py` | 同上 |

除上述算子适配外，其余源码无需修改。

### 2.3 安装

```bash
pip3 install -e .
pip3 install numpy==1.26.4
```

---

## 3. 快速验证

### 3.1 下载预训练模型

将预训练权重 `MatRIS_10M_OAM.pth.tar` 放到 `~/.cache/matris/`：

```bash
mkdir -p ~/.cache/matris/
cp MatRIS_10M_OAM.pth.tar ~/.cache/matris/
```

### 3.2 ASE Calculator（冒烟测试）

```python
from ase.build import bulk
import torch
from matris.applications.base import MatRISCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"
calc = MatRISCalculator(
    model='matris_10m_oam',  # matris_10m_oam / matris_10m_mp
    task='efsm',             # e / ef / efs / efsm
    device=device,
)

cu = bulk('Cu', a=5.43, cubic=True)
cu.calc = calc
print(cu.get_potential_energy())   # 能量 (eV)
print(cu.get_forces())             # 力 (eV/A)
print(cu.get_stress())             # 应力 (eV/A^3)
print(cu.get_magnetic_moments())   # 磁矩 (muB)
```

> 脚本说明：以上代码为上游官方 README 示例，未做修改。

### 3.3 结构优化（Cu，FIRE，500 步）

```bash
python relax.py
```

脚本使用 `StructOptimizer(model='matris_10m_oam', task='efsm', optimizer='FIRE')` 优化铜块体结构，日志输出 `Relax completed in <耗时> seconds`。

> 脚本说明：`relax.py` 为仓库测试脚本，基于上游官方示例编写，增加了计时日志并将收敛阈值收紧为 `fmax=6e-6`（官方示例为 0.05）；上游源码本身未做修改（算子适配见 2.2 节）。

### 3.4 分子动力学（NVT 300K，1000 步）

```bash
python md.py
```

日志输出 `MD completed in <耗时> seconds`。

> 脚本说明：`md.py` 为仓库测试脚本，基于上游官方示例编写、仅增加计时日志；上游源码本身未做修改。
> 运行上述结构优化与分子动力学脚本前，请设置环境变量 `MACA_VISIBLE_DEVICES=0`、`MACA_EXT_DIRECT_DISPATCH=0`。

---

## 4. 使用指南

### 4.1 结构优化

```python
from ase.build import bulk
import torch
from matris.applications.relax import StructOptimizer

matris_opt = StructOptimizer(
    model="matris_10m_oam",
    task="efsm",
    optimizer="FIRE",  # FIRE, BFGS ...
    device="cuda",
)

atom = bulk('Cu', a=5.43, cubic=True)
opt_result = matris_opt.relax(
    atoms=atom,          # pymatgen.Structure 或 ase.Atoms
    verbose=True,
    steps=500,
    fmax=0.000006,
    relax_cell=True,
    ase_filter="FrechetCellFilter",
)
```

### 4.2 分子动力学

```python
from ase.build import bulk
from matris.applications import MolecularDynamics

atom = bulk('Cu', a=5.43, cubic=True)
md = MolecularDynamics(
    atoms=atom,
    model="matris_10m_oam",
    ensemble="nvt",      # nvt, nve ...
    temperature=300,     # K
    timestep=1,          # fs
    trajectory="md_out.traj",
    logfile="md_out.log",
    loginterval=100,
    task="efsm",
    device="cuda",
)
md.run(1000)
```

### 4.3 使用提示

- 预训练模型可选：`matris_10m_omat`、`matris_10m_oam`、`matris_10m_mp`
- `task` 按需选择：`e`（能量）/ `ef` / `efs` / `efsm`（能量+力+应力+磁矩）

---

## 5. 常见问题

- **GPU 不可用**：确认容器已映射 `--device=/dev/dri --device=/dev/mxcd --group-add video`，宿主机 MXMACA 驱动正常
- **`jiterator` 报错**：确认已应用第 2.2 节的算子适配（`torch.jit.script` 替代）
- **模型加载失败**：确认权重位于 `~/.cache/matris/MatRIS_10M_OAM.pth.tar`
- **显存不足（OOM）**：减小结构尺寸或 MD 步数

---

## 6. 参考资源

- MatRIS 上游仓库：[https://github.com/HPC-AI-Team/MatRIS](https://github.com/HPC-AI-Team/MatRIS)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码。本文档所述源码适配修改（第 2.2 节）随本仓库一并提供，修改与再分发应遵守 MatRIS 的 BSD-3-Clause 许可证条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
