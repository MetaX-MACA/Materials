# 使用沐曦GPU运行EquiformerV2的使用手册

本手册指导用户在沐曦（MetaX）GPU上安装并运行 [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2)（高阶等变 Transformer，ICLR 2024），完成 OC20 S2EF 催化剂体系的能量与力预测训练和推理。

[EquiformerV2](https://github.com/atomicarchitects/equiformer_v2) 项目由 FAIR 团队以 MIT 许可证开源发布，本指南使用其提交 `9841d0e`，并依赖 [FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem) OCP 核心（提交 `5a7738f9`）。模型核心源码在沐曦 MXMACA 软件栈（原生兼容 CUDA API）上**无需修改**即可运行（数据下载/预处理脚本的适配见第 2.2 节）。您可通过上述仓库查阅项目源代码与许可证全文。

推荐环境：MACA PyTorch 镜像（torch2.4 + Python 3.10 + Ubuntu 22.04）。

---

## 1. 环境准备

### 1.1 启动容器

```bash
docker run -it --name equiformerv2 \
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

## 2. 安装 EquiformerV2

### 2.1 获取源码

```bash
git clone https://github.com/FAIR-Chem/fairchem
cd fairchem && git checkout 5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd && cd ..

git clone https://github.com/atomicarchitects/equiformer_v2
cd equiformer_v2 && git checkout 9841d0e && cd ..
```

### 2.2 源码适配说明

1. **OCP 注册导入（上游官方要求）**：EquiformerV2 官方环境配置文档要求，在 fairchem 的 `ocpmodels/common/utils.py` 第 329 行后添加两行，使训练器可找到自定义模型与训练逻辑：

   ```python
   import nets
   import oc20.trainer
   ```

   ```bash
   sed -i 'N;329i\        import nets' fairchem/ocpmodels/common/utils.py
   sed -i 'N;329i\        import oc20.trainer' fairchem/ocpmodels/common/utils.py
   ```

2. **数据脚本适配（仅小样本抽取用）**：fairchem 的 `scripts/download_data.py` 与 `scripts/preprocess_ef.py` 做了以下修改，用于抽取小样本数据（完整下载/预处理流程不受影响）：

   - 新增 `--num-data-per-worker` 参数：每个 worker 抽取指定数量的样本后停止
   - `download_data.py` 在下载文件已存在时跳过 `wget`

3. **其余源码零修改**：`equiformer_v2-9841d0e` 与 fairchem 其余核心代码未做沐曦特有适配，直接运行于 MXMACA 软件栈。

### 2.3 安装

```bash
# 1) 安装 fairchem OCP 核心
cd fairchem
pip3 install --upgrade pip
pip3 install -e ./
cd ..

# 2) 安装 EquiformerV2 依赖
pip3 install submitit==1.5.2 \
             ocp-models==0.0.3 \
             pyyaml==6.0.2 \
             matplotlib==3.9.2 \
             lmdb==1.5.1 \
             e3nn==0.5.5 \
             numba==0.60.0 \
             timm==1.0.11 \
             ase==3.23.0 \
             wandb==0.18.6 \
             tensorboard==2.18.0
```

---

## 3. 快速验证

### 3.1 数据准备

从 OC20 官方数据源下载 S2EF 数据（以 200K 训练集 400 样本为例）：

```bash
cd fairchem
python scripts/download_data.py --task s2ef --split "200k" \
    --num-workers 1 --num-data-per-worker 400 --ref-energy
cd ../equiformer_v2
```

> 脚本说明：`download_data.py` 为 fairchem 上游脚本的适配版，新增了 `--num-data-per-worker` 参数用于抽取小样本数据（见 2.2 节）；仓库测试脚本还在预处理阶段对 `scripts/preprocess_ef.py` 追加了 `xyz_logs = sorted(xyz_logs)` 以保证结果可复现。

### 3.2 训练（单卡，400 样本）

```bash
python -u -m torch.distributed.launch --nproc_per_node=1 main_oc20.py \
    --distributed \
    --num-gpus 1 \
    --mode train \
    --config-yml <equiformer_v2_N@12_L@6_M@2.yml路径> \
    --run-dir ./run/N@12_L@6_M@2 \
    --print-every 200 \
    --seed 123 \
    --amp
```

日志输出 `Total time taken: <耗时>` 即训练成功。

> 脚本说明：`main_oc20.py` 为 equiformer_v2 上游源码自带的训练/验证入口，未做修改。

### 3.3 推理（val_id，800 样本）

```bash
python -u -m torch.distributed.launch --nproc_per_node=1 main_oc20.py \
    --distributed \
    --num-gpus 1 \
    --mode validate \
    --checkpoint <eq2模型checkpoint路径> \
    --config-yml <equiformer_v2_N@12_L@6_M@2_epochs@30.yml路径> \
    --run-dir ./run/inference \
    --print-every 200 \
    --seed 123 \
    --amp
```

预训练模型可选：`eq2_83M_2M`、`eq2_31M_ec4_allmd`、`eq2_153M_ec4_allmd`。

---

## 4. 使用指南

### 4.1 训练

- 配置模板位于 `oc20/configs/s2ef/`（2M、all_md 等子目录），`equiformer_v2_N@12_L@6_M@2.yml` 为 83M 参数基础配置
- `--mode train` 训练，`--mode validate` 验证
- 多卡训练：`--nproc_per_node=<卡数>` + `--distributed --num-gpus <卡数>`

### 4.2 官方预训练 checkpoint

官方 README「Checkpoints」章节提供训练好的 checkpoint（`.pt` 文件）及对应配置，可直接通过 `--checkpoint` 加载用于 `--mode validate` 验证：

| 模型 | 数据集 | checkpoint | 对应 config |
|---|---|---|---|
| EquiformerV2 (83M) | S2EF-2M | `eq2_83M_2M.pt` | `oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2_epochs@30.yml` |
| EquiformerV2 (31M) | S2EF-All+MD | `eq2_31M_ec4_allmd.pt` | `oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml` |
| EquiformerV2 (153M) | S2EF-All+MD | `eq2_153M_ec4_allmd.pt` | `oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml` |

下载地址见上游仓库 README 的「Checkpoints」章节。

---

## 5. 常见问题

- **GPU 不可用**：确认容器已映射 `--device=/dev/dri --device=/dev/mxcd --group-add video`，宿主机 MXMACA 驱动正常
- **`cannot import name 'nets'` 等注册报错**：确认已完成第 2.2 节对 `ocpmodels/common/utils.py` 的修改
- **显存不足（OOM）**：减小配置文件中的 `batch_size`，或选用更小的模型配置（如 `eq2_31M`）
- **W&B 报错**：`export WANDB_MODE=offline`

---

## 6. 参考资源

- EquiformerV2 上游仓库：[https://github.com/atomicarchitects/equiformer_v2](https://github.com/atomicarchitects/equiformer_v2)
- EquiformerV2 论文：[https://arxiv.org/abs/2306.12059](https://arxiv.org/abs/2306.12059)
- FAIR-Chem/fairchem：[https://github.com/FAIR-Chem/fairchem](https://github.com/FAIR-Chem/fairchem)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码。本文档所述适配修改（第 2.2 节的注册导入为上游官方要求，数据脚本修改为小样本抽取功能）随本仓库一并提供，修改与再分发应遵守相应开源许可证的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
