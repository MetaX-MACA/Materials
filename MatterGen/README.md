# 使用沐曦GPU运行MatterGen的使用手册

本手册旨在指导用户在沐曦（MetaX）GPU上运行微软MatterGen模型，实现无机材料的生成式设计。

[MatterGen](https://github.com/microsoft/mattergen)项目由第三方以开源许可证发布，我方未对其源代码作任何修改。您可通过以下链接 https://github.com/microsoft/mattergen 查阅该项目的源代码、许可证全文、版权与归属声明及其他声明文件。

---

## 1. MatterGen简介

MatterGen是微软研究院开发的基于扩散模型的生成式AI模型，用于无机材料设计。与传统的高通量筛选不同，MatterGen能够直接根据目标性能生成晶体结构，实现材料的“逆向设计”。其核心功能包括：

- **无条件生成**：从预训练模型采样生成稳定的晶体结构
- **单条件生成**：通过微调，使生成过程受化学组成、空间群、带隙、磁密度等特定性质约束
- **多条件生成**：通过微调，使生成过程受化学组成、空间群、带隙、磁密度等多种特定性质约束

该模型已发表于*Nature*，并成功用于合成新型材料TaCr₂O₆，实验测得体积模量与设计值误差小于20%。
> 由于沐曦GPU对CUDA生态的良好兼容性，MatterGen这类基于PyTorch的模型理论上可以无缝迁移运行。

---

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

### 2.3 安装Git LFS

MatterGen仓库使用Git LFS存储模型权重和数据集，必须先安装：

```bash
sudo apt install git-lfs
git lfs install
```

验证安装：
```bash
git lfs --version
# 应输出类似 git-lfs/3.0.2
```

### 2.4 安装MatterSim仓库
MatterGen依赖MatterSim，按以下步骤安装MatterSim。

```bash
git clone https://github.com/microsoft/mattersim
cd mattersim
git checkout v1.2.5
sed -i 's/torchvision>=0\.17\.0/torchvision>=0.15.0/g' pyproject.toml #降低torchvision版本。
pip install .
```

### 2.5 安装MatterGen仓库
#### 2.5.1 下载MatterGen仓库
```bash
git clone https://github.com/microsoft/mattergen
cd mattergen
git checkout v1.0.3
```

#### 2.5.2 修改 MatterSim `pyproject.toml` 文件

由于maca-pytorch基础镜像已包含兼容的PyTorch环境，需要修改 `pyproject.toml` 中的 `torch` 相关组件，避免版本冲突。具体操作如下：
```bash
# 一些组件固定版本
"lmdb=1.8.1"
"monty>=2024.7.30 ",  # keep up-to-date together with pymatgen, atomate2
"numpy>=2.0",  # pin numpy before breaking changes in 2.0
"pymatgen==2024.11.13",
"SMACT==3.2.0",

# 注释以下各行。
#"mattersim>=1.1",     # 使用适配沐曦GPU的安装版本
#"torch==2.2.1+cu118; sys_platform == 'linux'",
#"torchvision==0.17.1+cu118; sys_platform == 'linux'",
#"torchaudio==2.2.1+cu118; sys_platform == 'linux'",
#"torch==2.4.1; sys_platform == 'darwin'",
#"torchvision==0.19.1; sys_platform == 'darwin'",
#"torchaudio==2.4.1; sys_platform == 'darwin'",
#"torch_cluster",
#"torch_geometric>=2.5",
#"torch_scatter",
#"torch_sparse",

#[tool.uv.sources]
#torch = { index = "pytorch_linux",  marker = "sys_platform == 'linux'" }
#torchvision = { index = "pytorch_linux",  marker = "sys_platform == 'linux'" }
#torchaudio = { index = "pytorch_linux",  marker = "sys_platform == 'linux'" }
#pyg-lib = [
#  { url = "https://data.pyg.org/whl/torch-2.2.0%2Bcu118/pyg_lib-0.4.0%2Bpt22cu118-cp310-cp310-linux_x86_64.whl",  marker = "sys_platform == 'linux'" },
#  { url = "https://data.pyg.org/whl/torch-2.4.0%2Bcpu/pyg_lib-0.4.0%2Bpt24-cp310-cp310-macosx_14_0_universal2.whl",  marker = "sys_platform == 'darwin'" }
#]
#torch_cluster = [
#  { url = "https://data.pyg.org/whl/torch-2.2.0%2Bcu118/torch_cluster-1.6.3%2Bpt22cu118-cp310-cp310-linux_x86_64.whl",  marker = "sys_platform == 'linux'" },
#  { url = "https://data.pyg.org/whl/torch-2.4.0%2Bcpu/torch_cluster-1.6.3-cp310-cp310-macosx_10_9_universal2.whl",  marker = "sys_platform == 'darwin'"  }
#]
#torch_scatter = [
#  { url = "https://data.pyg.org/whl/torch-2.2.0%2Bcu118/torch_scatter-2.1.2%2Bpt22cu118-cp310-cp310-linux_x86_64.whl",  marker = "sys_platform == 'linux'" },
#  { url = "https://data.pyg.org/whl/torch-2.4.0%2Bcpu/torch_scatter-2.1.2-cp310-cp310-macosx_10_9_universal2.whl",  marker = "sys_platform == 'darwin'"  }
#]
#torch_sparse = [
#  { url = "https://data.pyg.org/whl/torch-2.2.0%2Bcu118/torch_sparse-0.6.18%2Bpt22cu118-cp310-cp310-linux_x86_64.whl",  marker = "sys_platform == 'linux'" },
#  { url = "https://data.pyg.org/whl/torch-2.4.0%2Bcpu/torch_sparse-0.6.18-cp310-cp310-macosx_11_0_universal2.whl",  marker = "sys_platform == 'darwin'"  }
#]
#[[tool.uv.index]]
#name = "pytorch_linux"
#url = "https://download.pytorch.org/whl/cu118"
#explicit = true
```
#### 2.5.3 安装MatterGen
```bash
pip install .
```
> MatterGen官方推荐使用`uv`进行环境管理，也可按官方脚本安装。

#### 2.5.4 安装PyG库
从沐曦开发者社区[沐曦资源中心](https://developer.metax-tech.com/softnova/search?package_name=pyg)中下载对应操作系统和torch版本的maca-pyg包，并解压。
```bash
pip install maca-pyg-$PYG_VERSION/wheel/*.whl #PYG_VERSION为相应版本。
```
查看已安装torch组件
```bash
pip list | grep torch
```
预期可见以下组件已安装好。
```bash
causal_conv1d             1.5.4+metax3.8.2.2torch2.8
dropout_layer_norm        0.1+metax3.8.2.2torch2.8
flash_attn                2.6.3+metax3.8.2.2torch2.8
flash_attn_3              3.0.0+metax3.8.2.2torch2.8
flash_kda                 0.0.1+metax3.8.2.2torch2.8
flash-linear-attention    0.5.0+metax3.8.2.2torch2.8
flash_mla                 1.0.1+metax3.8.2.2torch2.8
flash_moba                2.0.0+metax3.8.2.2torch2.8
flash_prefill             1.0+metax3.8.2.2torch2.8
flashinfer                0.2.6+metax3.8.2.2torch2.8
fused_dense_lib           2.6.3+metax3.8.2.2torch2.8
hstu_attn                 1.0+metax3.8.2.2torch2.8
mamba_ssm                 2.2.4+metax3.8.2.2torch2.8
mctlassEx                 0.1.1+metax3.8.2.2torch2.8
pytorch-lightning         2.0.6
rotary_emb                0.1+metax3.8.2.2torch2.8
sageattention             2.0.1+metax3.8.2.2torch2.8
spargeattention           2.0.1+metax3.8.2.2torch2.8
spconv                    2.1.0+metax3.8.2.2torch2.8
torch                     2.8.0+metax3.8.2.2
torch_cluster             1.6.3
torch-ema                 0.3
torch-geometric           2.7.0
torch-runstats            0.2.0
torch_scatter             2.1.2
torch-sim-atomistic       0.6.1
torch_sparse              0.6.18
torch_spline_conv         1.2.2
torchaudio                2.4.1+metax3.8.2.2
torchcodec                0.6.0+metax3.8.2.2
torchmetrics              1.9.0
torchvision               0.15.1+metax3.8.2.2
xentropy_cuda_lib         0.1+metax3.8.2.2torch2.8
xformers                  0.0.22+metax3.8.2.2torch2.8
```

### 2.6 验证沐曦GPU可用性

运行以下Python命令检查GPU是否被正确识别：

```python
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.device_count())  # 显示可用GPU数量
print(torch.cuda.get_device_name(0))  # 显示GPU型号
```

沐曦GPU的软件栈MXMACA原生兼容CUDA API，因此`torch.cuda`调用应正常工作。

---

## 3. 下载预训练模型

MatterGen提供多个预训练checkpoint，可通过Git LFS或Hugging Face下载。

### 3.1 可用模型列表

| 模型名称 | 说明 |
|---------|------|
| `mattergen_base` | 在Alex-MP-20上训练的无条件基模型 |
| `mp_20_base` | 在MP-20上训练的无条件基模型 |
| `chemical_system` | 基于化学系统微调 |
| `space_group` | 基于空间群微调 |
| `dft_band_gap` | 基于DFT带隙微调 |
| `dft_mag_density` | 基于DFT磁密度微调 |
| `ml_bulk_modulus` | 基于ML预测的体积模量微调 |

### 3.2 下载模型权重

以`mattergen_base`为例：

```bash
git lfs pull -I checkpoints/mattergen_base --exclude=""
```

或通过Hugging Face自动下载（在生成命令中指定`--pretrained-name`时自动触发）。

---

## 4. 材料生成

### 4.1 无条件生成

从预训练基模型采样生成材料：

```bash
export STRUCTURES_PATH=/opt/mattergen/data-release/cifs/mattergen/mattergen_base
export PRETRAINED_MODELS_PATH=/opt/mattersim/pretrained_models

export MODEL_NAME=mattergen_base
export RESULTS_PATH=results/

mattergen-generate $RESULTS_PATH \
    --pretrained-name=$MODEL_NAME \
    --batch_size=16 \
    --num_batches=1 \
    --device=cuda
```

**参数说明**：
- `--batch_size`：批量大小，根据GPU显存调整，沐曦C600（144GB显存）可尝试更大批次
- `--num_batches`：生成批次数量
- `--device=cuda`：指定使用GPU

**输出文件**：
- `generated_crystals_cif.zip`：CIF格式结构文件
- `generated_crystals.extxyz`：EXTXYZ格式结构文件
- `generated_trajectories.zip`（可选）：完整去噪轨迹

### 4.2 单属性条件生成

以磁密度为目标属性（目标值0.15）为例：

```bash
export MODEL_NAME=dft_mag_density
export RESULTS_PATH="results/$MODEL_NAME/"

mattergen-generate $RESULTS_PATH \
    --pretrained-name=$MODEL_NAME \
    --batch_size=16 \
    --device=cuda \
    --properties_to_condition_on="{'dft_mag_density': 0.15}" \
    --diffusion_guidance_factor=2.0
```

**关键参数**：
- `--diffusion_guidance_factor`：分类器无关引导系数γ。设为0时为无条件生成；增大该值使生成样本更严格符合目标属性，但会降低多样性

### 4.3 多属性条件生成

联合条件生成示例（化学系统+凸包能量）：

```bash
export MODEL_NAME=chemical_system_energy_above_hull
export RESULTS_PATH="results/$MODEL_NAME/"

mattergen-generate $RESULTS_PATH \
    --pretrained-name=$MODEL_NAME \
    --batch_size=16 \
    --device=cuda \
    --properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-O'}" \
    --diffusion_guidance_factor=2.0
```

### 4.4 使用自定义训练模型

若使用自己训练的模型，将`--pretrained-name`替换为：

```bash
--model_path=/path/to/your/model
```

---

## 5. 生成材料评估

MatterGen提供评估脚本，使用MatterSim机器学习力场进行结构弛豫并计算新颖性、唯一性、稳定性等指标。

### 5.1 下载参考数据集

```bash
# 下载TRI2024校正参考数据集（推荐）
git lfs pull -I data-release/alex-mp/reference_TRI2024correction.gz --exclude=""
```

### 5.2 运行评估

```bash
mattergen-evaluate \
    --structures_path=$RESULTS_PATH \
    --relax=True \
    --structure_matcher='disordered' \
    --save_as="$RESULTS_PATH/metrics.json" \
    --reference_dataset_path="data-release/alex-mp/reference_TRI2024correction.gz" \
    --device=cuda
```

**参数说明**：
- `--relax=True`：使用MatterSim进行结构弛豫
- `--structure_matcher='disordered'`：处理无序结构匹配
- `--reference_dataset_path`：参考数据集路径（推荐TRI2024校正方案）
- `--device=cuda`：指定GPU

**可选参数**：
- `--potential_load_path="MatterSim-v1.0.0-5M.pth"`：使用更大的5M参数MatterSim模型
- `--structures_output_path="relaxed_structures.extxyz"`：保存弛豫后的结构
- `--save_detailed_as="detailed_metrics.json"`：保存逐结构详细指标

---

## 6. 模型训练与微调

### 6.1 数据集预处理

以MP-20数据集为例：

```bash
# 下载数据集
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets

# 预处理为模型可读格式
csv-to-dataset \
    --csv-folder datasets/mp_20/ \
    --dataset-name mp_20 \
    --cache-folder datasets/cache
```

对于Alex-MP-20数据集（约60万结构）：

```bash
git lfs pull -I data-release/alex-mp/alex_mp_20.zip --exclude=""
unzip data-release/alex-mp/alex_mp_20.zip -d datasets
csv-to-dataset \
    --csv-folder datasets/alex_mp_20/ \
    --dataset-name alex_mp_20 \
    --cache-folder datasets/cache
```

> 预处理Alex-MP-20耗时约1小时。

### 6.2 从头训练基模型

```bash
mattergen-train data_module=mp_20 ~trainer.logger
```

**参数说明**：
- `~trainer.logger`：禁用W&B日志（默认启用）
- 训练输出保存在`outputs/singlerun/${日期}/${时间}/`

**Apple Silicon用户**：需添加`~trainer.strategy trainer.accelerator=mps`，沐曦GPU使用CUDA加速，无需此参数。

### 6.3 属性微调

以磁密度属性微调为例：

```bash
export PROPERTY=dft_mag_density

mattergen-finetune \
    adapter.pretrained_name=mattergen_base \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY=$PROPERTY \
    ~trainer.logger \
    data_module.properties=["$PROPERTY"]
```

**多属性微调示例**：

```bash
export PROPERTY1=dft_mag_density
export PROPERTY2=dft_band_gap

mattergen-finetune \
    adapter.pretrained_name=mattergen_base \
    data_module=mp_20 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY1=$PROPERTY1 \
    +lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.$PROPERTY2=$PROPERTY2 \
    ~trainer.logger \
    data_module.properties=["$PROPERTY1","$PROPERTY2"]
```

### 6.4 添加自定义属性

如需在自有数据上微调：

1. 在`mattergen/common/utils/globals.py`的`PROPERTY_SOURCE_IDS`中添加属性名
2. 在训练CSV文件中添加对应列
3. 重新运行`csv-to-dataset`预处理
4. 在`mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/`中添加属性配置文件

---

## 7. 沐曦GPU性能优化建议

### 7.1 批次大小调优

沐曦C600配备144GB HBM3e显存，建议：

- **生成任务**：初始`batch_size=32`，逐步增加至显存利用率达90%左右
- **训练任务**：基模型训练推荐`batch_size=512`，需配合梯度累积

```bash
# 梯度累积示例
mattergen-train data_module=alex_mp_20 \
    trainer.accumulate_grad_batches=4 \
    ~trainer.logger
```

### 7.2 数据加载优化

设置DataLoader工作进程数以提高数据加载效率：

```bash
mattergen-train data_module=mp_20 \
    data_module.num_workers=8 \
    ~trainer.logger
```

### 7.3 混合精度训练

利用沐曦GPU对FP16/FP8精度的支持：

```bash
mattergen-train data_module=mp_20 \
    trainer.precision=16 \
    ~trainer.logger
```

---

## 8. 常见问题排查

### 8.1 GPU不可用

**现象**：`torch.cuda.is_available()`返回False

**排查步骤**：
1. 确认沐曦GPU驱动已正确安装
2. 确认MXMACA软件栈已配置
3. 检查PyTorch版本是否与CUDA兼容

沐曦GPU原生兼容CUDA生态，通常无需额外适配。

### 8.2 显存不足（OOM）

**解决方案**：
- 减小`--batch_size`
- 增加梯度累积步数
- 使用更小的MatterSim模型（如1M参数版本）

### 8.3 Git LFS文件未下载

**解决方案**：
```bash
# 强制拉取特定文件
git lfs pull -I checkpoints/mattergen_base --exclude=""
# 或拉取所有LFS文件
git lfs pull
```

---

## 9. 引用

若使用MatterGen进行科研工作，请引用：

```bibtex
@article{MatterGen2025,
  author  = {Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and Sordillo, Roberto and Sun, Lixin and Smith, Jake and Nguyen, Bichlien and Schulz, Hannes and Lewis, Sarah and Huang, Chin-Wei and Lu, Ziheng and Zhou, Yichi and Yang, Han and Hao, Hongxia and Li, Jielan and Yang, Chunlei and Li, Wenjie and Tomioka, Ryota and Xie, Tian},
  journal = {Nature},
  title   = {A generative model for inorganic materials design},
  year    = {2025},
  doi     = {10.1038/s41586-025-08628-5},
}
```

## 10. 参考资源

- GitHub仓库：[https://github.com/microsoft/mattergen](https://github.com/microsoft/mattergen)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码，且不涉及对其源代码的任何修改。您按照本文档配置、部署或使用相关软件时，应遵守适用许可证规定的条款及条件。相关软件的源代码、许可证全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
