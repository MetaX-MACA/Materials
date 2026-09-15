# 使用沐曦GPU运行ALIGNN的使用手册

本手册指导用户在沐曦（MetaX）GPU上安装并运行 [ALIGNN](https://github.com/usnistgov/alignn)（Atomistic Line Graph Neural Network，原子线图神经网络），完成材料性质预测模型的训练与推理。

[ALIGNN](https://github.com/usnistgov/alignn) 项目由美国国家标准与技术研究院（NIST）开发，遵循 NIST Terms of Use 条款发布，本指南使用其 2024.10.30 版本。上游源码在沐曦 GPU 上需做少量适配（详见「源码适配说明」）。您可通过 https://github.com/usnistgov/alignn 查阅项目源代码与许可条款全文。

推荐环境：MACA PyTorch 镜像（torch2.4 + Python 3.10 + Ubuntu 22.04）。

---

## 1. 环境准备

### 1.1 启动容器

```bash
docker run -it --name alignn \
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

## 2. 安装 ALIGNN

### 2.1 获取源码

```bash
git clone https://github.com/usnistgov/alignn
cd alignn
git checkout 2024.10.30
```

### 2.2 源码适配说明

ALIGNN 依赖 DGL，上游代码将 GPU 设备上的张量直接传给 `numpy` 转换接口。GPU 设备上的张量需先转为 CPU 张量才能转换为 numpy 数组，因此做了以下适配：

| 文件 | 修改内容 |
|---|---|
| `alignn/graphs.py` | `np.array(r)` → `np.array(r.cpu())`；`np.array(images)` → `np.array(images.cpu())` |
| `alignn/models/utils.py` | `np.diff(self.centers)` → `np.diff(self.centers.cpu())` |

另对 `alignn/tests/` 下 3 个测试文件做了配套适配（测试数据/设备处理）。除上述改动外，其余源码无需修改。

### 2.3 安装

```bash
pip3 install --upgrade pip
pip3 install jarvis_leaderboard jarvis-tools==2024.10.30 phonopy==2.39.0
pip3 install ./
pip3 install ase==3.24.0
```

---

## 3. 快速验证

以下命令均使用上游源码自带的 `train_alignn.py` 与 `alignn/examples/sample_data*` 示例数据，未做修改。

### 3.1 性质回归训练

```bash
cd alignn
train_alignn.py \
    --root_dir 'alignn/examples/sample_data' \
    --config 'alignn/examples/sample_data/config_example.json' \
    --output_dir=temp
```

### 3.2 多输出训练

```bash
train_alignn.py \
    --root_dir 'alignn/examples/sample_data_multi_prop' \
    --config 'alignn/examples/sample_data/config_example.json' \
    --output_dir=temp
```

### 3.3 力场训练

```bash
train_alignn.py \
    --root_dir 'alignn/examples/sample_data_ff' \
    --config 'alignn/examples/sample_data_ff/config_example_atomwise.json' \
    --output_dir='temp'
```

> 运行前请设置环境变量 `MACA_VISIBLE_DEVICES=0`。训练正常完成即表示沐曦 GPU 环境运行成功。

---

## 4. 使用指南

### 4.1 ALIGNN-FF 力场推理（ASE Calculator）

ALIGNN 提供基于 JARVIS-Tools 的力场接口，可在 ASE 中作为计算器使用（详见上游 README 的 ALIGNN-FF 章节）。

### 4.2 预训练模型

上游提供多个预训练模型（如 JARVIS-DFT 形成能、带隙等），下载后通过 `train_alignn.py --model <path>` 或 `run_alignn.py` 加载。

---

## 5. 常见问题

- **GPU 不可用**：确认容器已映射 `--device=/dev/dri --device=/dev/mxcd --group-add video`，宿主机 MXMACA 驱动正常
- **`numpy` 转换报错**：确认已应用第 2.2 节的 `.cpu()` 适配
- **DGL 版本冲突**：确认使用镜像自带的 DGL 版本，勿自行升级
- **显存不足（OOM）**：减小 config 中的 `batch_size`

---

## 6. 参考资源

- ALIGNN 上游仓库：[https://github.com/usnistgov/alignn](https://github.com/usnistgov/alignn)
- 沐曦开发者社区：[https://developer.metax-tech.com](https://developer.metax-tech.com)

---

本文档仅提供相关软件的配置与使用说明，不包含亦不分发前述软件的源代码或目标代码。本文档所述源码适配修改（第 2.2 节）随本仓库一并提供，修改与再分发应遵守 ALIGNN 的 NIST Terms of Use 条款及条件。相关软件的源代码、许可条款全文、版权与归属声明及其他项目文档，请以其官方网站或原始发布页面为准。

Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
