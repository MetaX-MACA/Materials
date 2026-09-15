# Material

本项目汇集了多个可用于**沐曦（MetaX）GPU** 的材料科学计算解决方案。我们专注于构建高性能、易适配的AI for Material 软件栈，助力您在材料科学等前沿领域的科研与开发工作。

## 📂 项目结构与说明

本文件夹以**聚合分发**方式提供多个相互独立的开源项目，各项目作为独立程序分别置于各自子目录中，我方仅将其聚合打包传输，并未将其组合为单一作品。各项目分别受其自带许可证约束，您在使用或分发任一项目前，须遵守该项目对应许可证的条款。如各项目内另附有我方提供的Patch文件或适配文件，系针对相应开源项目的修改或组合，此文件的许可证放置于对应文件夹中。


### 框架/应用支持

以下是目前已适配沐曦GPU的核心AI框架与科学计算模型概览：

| 模型/工具 | 类型 | 用途 |
| --- | --- | --- |
| **[ALIGNN](https://github.com/MetaX-MACA/Materials/tree/main/ALIGNN)** | 性质预测 | 基于原子图和原子线图建模二体与三体相互作用，用于形成能、带隙等材料性质预测 |
| **[Aviary](https://github.com/MetaX-MACA/Materials/tree/main/Aviary)** | 模型库 | 为多种材料发现模型提供统一接口，支持模型的训练、推理与对比 |
| **[EquiformerV2](https://github.com/MetaX-MACA/Materials/tree/main/EquiformerV2)** | 机器学习势 | 基于等变 Transformer 预测原子体系的能量和力，主要用于催化体系的结构—能量与力预测 |
| **[MACE](https://github.com/MetaX-MACA/Materials/tree/main/MACE)** | 机器学习势 | 用于机器学习原子间势的训练与推理，支持能量、原子力和应力预测，以及结构优化与分子动力学模拟 |
| **[MatGL](https://github.com/MetaX-MACA/Materials/tree/main/MatGL)** | 模型库 | 提供材料图神经网络、预训练性质预测模型和机器学习势，支持模型训练、推理与微调 |
| **[MatRIS](https://github.com/MetaX-MACA/Materials/tree/main/MatRIS)** | 机器学习势 | 预测材料体系的能量、原子力、应力和磁矩，支持结构优化与原子尺度模拟 |
| **[MatterGen](https://github.com/MetaX-MACA/Materials/tree/main/MatterGen)** | 结构生成 | 生成多样化的无机晶体候选结构，并支持根据化学体系或目标性质进行条件生成 |
| **[MatterSim](https://github.com/MetaX-MACA/Materials/tree/main/MatterSim)** | 机器学习势 | 预测材料体系的能量、原子力和应力，支持结构优化、分子动力学及声子等材料性质计算 |

>更多框架和应用正在持续适配与添加中，敬请关注。


---

## 🚀 快速开始

### 环境与依赖

*   **硬件**: 需具备沐曦（MetaX）GPU（如曦云C系列、曦索X系列科学计算GPU）。
*   **基础软件**: Linux操作系统，沐曦GPU运行时环境及MXMACA软件栈。
*   **主要依赖**: Python 3.8+，**PyTorch**, **PaddlePaddle**，以及各子项目特定的依赖库。

### 安装与使用

具体安装与使用步骤，请参阅各子项目文件夹内的独立指南（如`README.md`或`UserGuide`）。

---

## ⚠️ 合规与许可证声明

在使用本仓库中的任何资源前，请仔细阅读并遵守以下声明：

1.  **独立项目**：本仓库中的每个子项目均为独立的开源项目，保留其原有的许可证和版权声明。我方仅提供聚合分发服务，不对这些项目的功能、安全性或合规性做额外担保。
2.  **使用责任**：您有责任理解并遵守每个子项目自带的许可证条款。在使用或分发任何子项目时，请确保完全符合其许可证要求。
3.  **修改与组合**：如子项目内包含由我方提供的Patch文件或适配文件，这些特定文件的许可证将放置于对应的子文件夹中，请在使用时一并遵循。

---

## 🤝 贡献与反馈

欢迎通过GitHub Issues或Pull Requests提出建议、报告问题或贡献新的适配模型与框架。让我们一起推动国产算力生态与AI4S领域的革新！

**感谢您对沐曦GPU生态及AI4S发展的关注与支持！**

