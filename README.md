# Godcore Infrastructure / 浑沌 (Hundun)

## 项目名称来源 / Origin of the Name

**中**：本项目名称来源于《庄子·应帝王》中的典故：“南海之帝为儵，北海之帝为忽，中央之帝为浑沌。儵与忽时相与遇于浑沌之地，浑沌待之甚善。儵与忽谋报浑沌之德，曰：‘人皆有七窍以视听食息，此独无有，尝试凿之。’日凿一窍，七日而浑沌死。”
这个故事象征着从无序（混沌）到有序（开窍）的转化过程，也隐喻了在训练深度学习模型时，打破原有状态以注入知识和结构，正如为“浑沌”开窍。

**EN**: The name of this project is inspired by a story from the ancient Chinese philosophical text *Zhuangzi*: "The Emperor of the South Sea was called Shu (Brief), the Emperor of the North Sea was called Hu (Sudden), and the Emperor of the Central Region was called Hundun (Chaos). Shu and Hu from time to time came together for a meeting in the territory of Hundun, and Hundun treated them very generously. Shu and Hu discussed how they could repay his kindness. 'All men,' they said, 'have seven openings so they can see, hear, eat, and breathe. But Hundun alone doesn't have any. Let's try boring him some!' Every day they bored another hole, and on the seventh day Hundun died."
This story symbolizes the transformation from chaos to order. In the context of deep learning, it serves as a metaphor for the training process: breaking the original unordered state (chaos) to inject knowledge and structure.

## 简介 / Introduction

**中**：Godcore-infrastructure（浑沌）是一个双后端（MLX和CUDA）的模型检查点（Checkpointer）和数据加载器（Dataloader）基础设施项目。它提供了强大的特性，包括应对 `SIGINT/SIGTERM` 中断的紧急保存机制，支持断点续训，以及自适应噪声调度器。

**EN**: Godcore-infrastructure (Hundun) is a Dual-Backend (MLX/CUDA) Checkpointer and Dataloader Infrastructure. It provides robust features such as emergency saving mechanisms upon catching `SIGINT/SIGTERM` signals, seamless resume-training capabilities, and an adaptive noise scheduler.

## 主要特性 / Key Features

- **双后端支持 (Dual-Backend Support)**: 兼容 MLX (Apple Silicon) 和 CUDA (PyTorch)。/ Compatible with MLX (Apple Silicon) and CUDA (PyTorch).
- **可靠的检查点管理 (Robust Checkpointer)**: 统一管理模型权重和数据加载器状态，支持被意外终止时进行紧急保存。/ Unified management of model weights and dataloader states with emergency save capabilities on interrupt.
- **无缝断点续训 (Seamless Resume)**: 自动计算跳过已处理的 batch，恢复训练进度。/ Automatically calculates skipped batches and resumes training progress seamlessly.
- **自适应噪声调度器 (Adaptive Noise Scheduler)**: 基于正态分布采样进行自适应噪声注入的调度系统。/ An adaptive noise injection scheduling system based on normal distribution sampling.

## 安装 / Installation

**中**：可以通过 `pip` 在本地进行安装：
**EN**: You can install it locally via `pip`:

```bash
# 基础安装 / Basic installation
pip install .

# 安装 MLX 支持 / For MLX support
pip install .[mlx]

# 安装 CUDA (PyTorch) 支持 / For CUDA/PyTorch support
pip install .[cuda]
```

## 使用方法 / Usage

**中**：在您的训练脚本中使用 `Checkpointer` 进行状态管理：
**EN**: Use the `Checkpointer` in your training script for state management:

```python
from training.core.checkpoint import Checkpointer
from training.core.args import get_training_parser

# 解析参数 / Parse arguments
parser = get_training_parser()
args = parser.parse_args()

# 初始化检查点管理器 / Initialize checkpointer
checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix)
checkpointer.register_args(args)

# 注册模型和优化器 / Register models and optimizers
# checkpointer.register_model("my_model", model)
# checkpointer.register_optimizer("my_optim", optimizer)

# 恢复最新的检查点 / Resume from latest checkpoint
global_step = checkpointer.load_latest()
```
