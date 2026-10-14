> 日凿一窍，七日而浑沌死

[English Version](README.md)

# Godcore Infrastructure / 浑沌

Godcore-infrastructure（浑沌）是一个双后端（MLX和CUDA）的模型检查点（Checkpointer）和数据加载器（Dataloader）基础设施项目。它提供了强大的特性，包括应对 `SIGINT/SIGTERM` 中断的紧急保存机制，支持断点续训，以及自适应噪声调度器。

## 主要特性

- **双后端支持**: 兼容 MLX (Apple Silicon) 和 CUDA (PyTorch)。
- **可靠的检查点管理**: 统一管理模型权重和数据加载器状态，支持被意外终止时进行紧急保存。
- **无缝断点续训**: 自动计算跳过已处理的 batch，恢复训练进度。
- **自适应噪声调度器**: 基于正态分布采样进行自适应噪声注入的调度系统。

## 安装

可以通过 `pip` 在本地进行安装：

```bash
# 基础安装
pip install .

# 安装 MLX 支持
pip install .[mlx]

# 安装 CUDA (PyTorch) 支持
pip install .[cuda]
```

## 使用方法

在您的训练脚本中使用 `Checkpointer` 进行状态管理：

```python
from training.core.checkpoint import Checkpointer
from training.core.args import get_training_parser

# 解析参数
parser = get_training_parser()
args = parser.parse_args()

# 初始化检查点管理器
checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix)
checkpointer.register_args(args)

# 注册模型和优化器
# checkpointer.register_model("my_model", model)
# checkpointer.register_optimizer("my_optim", optimizer)

# 恢复最新的检查点
global_step = checkpointer.load_latest()
```
