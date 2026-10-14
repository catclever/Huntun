> 日凿一窍，七日而浑沌死

[中文版](README.zh-CN.md)

# Godcore Infrastructure / Hundun

Godcore-infrastructure (Hundun) is a Dual-Backend (MLX/CUDA) Checkpointer and Dataloader Infrastructure. It provides robust features such as emergency saving mechanisms upon catching `SIGINT/SIGTERM` signals, seamless resume-training capabilities, and an adaptive noise scheduler.

## Key Features

- **Dual-Backend Support**: Compatible with MLX (Apple Silicon) and CUDA (PyTorch).
- **Robust Checkpointer**: Unified management of model weights and dataloader states with emergency save capabilities on interrupt.
- **Seamless Resume**: Automatically calculates skipped batches and resumes training progress seamlessly.
- **Adaptive Noise Scheduler**: An adaptive noise injection scheduling system based on normal distribution sampling.

## Installation

You can install it locally via `pip`:

```bash
# Basic installation
pip install .

# For MLX support
pip install .[mlx]

# For CUDA/PyTorch support
pip install .[cuda]
```

## Usage

Use the `Checkpointer` in your training script for state management:

```python
from training.core.checkpoint import Checkpointer
from training.core.args import get_training_parser

# Parse arguments
parser = get_training_parser()
args = parser.parse_args()

# Initialize checkpointer
checkpointer = Checkpointer(out_dir=args.out_dir, prefix=args.ckpt_prefix)
checkpointer.register_args(args)

# Register models and optimizers
# checkpointer.register_model("my_model", model)
# checkpointer.register_optimizer("my_optim", optimizer)

# Resume from latest checkpoint
global_step = checkpointer.load_latest()
```
