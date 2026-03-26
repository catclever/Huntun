from setuptools import setup

# This setup file logically decouples the training/core submodule.
# It allows this independent repository to be installed via `pip install .`
# and flawlessly preserves the exact `training.core` Python namespace.

setup(
    name="godcore-infrastructure",
    version="0.1.0",
    description="Dual-Backend (MLX/CUDA) Checkpointer and Dataloader Infrastructure",
    author="Kael",
    packages=["training.core"],
    # Maps the 'training.core' namespace directly to the root of this git repository (.)
    package_dir={"training.core": "."},
    install_requires=[
        "numpy",
        "pandas",
        "pyarrow", # For parquet loading in dataloader
    ],
    extras_require={
        "mlx": ["mlx"],
        "cuda": ["torch", "safetensors"]
    }
)
