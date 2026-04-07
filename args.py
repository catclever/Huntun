import argparse

def get_training_parser(description: str = "Training Script"):
    """
    Returns a unified ArgumentParser pre-populated with common training arguments.
    Additional script-specific arguments can be added to it before parsing.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save a checkpoint every X steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for shuffling chunks and data. Leave as None for true randomness across runs.")
    parser.add_argument("--z_dim", type=int, default=1024, help="Dimension of the absolute truth anchor space (z_target)")
    parser.add_argument("--fusion_main_weight", type=float, default=1.0, help="Weight of the randomly selected primary embedding logic")
    parser.add_argument("--fusion_other_weight", type=float, default=0.1, help="Weight assigned to all other non-primary embeddings")
    parser.add_argument("--config_file", type=str, default="dataset_config.json", help="Path to JSON configuration file for Dataset & Models")
    parser.add_argument("--out_dir", type=str, default="checkpoints/run", help="Output directory for checkpoints")
    parser.add_argument("--ckpt_prefix", type=str, nargs="?", const="TIMESTAMP", default=None, help="Prefix for checkpoint folders")
    parser.add_argument("--tokenizer_id", type=str, default="Qwen/Qwen2.5-7B", help="HuggingFace Tokenizer ID")
    
    # Dual Resume Mechanism
    parser.add_argument("--continue", dest="auto_resume", action="store_true", help="Automatically find and resume from the latest checkpoint or emergency save")
    parser.add_argument("--resume_from", type=str, default=None, help="Explicitly specify the path to a checkpoint directory to resume from")
    
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Learning rate linear warmup steps")
    parser.add_argument("--data_dir", type=str, default="./embs", help="Local cache directory for ModelScope downloads")
    parser.add_argument("--keep_npz_cache", action="store_true", help="If set, downloaded .npz chunks are not deleted after they are consumed")
    
    # Auxiliary Flow Loss
    parser.add_argument("--x1_weight", type=float, default=0.5, help="Weight for x1-consistency loss (0=disabled)")
    parser.add_argument("--snap_ce_weight", type=float, default=0.1, help="Weight for snap cross-entropy loss (0=disabled)")
    parser.add_argument("--t_power", type=float, default=0.5, help="Power for high-t sampling bias (1.0=uniform, 0.5=sqrt bias)")
    
    return parser
