import os
import json
import signal
import sys
import argparse
from typing import Dict, Any



class ResumedDataLoader:
    """Wraps a dataloader to transparently skip a specified number of batches and maintain __len__."""
    def __init__(self, dataloader, skip_steps):
        self.dataloader = dataloader
        self.skip_steps = skip_steps

    def __iter__(self):
        it = iter(self.dataloader)
        for _ in range(self.skip_steps):
            try:
                next(it)
            except StopIteration:
                break
        return it

    def __len__(self):
        # We return the original length so tqdm displays correctly, 
        # though ideally it could be max(0, len(self.dataloader) - self.skip_steps)
        # Keeping original length keeps global epoch math consistent in some scripts.
        return len(self.dataloader)

    def set_epoch(self, epoch):
        if hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
            self.dataloader.sampler.set_epoch(epoch)

class Checkpointer:
    """
    A unified checkpointer that manages saving/loading Model weights and Dataloader states.
    It automatically catches SIGINT/SIGTERM to perform emergency saves before exiting.
    """
    def __init__(self, out_dir: str, prefix: str = None, keep_last_k: int = 5):
        self.out_dir = out_dir
        self.keep_last_k = keep_last_k
        
        if prefix == "TIMESTAMP":
            from datetime import datetime
            self.prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.prefix = "" if prefix is None else prefix
            
        os.makedirs(self.out_dir, exist_ok=True)
        
        self._models = {}
        self._dataloaders = {}
        self._optimizers = {}
        self._args = None
        self.current_step = 0

        # Register robust interrupt traps globally
        self._setup_signals()

    def register_model(self, name: str, model):
        """Register an MLX nn.Module to be saved/loaded as .safetensors"""
        self._models[name] = model
        
    def register_dataloader(self, name: str, dataloader):
        """Register a custom dataloader that implements state_dict/load_state_dict"""
        self._dataloaders[name] = dataloader
        
    def register_optimizer(self, name: str, optimizer):
        """Register an MLX optimizer to serialize its momentum/variance state dicts."""
        self._optimizers[name] = optimizer
        
    def register_args(self, args: argparse.Namespace):
        """Register parsed CLI arguments to be saved inside the checkpoint."""
        if hasattr(args, 'seed') and args.seed is None:
            import time
            args.seed = int(time.time() * 1000) % 1000000
            print(f"No seed provided. Auto-generated fresh random seed: {args.seed}")
        self._args = args

    def _setup_signals(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n[Checkpointer] Caught kill signal! Taking over to perform emergency save...")
        self.save(self.current_step, is_emergency=True)
        sys.exit(0)
        
    def set_step(self, step: int):
        """Explicitly sets the current global step counter for emergency checkpointing."""
        self.current_step = step

    def step(self):
        """Advances the internal step counter and automatically saves if save_steps is reached."""
        self.current_step += 1
        if self._args and hasattr(self._args, 'save_steps'):
            if self.current_step % self._args.save_steps == 0:
                self.save(self.current_step)
        return self.current_step

    def save(self, step: int, is_emergency: bool = False):
        """Saves registered models and dataloaders."""
        self.current_step = step
        base_name = f"step_{step}_emergency" if is_emergency else f"step_{step}"
        folder_name = f"{self.prefix}_{base_name}" if self.prefix else base_name
        save_path = os.path.join(self.out_dir, folder_name)
        
        if is_emergency:
            print(f"\n[Interrupt] Caught kill signal at step {step}! Saving emergency checkpoint to {save_path}...")
            
        os.makedirs(save_path, exist_ok=True)
        
        # Save models
        for name, model in self._models.items():
            if hasattr(model, "save_weights"):
                try:
                    model.save_weights(f"{save_path}/{name}.safetensors")
                except RuntimeError as e:
                    if "without a primitive" in str(e):
                        print(f"Warning: Could not save model '{name}' due to un-evaluated compilation arrays. Its state will be discarded for this emergency checkpoint.")
                    else:
                        raise e
            else:
                try:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        torch.save(model.state_dict(), f"{save_path}/{name}.pt")
                    else:
                        print(f"Warning: Unrecognized model type for {name}")
                except ImportError:
                    print(f"Warning: Model {name} looks like PyTorch but torch is not installed.")
                    
            # Check if model has a get_config method to save its architecture parameters
            if hasattr(model, "get_config"):
                with open(f"{save_path}/{name}_config.json", "w") as f:
                    json.dump(model.get_config(), f, indent=4)
                    
        # Save optimizers
        if self._optimizers:
            for name, opt in self._optimizers.items():
                if hasattr(opt, "state_dict"): # PyTorch optimizer
                    try:
                        import torch
                        torch.save(opt.state_dict(), f"{save_path}/{name}.pt")
                    except Exception as e:
                        print(f"Warning: Could not save PyTorch optimizer {name}: {e}")
                elif hasattr(opt, "state"): # MLX optimizer
                    try:
                        import mlx.core as mx
                        from mlx.utils import tree_flatten
                        flat_state = dict(tree_flatten(opt.state))
                        if flat_state: # don't save empty states
                            mx.save_safetensors(f"{save_path}/{name}.safetensors", flat_state)
                    except RuntimeError as e:
                        if "without a primitive" in str(e):
                            print(f"Warning: Could not save state for optimizer {name} due to un-evaluated compilation arrays.")
                        else:
                            raise e
                    except ImportError:
                        pass
            
        # Save dataloaders
        for name, loader in self._dataloaders.items():
            if hasattr(loader, "state_dict"):
                state = loader.state_dict()
                state["global_step"] = step
                with open(f"{save_path}/{name}.json", "w") as f:
                    json.dump(state, f)
                    
        # Save training args
        if self._args is not None:
            # We must convert argparse.Namespace to dict 
            with open(f"{save_path}/training_args.json", "w") as f:
                json.dump(vars(self._args), f, indent=4, ensure_ascii=False)
                
        if is_emergency:
            print(f"[Checkpoint] Emergency save completed at {save_path}. Safely exiting process.")
            sys.exit(0)
        else:
            print(f"[Checkpoint] Saved at step {step} to {save_path}")
            self._garbage_collect()
            
    def _garbage_collect(self):
        """Removes older checkpoints to save disk space, keeping only the latest `keep_last_k`."""
        if self.keep_last_k <= 0:
            return
            
        import shutil
        saved_ckpts = []
        for d in os.listdir(self.out_dir):
            d_path = os.path.join(self.out_dir, d)
            if not os.path.isdir(d_path):
                continue
                
            # Match current prefix
            if self.prefix and self.prefix != "TIMESTAMP":
                if not d.startswith(f"{self.prefix}_step_"):
                    continue
                    
            if "_step_" in d or d.startswith("step_"):
                # Parse step
                step = -1
                if "_step_" in d:
                    step_str = d.split("_step_")[1]
                else:
                    step_str = d.split("step_")[1]
                
                step_str = step_str.replace("_emergency", "")
                if step_str.isdigit():
                    step = int(step_str)
                        
                if step != -1:
                    saved_ckpts.append((step, d_path))
                    
        # Sort by step ascending
        saved_ckpts.sort(key=lambda x: x[0])
        
        # Delete if exceeding limits
        if len(saved_ckpts) > self.keep_last_k:
            for step, path in saved_ckpts[:-self.keep_last_k]:
                try:
                    shutil.rmtree(path)
                    print(f"  -> [GC] Removed old checkpoint: {path} to save disk space.")
                except Exception as e:
                    print(f"  -> [GC] Failed to remove {path}: {e}")
            
    def get_latest_checkpoint_path(self) -> str:
        """Finds the path to the checkpoint with the highest global_step."""
        if not os.path.exists(self.out_dir):
            return None
            
        latest_dir = None
        max_step = -1
        
        for d in os.listdir(self.out_dir):
            d_path = os.path.join(self.out_dir, d)
            if not os.path.isdir(d_path):
                continue
                
            # Filter strictly by prefix if provided
            if self.prefix and self.prefix != "TIMESTAMP":
                expected_emergency = f"{self.prefix}_latest_emergency"
                expected_step_prefix = f"{self.prefix}_step_"
                if d != expected_emergency and not d.startswith(expected_step_prefix):
                    continue
                    
            if d.endswith("latest_emergency") or "_step_" in d or d.startswith("step_"):
                # Always attempt to extract exact step from dataloader state first
                step = -1
                for file in os.listdir(d_path):
                    if file.endswith(".json") and "dataloader" in file:
                        try:
                            with open(os.path.join(d_path, file), "r") as f:
                                state = json.load(f)
                                step = state.get("global_step", -1)
                        except:
                            pass
                            
                if step == -1:
                    import re
                    m = re.search(r'step_(\d+)', d)
                    if m:
                        step = int(m.group(1))
                        
                if step > max_step:
                    max_step = step
                    latest_dir = d
                    
        if not latest_dir:
            return None
            
        # Extract the prefix to continue seamlessly
        if latest_dir.endswith("latest_emergency"):
            if latest_dir == "latest_emergency":
                self.prefix = ""
            else:
                self.prefix = latest_dir.replace("_latest_emergency", "")
        else:
            if latest_dir.startswith("step_"):
                self.prefix = ""
            else:
                self.prefix = latest_dir.split("_step_")[0]
                
        print(f"Auto-resume overriding future saves to use prefix: '{self.prefix}' (Extracted from {latest_dir})")
        return os.path.join(self.out_dir, latest_dir)

    def load_latest(self) -> int:
        """Automatically finds and loads the checkpoint with the highest global_step."""
        load_path = self.get_latest_checkpoint_path()
        if not load_path:
            print(f"No valid checkpoints found in {self.out_dir}. Starting fresh.")
            return 0
        return self.load(load_path)
        
    def load_args(self, load_path: str, current_args: argparse.Namespace) -> argparse.Namespace:
        """
        Loads training_args.json from the checkpoint and overrides current_args.
        Prints a warning for any arguments that were modified to match the checkpoint.
        """
        args_file = os.path.join(load_path, "training_args.json")
        if not os.path.exists(args_file):
            print(f"Warning: No training_args.json found in {load_path}. Proceeding with command line args.")
            return current_args
            
        with open(args_file, "r") as f:
            saved_args_dict = json.load(f)
            
        current_args_dict = vars(current_args)
        
        # Parse raw sys.argv to find which arguments were explicitly provided by the user
        import sys
        explicit_args = set()
        for arg in sys.argv[1:]:
            if arg.startswith("--"):
                explicit_args.add(arg[2:].split('=')[0])
                
        overridden_keys = []
        preserved_keys = []
        rejected_keys = []
        
        # Define arguments that are STRICTLY tied to model architecture or dataset shape
        # These CANNOT be changed during resume.
        STRICT_MODEL_ARGS = {"embed_dim", "hidden_dim", "num_layers", "num_heads", "max_len", "seq_len", "z_dim", "tokenizer_id"}
        
        for k, v in saved_args_dict.items():
            if k in current_args_dict and current_args_dict[k] != v:
                if k in explicit_args:
                    if k in STRICT_MODEL_ARGS:
                        # Reject user's attempt to override a strict model parameter during resume
                        rejected_keys.append(f"{k}: user requested {current_args_dict[k]}, but forced to keep saved {v}")
                        current_args_dict[k] = v
                    else:
                        # Allow user to override training hyperparameters (lr, batch_size, global_attention_ratio, etc)
                        preserved_keys.append(f"{k}: {v} (saved) -> {current_args_dict[k]} (CLI override)")
                else:
                    overridden_keys.append(f"{k}: {current_args_dict[k]} (default) -> {v} (saved)")
                    current_args_dict[k] = v
                    
        if overridden_keys:
            print("Restored the following arguments from checkpoint (defaults):")
            for msg in overridden_keys:
                print(f"  {msg}")
                
        if preserved_keys:
            print("Preserved the following command-line training arguments (overriding checkpoint):")
            for msg in preserved_keys:
                print(f"  {msg}")
                
        if rejected_keys:
            print("WARNING: Rejected attempt to change STRICT model parameters during resume:")
            for msg in rejected_keys:
                print(f"  {msg}")
                
        if overridden_keys:
            print(f" [WARNING] The following arguments were OVERRIDDEN to match the checkpoint:")
            for msg in overridden_keys:
                print(f"   - {msg}")
                
        return current_args
            
    def load(self, load_path: str) -> int:
        """Loads registered models and dataloaders, returning the resumed global_step."""
        print(f"Resuming from checkpoint {load_path}...")
        step = 0
        
        for name, model in self._models.items():
            try:
                if hasattr(model, "load_weights"):
                    target_file = f"{load_path}/{name}.safetensors"
                    # Compatibility specific for the adapter to fuser rename
                    if name == "sense_fuser" and not os.path.exists(target_file):
                        fallback_file = f"{load_path}/sense_adapter.safetensors"
                        if os.path.exists(fallback_file):
                            target_file = fallback_file
                            print(f"Info: Loaded legacy '{fallback_file}' for model '{name}'.")
                    model.load_weights(target_file)
                else:
                    import torch
                    if isinstance(model, torch.nn.Module):
                        target_file = f"{load_path}/{name}.pt"
                        model.load_state_dict(torch.load(target_file, weights_only=True))
            except Exception as e:
                print(f"Warning: Could not load weights for {name}: {e}")
                
        for name, opt in self._optimizers.items():
            if hasattr(opt, "load_state_dict"): # PyTorch
                try:
                    import torch
                    target_file = f"{load_path}/{name}.pt"
                    if os.path.exists(target_file):
                        opt.load_state_dict(torch.load(target_file, weights_only=True))
                except Exception as e:
                    print(f"Warning: Could not load state for PyTorch optimizer {name}: {e}")
            elif hasattr(opt, "state"): # MLX
                try:
                    import mlx.core as mx
                    from mlx.utils import tree_unflatten
                    target_file = f"{load_path}/{name}.safetensors"
                    if os.path.exists(target_file):
                        # Load flat safetensors and rebuild the nested MLX dict structure
                        flat_state = list(mx.load(target_file).items())
                        opt.state = tree_unflatten(flat_state)
                except Exception as e:
                    print(f"Warning: Could not load state for MLX optimizer {name}: {e}")
                
        for name, loader in self._dataloaders.items():
            if hasattr(loader, "load_state_dict"):
                try:
                    with open(f"{load_path}/{name}.json", "r") as f:
                        state = json.load(f)
                        loader.load_state_dict(state)
                        # Extract the global step tracked within the dataloader state
                        step = state.get("global_step", 0)
                except Exception as e:
                    print(f"Warning: Could not load state for dataloader {name}: {e}")
                    
        # Fallback for step parsing if dataloader state missing
        if step == 0:
            import re
            m = re.search(r'step_(\d+)', load_path)
            if m:
                step = int(m.group(1))
                
        self.current_step = step
        return step


    def get_skip_steps(self, resumed_step: int, dataloader_len: int) -> int:
        """
        Calculates how many batches to skip in the current epoch when resuming.
        """
        if dataloader_len == 0:
            return 0
        return resumed_step % dataloader_len

    def setup_epoch_seed(self, epoch: int, base_seed: int = None):
        """
        Sets deterministic pseudo-random seeds for the current epoch to ensure reproducible dataloader shuffling.
        If base_seed is None, falls back to 42.
        """
        try:
            import mlx.core as mx
        except ImportError:
            mx = None
        import numpy as np
        import torch
        import random
        
        epoch_seed = (base_seed if base_seed is not None else 42) + epoch
        
        if mx is not None:
            mx.random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        random.seed(epoch_seed)
        print(f"Set epoch deterministic seed to {epoch_seed}")

    def wrap_dataloader(self, dataloader, epoch: int, global_step: int):
        """
        Sets the seed for the epoch and returns a wrapped dataloader that automatically skips 
        the already processed batches if resuming mid-epoch.
        """
        base_seed = self._args.seed if hasattr(self._args, 'seed') else None
        self.setup_epoch_seed(epoch, base_seed)
        
        # Only skip if we are on the exact start_epoch that we resumed from
        start_epoch = global_step // len(dataloader) if len(dataloader) > 0 else 0
        skip_steps = 0
        if epoch == start_epoch:
            skip_steps = self.get_skip_steps(global_step, len(dataloader))
            
        if skip_steps > 0:
            print(f"Checkpointer: Auto-skipping first {skip_steps} batches of Epoch {epoch+1}")
            return ResumedDataLoader(dataloader, skip_steps)
        return dataloader
