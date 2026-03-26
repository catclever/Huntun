def linear_warmup_schedule(global_step: int, peak_lr: float, warmup_steps: int) -> float:
    """
    Computes a linear learning rate warmup absolute schedule based on the global step.
    Prevents amnesic restarts when resuming from checkpoints by ignoring optimizer state.
    
    Args:
        global_step (int): The absolute training steps globally executed so far.
        peak_lr (float): The target learning rate after warmup finishes.
        warmup_steps (int): The duration (in steps) of the linear ramp.
        
    Returns:
        float: The calculated learning rate at the current step.
    """
    if warmup_steps <= 0:
        return peak_lr
        
    if global_step < warmup_steps:
        return peak_lr * (global_step / warmup_steps)
        
    return peak_lr
