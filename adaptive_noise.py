import torch
import math

class AdaptiveNoiseScheduler:
    def __init__(self, 
                 warmup_steps=0, 
                 max_noise_ratio=0.5, 
                 noise_variance=0.05, 
                 loss_range=(0.3, 0.6), 
                 ema_decay=0.99):
        """
        Adaptive Noise Scheduler based on Normal distribution sampling.
        
        Args:
            warmup_steps: Number of global steps before any noise is injected.
            max_noise_ratio: The maximum mean value (R_max) for the noise ratio distribution.
            noise_variance: The variance (sigma^2) of the Normal distribution for sampling the noise ratio.
            loss_range: Tuple (L_low, L_high). If EMA loss <= L_low, mean is max_noise_ratio. If EMA loss >= L_high, mean is 0.
            ema_decay: The decay factor for the Exponential Moving Average of the loss.
        """
        self.warmup_steps = warmup_steps
        self.max_noise_ratio = max_noise_ratio
        # We need standard deviation for torch.randn
        self.noise_std = math.sqrt(noise_variance) 
        self.loss_low, self.loss_high = loss_range
        self.ema_decay = ema_decay
        
        self.ema_loss = None
        self.current_mu = 0.0

    def step(self, current_loss):
        """
        Update the EMA loss with the current step's loss.
        """
        if self.ema_loss is None:
            self.ema_loss = current_loss
        else:
            self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * current_loss
            
    def _compute_mu(self):
        """
        Map the current EMA loss to the mean (mu) of the noise distribution.
        """
        if self.ema_loss is None:
            return 0.0
            
        if self.ema_loss <= self.loss_low:
            return self.max_noise_ratio
        elif self.ema_loss >= self.loss_high:
            return 0.0
        else:
            # Linear interpolation
            # When loss is closer to loss_low, mu is closer to max_noise_ratio
            ratio = (self.loss_high - self.ema_loss) / (self.loss_high - self.loss_low)
            return self.max_noise_ratio * ratio

    def inject_noise(self, z_target, global_step):
        """
        Injects noise into the input tensor based on the adaptive stochastic mechanism.
        
        Args:
            z_target: The pure input tensor (e.g., past context).
            global_step: Current global training step.
            
        Returns:
            The optionally noisy tensor, and the sampled noise ratio (for logging).
        """
        if global_step < self.warmup_steps:
            return z_target, 0.0
            
        self.current_mu = self._compute_mu()
        
        # Sample the noise ratio r ~ N(mu, sigma^2)
        # We just need one ratio per batch step
        r = torch.normal(mean=self.current_mu, std=self.noise_std, size=(1,)).item()
        
        if r <= 0:
            # No noise injected (pure Teacher Forcing)
            return z_target, r
            
        # Optional: clamp r so it doesn't get ridiculously large due to normal distribution tail
        r = min(r, self.max_noise_ratio * 1.5)
        
        # Inject relative noise proportional to r and the actual variance of the signal
        # IMPORTANT: Calculate std across the batch and sequence dimensions (0, 1), NOT across the feature dimension (2)!
        # Otherwise, we capture the internal LayerNorm variance (~1.0) instead of the semantic variance between characters!
        signal_std = z_target.std(dim=(0, 1)).mean().item() + 1e-8 
        noise = torch.randn_like(z_target) * r * signal_std
        z_noisy = z_target + noise
        
        return z_noisy, r
