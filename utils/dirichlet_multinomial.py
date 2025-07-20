import torch
from typing import Optional, Tuple
from einops import repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class DirichletMultinomial:
    def __init__(self, alpha: float, device: Optional[torch.device] = None, generator: Optional[torch.Generator] = None):
        """
        Dirichlet-Multinomial distribution with low-rank parameterization
        Args:
            alpha: float (concentration parameter)
            device: torch.device
            generator: torch.Generator
        """
        self.alpha = alpha
        self.device = device
        self.generator = generator
        
    def __call__(self, shape: Tuple[int, int, int], R_src: int, R_tgt: Optional[int] = None):
        """
        Sample indices from the Dirichlet-Multinomial distribution
        Args:
            shape: tuple, Batch elements, Dirichlet components, Expansion factor (B, N, K)
            R_src: int, number of samples to draw for src
            R_tgt: int, number of samples to draw for tgt (optional)
        Returns:
            src: Tensor, sampled indices from the multinomial distribution (B, R_src)
            tgt: Tensor, sampled indices from the complementary multinomial distribution (B, R_tgt)
        """
        B, N, K = shape
        # N components with shared concentration parameter alpha
        alpha = self.alpha * torch.ones((B, N), device=self.device)
        # sample from low-rank Dirichlet-Multinomial
        src_prior = torch._sample_dirichlet(alpha, generator=self.generator)
        src_prior = src_prior.repeat_interleave(K, dim=-1)
        src = torch.multinomial(src_prior, num_samples=R_src, replacement=False, generator=self.generator)
        tgt = None
        # sample again, but avoid repeating the same samples
        if exists(R_tgt):
            tgt_prior = torch._sample_dirichlet(alpha, generator=self.generator)
            tgt_prior = tgt_prior.repeat_interleave(K, dim=-1)
            tgt_prior = tgt_prior.scatter(1, src, 0) # avoid repeating src samples
            tgt = torch.multinomial(tgt_prior, num_samples=R_tgt, replacement=False, generator=self.generator)
        return src, tgt
    
    @staticmethod
    def apply_masking(x: torch.Tensor, mask: torch.LongTensor):
        """
        Gather the input tensor x using the provided mask.
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Mask tensor of shape (B, M)
        Returns:
            masked: Masked tensor of shape (B, M, D)
        """
        expanded_mask = repeat(mask, 'b n -> b n d', d = x.size(-1))
        masked = x.gather(1, expanded_mask)
        return masked
        