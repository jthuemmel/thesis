import torch
from typing import Optional, Tuple, Sequence

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
    

class HierarchicalDirichletMultinomial:
    def __init__(
        self,
        concentrations: Sequence[float],
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Hierarchical Dirichlet-Multinomial with independent Dirichlet priors over multiple axes.
        Builds a joint sampling distribution by taking the outer product of per-axis Dirichlet samples,
        allowing full support across the combined component space.

        Args:
            concentrations: list of concentration parameters for each axis (excluding batch axis)
            device: torch.device to place tensors on
            generator: torch.Generator for reproducibility
        """
        self.concentrations = concentrations
        self.device = device
        self.generator = generator

    def _sample_prior(self, B: int, dims: Sequence[int]) -> torch.Tensor:
        """
        Internal: sample Dirichlet weights per axis and combine them via outer products.
        Returns a flattened prior tensor of shape (B, prod(dims)), with full-rank support.
        """
        factors = []
        for c, D in zip(self.concentrations, dims):
            concentration = c * torch.ones((B, D), device=self.device)
            factor = torch._sample_dirichlet(concentration, generator=self.generator)
            factors.append(factor)

        # Combine factors across axes via elementwise multiplication in broadcasted form
        num_axes = len(dims)
        joint = factors[0].view(B, *dims[:1], *[1] * (num_axes - 1))
        for i in range(1, num_axes):
            shape_i = [1] * num_axes
            shape_i[i] = dims[i]
            joint = joint * factors[i].view(B, *shape_i)

        return joint.flatten(start_dim=1)

    def __call__(
        self,
        shape: Tuple[int, ...],
        R_src: int,
        R_tgt: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample index sets from the hierarchical Dirichlet-Multinomial distribution.

        Args:
            shape: tuple where shape[0] is batch size B and shape[1:] are component sizes for each axis
            R_src: number of samples to draw for source indices
            R_tgt: number of samples to draw for target indices (optional)

        Returns:
            src: Tensor of shape (B, R_src) with sampled flattened indices
            tgt: Tensor of shape (B, R_tgt) or None
        """
        B, *dims = shape
        assert len(dims) == len(self.concentrations), \
            f"Expected {len(self.concentrations)} axes, got {len(dims)} in shape"

        # Source sampling
        src_prior = self._sample_prior(B, dims)
        src = torch.multinomial(src_prior, num_samples=R_src, replacement=False, generator=self.generator)
        tgt = None

        if exists(R_tgt):
            # Target sampling with exclusion of source picks
            tgt_prior = self._sample_prior(B, dims)
            tgt_prior = tgt_prior.scatter(1, src, 0.0)
            tgt = torch.multinomial(tgt_prior, num_samples=R_tgt, replacement=False, generator=self.generator)

        return src, tgt
