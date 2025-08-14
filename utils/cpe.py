import torch
import math
from einops import rearrange
from typing import Optional, List, Tuple

class ContinuousPositionalEmbedding(torch.nn.Module):
    """
    Continuous sinusoidal positional encoding for N-dimensional positions.
    """
    def __init__(self, dim_per_coord: int, wavelengths: List[Tuple[int, int]] = [(1., 256)], model_dim: Optional[int] = None):
        """
        Args:
            dim_per_coord (int): Number of frequency dimensions per coordinate.
            wavelengths (list): Tuple of (min, max) wavelengths for each coordinate.
            model_dim (int): (Optional) Dimension to project the embedding to.
        """
        super().__init__()
        d_half = dim_per_coord // 2

        # Precompute per-coordinate frequency factors
        freqs = torch.stack([
            torch.exp(math.log(2 * math.pi) - math.log(lmin) - torch.linspace(0, 1, d_half) * (math.log(lmax) - math.log(lmin)))
            for lmin, lmax in wavelengths
            ])

        # register buffer and optional projection
        self.register_buffer("freqs", freqs)  # shape (n_coords, dim_per_coord // 2)
        self.embedding_dim = len(wavelengths) * (d_half * 2) #make sure the embedding dim is correct even if d_half rounds
        self.proj = torch.nn.Identity() if model_dim is None else torch.nn.Linear(self.embedding_dim, model_dim)

    def forward(self, positions: torch.Tensor):
        """
        Args:
            positions (torch.Tensor): Tensor of shape (..., n_coords) in [0, max_positions] per axis.
        Returns:
            torch.Tensor: Positional embedding of shape (..., model_dim or embedding_dim).
        """
        angles = torch.einsum("...i, i d -> ...i d", positions, self.freqs)
        emb = torch.stack((angles.sin(), angles.cos()), dim=-1)
        emb = rearrange(emb, "... n d two -> ... (n d two)")
        return self.proj(emb)