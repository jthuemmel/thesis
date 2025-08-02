import torch
import math
from einops import rearrange
from typing import Optional, List

class ContinuousPositionalEmbedding(torch.nn.Module):
    """
    Continuous sinusoidal positional encoding for N-dimensional positions.
    """
    def __init__(self, dim_per_coord: int, max_positions: List[int] = [256], model_dim: Optional[int] = None):
        """
        Args:
            dim_per_coord (int): Number of frequency dimensions per coordinate.
            max_positions (list): Maximum position value for each coordinate (e.g. 256 for 256x256 grid).
            model_dim (int): (Optional) Dimension to project the embedding to.
        """
        super().__init__()
        self.n_coords = len(max_positions)
        self.dim_per_coord = dim_per_coord
        self.embedding_dim = self.n_coords * dim_per_coord

        if model_dim is not None:
            self.proj = torch.nn.Linear(self.embedding_dim, model_dim)
        else:
            self.proj = torch.nn.Identity()

        # Precompute per-coordinate frequency factors
        freqs = [
            torch.exp(torch.arange(0, dim_per_coord, 2).float() * (-math.log(mp) / dim_per_coord))
            for mp in max_positions
        ]  # list of (dim_per_coord//2,)
        self.register_buffer("freqs", torch.stack(freqs))  # shape (n_coords, dim_per_coord // 2)

    def forward(self, positions: torch.Tensor):
        """
        Args:
            positions (torch.Tensor): Tensor of shape (..., n_coords) in [0, max_positions] per axis.
        Returns:
            torch.Tensor: Positional embedding of shape (..., model_dim or embedding_dim).
        """
        if positions.shape[-1] != self.n_coords:
            raise ValueError(f"Expected last dim {self.n_coords}, got {positions.shape[-1]}")

        # Apply per-coordinate frequency bands: (..., n_coords, dim_per_coord // 2)
        angles = torch.einsum("...i, i d -> ...i d", positions, self.freqs)

        # (..., n_coords, dim_per_coord // 2, 2)
        emb = torch.stack((angles.sin(), angles.cos()), dim=-1)

        # (..., embedding_dim)
        emb = rearrange(emb, "... n d two -> ... (n d two)")

        return self.proj(emb)