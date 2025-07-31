import torch
import math
from einops import rearrange

class ContinuousPositionalEmbedding(torch.nn.Module):
    """
    Continuous sinusoidal positional encoding for N-dimensional positions.
    
    The overall embedding dimension is split equally across the N coordinate dimensions.
    Each coordinate is encoded with sine and cosine functions at multiple frequencies.
    """
    def __init__(self, model_dim: int, dim_per_coord: int, n_coords: int, max_positions: float = 10000.0, learnable: bool = False):
        """
        Args:
            model_dim (int): The embedding dimension of the model.
            dim_per_coord (int): The number of dimensions per coordinate.
            n_coords (int): The number of coordinates.
            max_positions (float): The maximum position
        """
        super().__init__()
        self.n_coords = n_coords
        self.max_positions = max_positions
        embedding_dim = n_coords * dim_per_coord

        # Projection layers
        if learnable:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim), 
                torch.nn.SiLU(), 
                torch.nn.Linear(embedding_dim, model_dim)
                )
        else:
            assert model_dim == embedding_dim, "model_dim must be equal to the embedding dimension: {n_coords} * {dim_per_coord}"
            self.proj = torch.nn.Identity()

        # Precompute frequency scaling factors for one coordinate.
        self.register_buffer(
            "freqs",
            torch.exp(torch.arange(0, dim_per_coord, 2).float() * (-math.log(max_positions) / dim_per_coord))
        )

    def forward(self, positions: torch.Tensor):
        """
        Args:
            positions (torch.Tensor): A tensor of continuous positions with shape (..., n_coords) in [0, 1].
        Returns:
            torch.Tensor: The sinusoidal embeddings with shape (..., embedding_dim).
        """
        if positions.shape[-1] != self.n_coords:
            raise ValueError(f"Expected last dimension size {self.n_coords}, got {positions.size(-1)}")
        
        # Compute angles for each coordinate: shape (..., n_coords, dim_per_coord/2)
        angles = torch.einsum("...i, d -> ...id", positions * self.max_positions, self.freqs)
        
        # Stack sin and cos embeddings: shape (..., n_coords, dim_per_coord/2, 2)
        emb = torch.stack((angles.sin(), angles.cos()), dim=-1)
        
        # Use einops.rearrange to interleave and flatten the last two dimensions:
        # From shape (..., n_coords, dim_per_coord/2, 2) to (..., embedding_dim)
        emb = rearrange(emb, '... n d two -> ... (n d two)')
        
        # Apply projection layers
        emb = self.proj(emb)

        return emb