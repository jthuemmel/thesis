import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init, Identity
from einops import rearrange, repeat, reduce
from dataclasses import dataclass
from utils.networks import Interface
from utils.cpe import ContinuousPositionalEmbedding
from utils.components import Attention, GatedFFN, ConditionalLayerNorm, ConditioningNetwork

class StochasticWeatherField(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        # Latents
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)

        # Coordinates
        self.coordinates = ContinuousPositionalEmbedding(dim_per_coord=cfg.dim_coords, max_positions=[6, cfg.num_features, 16, 30], model_dim=cfg.dim)        

        # I/O
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.norm_in = ConditionalLayerNorm(cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)

         # Self-conditioning networks
        self.query_network = ConditioningNetwork(cfg.dim)
        self.latent_network = ConditioningNetwork(cfg.dim)

        # Noise projection
        self.proj_noise = Linear(cfg.dim_noise, cfg.dim_noise) if cfg.dim_noise is not None else Identity()
        self.dim_noise = cfg.dim_noise

        # Interface Network
        self.network = ModuleList([
            Interface(cfg.dim, cfg.num_compute_blocks, dim_heads= cfg.dim_heads, dim_ctx=cfg.dim_noise) 
            for _ in range(cfg.num_layers)
        ])

        # Initialization
        self.apply(self.base_init)
        self.apply(self.zero_init)

    @staticmethod
    def base_init(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std = 0.02)
        if isinstance(m, Embedding):
            init.trunc_normal_(m.weight, std = 0.02)

    @staticmethod
    def zero_init(m):
        if isinstance(m, Attention):
            torch.nn.init.zeros_(m.to_out.weight)
        if isinstance(m, GatedFFN):
            torch.nn.init.zeros_(m.to_out.weight)
        if isinstance(m, ConditionalLayerNorm) and m.linear is not None:
            torch.nn.init.zeros_(m.linear.weight)

    def forward(self, 
            src: torch.Tensor, 
            src_coords: torch.Tensor, 
            tgt_coords: torch.Tensor,
            num_steps: int = 1,
            ):
        q, z, noise = None, None, None
        if self.dim_noise is not None:
            noise = torch.randn(src.size(0), 1, self.dim_noise, device=src.device, dtype=src.dtype) 
            
        for _ in range(num_steps):
            # Step through the model
            pred, q, z = self.step(src, src_coords, tgt_coords, q_prev=q, z_prev=z, noise=noise)
            q = q.detach()  # Detach query to avoid backprop across steps
            z = z.detach()
        return pred

    def step(self, 
            src: torch.Tensor, 
            src_coords: torch.Tensor, 
            tgt_coords: torch.Tensor,
            q_prev: torch.Tensor = None,
            z_prev: torch.Tensor = None,
            noise: torch.Tensor = None
            ):
        # Initialize src tokens and add positional embeddings
        src = self.proj_in(src)
        src = self.norm_in(src) + self.coordinates(src_coords)
        # Initialize latent tokens with self conditioning
        z_init = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        z = self.latent_network(z_init, z_prev) 
        # Initialize query tokens with self conditioning
        q_init = self.coordinates(tgt_coords)
        q = self.query_network(q_init, q_prev)
        # Shared noise projection
        noise = self.proj_noise(noise) if noise is not None else None
        # Interface network computation over src and query
        x = torch.cat([src, q], dim = 1)
        for block in self.network:
            x, z = block(x, z, noise)
        _, q = x.split([src_coords.size(1), tgt_coords.size(1)], dim = 1)
        # Decode query only
        pred = self.proj_out(q)
        # Return prediction, query, and latent states
        return pred, q, z

