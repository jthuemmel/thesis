import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init, Identity
from einops import rearrange, repeat, reduce
from dataclasses import dataclass
from utils.networks import Interface
from utils.cpe import ContinuousPositionalEmbedding
from utils.components import Attention, GatedFFN, ConditionalLayerNorm

class NeuralWeatherField(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        # Latents
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)

        # Coordinates
        self.coordinates = ContinuousPositionalEmbedding(dim_per_coord=cfg.dim_coords, max_positions=[6, cfg.num_features, 16, 30], model_dim=cfg.dim)        
        
        # Query
        self.query = GatedFFN(cfg.dim)

        # I/O
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.norm_in = ConditionalLayerNorm(cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)
        self.proj_noise = Linear(cfg.dim_noise, cfg.dim_noise) if cfg.dim_noise > 1 else Identity()
        
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
        if isinstance(m, ConditionalLayerNorm):
            torch.nn.init.zeros_(m.linear.weight)

    def forward(self, 
                src: torch.Tensor, 
                src_coords: torch.Tensor, 
                tgt_coords: torch.Tensor,
                noise: torch.Tensor = None
                ):
        # Latent
        z = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        
        # Encode src
        src = self.proj_in(src)
        src = self.norm_in(src) + self.coordinates(src_coords)

        # Query tgt 
        tgt = self.coordinates(tgt_coords)
        tgt = tgt + self.query(tgt)

        # Interface network computation over src and tgt
        x = torch.cat([src, tgt], dim = 1)
        noise = self.proj_noise(noise) if noise is not None else None
        for block in self.network:
            x, z = block(x, z, noise)
        
        # Decode tgt only
        _, tgt = x.split([src_coords.size(1), tgt_coords.size(1)], dim = 1)
        tgt = self.proj_out(tgt)
        return tgt, z
