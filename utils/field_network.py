import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init, LayerNorm
from einops import rearrange, repeat, reduce
from dataclasses import dataclass
from utils.networks import Interface
from utils.cpe import ContinuousPositionalEmbedding
from utils.components import *

class StochasticWeatherField(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        # Latents
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)

        # Coordinates
        self.coords = ContinuousPositionalEmbedding(cfg.dim_coords, [8, 8, 16, 32], cfg.dim)
        self.query_ffn = GatedFFN(cfg.dim)

        # I/O
        self.proj_in = Embedding(cfg.num_features, cfg.dim * cfg.dim_in)
        self.norm_in = ConditionalLayerNorm(cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)

         # Self-conditioning networks
        self.query_conditioning = ConditioningNetwork(cfg.dim)
        self.latent_conditioning = ConditioningNetwork(cfg.dim)

        # Noise projection
        self.proj_noise = Linear(cfg.dim_noise, cfg.dim) if cfg.dim_noise is not None else None
        self.dim_noise = cfg.dim_noise
        self.generator = None

        # Interface Network
        self.network = ModuleList([
            Interface(cfg.dim, cfg.num_compute_blocks, dim_heads= cfg.dim_heads, dim_ctx=None) 
            for _ in range(cfg.num_layers)
        ])

        # Initialization
        self.apply(self.base_init)
        #self.apply(self.zero_init)    

    @staticmethod
    def base_init(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
            if m.bias is not None:
                init.zeros_(m.bias)

        if isinstance(m, Embedding):
            init.trunc_normal_(m.weight, std = get_weight_std(m.weight))

        if isinstance(m, LayerNorm):
            if m.bias is not None:
                init.zeros_(m.bias)
            if m.weight is not None:
                init.ones_(m.weight)

        if isinstance(m, ConditionalLayerNorm) and m.linear is not None:
            torch.nn.init.trunc_normal_(m.linear.weight, std = 1e-8)

    @staticmethod
    def zero_init(m):
        if isinstance(m, Attention):
            torch.nn.init.trunc_normal_(m.to_out.weight, std = 1e-8)
        if isinstance(m, GatedFFN):
            torch.nn.init.trunc_normal_(m.to_out.weight, std = 1e-8)

    def forward(self, 
            src: torch.Tensor, 
            src_coords: torch.Tensor, 
            tgt_coords: torch.Tensor,
            num_steps: int = 1,
            ):
        q, z = None, None
        for _ in range(num_steps):
            pred, q, z = self.step(src, src_coords, tgt_coords, q_prev=q, z_prev=z)
            q = q.detach()
            z = z.detach()
        return pred
    
    def noise_like(self, src: torch.Tensor):
        if self.dim_noise is not None:
            return torch.randn(src.size(0), 1, self.dim_noise, device=src.device, dtype=src.dtype, generator = self.generator)
        return None
    
    def to_x(self, src, src_coords):
        src_var = src_coords[:, :, 1] # variable index from [B, N, C]
        w = self.proj_in(src_var)
        w = rearrange(w, "b n (d i) -> b n d i", i = src.size(-1))
        x = torch.einsum("b n i, b n d i -> b n d", src, w)
        x = self.norm_in(x) + self.coords(src_coords)
        return x
    
    def to_q(self, tgt_coords, q_prev):
        q_init = self.coords(tgt_coords)
        q_init = q_init + self.query_ffn(q_init)
        q = self.query_conditioning(q_init, q_prev)
        return q
               
    def to_z(self, src, z_prev):
        z_init = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        if self.dim_noise is not None:
            noise = self.noise_like(src)
            z_noise = self.proj_noise(noise)
            z_init = torch.cat([z_init, z_noise], dim = 1)
        z = self.latent_conditioning(z_init, z_prev)
        return z

    def step(self, 
            src: torch.Tensor, 
            src_coords: torch.Tensor, 
            tgt_coords: torch.Tensor,
            q_prev: torch.Tensor = None,
            z_prev: torch.Tensor = None,
            ):
        
        x = self.to_x(src, src_coords)
        q = self.to_q(tgt_coords, q_prev)
        z = self.to_z(src, z_prev)
        
        # Interface network over src and query
        combined = torch.cat([x, q], dim = 1)
        for block in self.network:
            combined, z = block(combined, z)
        _, q = combined.split([src_coords.size(1), tgt_coords.size(1)], dim = 1)

        # Decode query only
        pred = self.proj_out(q)

        # Return prediction, query, and latent states
        return pred, q, z

