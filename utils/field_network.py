import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init
from einops import rearrange, repeat, reduce
from dataclasses import dataclass
from utils.networks import Interface
from utils.components import Attention, GatedFFN, ConditionalLayerNorm
from typing import Optional

class Coordinates(Module):
    def __init__(self, dim: int, dim_coords: int, num_coords: int):
        super().__init__()
        self.embedding = Embedding(num_coords, dim_coords)
        self.linear = Linear(dim_coords, dim, bias = True)

    def forward(self, idx: torch.LongTensor):
        return self.linear(self.embedding(idx))

class NeuralWeatherField(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        # Coordinates
        self.coordinates = Coordinates(dim=cfg.dim, dim_coords=cfg.dim_coords, num_coords=cfg.num_tokens)
        self.ffn_query = GatedFFN(cfg.dim)

        # I/O
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.norm_in = ConditionalLayerNorm(cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)

        # Interface Network
        self.network = ModuleList([
            Interface(cfg.dim, cfg.num_compute_blocks, dim_heads= cfg.dim_heads) 
            for _ in range(cfg.num_layers)
        ])

        # Latents
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)
        self.norm_ctx = ConditionalLayerNorm(cfg.dim)
        self.ffn_ctx = GatedFFN(cfg.dim)

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

    def initialize_data(self, src: torch.Tensor, src_coords: torch.Tensor, tgt_coords: torch.Tensor):
        src = self.proj_in(src)
        src = self.norm_in(src) + self.coordinates(src_coords)
        query = self.coordinates(tgt_coords)
        query = query + self.ffn_query(query)
        x = torch.cat([src, query], dim = 1)
        return x

    def initialize_latents(self, ctx: torch.Tensor):
        z = repeat(self.latents.weight, "z d -> b z d", b = ctx.size(0))
        ctx = ctx + self.ffn_ctx(ctx.detach())
        z = z + self.norm_ctx(ctx)
        return z

    def forward(self, 
                src: torch.Tensor, 
                src_coords: torch.Tensor, 
                tgt_coords: torch.Tensor, 
                ctx: torch.Tensor):
        x = self.initialize_data(src, src_coords, tgt_coords)
        z = self.initialize_latents(ctx)
        for block in self.network:
            x, z = block(x, z)
        x = self.proj_out(x)
        return x, z

        