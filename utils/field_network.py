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

        # Noise conditioning
        self.proj_noise = Linear(cfg.dim_noise, cfg.dim_noise, bias = True) # bias here introduces a shared ctx for all CLN layers in the network
        self.noise_dim = cfg.dim_noise

        # Interface Network
        self.network = ModuleList([
            Interface(cfg.dim, cfg.num_compute_blocks, dim_heads= cfg.dim_heads, dim_ctx=cfg.dim_noise) 
            for _ in range(cfg.num_layers)
        ])

        # Latents
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)
        self.norm_latents = ConditionalLayerNorm(cfg.dim)
        self.ffn_latent = GatedFFN(cfg.dim)

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
                latents: Optional[torch.Tensor] = None,
                noise: Optional[torch.Tensor] = None
                ):
        # Encoder
        src = self.proj_in(src)
        src = self.norm_in(src) + self.coordinates(src_coords)
        # Query embedding
        query = self.coordinates(tgt_coords)
        query = query + self.ffn_query(query)
        # Latent with self-conditioning
        z = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        latents = latents if latents is not None else torch.zeros_like(z)
        latents = latents + self.ffn_latent(latents)
        z = z + self.norm_latents(latents)
        # Shared noise encoder
        noise = noise if noise is not None else src.new_zeros(self.noise_dim)
        noise = self.proj_noise(noise)
        # Interface network computation
        x = torch.cat([src, query], dim = 1)
        for block in self.network:
            x, z = block(x, z, noise)
        # Split target and project to out dim
        _, tgt = x.split([src.size(1), query.size(1)], dim = 1)
        tgt = self.proj_out(tgt)
        return tgt, z

        