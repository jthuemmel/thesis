import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init
from einops import rearrange, repeat, reduce
from dataclasses import dataclass
from utils.networks import Interface
from utils.cpe import ContinuousPositionalEmbedding
from utils.components import Attention, GatedFFN, ConditionalLayerNorm
from typing import Optional

class NeuralWeatherField(Module):
    def __init__(self, decoder_cfg: dataclass, encoder_cfg: Optional[dataclass] = None):
        super().__init__()
        D = decoder_cfg.dim
        L = decoder_cfg.num_latents
        I = decoder_cfg.dim_in
        O = decoder_cfg.dim_out
        C = decoder_cfg.dim_coords
        
        # Latents
        self.latent_embedding = Embedding(L, D)

        # Coordinates
        self.src_coords = ContinuousPositionalEmbedding(model_dim=D, dim_per_coord=C, n_coords=4, learnable=True)
        self.tgt_coords = ContinuousPositionalEmbedding(model_dim=D, dim_per_coord=C, n_coords=4, learnable=True)

        # I/O
        self.proj_in = Linear(I, D)
        self.norm_in = ConditionalLayerNorm(D)
        self.proj_out = Linear(D, O)

        # Interface Networks
        self.decoder = ModuleList([
            Interface(D, decoder_cfg.num_compute_blocks, dim_heads= decoder_cfg.dim_heads, dim_ctx=decoder_cfg.dim_noise) 
            for _ in range(decoder_cfg.num_layers)
        ])

        if encoder_cfg is None:
            self.encoder = None
        else:
            self.encoder = ModuleList([
                Interface(D, encoder_cfg.num_compute_blocks, dim_heads=encoder_cfg.dim_heads, dim_ctx=encoder_cfg.dim_noise) 
                for _ in range(encoder_cfg.num_layers)
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
                ):
        # Latent
        z = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))

        # Encoder
        src = self.proj_in(src)
        src = self.norm_in(src) + self.src_coords(src_coords)

        if self.encoder is not None:
            for block in self.encoder:
                src, z = block(src, z, None)

        # Query embedding
        tgt = self.tgt_coords(tgt_coords)
        x = torch.cat([src, tgt], dim = 1)
        
        # Interface network computation
        for block in self.decoder:
            x, z = block(x, z, None)

        # Split target and project to out dim
        _, tgt = x.split([src_coords.size(1), tgt_coords.size(1)], dim = 1)
        tgt = self.proj_out(tgt)
        return tgt, z

        