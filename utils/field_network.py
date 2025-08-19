import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init, LayerNorm
from einops import rearrange, repeat, pack, unpack
from dataclasses import dataclass
from utils.networks import Interface
from utils.cpe import ContinuousPositionalEmbedding
from utils.components import *

class StochasticWeatherField(Module):
    def __init__(self, model_cfg: dataclass):
        super().__init__()
        cfg = model_cfg.decoder
        encoder_cfg = model_cfg.encoder

        # Latents
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)

        # Coordinates
        self.x_coords = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, cfg.dim)
        self.q_coords = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, cfg.dim)
        self.query_ffn = GatedFFN(cfg.dim)

        # I/O
        self.film = Embedding(cfg.num_features, cfg.dim_coords)
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.norm_in = ConditionalLayerNorm(cfg.dim, dim_ctx= cfg.dim_coords)
        self.proj_out1 = Linear(cfg.dim, cfg.dim_out)
        self.norm_out = ConditionalLayerNorm(cfg.dim_out, dim_ctx= cfg.dim_coords)
        self.proj_out2 = Linear(cfg.dim_out, cfg.dim_out)

         # Self-conditioning networks
        self.query_conditioning = ConditioningNetwork(cfg.dim)
        self.latent_conditioning = ConditioningNetwork(cfg.dim)

        # Noise projection
        self.proj_noise = Linear(cfg.dim_noise, cfg.dim) if cfg.dim_noise is not None else None
        self.dim_noise = cfg.dim_noise
        self.generator = None

        # Interface Networks
        self.encoder = ModuleList([
            Interface(encoder_cfg.dim, encoder_cfg.num_compute_blocks, dim_ctx=None) 
            for _ in range(encoder_cfg.num_layers)
        ]) if encoder_cfg else None

        self.decoder = ModuleList([
            Interface(cfg.dim, cfg.num_compute_blocks, dim_ctx=None) 
            for _ in range(cfg.num_layers)
        ])

        # Initialization
        self.apply(self.base_init)
        self.apply(self.zero_init)    

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
        x = self.proj_in(src)
        ctx = self.film(src_coords[..., 1])
        x = self.norm_in(x, ctx) + self.x_coords(src_coords)
        return x
    
    def to_q(self, tgt_coords, q_prev):
        q_init = self.q_coords(tgt_coords)
        q_init = q_init + self.query_ffn(q_init)
        q = self.query_conditioning(q_init, q_prev)
        return q
               
    def to_z(self, src, z_prev):
        z_init = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        if self.dim_noise is not None:
            noise = self.noise_like(src)
            z_noise = self.proj_noise(noise)
            z_init, _ = pack([z_init, z_noise], "b * d")
        z = self.latent_conditioning(z_init, z_prev)
        return z

    def to_out(self, q, tgt_coords):
        ctx = self.film(tgt_coords[..., 1])
        q = self.proj_out1(q)
        q = self.norm_out(q, ctx)
        return self.proj_out2(q)

    def step(self, 
            src: torch.Tensor, 
            src_coords: torch.Tensor, 
            tgt_coords: torch.Tensor,
            q_prev: torch.Tensor = None,
            z_prev: torch.Tensor = None,
            ):
        # Input embedding
        x = self.to_x(src, src_coords)
        z = self.to_z(src, z_prev)
        q = self.to_q(tgt_coords, q_prev)

        #Interface network over src only
        if self.encoder is not None:
            for block in self.encoder:
                x, z = block(x, z)

        # Interface network over src and query
        packed, packed_shape = pack([x, q], "b * d")
        for block in self.decoder:
            packed, z = block(packed, z)
        _, q = unpack(packed, packed_shape, "b * d")

        # Decode query only
        pred = self.to_out(q, tgt_coords)

        # Return prediction, query, and latent states
        return pred, q, z

