import torch
from torch.nn import Embedding, Module, ModuleList, Linear, init, LayerNorm
from einops import rearrange, repeat, pack, unpack
from dataclasses import dataclass
from utils.cpe import ContinuousPositionalEmbedding
from utils.components import *

class WeatherField(Module):
    def __init__(self, model_cfg: dataclass):
        super().__init__()
        cfg = model_cfg.decoder

        self.position_embedding = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, cfg.dim)
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)

        self.perceiver = Perceiver(cfg.dim, cfg.num_layers)
        self.to_x = SegmentLinear(cfg.dim_in, cfg.dim, cfg.num_features)
        self.to_out = SegmentLinear(cfg.dim, cfg.dim_out, cfg.num_features)
        self.to_q = SegmentLinear(cfg.dim, cfg.dim, cfg.num_features)

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
            ):
        pos_tgt, var_tgt = tgt_coords[..., (0, 2, 3)], tgt_coords[..., 1]
        pos_src, var_src = src_coords[..., (0, 2, 3)], src_coords[..., 1]
        z = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        x = self.to_x(src, var_src) + self.position_embedding(pos_src)
        q = self.to_q(self.position_embedding(pos_tgt), var_tgt)
        q, z = self.perceiver(x, z, q)
        q = self.to_out(q, var_tgt)
        return q