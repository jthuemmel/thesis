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
        embedding_dim = (len(cfg.wavelengths) + 1) * cfg.dim_coords

        self.position_embedding = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, None)
        self.feature_embedding = Embedding(cfg.num_features, cfg.dim_coords)
        self.latent_embedding = Embedding(cfg.num_latents, cfg.dim)

        self.encoder = ModuleList([TransformerBlock(cfg.dim, dim_heads=cfg.dim_heads) for _ in range(cfg.num_layers)])
        self.decoder = TransformerBlock(cfg.dim, dim_heads=cfg.dim_heads)

        self.proj_src = SegmentLinear(cfg.dim_in, cfg.dim_in, cfg.num_features)
        self.proj_x = SegmentLinear(embedding_dim + cfg.dim_in, cfg.dim, cfg.num_features)
        self.norm_x = LayerNorm(cfg.dim)
        self.proj_q = SegmentLinear(embedding_dim, cfg.dim, cfg.num_features)
        self.proj_out = SegmentLinear(cfg.dim, cfg.dim_out, cfg.num_features)

        
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
        var_src, var_tgt = src_coords[..., 1], tgt_coords[..., 1]

        # concatenate embedded inputs, positions and features
        x, _ = pack([self.proj_src(src, var_src),
                  self.feature_embedding(var_src),
                  self.position_embedding(src_coords[..., (0, 2, 3)])], 
                  'b n *')
        
        q, _ = pack([self.position_embedding(tgt_coords[..., (0, 2, 3)]),
                     self.feature_embedding(var_tgt)],
                    'b m *')
        
        x = self.proj_x(x, var_src)
        x = self.norm_x(x)
        query = self.proj_q(q, var_tgt)
        latent = repeat(self.latent_embedding.weight, "z d -> b z d", b = src.size(0))
        for block in self.encoder:
            kv, _ = pack([x, latent], 'b * d')
            latent = block(latent, kv)
        query = self.decoder(query, latent)
        query = self.proj_out(query, var_tgt)
        return query