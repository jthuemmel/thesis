import torch
from utils.components import *

class MaskedPredictor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # embeddings
        self.latent_tokens = torch.nn.Embedding(cfg.num_latents, cfg.dim)
        self.mask_token = torch.nn.Embedding(1, cfg.dim)
        self.coordinate_embedding = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, cfg.dim)

        # grouped linear projections
        self.proj_in = GroupLinear(cfg.dim_in, cfg.dim, cfg.num_features)
        self.proj_out = GroupLinear(cfg.dim, cfg.dim_out, cfg.num_features)

        # Transformer blocks
        self.network = torch.nn.Sequential(*[
            InterfaceBlock(cfg.dim, dim_heads=cfg.dim_heads, num_blocks= cfg.num_compute_blocks)
            for _ in range(cfg.num_layers)
            ])
        
        # Initialization
        self.apply(self.base_init)

    @staticmethod
    def base_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
        if isinstance(m, torch.nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)
        if isinstance(m, ConditionalLayerNorm) and m.linear is not None:
            torch.nn.init.trunc_normal_(m.linear.weight, std = 1e-8)

    def forward(self, tokens: torch.FloatTensor, visible: torch.BoolTensor, coordinates: torch.LongTensor) -> torch.FloatTensor:
        # embed tokens per-group
        src = self.proj_in(tokens, group_by = coordinates[..., 0])
        # src where visible else mask
        x = torch.where(visible, src, self.mask_token.weight)
        # add position codes
        x = x + self.coordinate_embedding(coordinates)
        # expand latent vectors
        latents = self.latent_tokens.weight.expand(tokens.size(0), -1, -1)
        # process
        x, latents = self.network((x, latents))
        # project per-group
        out = self.proj_out(x, group_by = coordinates[..., 0])
        return out
    