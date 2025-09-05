import torch
from utils.components import *

class WeatherField(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # embeddings
        self.latent_embedding = torch.nn.Embedding(cfg.num_latents, cfg.dim)
        self.world_embedding = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, cfg.dim)

        # grouped linear projections
        self.proj_in = GroupLinear(cfg.dim_in, cfg.dim, cfg.num_features)
        self.proj_out = GroupLinear(cfg.dim, cfg.dim_out, cfg.num_features)

        # Transformer blocks
        self.encoder = torch.nn.ModuleList([TransformerBlock(cfg.dim, dim_heads=cfg.dim_heads) for _ in range(cfg.num_layers)])
        self.decoder = TransformerBlock(cfg.dim, dim_heads=cfg.dim_heads, has_skip=False)
        
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

    def forward(self, tokens, visible, shape):
        # fancy indices
        batch = torch.arange(tokens.size(0), device=tokens.device).view(-1, 1)
        indices = torch.arange(tokens.size(1), device=tokens.device).expand(tokens.size(0), -1)
        
        # positional embedding for all available coordinates
        coordinates = torch.stack(torch.unravel_index(indices, shape), dim = -1)
        world = self.world_embedding(coordinates)
        
        # embed visible values and add their positional codes
        src = self.proj_in(tokens[batch, visible], group_by = coordinates[batch, visible, 0])
        src = src + world[batch, visible]
        
        # update latents given src and latents
        latents = self.latent_embedding.weight.expand(tokens.size(0), -1, -1)
        for perceiver in self.encoder:
            kv = torch.cat([src, latents], dim = 1)
            latents = perceiver(latents, kv)

        # update world given latents
        out = self.decoder(world, latents)
        out = self.proj_out(out, group_by = coordinates[batch, visible, 0])
        return out