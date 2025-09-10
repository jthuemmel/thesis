import torch
from utils.components import *

class MaskedPredictor(torch.nn.Module):
    def __init__(self, model, world):
        super().__init__()
        # embeddings
        self.latent_tokens = torch.nn.Embedding(model.num_latents, model.dim)
        self.mask_token = torch.nn.Embedding(1, model.dim)
        self.coordinate_embedding = ContinuousPositionalEmbedding(model.dim_coords, model.wavelengths, model.dim)

        # grouped linear projections
        self.proj_in = GroupLinear(model.dim_in, model.dim, world.token_sizes['v'])
        self.proj_out = GroupLinear(model.dim, model.dim_out, world.token_sizes['v'])

        # Transformer blocks
        self.network = torch.nn.Sequential(*[
            InterfaceBlock(model.dim, dim_heads=model.dim_heads, num_blocks= model.num_compute_blocks)
            for _ in range(model.num_layers)
            ])
        
        # world attributes
        coordinates = torch.stack(torch.unravel_index(torch.arange(world.num_tokens), world.token_shape), dim=-1)
        modality_idx = coordinates[..., world.field_layout.index('v')]
        self.register_buffer("coordinates", coordinates)
        self.register_buffer("modality_idx", modality_idx)

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

    def forward(self, tokens: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
        # embed tokens per-group
        src = self.proj_in(tokens, group_by = self.modality_idx)
        # src where visible else mask
        x = torch.where(visible, src, self.mask_token.weight)
        # add position codes
        x = x + self.coordinate_embedding(self.coordinates)
        # expand latent vectors
        latents = self.latent_tokens.weight.expand(tokens.size(0), -1, -1)
        # process
        x, latents = self.network((x, latents))
        # project per-group
        out = self.proj_out(x, group_by = self.modality_idx)
        return out
    