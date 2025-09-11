import torch
from utils.components import *
from utils.index_linear import SegmentLinear

class MaskedPredictor(torch.nn.Module):
    def __init__(self, model, world):
        super().__init__()
         # coordinates
        self.coordinate_embedding = ContinuousPositionalEmbedding(model.dim_coords, model.wavelengths, model.dim)
        self.register_buffer("coordinates", torch.stack(torch.unravel_index(torch.arange(world.num_tokens), world.token_shape), dim=-1))

        # learnable tokens
        self.latent_tokens = torch.nn.Parameter(torch.empty(1, model.num_latents, model.dim))
        self.mask_token = torch.nn.Parameter(torch.empty(1, 1, model.dim))
        
        # variable-wise segmented linear projections
        self.proj_in = SegmentLinear(model.dim_in, model.dim, self.coordinates[..., world.field_layout.index('v')])
        self.proj_out = SegmentLinear(model.dim, model.dim_out, self.coordinates[..., world.field_layout.index('v')])

        # Transformer blocks
        self.network = torch.nn.Sequential(*[
            InterfaceBlock(model.dim, dim_heads=model.dim_heads, num_blocks= model.num_compute_blocks, use_checkpoint=model.use_checkpoint)
            for _ in range(model.num_layers)
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
        if isinstance(m, torch.nn.Parameter):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
        if isinstance(m, torch.nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)
        if isinstance(m, ConditionalLayerNorm) and m.linear is not None:
            torch.nn.init.trunc_normal_(m.linear.weight, std = 1e-8)

    def forward(self, tokens: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
        src = self.proj_in(tokens)
        # src where visible else mask
        x = torch.where(visible, src, self.mask_token)
        # add position codes
        x = x + self.coordinate_embedding(self.coordinates)
        # expand latent vectors
        latents = self.latent_tokens.expand(tokens.size(0), -1, -1)
        # process
        x, latents = self.network((x, latents))
        out = self.proj_out(x)
        return out
    