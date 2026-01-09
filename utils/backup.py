import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # I/O
        self.to_tokens = EinMix(
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} di", 
            weight_shape=f'v di {world.patch_pattern}', 
            di = network.dim_in, **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} e", 
            weight_shape=f'e v {world.patch_pattern} d', 
            d = network.dim, e = network.num_tails, 
            **world.patch_sizes, **world.token_sizes 
            )

        # position embedding
        self.positions = ContinuousPositionalEmbedding(network.dim_coords, network.wavelengths, network.dim - network.dim_in)
        self.register_buffer(
            "coordinates",
            torch.stack(
                torch.unravel_index(indices = torch.arange(world.num_tokens), shape = world.token_shape),
                dim = -1)
        )

        # learnable embeddings
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.queries = torch.nn.Embedding(world.num_tokens, network.dim_in)

        # flamingo transformer
        self.transformer = torch.nn.ModuleList([
            TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                drop_path = network.drop_path,
                ) 
                for _ in range(network.num_layers)
            ])
        self.write = TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                has_skip = False
                )

        # Weight initialization
        self.apply(self.base_init)
    
    @staticmethod
    def base_init(m):
        '''Explicit weight initialization'''
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # embedding
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # einmix
        if isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # conditional layer norm
        if isinstance(m, ConditionalLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None: # CLN weight close to 0
                torch.nn.init.trunc_normal_(m.weight, std = 1e-7)

    def forward(self, fields, context):
        # linear embedding + token-wise bias
        tokens = self.to_tokens(fields)
        # positional codes
        positions = self.positions(self.coordinates)
        # expand shapes (d = dc + di)
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = tokens.size(0))
        queries = einops.repeat(self.queries.weight, 'n di -> b n di', b = tokens.size(0))
        positions = einops.repeat(positions, "n dc -> b n dc", b = tokens.size(0))
        context = einops.repeat(context, 'b n -> b n d', d = latents.size(-1))
        # concatenate position codes
        tokens = torch.cat([tokens, positions], dim = -1)
        queries = torch.cat([queries, positions], dim = -1)
        # encode only context tokens
        tokens = tokens.gather(1, context)
        # apply flamingo-style transformer
        for block in self.transformer:
            kv = torch.cat([tokens, latents], dim = 1)
            latents = block(q = latents, kv = kv)
        queries = self.write(q = queries, kv = latents)
        # linear tail prediction
        fields = self.to_fields(queries)
        return fields