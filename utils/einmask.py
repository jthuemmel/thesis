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
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'v d {world.patch_pattern}', 
            bias_shape=f'{world.flat_token_pattern} d', # token-wise bias
            d = network.dim, **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} e", 
            weight_shape=f'e v {world.patch_pattern} d', 
            d = network.dim, e = network.num_tails, 
            **world.patch_sizes, **world.token_sizes 
            )
        
        # embeddings
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.queries = torch.nn.Embedding(world.num_tokens, network.dim)

        # flamingo
        self.transformer = torch.nn.ModuleList([
            TransformerBlock(
                dim =network.dim, 
                dim_heads=network.dim_heads, 
                dim_ctx = network.dim_noise
                ) 
                for _ in range(network.num_layers)
            ])
        self.write = TransformerBlock(
                dim =network.dim, 
                dim_heads=network.dim_heads, 
                dim_ctx = network.dim_noise,
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
                torch.nn.init.trunc_normal_(m.bias, std = 0.02)
        # conditional layer norm
        if isinstance(m, ConditionalLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None: # CLN weight close to 0
                torch.nn.init.trunc_normal_(m.weight, std = 1e-7)

    def forward(self, fields, context):
        # linear embedding + token-wise bias
        tokens = self.to_tokens(fields)
        # expand shapes
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = tokens.size(0))
        queries = einops.repeat(self.queries.weight, 'n d -> b n d', b = tokens.size(0))
        context = einops.repeat(context, 'b n -> b n d', d = tokens.size(-1))
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
