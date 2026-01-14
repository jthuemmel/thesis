import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # I/O
        self.to_tokens = EinMix(
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'v {world.patch_pattern} d', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} e", 
            weight_shape=f'v d {world.patch_pattern} e', 
            d = network.dim, 
            e = default(network.num_tails, 1), 
            **world.patch_sizes, **world.token_sizes 
            )        

        # maybe context projection
        if default(network.dim_noise, 0) > 0:
            self.to_context = GatedFFN(network.dim_noise)

        # learnable embedddings
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.coordinates = torch.nn.Embedding(world.num_tokens, network.dim)

        # latent transformer
        self.encoder = torch.nn.ModuleList([
            TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                drop_path = network.drop_path,
                ) 
                for _ in range(network.num_layers)
            ])
        self.decoder = TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
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

    def forward(self, fields: torch.FloatTensor, visible: torch.LongTensor, E: int = 1, rng: torch.Generator = None) -> torch.FloatTensor:
        # visible: (S, B, v) or (B, v)
        if visible.ndim == 2: visible = visible.unsqueeze(0)
        S, B = (visible.size(0), visible.size(1))

        # expand to ensemble form
        xs = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        vs = einops.repeat(visible, 's b ... -> s (b e) ... d', e = E, b = B, s = S, d = self.latents.embedding_dim)

        # maybe sample noise
        if exists(self.to_context):
            eta = torch.randn((S, B * E, 1, self.network.dim_noise), generator = rng, device = fields.device)
        else:
            eta = [None] * S

        # iterate
        for s in range(S):
            # step
            xs = self._step(fields=xs, visible=vs[s], noise=eta[s])
            # detach gradients unless it is the last step
            if s < S - 1:
                xs = xs.detach()
        
        # rearrange to ensemble form
        xs = einops.rearrange(xs, "(b e) ... k -> b ... (e k)", e = E, b = B)
        return xs
    
    def _step(self, fields: torch.FloatTensor, visible: torch.LongTensor, noise: torch.FloatTensor) -> torch.FloatTensor:
        # prepare coordinates and latents
        coords = einops.repeat(self.coordinates.weight, 'n d -> b n d', b = fields.size(0))
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = fields.size(0))

        # embed tokens and select visible ones
        tokens = self.to_tokens(fields) + coords
        tokens = tokens.gather(1, visible)

        # maybe embed noise as context
        context = self.to_context(noise) if exists(self.to_context) else None        
        
        # latent transformer
        for block in self.encoder:
            latents = block(q = latents, kv = torch.cat([tokens, latents], dim = 1), ctx = context)
        predictions = self.decoder(q = coords, kv = latents, ctx = context)

        # back to fields
        predictions = self.to_fields(predictions)
        return predictions