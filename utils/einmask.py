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

        # pre-compute positional embedding
        self.cpe = ContinuousPositionalEmbedding(network.dim_coords, network.wavelengths, model_dim= None)
        idcs = torch.unravel_index(indices = torch.arange(world.num_tokens), shape = world.token_shape)
        idcs = torch.stack(idcs, dim = -1)
        pos = self.cpe(idcs)
        self.register_buffer('coordinates', pos)

        # I/O
        self.to_tokens = EinMix(
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'v {world.patch_pattern} d', 
            bias_shape='v d',
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} e", 
            weight_shape=f'v d {world.patch_pattern} e', 
            bias_shape='v e',
            d = network.dim, 
            e = default(network.num_tails, 1), 
            **world.patch_sizes, **world.token_sizes 
            )
        
        # maybe context projection
        if default(network.dim_noise, 0) > 0:
            self.to_context = GatedFFN(network.dim_noise, bias= True)

        # positional projections
        self.to_positions = torch.nn.Linear(self.cpe.embedding_dim, network.dim)
        self.to_queries = torch.nn.Linear(self.cpe.embedding_dim, network.dim)

        # latent transformer
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.encoder = torch.nn.ModuleList([
            TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                drop_path = network.drop_path,
                bias=True,
                ) 
                for _ in range(network.num_layers)
            ])
        self.decoder = TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                has_skip = False,
                bias=True,
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
        # expand embeddings to batch size
        B = fields.size(0)
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = B)
        coords = einops.repeat(self.coordinates, "n d -> b n d", b = B)

        # prepare tokens and queries        
        tokens = self.to_positions(coords) + self.to_tokens(fields)
        queries = self.to_queries(coords)

        # maybe prepare context
        context = self.to_context(noise) if exists(self.to_context) else None        

        # only encode visible tokens
        tokens = tokens.gather(1, visible)

        # apply flamingo-style transformer
        for block in self.encoder:
            latents = block(q = latents, kv = torch.cat([tokens, latents], dim = 1), ctx = context)
        queries = self.decoder(q = queries, kv = latents, ctx = context)
        
        # predict full fields
        queries = self.to_fields(queries)
        return queries