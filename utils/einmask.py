import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *
from utils.random_fields import RandomField

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # maybe use generative noise
        self.noise_generator = RandomField(network.dim, world) if default(network.num_tails, 1) <= 1 else None

        # I/O
        self.to_tokens = EinMix(
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'{world.patch_pattern} v d', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} k", 
            weight_shape=f'd v {world.patch_pattern} k', 
            d = network.dim, 
            k = default(network.num_tails, 1), 
            **world.patch_sizes, **world.token_sizes 
            )
                
        # positional embeddings
        self.src_positions = ContinuousPositionalEmbedding(
            dim_per_coord=network.dim_coords, 
            wavelengths=[(1, 2 * k) for k in world.token_shape],
            model_dim=network.dim
        )

        self.tgt_positions = ContinuousPositionalEmbedding(
            dim_per_coord=network.dim_coords, 
            wavelengths=[(1, 2 * k) for k in world.token_shape],
            model_dim=network.dim
        )
        
        # pre-computed coordinates
        self.register_buffer('indices', torch.arange(world.num_tokens))
        self.register_buffer("coordinates", torch.stack(
            torch.unravel_index(indices = self.indices, shape = world.token_shape), 
            dim = -1)
            )        
        
        # learnable latents
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)

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
                has_skip=False,
                )

        # Weight initialization
        self.apply(self.base_init)
    
    @staticmethod
    def base_init(m):
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

    def forward(self, 
                fields: torch.FloatTensor, 
                srcs: torch.LongTensor, 
                tgts: Optional[torch.LongTensor] = None,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        # define shapes
        B = fields.size(0)
        D = self.network.dim
        K = default(self.network.num_tails, 1)
        E = default(members, 1)
        S = srcs.size(0) if srcs.ndim > 2 else 1
        
        # maybe expand srcs/tgts
        if tgts is None: tgts = self.indices.expand(S, B, -1)
        if tgts.ndim == 2: tgts = tgts.expand(S, -1, -1)
        if srcs.ndim == 2: srcs = srcs.expand(S, -1, -1)

        # expand indices and coordinates
        src_idx = einops.repeat(srcs, 's b ... -> s (b e) ... d', d = D, e = E, b = B, s = S)
        tgt_idx = einops.repeat(tgts, 's b ... -> s (b e) ... d', d = D, e = E, b = B, s = S)
        src_coo = einops.repeat(self.coordinates[srcs], 's b ... -> s (b e) ...', e = E, b = B, s = S)
        tgt_coo = einops.repeat(self.coordinates[tgts], 's b ... -> s (b e) ...', e = E, b = B, s = S)

        # embed full fields as tokens
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        tokens = self.to_tokens(fields)

        # iterate
        for s in range(S):
            # prepare context and queries
            context = tokens.gather(1, src_idx[s]) + self.src_positions(src_coo[s])
            queries = self.tgt_positions(tgt_coo[s])

            # maybe condition on noise
            if exists(self.noise_generator):
                context, noise = self.noise_generator(context, rng = rng)
                queries = queries + noise.gather(1, tgt_idx[s])
                context = context + noise.gather(1, src_idx[s])

            # map context to latents
            latents = einops.repeat(self.latents.weight, '... -> (b e) ...', b = B, e = E)
            for read in self.encoder:
                kv = torch.cat([context, latents], dim = 1)
                latents = read(q = latents, kv = kv)

            # map latents to queries
            queries = self.decoder(q = queries, kv = latents)

            # update tokens
            tokens = tokens.scatter(1, tgt_idx[s], queries)

        # map all tokens back to fields
        fields = self.to_fields(tokens)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... k -> b ... (e k)", e = E, b = B, k = K)
        return fields