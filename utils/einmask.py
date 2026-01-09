import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *
from utils.random_fields import *

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # noise generator
        s = torch.tensor([500, 1000, 2000])
        t = torch.tensor([1, 2, 4])
        self.noise_generator = SphericalDiffusionNoise(
            num_channels = len(s) * len(t),
            num_steps = 6,
            num_lat = 45,
            num_lon = 90,
            horizontal_length = einops.repeat(s, 's -> (s t)', t = len(t)),
            temporal_length = einops.repeat(t, 't -> (s t)', s = len(s)),
            lat_slice=slice(14, -15, 1), # 16 latitudes ~ -32 to 32
            lon_slice=slice(0, 60, 2) # 30 longitudes ~ 90 to 330
        )

        # pre-compute positional embedding
        idcs = torch.unravel_index(indices = torch.arange(world.num_tokens), shape = world.token_shape)
        idcs = torch.stack(idcs, dim = -1)
        pos = ContinuousPositionalEmbedding(network.dim_coords, network.wavelengths, model_dim= None)
        self.register_buffer('coordinates', pos(idcs))

        # I/O
        self.token_embedding = EinMix(
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} di", 
            weight_shape=f'{world.patch_pattern} v di', 
            di = network.dim_in, **world.patch_sizes, **world.token_sizes
            )
        
        self.token_predictor = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} k", 
            weight_shape=f'd v {world.patch_pattern} k', 
            d = network.dim, k = network.num_tails, 
            **world.patch_sizes, **world.token_sizes 
            )
        
        self.context_embedding = torch.nn.Linear(
            in_features = pos.embedding_dim + self.noise_generator.nchannels,
            out_features = network.dim - network.dim_in,
            )

        # learnable parameters (Embedding for convenient initialization)
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.queries = torch.nn.Embedding(world.num_tokens, network.dim_in)

        # latent transformer
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

    def forward(self, fields: torch.FloatTensor, visible: torch.LongTensor, E: int = 1, rng: torch.Generator = None) -> torch.FloatTensor:
        # visible: (S, B, v) or (B, v)
        if visible.ndim == 2:
            S, B = (1, visible.size(0))
            visible = visible.unsqueeze(0)
        else:
            S, B = (visible.size(0), visible.size(1))

        # expand to ensemble form
        xs = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        vs = einops.repeat(visible, 's b ... -> s (b e) ...', e = E, b = B, s = S)

        # iterate
        for s in range(S):
            # sample noise
            eta = self.noise_generator(shape = (B * E,), rng = rng)
            eta = einops.repeat(eta, f"be c t h w -> be {self.world.flat_token_pattern} c", **self.world.token_sizes)
            # step
            xs = self.step(fields=xs, visible=vs[s], noise=eta)
            # detach gradients unless it is the last step
            if s < S - 1:
                xs = xs.detach()
        
        # rearrange to ensemble form
        xs = einops.rearrange(xs, "(b e) ... k -> b ... (e k)", e = E, b = B, k = self.network.num_tails)
        return xs

    def step(self, fields: torch.FloatTensor, visible: torch.LongTensor = None, noise: torch.FloatTensor = None) -> torch.FloatTensor:    
        # expand learnable codes
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = fields.size(0))
        queries = einops.repeat(self.queries.weight, 'n di -> b n di', b = fields.size(0))
        positions = einops.repeat(self.coordinates, "n dc -> b n dc", b = fields.size(0))        

        # maybe combine context information
        context = torch.cat([positions, noise], dim = -1) if exists(noise) else positions

        # linear embeddings
        context = self.context_embedding(context)
        tokens = self.token_embedding(fields)

        # concatenate context to tokens and queries
        tokens = torch.cat([tokens, context], dim = -1)
        queries = torch.cat([queries, context], dim = -1)

        # encode only visible tokens
        if exists(visible):
            visible = einops.repeat(visible, 'b n -> b n d', d = tokens.size(-1))
            tokens = tokens.gather(1, visible)
        
        # apply flamingo-style transformer
        for block in self.transformer:
            kv = torch.cat([tokens, latents], dim = 1)
            latents = block(q = latents, kv = kv)
        queries = self.write(q = queries, kv = latents)

        # decode tokens
        queries = self.token_predictor(queries)
        return queries
