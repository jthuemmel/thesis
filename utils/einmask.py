import einops
import torch

from einops.layers.torch import EinMix, Rearrange
from utils.config import *
from utils.components import *
from utils.random_fields import SphericalDiffusionNoise

class LatentModel(torch.nn.Module):
    def __init__(self, network: NetworkConfig):
        super().__init__()
        DI = network.dim_in + network.dim_coords
        DN = network.dim_noise
        D = network.dim
        NH = network.dim // network.dim_heads

        # latent tokens
        self.latents = torch.nn.Embedding(network.num_latents, D)

        # map values to latents
        self.norm_values = AdaptiveLayerNorm(DI)
        self.norm_queries = AdaptiveLayerNorm(DI, DN)
        self.value_encoder = TransformerBlock(dim=D, dim_kv= DI, num_heads= default(network.num_encoder_heads, NH))
        self.query_encoder = TransformerBlock(dim=D, dim_kv= DI, num_heads= default(network.num_encoder_heads, NH))

        # process latents
        self.processor = torch.nn.ModuleList([
             TransformerBlock(dim=D) 
             for _ in range(default(network.num_compute_blocks, 1))
             ])
        
        # map latents to queries
        self.query_decoder = torch.nn.ModuleList([
             TransformerBlock(dim=DI, dim_kv= D, num_heads= default(network.num_decoder_heads, NH), dim_ctx= DN)
             for _ in range(default(network.num_write_blocks, 1))
            ])

    def forward(self, values: torch.Tensor, queries: torch.Tensor, ctx: torch.Tensor = None) -> torch.Tensor:
        # expand latents
        latents = einops.repeat(self.latents.weight, '... -> b ...', b=values.size(0))
        
        # extract separate latent representations
        query_latents, value_latents = latents.chunk(2, dim = 1)
        query_latents = self.query_encoder(query_latents, kv = self.norm_queries(queries, ctx = ctx))
        value_latents = self.value_encoder(value_latents, kv = self.norm_values(values))
        latents = torch.cat([query_latents, value_latents], dim = 1)

        # jointly update latents
        for compute in self.processor:
             latents = compute(latents)

        # update queries
        for write in self.query_decoder:
            queries = write(queries, kv = latents, ctx = ctx)

        return queries

class FieldDecoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.world = world
        self.to_fields = EinMix(
            f'b ({world.token_pattern}) do -> (b k) ({world.flat_pattern})',
            weight_shape=f'k v {world.patch_pattern} do',
            **world.token_sizes, **world.kernel_sizes, 
            do= network.dim_in + network.dim_coords, k = default(network.num_tails, 1)
            )

        self.unflatten_fields = Rearrange(
            f'b ({world.flat_pattern}) -> b {world.field_pattern}',
            **world.token_sizes, **world.patch_sizes
            )

        self._build_fold_index(world)

    def _build_fold_index(self, world: WorldConfig):
        ts = world.token_sizes
        ks = world.kernel_sizes
        
        grid_strides = {}
        acc = 1
        for ax in reversed(world.layout):
            grid_strides[ax] = acc
            acc *= world.field_sizes[ax]

        idx = torch.zeros([*ts.values(), *ks.values()], dtype=torch.long)

        for ax in world.layout:
            n0 = torch.arange(ts[ax]) * world.patch_sizes[2*ax]
            dp = torch.arange(ks[2*ax])
            n0 = einops.repeat(n0, f'{ax} -> {world.token_pattern} {world.patch_pattern}', **ts, **ks)
            dp = einops.repeat(dp, f'{2*ax} -> {world.token_pattern} {world.patch_pattern}', **ts, **ks)
            idx += (n0 + dp).clamp(0, world.field_sizes[ax] - 1) * grid_strides[ax]

        self.register_buffer('idx', idx.flatten())

    def forward(self, tgt: torch.FloatTensor):
        B = tgt.size(0)
        tgt = self.to_fields(tgt)
        predicted_fields = torch.scatter_reduce(
            tgt.new_empty((B, self.world.num_elements)), 
            1, 
            self.idx.expand(B, -1), 
            tgt, 
            reduce='mean', 
            include_self=False
            )
        return self.unflatten_fields(predicted_fields)

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # stochasticity
        T, H, W = tuple(world.token_sizes[ax] for ax in ("t", "h", "w"))

        horizontal = torch.tensor(default(network.grf_horizontal, [512]), dtype = torch.float32)
        temporal = torch.tensor(default(network.grf_temporal, [1]), dtype = torch.float32)
        channels = int(default(network.grf_channels, 1))
        
        num_fields = channels * len(horizontal) * len(temporal)
        nlat = 180 // world.patch_sizes['hh']
        nlon = 360 //  world.patch_sizes['ww']

        self.noise_generator = SphericalDiffusionNoise(
                num_channels=num_fields,
                num_lat=nlat,
                num_lon=nlon,
                num_steps=T,
                horizontal_length= einops.repeat(horizontal, 'h -> (c h t)', h = len(horizontal), t = len(temporal), c = channels),
                temporal_length=einops.repeat(temporal, 't -> (c h t)', h = len(horizontal), t = len(temporal), c = channels),
                lat_slice= slice((nlat - H) // 2, (nlat + H) // 2), # centered on the equator
                lon_slice=slice(0, 2 * W, 2) # 2 degree step
        )
        
        # I/O
        self.to_noise = EinMix(
                  pattern = f'... f t h w -> ... ({world.token_pattern}) dn',
                  weight_shape = 'v f dn',
                  f = num_fields, dn = default(network.dim_noise, network.dim_in), **world.token_sizes
                )

        self.to_tokens = EinMix(
                  f'b {world.field_pattern} -> b ({world.token_pattern}) di',
                  weight_shape= f'v {world.patch_pattern} di',
                  **world.patch_sizes, **world.token_sizes, di = network.dim_in
                  )
                
        
        self.to_fields = FieldDecoder(network, world)
        
        # embeddings
        self.mask_codes = torch.nn.Embedding(2, network.dim_in)
        self.position_codes = torch.nn.Embedding(world.num_tokens, network.dim_coords)
        
        # latent transformer
        self.transformer = LatentModel(network)

        # weight initialization
        self.apply(self.base_init)

        # maybe compile
        if True:
             self.transformer.compile()
             self.to_fields.compile()
             self.to_tokens.compile()
             self.to_noise.compile()


    @staticmethod
    def base_init(m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std = 0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        # embedding
        elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # AdaLN
        elif isinstance(m, AdaptiveLayerNorm):
                torch.nn.init.zeros_(m.bias)
                if exists(m.weight):
                    torch.nn.init.trunc_normal_(m.weight, std = 1e-4)
        # einmix
        elif isinstance(m, EinMix):
                torch.nn.init.trunc_normal_(m.weight, std = 0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, 
            fields: torch.FloatTensor, 
            visible: torch.BoolTensor, 
            members: Optional[int] = None, 
            rng: Optional[torch.Generator] = None
            ) -> torch.FloatTensor:
        B = fields.size(0)
        E = default(members, 1)

        # expand to ensemble
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E)
        visible = einops.repeat(visible, 'b ... -> (b e) ...', e = E)
        positions = einops.repeat(self.position_codes.weight, '... -> (b e) ...', e = E, b = B)

        # create mask representation
        mask = self.mask_codes(visible.long())

        # embed tokens
        tokens = self.to_tokens(fields)

        # create random field
        noise = self.noise_generator((B*E,), rng).to(tokens.dtype)
        noise = self.to_noise(noise)

        # add mask
        masked_tokens = mask + torch.where(visible[..., None], tokens, 0)

        # combine with position codes
        values = torch.cat([masked_tokens, positions], dim = -1)
        queries = torch.cat([mask, positions], dim = -1)

        # latent transformer
        predicted_tokens = self.transformer(values, queries, noise)
        
        # map all predicted_tokens back to fields
        predicted_fields = self.to_fields(predicted_tokens)

        # rearrange to ensemble last
        predicted_fields = einops.rearrange(predicted_fields, "(b e) ... -> b ... e", b = B)
        return predicted_fields