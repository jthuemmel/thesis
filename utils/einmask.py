import einops
import torch

from einops.layers.torch import EinMix, Rearrange
from utils.config import *
from utils.components import *
from utils.random_fields import SphericalDiffusionNoise

class LatentModel(torch.nn.Module):
    def __init__(self, network: NetworkConfig):
        super().__init__()
        # latent tokens
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)

        # map src to latents
        self.to_read = torch.nn.Sequential(torch.nn.Linear(network.dim_in, network.dim), AdaptiveLayerNorm(network.dim))
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads)
              for _ in range(default(network.num_read_blocks, 1))
              ])
        
        # map latents to tgt
        self.to_write = torch.nn.Sequential(torch.nn.Linear(network.dim, network.dim_out), AdaptiveLayerNorm(network.dim_out))
        self.decoder = torch.nn.ModuleList([
             TransformerBlock(dim=network.dim_out, dim_ctx=  network.dim_noise, num_heads= network.num_decoder_heads)
             for _ in range(default(network.num_write_blocks, 1))
            ])
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, E = src.size(0), ctx.size(0) // src.size(0)

        # expand latents
        latents = einops.repeat(self.latents.weight, '... -> b ...', b= B)
        
        # read src into latents
        src = self.to_read(src)
        for read in self.encoder:
             latents = read(latents, kv = torch.cat([src, latents], dim = 1))
        
        # expand latents and tgt to match the stochastic context
        tgt = einops.repeat(tgt, 'b ... -> (b e) ...', e = E)
        latents = einops.repeat(latents, 'b ... -> (b e) ...', e = E)

        # write latents to tgt
        latents = self.to_write(latents)
        for write in self.decoder:
            tgt = write(tgt, kv = torch.cat([tgt, latents], dim = 1), ctx = ctx)

        return tgt

class FieldDecoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.world = world
        self.to_fields = EinMix(
            f'b ({world.token_pattern}) do -> (b k) ({world.flat_pattern})',
            weight_shape=f'k v {world.patch_pattern} do',
            **world.token_sizes, **world.kernel_sizes, 
            do= network.dim_out, k = default(network.num_tails, 1)
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
                sigma=default(network.grf_sigma, 1.0),
                horizontal_length= einops.repeat(horizontal, 'h -> (c h t)', h = len(horizontal), t = len(temporal), c = channels),
                temporal_length=einops.repeat(temporal, 't -> (c h t)', h = len(horizontal), t = len(temporal), c = channels),
                lat_slice= slice((nlat - H) // 2, (nlat + H) // 2), # centered on the equator
                lon_slice=slice(0, 2 * W, 2) # 2 degree step
        )
        
        I/O
        self.from_noise = EinMix(
                  pattern = f'... f t h w -> ... ({world.token_pattern}) dn',
                  weight_shape = 'v f dn',
                  f = num_fields, dn = network.dim_noise, **world.token_sizes
                )

        # self.from_noise = GatedFFN(network.dim_noise)

        self.to_tokens = EinMix(
                  f'b {world.field_pattern} -> b ({world.token_pattern}) di',
                  weight_shape= f'v {world.patch_pattern} di',
                  **world.patch_sizes, **world.token_sizes, di = network.dim_in
                  )
                
        self.to_fields = FieldDecoder(network, world)
        
        # embeddings
        self.src_codes = torch.nn.Embedding(2, network.dim_in)
        self.tgt_codes = torch.nn.Embedding(2, network.dim_out)
        self.src_positions = torch.nn.Embedding(world.num_tokens, network.dim_in)
        self.tgt_positions = torch.nn.Embedding(world.num_tokens, network.dim_out)
        
        # transformer
        self.transformer = LatentModel(network)

        # weight initialization
        self.apply(self.base_init)
        self.apply(self.zero_init)

        # maybe compile
        if network.kwargs.get('compile', True):
             self.transformer.compile()
             self.to_fields.compile()
             self.to_tokens.compile()
             self.from_noise.compile()

    @staticmethod
    def zero_init(m: torch.nn.Module):
        if isinstance(m, AdaptiveLayerNorm):
            if exists(m.proj):
                torch.nn.init.trunc_normal_(m.proj.weight, std = 1e-5)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, GatedFFN):
            torch.nn.init.zeros_(m.to_out.weight)

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
        # einmix
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, 
            fields: torch.FloatTensor, 
            mask: torch.BoolTensor, 
            members: Optional[int] = None, 
            rng: Optional[torch.Generator] = None,
            **kwargs
            ) -> torch.FloatTensor:
        B = fields.size(0)
        E = default(members, 1)

        # embed tokens
        tokens = self.to_tokens(fields)        

        # create random fields
        noise = self.noise_generator((B*E,), rng).to(tokens.dtype)
        ctx = self.from_noise(noise)
        # noise = torch.randn((B * E, 1, self.network.dim_noise), device = tokens.device, dtype = tokens.dtype)
        # ctx = self.from_noise(noise) + noise

        # mask
        tgt = self.tgt_codes(mask.long()) + self.tgt_positions.weight
        src = self.src_codes(mask.long()) + self.src_positions.weight
        src = torch.where(
            condition = mask[..., None], 
            input = tokens + src, 
            other = src
            )
        
        # transformer
        predicted_tokens = self.transformer(src, tgt, ctx = ctx)

        # map all predicted_tokens back to fields
        predicted_fields = self.to_fields(predicted_tokens)

        # rearrange to ensemble last
        predicted_fields = einops.rearrange(predicted_fields, "(b e) ... -> b ... e", b = B)
        return predicted_fields