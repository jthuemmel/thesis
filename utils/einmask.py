import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *
from utils.random_fields import RandomField

class EinDecoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # config attributes 
        c = network.dim_out
        k = default(network.num_tails, 1)
        vv = world.patch_sizes['vv']
        v = world.token_sizes['v']
        groups = v * vv * k

        # project tokens to low dimensional space before upsampling
        self.token_to_grid = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> b (v vv k c) t h w",
            weight_shape=f"v vv k c d",
            d = network.dim, c = c, k = k, vv = vv,
            **world.token_sizes
        )

        # small CNN for post-processing
        self.cnn = torch.nn.ModuleList([
            ConvNextBlock(c * groups, (3, 7, 7), groups) 
            for _ in range(default(network.num_cnn_blocks, 0))
            ])

        # upsample grid via interpolate + depthwise separable convolution
        self.upsample = ConvInterpolate(c * groups, (3, 5, 5), 
                                        num_groups= c * groups,
                                        out_size= tuple(world.field_sizes[ax] for ax in ['t', 'h', 'w']),
                                        mode='nearest-exact')
        
        # pointwise projection to output
        self.grid_to_field = EinMix(
            f'b (v vv k c) (t tt) (h hh) (w ww) -> b {world.field_pattern} k',
            weight_shape="v vv k c",
            k = k, c = c, **world.patch_sizes, **world.token_sizes
        )

    def forward(self, x: torch.FloatTensor):
        x = self.token_to_grid(x)
        for block in self.cnn:
            x = block(x)
        x = self.upsample(x)
        x = self.grid_to_field(x)
        return x

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # I/O
        self.to_tokens = EinMix(
            pattern=f"b {world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'{world.patch_pattern} v d', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        if exists(network.num_cnn_blocks):
            self.to_fields = EinDecoder(network, world)  
        else:
            self.to_fields =EinMix(
                pattern=f"b {world.flat_token_pattern} d -> b {world.field_pattern} k", 
                weight_shape=f'd v {world.patch_pattern} k', 
                d = network.dim, 
                k = default(network.num_tails, 1), 
                **world.patch_sizes, **world.token_sizes 
                )
                
        # noise mapping
        if default(network.num_tails, 1) > 1:
            self.noise_generator = None
        else:
            self.noise_generator = RandomField(network.dim, world, has_ffn=False)
        
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
            TransformerBlock(network.dim, dim_ctx=network.dim_noise)
            for _ in range(default(network.num_read_blocks, 1))
        ])

        self.processor = torch.nn.ModuleList([
            TransformerBlock(network.dim, drop_path=network.drop_path, dim_ctx=network.dim_noise)
            for _ in range(default(network.num_compute_blocks, 1))
        ])
        
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(network.dim, dim_ctx=network.dim_noise)
            for n in range(default(network.num_write_blocks, 1))
        ])

        # Weight initialization
        self.apply(self.base_init)
        self.apply(self.zero_init)

    @staticmethod
    def zero_init(m: torch.nn.Module):
        # residual blocks zero out their last layer 
        if isinstance(m, (TransformerBlock, ConvNextBlock, ConvInterpolate)):
            for name, sm in m.named_modules():
                if "_out" in name and hasattr(sm, 'weight'):
                    torch.nn.init.trunc_normal_(sm.weight, std = 1e-7)

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
                torch.nn.init.trunc_normal_(m.bias, std = 0.02)
        # convolution
        elif isinstance(m, torch.nn.Conv3d):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # conditional layer norm
        elif isinstance(m, ConditionalLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None: # CLN weight close to 0
                torch.nn.init.trunc_normal_(m.weight, std = 1e-7)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                srcs: List[torch.LongTensor] | torch.LongTensor, 
                tgts: List[torch.LongTensor] | torch.LongTensor = None,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        B = fields.size(0)
        D = self.network.dim
        K = default(self.network.num_tails, 1)
        E = default(members, 1)

        # expand to ensemble form
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        latents = einops.repeat(self.latents.weight, '... -> (b e) ...', b = B, e = E)
        coo = einops.repeat(self.coordinates, '... -> (b e) ...', b = B, e = E)
        src_idx = einops.repeat(srcs, 'b ... -> (b e) ... d', d = D, e = E, b = B)
        
        # embed full fields as tokens
        tokens = self.to_tokens(fields).gather(1, src_idx)

        # prepare tgt and src
        tgt = self.tgt_positions(coo)
        src = self.src_positions(coo).scatter_add_(1, src_idx, tokens)

        # maybe add random field
        if exists(self.noise_generator):
            noise = self.noise_generator(shape = (B * E,), rng = rng).to(src.dtype)
            tgt = tgt + noise
            src = src + noise

        # map src to latents
        for read in self.encoder:
            latents = read(q = latents, kv = torch.cat([src, latents], dim = 1))

        # process latents
        for process in self.processor:
            latents = process(q = latents)

        # map latents to tgt
        for write in self.decoder:
            tgt = write(q = tgt, kv = latents)

        # map all tokens back to fields
        fields = self.to_fields(tgt)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... k -> b ... (e k)", e = E, b = B, k = K)
        return fields