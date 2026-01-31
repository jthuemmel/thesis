import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *
from utils.random_fields import RandomField
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
                
        # noise mapping
        if default(network.num_tails, 1) > 1:
            self.to_noise = None
            self.noise_generator = None
        elif exists(network.dim_noise):
            self.to_noise = GatedFFN(network.dim_noise)
            self.noise_generator = None
        else:
            self.noise_generator = RandomField(network.dim, world, has_ffn=False)
            self.to_noise = None
        
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
            TransformerBlock(network.dim, drop_path=0.0, dim_ctx=network.dim_noise)
            for _ in range(network.num_read_blocks)
        ])

        self.processor = torch.nn.ModuleList([
            TransformerBlock(network.dim, drop_path=network.drop_path, dim_ctx=network.dim_noise)
            for _ in range(network.num_compute_blocks)
        ])
        
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(network.dim, drop_path=0.0, dim_ctx=network.dim_noise)
            for _ in range(network.num_write_blocks)
        ])

        # Weight initialization
        self.apply(self.base_init)

    @staticmethod
    def base_init(m: torch.nn.Module):
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
                srcs: List[torch.LongTensor] | torch.LongTensor, 
                tgts: List[torch.LongTensor] | torch.LongTensor = None,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        B = fields.size(0)
        D = self.network.dim
        K = default(self.network.num_tails, 1)
        E = default(members, 1)
        tgts = default(tgts, self.indices.expand(B, -1))

        # expand to ensemble form
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        latents = einops.repeat(self.latents.weight, '... -> (b e) ...', b = B, e = E)
        coo = einops.repeat(self.coordinates, '... -> (b e) ...', b = B, e = E)
        src_idx = einops.repeat(srcs, 'b ... -> (b e) ... d', d = D, e = E, b = B)
        tgt_idx = einops.repeat(tgts, 'b ... -> (b e) ... d', d = D, e = E, b = B)
        
        # embed full fields as tokens
        tokens = self.to_tokens(fields) + self.src_positions(coo)

        # gather tokens visible at this step
        context = tokens.gather(1, src_idx)

        # prepare queries
        queries = self.tgt_positions(coo).gather(1, tgt_idx)

        # maybe create functional noise
        if exists(self.to_noise):
            ctx = torch.randn(B * E, 1, self.network.dim_noise, generator = rng, device = fields.device)
            ctx = self.to_noise(ctx)
        elif exists(self.noise_generator):
            context, noise = self.noise_generator(context, rng = rng)
            queries = queries + noise.gather(1, tgt_idx)
            context = context + noise.gather(1, src_idx)
            ctx = None
        else:
            ctx = None

        # map context to latents
        for read in self.encoder:
            latents = read(q = latents, kv = torch.cat([context, latents]), ctx = ctx)

        # process latents
        for process in self.processor:
            latents = process(q = latents, ctx = ctx)

        # map latents to queries
        for write in self.decoder:
            queries = write(q = queries, kv = torch.cat([queries, latents], dim = 1), ctx = ctx)

        # scatter tokens predicted at this step
        tokens = tokens.scatter(1, tgt_idx, queries)

        # map all tokens back to fields
        fields = self.to_fields(tokens)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... k -> b ... (e k)", e = E, b = B, k = K)
        return fields