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
        self.noise_generator = RandomField(network.dim, world, has_ffn=True) if default(network.num_tails, 1) <= 1 else None

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

    @staticmethod
    def freeze_weights(m):
        for param in m.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_weights(m):
        for param in m.parameters():
            param.requires_grad = True

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

        # default tgts as all indices    
        tgts = default(tgts, self.indices.expand(B, -1))

        # embed full fields as tokens
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        tokens = self.to_tokens(fields)

        # ensure srcs/tgts are lists
        if not isinstance(srcs, list): srcs = [srcs]
        if not isinstance(tgts, list): tgts = [tgts] * len(srcs)

        # iterate
        for src, tgt in zip(srcs, tgts, strict=True):
            # expand src/tgt/latents to ensemble form
            src = einops.repeat(srcs, 'b ... -> (b e) ...', e = E, b = B)
            tgt = einops.repeat(tgts, 'b ... -> (b e) ...', e = E, b = B)
            latents = einops.repeat(self.latents.weight, '... -> (b e) ...', b = B, e = E)

            # gather tokens visible at this step
            context = tokens.gather(1, einops.repeat(src, '... -> ... d', d = D))

            # add position codes and prepare queries
            context = context + self.src_positions(self.coordinates[src])
            queries = self.tgt_positions(self.coordinates[tgt])

            # maybe condition on noise
            if exists(self.noise_generator):
                context, noise = self.noise_generator(context, rng = rng)
                queries = queries + noise.gather(1, einops.repeat(tgt, '... -> ... d', d = D))
                context = context + noise.gather(1, einops.repeat(src, '... -> ... d', d = D))

            # map context to latents
            for read in self.encoder:
                latents = read(q = latents, kv = torch.cat([context, latents], dim = 1))

            # map latents to queries
            queries = self.decoder(q = queries, kv = latents)

            # scatter tokens predicted at this step
            tokens = tokens.scatter(1, einops.repeat(tgt, '... -> ... d', d = D), queries)

        # map all tokens back to fields
        fields = self.to_fields(tokens)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... k -> b ... (e k)", e = E, b = B, k = K)
        return fields