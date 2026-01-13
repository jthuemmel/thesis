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
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} di", 
            weight_shape=f'v di {world.patch_pattern}', 
            di = network.dim_in, **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} e", 
            weight_shape=f'e v {world.patch_pattern} d', 
            d = network.dim, e = default(network.num_tails, 1), 
            **world.patch_sizes, **world.token_sizes 
            )
        
        if network.dim_noise > 0:
            self.to_context = GatedFFN(network.dim_noise, bias= False)

        # learnable embeddings
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.queries = torch.nn.Embedding(world.num_tokens, network.dim_in)
        self.positions = torch.nn.Embedding(world.num_tokens, network.dim - network.dim_in)

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
        if visible.ndim == 2: visible = visible.unsqueeze(0)
        S, B = (visible.size(0), visible.size(1))

        # expand to ensemble form
        xs = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        vs = einops.repeat(visible, 's b ... -> s (b e) ... d', e = E, b = B, s = S, d = self.latents.embedding_dim)

        # maybe sample noise
        if exists(self.to_context):
            eta = torch.randn((S, B * E, self.network.dim_noise), generator = rng, device = fields.device)
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
        # prepare tokens
        tokens = self.to_tokens(fields)
        
        # expand to batch size
        positions = einops.repeat(self.positions.weight, "n dc -> b n dc", b = tokens.size(0))
        queries = einops.repeat(self.queries.weight, 'n di -> b n di', b = tokens.size(0))
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = tokens.size(0))

        # maybe prepare context
        if exists(self.to_context):
            context = self.to_context(noise)
        else:
            context = None

        # augment tokens and queries with positions
        tokens = torch.cat([tokens, positions], dim = -1)        
        queries = torch.cat([queries, positions], dim = -1)

        # only encode visible tokens
        tokens = tokens.gather(1, visible)

        # apply flamingo-style transformer
        for block in self.encoder:
            kv = torch.cat([tokens, latents], dim = 1)
            latents = block(q = latents, kv = kv, ctx = context)
        queries = self.decoder(q = queries, kv = latents, ctx = context)
        
        # linear prediction
        prediction = self.to_fields(prediction)
        return prediction