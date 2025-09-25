import torch
import einops
from utils.components import *
from einops.layers.torch import EinMix

class MaskedPredictor(torch.nn.Module):
    def __init__(self, model, world, generator: torch.Generator = None):
        super().__init__()
        self.model_cfg = model
        self.world_cfg = world
        self.generator = generator
        
        # noise
        if exists(model.dim_noise):
            self.noise_embedding = ContinuousPositionalEmbedding(model.dim_noise, model.wavelengths, model.dim_noise)
        else:
            self.noise_embedding = None

        # per-variable linear projections
        self.norm_in = ConditionalLayerNorm(model.dim)
        self.proj_in = EinMix(
            pattern = f'{world.flatland_pattern} -> b {world.flat_token_pattern} d',
            weight_shape = f'v {world.patch_pattern} d', 
            bias_shape = 'v d',
            d = model.dim, **world.patch_sizes, **world.token_sizes
            )
        
        self.proj_out = EinMix(
            pattern = f'b {world.flat_token_pattern} d -> {world.flatland_pattern} e',
            weight_shape = f'v {world.patch_pattern} e d',
            d = model.dim, e = world.num_tails, **world.patch_sizes, **world.token_sizes
            )

        # learnable tokens
        self.positions = torch.nn.Embedding(world.num_tokens, model.dim)
        self.latents = torch.nn.Embedding(model.num_latents, model.dim)
        self.masks = torch.nn.Embedding(1, model.dim)

        # Transformer
        self.network = torch.nn.ModuleList([
            InterfaceBlock(model.dim, 
                           dim_heads=model.dim_heads, 
                           num_blocks= model.num_compute_blocks, 
                           dim_ctx = model.dim_noise,
                           use_checkpoint=model.use_checkpoint
                           )
            for _ in range(model.num_layers)
            ])
        
        # Initialization
        self.apply(self.base_init)
        self.apply(self.zero_init)
    
    @staticmethod
    def base_init(m):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # embedding
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
        # einmix
        if isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # layer norm
        if isinstance(m, torch.nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)

    @staticmethod
    def zero_init(m):
        if isinstance(m, ConditionalLayerNorm) and m.linear is not None:
            torch.nn.init.trunc_normal_(m.linear.weight, std = 1e-8)

    def forward(self, tokens: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
        B, E = tokens.size(0), self.world_cfg.num_ens
        if E < 2:
            return self.step(tokens, visible)
        u = torch.rand((B, E), device = tokens.device, generator = self.generator)
        u = (u + torch.linspace(0, 1, B, device = tokens.device).view(-1, 1)) % 1 # stratification
        u = einops.rearrange(u, 'b e -> (b e) 1 1')
        tokens = einops.repeat(tokens, 'b ... -> (b e) ...', e = E)
        visible = einops.repeat(visible, 'b ... -> (b e) ...', e = E)
        prediction = self.step(tokens, visible, u)
        prediction = einops.rearrange(prediction, '(b e) ... t -> b ... (e t)', e = E)
        return prediction

    def step(self, tokens: torch.FloatTensor, visible: torch.BoolTensor, noise: torch.FloatTensor = None) -> torch.FloatTensor:
        # tokens: shape [B, N, D], visible: shape [B, N, 1]
        src = self.proj_in(tokens)
        x = torch.where(visible, src, self.masks.weight)
        x = self.norm_in(x) + self.positions.weight
        z = self.latents.weight.expand(tokens.size(0), -1, -1)
        ctx = self.noise_embedding(noise) if exists(self.noise_embedding) else None
        for block in self.network:
            x, z = block(x, z, ctx = ctx)
        out = self.proj_out(x)
        return out
    