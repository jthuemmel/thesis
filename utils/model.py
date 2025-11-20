import torch
import einops
from utils.components import *
from utils.backbones import *
from einops.layers.torch import EinMix

TRANSFORMERS = {'perceiver': Perceiver, 'interface': InterfaceNetwork, 'vit': ViT, 'flamingo': Flamingo}

class MaskedPredictor(torch.nn.Module):
    '''Masked Predictor based on Recurrent Interface Networks (RINs)'''
    def __init__(self, model, world):
        super().__init__()
        # Attributes
        self.dim_noise = model.dim_noise
        
        # Learnable tokens
        self.positions = torch.nn.Embedding(world.num_tokens, model.dim)
        self.latents = torch.nn.Embedding(model.num_latents, model.dim)
        self.masks = torch.nn.Embedding(1, model.dim)
        
        # Projections for latents and noise
        self.proj_noise = torch.nn.Sequential(
            GatedFFN(model.dim_noise), torch.nn.LayerNorm(model.dim_noise)
        ) if exists(model.dim_noise) else torch.nn.Identity()
        self.proj_latents = SelfConditioning(dim=model.dim)
        
        # Per-variable I/O projections
        self.norm_in = torch.nn.LayerNorm(model.dim)
        self.proj_in = EinMix(
            pattern = f'{world.flatland_pattern} -> b {world.flat_token_pattern} d',
            weight_shape = f'v {world.patch_pattern} d', 
            bias_shape = 'v d',
            d = model.dim, **world.patch_sizes, **world.token_sizes
            )
        
        self.proj_out = EinMix(
            pattern = f'b {world.flat_token_pattern} d -> {world.flatland_pattern}',
            weight_shape = f'v {world.patch_pattern} d',
            d = model.dim, **world.patch_sizes, **world.token_sizes
            )
        
        # Transformer
        self.transformer = TRANSFORMERS[getattr(model, 'backbone', 'perceiver')](model, world)
        
        # Weight initialization
        self.apply(self.base_init)
    
    @staticmethod
    def base_init(m):
        '''Explicit weight initialization for all components'''
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
        # conditional layer norm
        if isinstance(m, ConditionalLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None: # CLN weight close to 0
                torch.nn.init.trunc_normal_(m.weight, std = 1e-7)
        # self conditioning
        if isinstance(m, SelfConditioning):
            torch.nn.init.trunc_normal_(m.scale, std = 1e-7)

    def step(self, 
             tokens: torch.FloatTensor, 
             mask: torch.BoolTensor, 
             latents: torch.FloatTensor = None,
             noise: torch.FloatTensor = None
             ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # input projection
        x = self.proj_in(tokens)
        # apply mask
        x = torch.where(mask, self.masks.weight, x)
        # add positional embeddings
        x = self.norm_in(x) + self.positions.weight
        # self-condition latents
        z = self.proj_latents(initial = self.latents.weight.expand(tokens.size(0), -1, -1), 
                              previous = latents)
        # shared noise projection
        if exists(noise): 
            noise = self.proj_noise(noise)
        # transformer
        x, z = self.transformer(x = x, z = z, ctx = noise)
        # output projection
        x = self.proj_out(x)
        # copy over unmasked tokens
        x = torch.where(mask, x, tokens) 
        return x, z
    
    def forward(self, 
                tokens: torch.FloatTensor, 
                masks: torch.BoolTensor, 
                E: int, 
                rng: torch.Generator = None
                ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Args:
            tokens: (B, N, C_in) Tensor of input tokens
            masks: (S, B, N) BoolTensor of masks for S steps
            E: Number of ensemble members
            rng: torch.Generator for random number generation
        Returns:
            x: (B, N, C_out, E) Tensor of predicted tokens
            z: (B, L, D, E) Tensor of latent variables after processing
        '''
        S, B, N = masks.size()
        
        # parallelise ensemble processing
        fs = torch.randn((S, B * E, 1, self.dim_noise), device = tokens.device, generator = rng)
        xs = einops.repeat(tokens, "b n c -> (b e) n c", e = E, b = B, n = N)
        ms = einops.repeat(masks, 's b n -> s (b e) n 1', e = E, b = B, n = N, s = S)

        # iterate
        zs = None
        for s in range(S):
            xs, zs = self.step(tokens = xs, mask = ms[s], latents = zs, noise = fs[s])
            # detach gradients unless it is the last step
            if s < S - 1:
                xs, zs = (xs.detach(), zs.detach())
        
        # rearrange to ensemble form
        xs = einops.rearrange(xs, "(b e) n c -> b n c e", e = E, b = B, n = N)
        zs = einops.rearrange(zs, "(b e) l d -> b l d e", e = E, b = B)
        return xs, zs