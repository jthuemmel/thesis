import torch
from utils.components import *
from einops.layers.torch import EinMix

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
        self.proj_noise = GatedFFN(dim=model.dim_noise) if exists(model.dim_noise) else torch.nn.Identity()
        self.proj_latents = SelfConditioning(dim=model.dim)
        
        # Per-variable I/O projections
        self.norm_in = ConditionalLayerNorm(model.dim)
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
        self.interface_network = torch.nn.ModuleList([
            InterfaceBlock(model.dim, 
                           dim_heads=model.dim_heads, 
                           num_blocks= model.num_compute_blocks, 
                           dim_ctx = model.dim_noise,
                           use_checkpoint=model.use_checkpoint)
            for _ in range(model.num_layers)
            ])
        
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
            if m.weight is not None:
                torch.nn.init.zeros_(m.weight)

    def forward(self, 
             tokens: torch.FloatTensor, 
             mask: torch.BoolTensor, 
             latents: torch.FloatTensor = None,
             noise: torch.FloatTensor = None
             ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
        Args:
            tokens: (B, N, C_in) Tensor of input tokens
            mask: (B, N, 1) BoolTensor indicating masked tokens
            latents: (B, L, D) Tensor of latent variables (optional)
            noise: (B, 1, C_noise) Tensor of noise conditioning (optional)
        Returns:
            x: (B, N, C_out) Tensor of predicted tokens
            z: (B, L, D) Tensor of latent variables after processing
        '''
        # input projection
        x = self.proj_in(tokens)
        # apply mask
        x = torch.where(mask, self.masks.weight, x)
        # add positional embeddings
        x = self.norm_in(x) + self.positions.weight
        # self-condition latents
        z_init = self.latents.weight.expand(tokens.size(0), -1, -1)
        z = self.proj_latents(z_init, previous = latents)
        # shared noise projection
        if exists(noise): noise = self.proj_noise(noise)
        # transformer
        for block in self.interface_network: 
            x, z = block(x, z, ctx = noise)
        # output projection
        x = self.proj_out(x)
        # copy over unmasked tokens
        x = torch.where(mask, x, tokens)
        return x, z