import einops
import torch

from einops.layers.torch import EinMix
from utils.config import *
from utils.components import *
from utils.loss_fn import *
    
class EinAR(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # config attributes
        self.network = network
        self.world = world

        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(network.num_latents, network.dim), std = network.dim ** -0.5))
        self.src_positions = torch.nn.Parameter(init_sincos_positions(network.dim, world= world))

        # I/O
        self.to_tokens = torch.nn.Sequential(
            EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) c',
                weight_shape = f'v {world.patch_pattern} c', 
                c = network.dim, **world.token_sizes, **world.patch_sizes),
            torch.nn.RMSNorm(network.dim)
        )

        self.to_output = torch.nn.Sequential(
            EinMix(f'b ({world.token_pattern}) d -> (k b) {world.field_pattern}',
                   weight_shape = f'k v {world.patch_pattern} d',
                   d = network.dim, k = network.num_tails, 
                   **world.patch_sizes, **world.token_sizes),
            GaussianSmoothing3D(world.field_shape[0], kernel_size= 5, sigma= 1.),
            Rearrange('(k b) ... -> k b ...', k = network.num_tails)
        )
        
        # Encoder
        self.predictor = torch.nn.ModuleList([
                TransformerBlock(dim= network.dim, drop_path= network.drop_path) 
                for _ in range(default(network.num_layers, 1))
                ])
        
        # weight initialization
        self.apply(self.base_init)
        
    def base_init(self, m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear) or isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)            
   
    def forward(self, fields: torch.FloatTensor, num_steps: int = 1):
        # tokenize input
        src = self.to_tokens(fields) + self.src_positions
        
        # roll-out
        predictions = []
        for _ in range(num_steps):
            # add cls tokens
            cls = einops.repeat(self.latent_tokens, 'z d -> b z d', b= src.size(0))
            latents, shape = einops.pack([src, cls], 'b * d')
            
            # transformer stack
            for predict in self.predictor:
                latents = predict(latents)
            
            # forward src tokens
            src, cls = einops.unpack(latents, shape, 'b * d')
            
            # decode
            pred = self.to_output(src)
            predictions.append(pred)
        
        return einops.pack(predictions, 'k b v * h w')[0]