import einops
import torch

from einops.layers.torch import EinMix
from utils.config import *
from utils.components import *
from utils.loss_fn import *


class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # config attributes
        self.network = network
        self.world = world

        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(network.num_latents, network.dim), std = network.dim ** -0.5))
        self.mask_token = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(network.dim_out), std = network.dim_out.size(-1) ** -0.5))
        self.src_positions = torch.nn.Parameter(init_sincos_positions(network.dim, world= world))
        self.tgt_positions = torch.nn.Parameter(init_sincos_positions(network.dim_out, world= world))

        # I/O
        self.to_tokens = torch.nn.Sequential( # maybe keep vv out of c here for delayed mixing
            EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) c',
                weight_shape = f'v {world.patch_pattern} c', 
                c = network.dim_in, **world.token_sizes, **world.patch_sizes),
            EinMix(f'b ({world.token_pattern}) c -> b ({world.token_pattern}) d',
                weight_shape = f'v d c',
                d = network.dim, c = network.dim_in, **world.token_sizes),
            torch.nn.RMSNorm(network.dim)
        )
        
        self.to_decoder = torch.nn.Sequential(
            torch.nn.RMSNorm(network.dim),
            torch.nn.Linear(network.dim, network.dim_out, bias = False),
        )

        self.to_output = torch.nn.Sequential(
            EinMix(f'b ({world.token_pattern}) d -> (k b) {world.field_pattern}',
                   weight_shape = f'k v {world.patch_pattern} d',
                   d = network.dim_out, k = network.num_tails, **world.patch_sizes, **world.token_sizes),
            GaussianSmoothing3D(world.field_shape[0], kernel_size= 5, sigma= 1.),
            Rearrange('(k b) ... -> k b ...', k = network.num_tails)
        )
        
        # Encoder / Decoder
        self.encoder = torch.nn.ModuleList([
                TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads, drop_path= network.drop_path) 
                for _ in range(default(network.num_read_blocks, 1))
                ])
        
        self.decoder = torch.nn.ModuleList([
                TransformerBlock(dim= network.dim_out, num_heads= network.num_decoder_heads) 
                for _ in range(default(network.num_write_blocks, 1))
                ])
        
        # weight initialization
        self.apply(self.base_init)
        
    def base_init(self, m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear) or isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)            
   
    def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
        B = fields.size(0)
        
        # tokenize and add position codes
        tokens = self.to_tokens(fields) + self.src_positions

        # select visible
        src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = B)

        # pad with latent tokens
        latents = einops.repeat(self.latent_tokens, 'z d -> b z d', b = B)
        latents, shape = einops.pack([src, latents], 'b * d')

        # jointly encode latents and visible
        for read in self.encoder:
            latents = read(latents)

        # project to decoder dim with optional bottleneck
        if self.network.kwargs.get('bottleneck', False):
            _, latents = einops.unpack(latents, shape, 'b * d')
        latents = self.to_decoder(latents)

        # create queries from position codes and add mask token
        tgt = einops.repeat(self.tgt_positions, 'n d -> b n d', b = B)
        tgt = tgt + self.mask_token

        # cross-attention decoder
        for write in self.decoder:
            tgt = write(tgt, latents)

        # prediction head
        pred = self.to_output(tgt)
        return pred
    
