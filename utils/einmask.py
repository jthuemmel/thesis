import einops
import torch

from einops.layers.torch import EinMix
from utils.config import *
from utils.components import *
from utils.loss_fn import *

def init_sincos_positions(dim: int, world: WorldConfig):
    # integer indices
    coordinates = torch.stack(torch.unravel_index(indices = torch.arange(world.num_tokens), shape = world.token_shape), dim = -1)
    # log wavelengths
    log_wavelengths = torch.as_tensor(world.token_shape).log()
    # only encode shape dimensions with actual size
    valid = log_wavelengths > 0
    log_wavelengths = log_wavelengths[valid]
    coordinates = coordinates[:, valid]
    # space the frequencies according to the required number of bands
    negative_spacing = torch.linspace(0, -1, dim // (coordinates.size(-1) * 2))
    # calculate the sin/cos embeddings:
    frequencies = torch.exp(negative_spacing * log_wavelengths[..., None])
    angles = torch.einsum("n i, i d -> n i d", coordinates, frequencies) # overflows fp16, be careful
    positions = einops.rearrange([angles.sin(), angles.cos()], 'two n i d -> n (two i d)')
    # avoid uneven dimensions by zero-padding
    positions = torch.nn.functional.pad(positions, (0, dim - positions.size(-1))) 
    return positions

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(torch.zeros(network.num_latents, network.dim))
        self.mask_token = torch.nn.Parameter(torch.zeros(network.dim_out))
        self.src_positions = torch.nn.Parameter(init_sincos_positions(network.dim, world= world))
        self.tgt_positions = torch.nn.Parameter(init_sincos_positions(network.dim_out, world= world))

        # I/O 
        self.to_tokens = torch.nn.Sequential(
            EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) d',
                   weight_shape = f'v {world.patch_pattern} d',
                   d = network.dim_in, **world.token_sizes, **world.patch_sizes),
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
                TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads) 
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
        if isinstance(m, EinMask):
            torch.nn.init.trunc_normal_(m.mask_token, std = m.mask_token.size(-1) ** -0.5)
            torch.nn.init.trunc_normal_(m.latent_tokens, std = m.latent_tokens.size(-1) ** -0.5)
   
    def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
        B = fields.size(0)
        # tokenize and add position codes
        tokens = self.to_tokens(fields) + self.src_positions

        # select visible
        src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = B)
        
        # latent encoder
        latents = einops.repeat(self.latent_tokens, 'z d -> b z d', b = B)
        latents, shape = einops.pack([src, latents], 'b * d')
        for read in self.encoder:
            latents = read(latents)

        # project latents to decoder
        _, latents = einops.unpack(latents, shape, 'b * d')
        latents = self.to_decoder(latents)

        # create queries from mask tokens and scatter src tokens
        tgt = einops.repeat(self.mask_token, 'd -> b n d', b = B, n = tokens.size(1))
        tgt = tgt + self.tgt_positions

        # decoder
        tgt, shape = einops.pack([tgt, latents], 'b * d')
        for write in self.decoder:
            tgt = write(tgt)

        # prediction head
        tgt, _ = einops.unpack(latents, shape, 'b * d')
        pred = self.to_output(tgt)
        return pred
    