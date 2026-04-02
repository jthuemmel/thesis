import einops
import torch

from einops.layers.torch import EinMix
from utils.config import *
from utils.components import *

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # learnable parameters
        self.latent_tokens = torch.nn.Embedding(network.num_latents, network.dim_in)
        self.mask_token = torch.nn.Embedding(2, network.dim_out)
        self.tgt_positions = torch.nn.Parameter(self.init_sincos_positions(network.dim_out))
        self.src_positions = torch.nn.Parameter(self.init_sincos_positions(network.dim_in))

        # I/O
        self.field_encoder = EinMix(
            pattern = f'b {world.field_pattern} -> b ({world.token_pattern}) di', 
            weight_shape= f'v {world.patch_pattern} di',
            **world.patch_sizes, **world.token_sizes, di = network.dim_in
            )
        self.field_decoder = FieldDecoder(network, world)
        
        # Encoder
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_in, num_heads= network.num_encoder_heads) 
              for _ in range(default(network.num_read_blocks, 1))
              ])
        
        # Mask decoder
        self.to_decoder = torch.nn.Sequential(torch.nn.Linear(network.dim_in, network.dim_out, bias = False), torch.nn.RMSNorm(network.dim_out))
        self.decoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_out, num_heads= network.num_decoder_heads) 
              for _ in range(default(network.num_write_blocks, 1))
              ])

        # weight initialization
        self.apply(self.base_init)

    def init_sincos_positions(self, dim: int):
        # integer indices
        coordinates = torch.stack(torch.unravel_index(indices = torch.arange(self.world.num_tokens), shape = self.world.token_shape), dim = -1)
        # log wavelengths
        log_wavelengths = torch.as_tensor(self.world.token_shape).log()
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
        
    @staticmethod
    def base_init(m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)
        # embedding
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # einmix
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                **kwargs
                ) -> torch.FloatTensor:
        B = visible.size(0)
        tokens = self.field_encoder(fields) + self.src_positions

        src = einops.rearrange(tokens[visible], '(b n) d -> b n d', b = B)
        latents = einops.repeat(self.latent_tokens.weight, 'z d -> b z d', b = B)
        for read in self.encoder:
            latents = read(latents, kv = torch.cat([latents, src], dim = 1))
            
        latents = self.to_decoder(latents)
        tgt = self.mask_token(visible.long()) + self.tgt_positions
        for write in self.decoder:
            tgt = write(tgt, kv = torch.cat([latents, tgt], dim = 1))

        tgt = self.field_decoder(tgt)
        tgt = einops.rearrange(tgt, '(b e) ... -> b ... e', b = B)
        return tgt
