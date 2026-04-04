import einops
import torch

from einops.layers.torch import EinMix, Rearrange
from utils.config import *
from utils.components import *

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(torch.zeros((network.num_latents, network.dim)))
        self.tgt_positions = torch.nn.Parameter(self.init_sincos_positions(network.dim_out))
        self.src_positions = torch.nn.Parameter(self.init_sincos_positions(network.dim_in))

        # I/O        
        self.src_encoder = EinMix(pattern = f'b {world.field_pattern} -> b ({world.token_pattern}) d', 
                   weight_shape= f'v {world.patch_pattern} d',
                   **world.patch_sizes, **world.token_sizes, d = network.dim_in
                )
        self.tgt_decoder = FieldDecoder(network.dim_out, world = world, num_tails= network.num_tails)

        
        # Transformer
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim, dim_kv = network.dim_in, num_heads= network.num_encoder_heads) 
              for _ in range(default(network.num_read_blocks, 1))
              ])
        
        self.processor = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim) 
              for _ in range(default(network.num_compute_blocks, 1))
              ])
        
        self.decoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_out, dim_kv = network.dim, num_heads= network.num_decoder_heads) 
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
        # einmask
        elif isinstance(m, EinMask):
            torch.nn.init.trunc_normal_(m.latent_tokens, std = 0.02)
   
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor,
                **kwargs
                ) -> torch.FloatTensor:
        latents = self.encode(fields, visible) # variable-size cannot run compiled
        tgt = self.predict(latents) # fixed-size can
        return tgt
    
    def encode(self, fields: torch.FloatTensor, visible: torch.BoolTensor,):
        B = visible.size(0)
        tokens = self.src_encoder(fields) + self.src_positions
        src = einops.rearrange(tokens[visible], '(b m) d -> b m d', b = B) # select visible tokens
        latents = einops.repeat(self.latent_tokens, 'z d -> b z d', b = B)
        for read in self.encoder:
            latents = read(latents, kv = src)
        return latents

    @torch.compile()
    def predict(self, latents: torch.FloatTensor):
        B = latents.size(0)
        # latent processor
        for compute in self.processor:
            latents = compute(latents)
        # Decoder
        tgt = einops.repeat(self.tgt_positions, 'n d -> b n d', b= B)
        for write in self.decoder:
            tgt = write(tgt, kv = latents)
        tgt = self.tgt_decoder(tgt)
        tgt = einops.rearrange(tgt, '(b e) ... -> b ... e', b = B)
        return tgt