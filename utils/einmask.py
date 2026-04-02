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
        self.mask_token = torch.nn.Parameter(torch.zeros((network.dim_in,)))
        self.position_codes = torch.nn.Parameter(self.init_sincos_positions(network.dim_in))

        # I/O
        self.field_encoder = EinMix(
            pattern = f'b {world.field_pattern} -> b ({world.token_pattern}) di', 
            weight_shape= f'v {world.patch_pattern} di',
            **world.patch_sizes, **world.token_sizes, di = network.dim_in
            )
        self.field_decoder = FieldDecoder(network, world)
        
        # Encoder
        self.noise_encoder = GatedFFN(network.dim_ctx)
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_in, num_heads= network.num_encoder_heads) 
              for _ in range(default(network.num_read_blocks, 1))
              ])
        
        # Mask decoder
        self.to_decoder = torch.nn.Linear(network.dim_in, network.dim_out, bias = False)
        self.decoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_out, dim_ctx = network.dim_ctx, num_heads= network.num_decoder_heads) 
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
        # adaLN
        elif isinstance(m, AdaptiveLayerNorm):
            torch.nn.init.zeros_(m.bias)
            if exists(m.weight):
                torch.nn.init.trunc_normal_(m.weight, std = 1e-5)
        elif isinstance(m, EinMask):
            torch.nn.init.trunc_normal_(m.mask_token, std = 0.02)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                members: int = 1,
                rng: Optional[torch.Generator] = None,
                ) -> torch.FloatTensor:
        B = visible.size(0)
        # tokenize
        tokens = self.field_encoder(fields) + self.position_codes

        # encode visible
        src = einops.rearrange(tokens[visible], '(b n) d -> b n d', b = B)
        for read in self.encoder:
            src = read(src)

        # pad with mask tokens
        tgt = (self.mask_token + self.position_codes).type_as(src)
        tgt = torch.masked_scatter(
            input = tgt, # for all masked locations
            mask = visible[..., None], # where visible is True
            source = src # copy src elements over
            )
    
        # stochastic conditioning
        tgt = einops.repeat(tgt, 'b n d -> (b e) n d', e = members)
        noise = torch.randn([B * members, 1, self.network.dim_ctx], device = tgt.device, dtype = tgt.dtype, generator = rng)
        noise = self.noise_encoder(noise)

        # decode (all tokens | noise)
        tgt = self.to_decoder(tgt)
        for write in self.decoder:
            tgt = write(tgt, ctx = noise)

        # tokens -> field and ensemble -> last
        tgt = self.field_decoder(tgt)
        tgt = einops.rearrange(tgt, '(b e) ... -> b ... e', b = B)
        return tgt
