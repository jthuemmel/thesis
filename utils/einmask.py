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
        self.global_tokens = torch.nn.Parameter(torch.zeros((default(network.num_latents, 1), network.dim_in)))
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
              TransformerBlock(dim= network.dim_in, dim_ctx= network.dim_ctx, num_heads= network.num_encoder_heads) 
              for _ in range(default(network.num_read_blocks, 1))
              ])
        
        # Mask decoder
        self.to_decoder = torch.nn.Linear(network.dim_in, network.dim_out, bias = False)
        self.decoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_out, dim_ctx= None, num_heads= network.num_decoder_heads) 
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
            torch.nn.init.trunc_normal_(m.global_tokens, std = 0.02)
            torch.nn.init.trunc_normal_(m.mask_token, std = 0.02)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                members: int = 1,
                rng: Optional[torch.Generator] = None,
                ) -> torch.FloatTensor:
        # tokenize
        tokens = self.field_encoder(fields) + self.position_codes

        # select visible and expand to ensemble size
        src = einops.repeat(tokens[visible], '(b n) d -> (b e) n d', e = members, b = tokens.size(0))
        visible = einops.repeat(visible, 'b n -> (b e) n ()', e = members)
        latents = einops.repeat(self.global_tokens, 'm d -> b m d', b = src.size(0))

        # stochastic conditioning
        noise = torch.randn([src.size(0), 1, self.network.dim_ctx], device = src.device, dtype = src.dtype, generator = rng)
        noise = self.noise_encoder(noise)

        # add global latent tokens
        src, ps = einops.pack([src, latents], 'b * d')

        # encode(visible tokens | noise)
        for read in self.encoder:
            src = read(src, ctx = noise)
        
        # unpack latents before scatter
        src, latents = einops.unpack(src, ps, 'b * d')

        # pad with mask tokens
        tgt = torch.masked_scatter(
            input = self.mask_token.type_as(src),
            mask = visible, # where visible is True
            source = src # copy src elements over
            )
        
        # pack latents again
        tgt, ps = einops.pack([tgt + self.position_codes, latents], 'b * d')

        # decode all tokens
        tgt = self.to_decoder(tgt)
        for write in self.decoder:
            tgt = write(tgt)

        # unpack latents again
        tgt, _ = einops.unpack(tgt, ps, 'b * d')

        # tokens -> field and ensemble -> last
        tgt = self.field_decoder(tgt)
        tgt = einops.rearrange(tgt, '(b e) ... -> b ... e', e = members)
        return tgt
