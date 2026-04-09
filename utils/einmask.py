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

def reparameterize(mu: torch.FloatTensor, sigma: torch.FloatTensor, rng: Optional[torch.Generator] = None):
    sigma = torch.nn.functional.softplus(sigma)
    kl = -0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2)).sum(dim=-1).mean()
    x_hat = mu + torch.randn_like(sigma, generator = rng) * sigma
    return x_hat, kl

class EinVAE(torch.nn.Module):
    def __init__(self, latent_dim: int, world: WorldConfig):
        super().__init__()
        self.world = world
        self.src_encoder = torch.nn.Sequential(
                EinMix(pattern = f'b {world.field_pattern} -> b ({world.token_pattern}) do', 
                    weight_shape= f'v {world.patch_pattern} do',
                    **world.patch_sizes, **world.token_sizes, do = world.dim_tokens),
                torch.nn.SiLU(),
                EinMix(f'b ({world.token_pattern}) di -> two b ({world.token_pattern}) do', 
                        weight_shape = f'two v do di',
                        **world.token_sizes, di = world.dim_tokens, do = latent_dim, two = 2),
            )

        self.tgt_decoder = torch.nn.Sequential(
            EinMix(f'b ({world.token_pattern}) di -> b ({world.token_pattern}) do',
                    weight_shape= 'v do di', 
                    bias_shape = f'{world.token_pattern} do',
                    di= latent_dim, do= world.dim_tokens, **world.token_sizes),
            TransformerBlock(world.dim_tokens, num_heads= 8),
            EinMix(f'b ({world.token_pattern}) di -> b {world.field_pattern}',
                    weight_shape = f'v {world.patch_pattern} di',
                    **world.token_sizes, **world.patch_sizes, di = world.dim_tokens),
            GaussianSmoothing3D(world.field_shape[0], sigma = 1.)
        )
        
        # weight initialization
        self.apply(self.base_init)
    
    def base_init(self, m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)
        # einmix (care with the weight shapes)
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias) and m.bias.ndim == 1: # initialized as a linear bias
                torch.nn.init.zeros_(m.bias)
            elif exists(m.bias) and m.bias.ndim > 1: # initialized as a positional bias
                m.bias = torch.nn.Parameter(init_sincos_positions(m.bias.size(-1), world= self.world).reshape_as(m.bias).contiguous())
    
    @torch.compile()
    def forward(self, fields: torch.FloatTensor, members: Optional[int] = 1, rng: Optional[torch.Generator] = None):
        mu, sigma = self.src_encoder(fields)
        mu = einops.repeat(mu, 'b ... -> (b e) ...', e = members)
        sigma = einops.repeat(sigma, 'b ... -> (b e) ...', e = members)
        x_hat, kl = reparameterize(mu, sigma, rng)
        x_hat = self.tgt_decoder(x_hat)
        x_hat = einops.rearrange(x_hat, '(b e) ... -> b ... e', e = members)
        return x_hat, kl
    
class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(torch.zeros((network.num_latents, network.dim)))
        self.tgt_positions = torch.nn.Parameter(init_sincos_positions(network.dim_out, world= world))
        self.src_positions = torch.nn.Parameter(init_sincos_positions(network.dim_in, world= world))

        # I/O        
        self.tokenizer = EinVAE(network.dim_in, world)
        self.predictor = torch.nn.Sequential(
            torch.nn.RMSNorm(network.dim_out),
            EinMix('b n d -> two b n c', 'two c d', two = 2, c = network.dim_in, d = network.dim_out)
            )
        
        # Perceiver
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
        
    def base_init(self, m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)
        # embedding
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
        # einmix
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)
        # einmask
        elif isinstance(m, EinMask):
            torch.nn.init.trunc_normal_(m.latent_tokens, std = m.latent_tokens.size(-1) ** -0.5)
            #torch.nn.init.trunc_normal_(m.mask_tokens, std = m.mask_tokens.size(-1) ** -0.5)
   
    @torch.compile()
    def to_tokens(self, fields: torch.FloatTensor, rng: Optional[torch.Generator] = None):
        z = self.tokenizer.src_encoder(fields) 
        x_hat, _ = reparameterize(z[0], z[1], rng)
        return x_hat 

    def read(self, tokens: torch.FloatTensor, visible: torch.BoolTensor,):
        B = visible.size(0)
        tokens = tokens + self.src_positions # add position codes
        src = einops.rearrange(tokens[visible], '(b m) d -> b m d', b = B) # select visible tokens
        z = einops.repeat(self.latent_tokens, 'n d -> b n d', b = B) # expand latents
        for block in self.encoder:
            z = block(z, kv = src) # cross-attend tokens into latents
        return z
    
    @torch.compile()
    def process(self, latents: torch.FloatTensor):
        B = latents.size(0)
        for block in self.processor:
            latents = block(latents) # latent self-attention
        queries = einops.repeat(self.tgt_positions, 'n d -> b n d', b = B) # expand all tgt positions
        for block in self.decoder:
            queries = block(queries, kv = latents) # cross-attend latents into queries
        mu, sigma = self.predictor(queries)
        return mu, sigma
    
    @torch.compile()
    def to_fields(self, mu: torch.FloatTensor, sigma: torch.FloatTensor, rng: Optional[torch.Generator] = None):
        tgt, _ = reparameterize(mu, sigma, rng)
        tgt = self.tokenizer.tgt_decoder(tgt)
        return tgt

    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor,
                members: Optional[int] = 1, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        fields = einops.repeat(fields, 'b ... -> (b e) ...', e = members)
        visible = einops.repeat(visible, 'b ... -> (b e) ...', e = members)
        tokens = self.to_tokens(fields, rng)# fields -> tokens 
        latents = self.read(tokens, visible) # visible tokens -> latents
        mu, sigma = self.process(latents) # latents -> predicted distribution
        x_hat = self.to_fields(mu, sigma, rng)
        x_hat = einops.rearrange(x_hat, '(b e) ... -> b ... e', e = members)
        latent_loss = f_gaussian_crps(tokens, mu, torch.nn.functional.softplus(sigma))[visible.logical_not()].mean()
        return x_hat, latent_loss
