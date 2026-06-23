import einops
import torch

from einops.layers.torch import EinMix
from utils.config import *
from utils.components import *
from utils.loss_fn import *
from utils.random_fields import RandomField

class EinMask_ENS(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # config attributes
        self.network = network
        self.world = world

        # default output dimension
        DO = default(network.dim_out, network.dim)
        DN = default(network.dim_noise, network.dim)
        
        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(torch.nn.init.trunc_normal_(torch.zeros(network.num_latents, network.dim), std = network.dim ** -0.5))
        self.src_positions = torch.nn.Parameter(init_sincos_positions(network.dim, shape = world.token_shape))
        self.tgt_positions = torch.nn.Parameter(init_sincos_positions(DO, shape = world.token_shape))
        
        # maybe noise generator
        if default(world.ens_size, 1) > 1:
            self.random_field = RandomField(network, world)
            self.noise_to_encoder = torch.nn.Sequential(torch.nn.Linear(DN, network.dim, bias = False), torch.nn.RMSNorm(network.dim))
            self.noise_to_decoder = torch.nn.Sequential(torch.nn.Linear(DN, DO, bias = False), torch.nn.RMSNorm(DO))
        else:
            self.random_field = None
            self.noise_to_decoder = torch.nn.Identity()
            self.noise_to_encoder = torch.nn.Identity()

        # I/O
        self.fields_to_tokens = torch.nn.Sequential(
            EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) c',
                weight_shape = f'v {world.patch_pattern} c', 
                c = network.dim, **world.token_sizes, **world.patch_sizes),
            torch.nn.RMSNorm(network.dim)
        )

        self.tokens_to_fields = EinMix(f'b ({world.token_pattern}) d -> (b k) {world.field_pattern}',
                                weight_shape = f'k v {world.patch_pattern} d', 
                                d = DO, k = default(network.num_tails, 1), **world.patch_sizes, **world.token_sizes)

        self.smoothing = GaussianSmoothing3D(world.field_shape[0], kernel_size= 5, sigma= 1., padding_mode= 'zeros')
        
        # encoder / decoder
        self.encoder = torch.nn.ModuleList([
                TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads, drop_path= network.drop_path) 
                for _ in range(default(network.num_read_blocks, 1))
                ])
        
        self.decoder = torch.nn.ModuleList([
                TransformerBlock(dim= DO, num_heads= network.num_decoder_heads) 
                for _ in range(default(network.num_write_blocks, 1))
                ])
        
        self.latent_to_decoder = torch.nn.Sequential(torch.nn.Linear(network.dim, DO, bias = False), torch.nn.RMSNorm(DO))

        # weight initialization
        self.apply(self.base_init)
        
    def base_init(self, m: torch.nn.Module):
        if isinstance(m, EinMix) or isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)
   
    def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor, rng: torch.Generator = None) -> torch.FloatTensor:
        B, E = fields.size(0), default(self.world.ens_size, 1)

        # ensemble expansion
        fields = einops.repeat(fields, 'b ... -> (b e) ...', b = B, e = E)
        visible = einops.repeat(visible, 'b ... -> (b e) ...', b = B, e = E)
        
        # tokenize and add position codes
        tokens = self.fields_to_tokens(fields) + self.src_positions

        # maybe sample random noise
        noise = self.random_field(tokens, rng) if exists(self.random_field) else tokens.new_zeros((B * E, 1, 1))

        # add noise embedding and select visible
        tokens = tokens + self.noise_to_encoder(noise)
        src = einops.rearrange(tokens[visible], '(b m) ... -> b m ...', b = fields.size(0))

        # pad with latent tokens
        latents = einops.repeat(self.latent_tokens, '... -> (b e) ...', b = B, e = E)
        latents, ps = einops.pack([src, latents], 'b * d')

        # self-attention encoder
        for read in self.encoder:
            latents = read(latents)

        # maybe bottleneck
        if self.network.kwargs.get('bottleneck', False):
            _, latents = einops.unpack(latents, ps, 'b * d')

        # project to decoder dim
        latents = self.latent_to_decoder(latents)
        queries = self.noise_to_decoder(noise) + self.tgt_positions

        # flamingo decoder
        for write in self.decoder:
            queries = write(queries, torch.cat([queries, latents], dim = 1))

        # prediction head
        queries = self.tokens_to_fields(queries)

        #maybe smoothing
        if self.network.kwargs.get('smoothing', False):
            queries = self.smoothing(queries)
        
        # reshape to ensemble last
        return einops.rearrange(queries, '(b e) ... -> b ... e', b = B)

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # config attributes
        self.network = network
        self.world = world

        # default I/O dimensions
        DO = default(network.dim_out, network.dim)
        DI = default(network.dim_in, network.dim)

        # learnable parameters
        self.latent_tokens = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.zeros(network.num_latents, network.dim), std = network.dim ** -0.5)
            )
        self.mask_token = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(torch.zeros(DO), std = DO ** -0.5)
            )
        self.src_positions = torch.nn.Parameter(
            init_sincos_positions(network.dim, shape = world.token_shape)
            )
        self.tgt_positions = torch.nn.Parameter(
            init_sincos_positions(DO, shape = world.token_shape)
            )

        # I/O
        self.to_tokens = torch.nn.Sequential(
            EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) c',
                weight_shape = f'v {world.patch_pattern} c', 
                c = DI, **world.token_sizes, **world.patch_sizes),
            EinMix(f'b ({world.token_pattern}) c -> b ({world.token_pattern}) d',
                weight_shape = f'v d c',
                d = network.dim, c = DI, **world.token_sizes),
            torch.nn.RMSNorm(network.dim)
        )
        
        self.to_decoder = torch.nn.Sequential(
            torch.nn.RMSNorm(network.dim),
            torch.nn.Linear(network.dim, DO, bias = False),
        )

        self.to_output = torch.nn.Sequential(
            EinMix(f'b ({world.token_pattern}) d -> (k b) {world.field_pattern}',
                   weight_shape = f'k v {world.patch_pattern} d',
                   d = DO, k = network.num_tails, **world.patch_sizes, **world.token_sizes),
            GaussianSmoothing3D(world.field_shape[0], kernel_size= 5, sigma= 1.),
            Rearrange('(k b) ... -> k b ...', k = network.num_tails)
        )
        
        # Encoder / Decoder
        self.encoder = torch.nn.ModuleList([
                TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads, drop_path= network.drop_path) 
                for _ in range(default(network.num_read_blocks, 1))
                ])
        
        self.decoder = torch.nn.ModuleList([
                TransformerBlock(dim= DO, num_heads= network.num_decoder_heads) 
                for _ in range(default(network.num_write_blocks, 1))
                ])
        
        # weight initialization
        self.apply(self.base_init)
        
    def base_init(self, m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear) or isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = m.weight.size(-1) ** -0.5)
            if exists(m.bias):
                torch.nn.init.zeros_(m.bias)            
   
    def forward(self, fields: torch.FloatTensor, visible: torch.BoolTensor, rng: torch.Generator = None) -> torch.FloatTensor:
        B = fields.size(0)
        
        # tokenize and add position codes
        tokens = self.to_tokens(fields) + self.src_positions

        # create queries from position codes and add mask token
        tgt = einops.repeat(self.tgt_positions, 'n d -> b n d', b = B)
        tgt = tgt + self.mask_token

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

        # cross-attention decoder
        for write in self.decoder:
            tgt = write(tgt, latents)

        # prediction head
        pred = self.to_output(tgt)
        return pred
