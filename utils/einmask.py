import einops
import torch

from einops.layers.torch import *
from utils.config import *
from utils.components import *
from torch.nn.functional import softplus

class Encoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world
        self.from_fields = torch.nn.Sequential(
            EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) di', 
                   weight_shape= f'v {world.patch_pattern} di',
                   **world.patch_sizes, **world.token_sizes, di = network.dim_in),
            torch.nn.RMSNorm(network.dim_in)
        )
        self.to_output = torch.nn.Linear(network.dim_in, network.dim, bias = False)
        self.latents = torch.nn.Embedding(network.num_latents, network.dim_in)
        self.positions = torch.nn.Embedding(world.num_tokens, network.dim_in)
        self.mask_codes = torch.nn.Embedding(2, network.dim_in)
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_in, num_heads= network.num_encoder_heads)
              for _ in range(default(network.num_read_blocks, 1))
              ])
        
    @torch.compiler.disable()
    def apply_mask(self, x: torch.FloatTensor, visible: torch.BoolTensor) -> torch.FloatTensor:
        return torch.where(visible[..., None], x, 0)
    
    def forward(self,
               fields: torch.FloatTensor, 
               visible: torch.BoolTensor, 
               ):
        tokens = self.from_fields(fields)
        x = self.apply_mask(tokens, visible)
        src = x + self.mask_codes(visible.long()) + self.positions.weight 
        latents = einops.repeat(self.latents.weight, '... -> b ...', b = src.size(0))
        for read in self.encoder:
            latents = read(latents, torch.cat([src, latents], dim = 1))
        return self.to_output(latents)

class Predictor(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world
        self.positions = torch.nn.Embedding(world.num_tokens, network.dim)
        self.mask_codes = torch.nn.Embedding(2, network.dim)
        self.predictor = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads)
              for _ in range(default(network.num_compute_blocks, 1))
              ])
        
    def forward(self, latents: torch.FloatTensor, targets: torch.BoolTensor):
        tgt = self.mask_codes(targets.long()) + self.positions.weight
        for compute in self.predictor:
            latents = compute(latents, torch.cat([tgt, latents], dim = 1))
        return latents

class Decoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world
        self.to_fields = FieldDecoder(network, world)  
        self.queries = torch.nn.Embedding(world.num_tokens + default(network.num_cls, 1), network.dim_out * 2)
        self.bottleneck = TransformerBlock(dim = network.dim_out * 2, dim_kv= network.dim, num_heads=network.num_decoder_heads)
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(dim=network.dim_out) 
            for _ in range(default(network.num_write_blocks, 1))
            ])
        
    def forward(self, 
               latents: torch.FloatTensor,
               members: Optional[int] = None, 
               rng: Optional[torch.Generator] = None
               ) -> torch.FloatTensor:
        B = latents.size(0)
        E = default(members, 1)

        # read from latents
        queries = einops.repeat(self.queries.weight, '... -> b ...', b = B)
        queries = self.bottleneck(queries, kv = latents)

        # reparametrization
        mu, sigma = einops.repeat(queries, 'b n (two d) -> two (b e) n d', e = E, two = 2)
        x = mu + torch.randn_like(sigma, generator = rng) * softplus(sigma)

        for write in self.decoder:
            x = write(x)

        # split off register tokens and map to fields
        x = x[:, :self.world.num_tokens] 
        fields = self.to_fields(x)
        fields = einops.rearrange(fields, "(b e) ... -> b ... e", b = B, e = E)
        return fields

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.encoder = Encoder(network, world)
        self.predictor = Predictor(network, world)
        self.decoder = Decoder(network, world)

        # weight initialization
        self.apply(self.base_init)

        # maybe compile
        if network.kwargs.get('compile', True):
            self.encoder.compile()
            self.decoder.compile()
            self.predictor.compile()

    @staticmethod
    def freeze_weights(m: torch.nn.Module):
        for param in m.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_weights(m: torch.nn.Module):
        for param in m.parameters():
            param.requires_grad = True

    @staticmethod
    def base_init(m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # embedding
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # einmix
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                targets: torch.BoolTensor,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None,
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        latents = self.encoder(fields, visible)
        latents = self.predictor(latents, targets)
        prediction = self.decoder(latents, members, rng)
        return prediction, latents
    