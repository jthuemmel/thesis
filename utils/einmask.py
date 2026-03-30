import einops
import torch

from einops.layers.torch import *
from utils.config import *
from utils.components import *
from torch.nn.functional import softplus

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()

        # I/O
        self.to_tokens =  EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) di', 
                    weight_shape= f'v {world.patch_pattern} di',
                    **world.patch_sizes, **world.token_sizes, di = network.dim_in
                    )
        
        self.to_fields = FieldDecoder(network, world)

        # linear projections
        self.to_bottleneck = torch.nn.Linear(network.dim, network.dim_out * 2, bias = False)
        self.to_predictor = torch.nn.Linear(network.dim_in, network.dim, bias = False)

        # embeddings
        self.positions = torch.nn.Embedding(world.num_tokens, network.dim_in)
        self.queries = torch.nn.Embedding(world.num_tokens, network.dim)

        # Transformer components
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_in) for _ in range(default(network.num_read_blocks, 1))
              ])
        
        self.predictor = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim) for _ in range(default(network.num_compute_blocks, 1))
              ])
        
        self.bottleneck = TransformerBlock(dim = network.dim_out * 2, num_heads=network.num_decoder_heads)
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(dim=network.dim_out) for _ in range(default(network.num_write_blocks, 1))
            ])

        # weight initialization
        self.apply(self.base_init)

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
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None,
                ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        latents = self.encode(fields, visible)
        latents = self.predict(latents, visible)
        ensemble = self.decode(latents, members, rng)
        return ensemble, latents
    
    def encode(self,
               fields: torch.FloatTensor, 
               visible: torch.BoolTensor, 
               ) -> torch.FloatTensor:
        tokens = self.to_tokens(fields) + self.positions.weight
        x = einops.rearrange(tokens[visible], '(b n) d -> b n d', b = tokens.size(0))
        for read in self.encoder:
            x = read(x)
        x = self.to_predictor(x)
        return x
    
    def predict(self,
               latents: torch.FloatTensor, 
               visible: torch.BoolTensor, 
               ) -> torch.FloatTensor:
        queries = self.queries.weight.type_as(latents)
        x = queries.masked_scatter(visible[..., None], latents)
        for compute in self.predictor:
            x = compute(x)
        return x
    
    def decode(self, 
               latents: torch.FloatTensor,
               members: Optional[int] = None, 
               rng: Optional[torch.Generator] = None
               ) -> torch.FloatTensor:
        latents = latents.clone().detach()
        x = self.to_bottleneck(latents)
        latents = self.bottleneck(latents)
        mu, sigma = einops.repeat(latents, 'b n (two d) -> two (b e) n d', e = default(members, 1), two = 2)
        x = mu + torch.randn_like(sigma, generator = rng) * softplus(sigma)
        for write in self.decoder:
            x = write(x)
        fields = self.to_fields(x)
        fields = einops.rearrange(fields, "(b e) ... -> b ... e", e = default(members, 1))
        return fields