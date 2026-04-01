import einops
import torch

from einops.layers.torch import *
from utils.config import *
from utils.components import *
from torch.nn.functional import softplus

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # Encoder
        self.tokenizer =  EinMix(
            f'b {world.field_pattern} -> b ({world.token_pattern}) di', 
            weight_shape= f'v {world.patch_pattern} di', 
            **world.patch_sizes, **world.token_sizes, di = network.dim_in
            )
        
        self.positions = torch.nn.Parameter(torch.zeros(1, world.num_tokens, network.dim_in))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, network.dim_in))
    
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_in) for _ in range(default(network.num_read_blocks, 1))
              ])
        
        # Predictor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(network.dim_in, network.dim, bias = False),
            *[TransformerBlock(dim= network.dim) for _ in range(default(network.num_compute_blocks, 1))],
            torch.nn.RMSNorm(network.dim),
            torch.nn.Linear(network.dim, network.dim_in, bias = False)
        )

        # weight initialization
        self.apply(self.base_init)

        # maybe compile
        self.predictor.compile()

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
        # explicit parameters
        elif isinstance(m, EinMask):
            torch.nn.init.trunc_normal_(m.mask_token, std = 0.02)
            torch.nn.init.trunc_normal_(m.positions, std = 0.02)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                **kwargs: dict
                ) -> torch.FloatTensor:
        src = self.encode(fields, visible)
        tgt = self.predict(src, visible)
        return tgt

    def encode(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                ) -> torch.FloatTensor:
        # map fields to tokens and add position codes
        tokens = self.tokenizer(fields) + self.positions

        # encode visible tokens only
        src = einops.rearrange(tokens[visible], '(b n) d -> b n d', b = tokens.size(0), d = tokens.size(-1))
        for read in self.encoder:
            src = read(src)

        return src

    def predict(self, 
                src: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                ) -> torch.FloatTensor:
        tgt = torch.masked_scatter(
            input = self.mask_token.type_as(src), # pad with mask tokens
            mask = visible[..., None], # if visible is True
            source = src # copy src elements over
            )
        # add position codes
        tgt = tgt + self.positions
        
        # predict masked locations
        tgt = self.predictor(tgt)

        return tgt
    