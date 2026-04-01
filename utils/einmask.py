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
        self.tokenizer = EinMix(
            pattern = f'b {world.field_pattern} -> b ({world.token_pattern}) di', 
            weight_shape= f'v {world.patch_pattern} di', 
            **world.patch_sizes, **world.token_sizes, di = network.dim_in
            )
        
        self.positions = torch.nn.Parameter(torch.zeros(1, world.num_tokens, network.dim_in))
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, network.dim_in))
    
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim_in, dim_ctx= network.dim_ctx) for _ in range(default(network.num_read_blocks, 1))
              ])
        
        # Predictor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(network.dim_in, network.dim_out, bias = False),
            *[TransformerBlock(dim= network.dim_out) for _ in range(default(network.num_write_blocks, 1))],
            FieldDecoder(network, world)
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
        elif isinstance(m, AdaptiveLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.trunc_normal_(m.weight, std = 1e-6)
        # explicit parameters
        elif isinstance(m, EinMask):
            torch.nn.init.trunc_normal_(m.mask_token, std = 0.02)
            torch.nn.init.trunc_normal_(m.positions, std = 0.02)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                members: Optional[int] = None,
                rng: Optional[torch.Generator] = None,
                ) -> torch.FloatTensor:
        # tokenize
        tokens = self.tokenizer(fields) + self.positions

        # encode visible tokens only
        src = einops.rearrange(tokens[visible], '(b n) d -> b n d', b = tokens.size(0), d = tokens.size(-1))

        # ensemble expansion
        src = einops.repeat(src, 'b n d -> (b e) n d', e = default(members, 1))
        visible = einops.repeat(visible, 'b n -> (b e) n ()', e = default(members, 1))

        # functional noise
        noise = torch.randn([src.size(0), 1, src.size(-1)], device = src.device, dtype = src.dtype, generator = rng)

        # stochastic encoder
        for read in self.encoder:
            src = read(src, ctx = noise)

        # pad with mask tokens
        tgt = torch.masked_scatter(
            input = (self.mask_token + self.positions).type_as(src),
            mask = visible, # where visible is True
            source = src # copy src elements over
            )
        
        # predict masked locations
        tgt = self.predictor(tgt)
        return tgt
