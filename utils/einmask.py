import einops
import torch

from einops.layers.torch import *
from utils.config import *
from utils.components import *
from torch.nn.functional import softplus
        
class EinEncoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # embeddings
        self.masks = torch.nn.Embedding(2, network.dim_in)
        self.positions = torch.nn.Embedding(world.num_tokens, network.dim_in)
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)

        # encoder
        self.input_to_src = torch.nn.Sequential(torch.nn.Linear(network.dim_in, network.dim), torch.nn.RMSNorm(network.dim)) 
        self.encoder = torch.nn.ModuleList([
              TransformerBlock(dim= network.dim, num_heads= network.num_encoder_heads)
              for _ in range(default(network.num_read_blocks, 1))
              ])

    def forward(self, src: torch.FloatTensor, mask: torch.BoolTensor):
        latents = einops.repeat(self.latents.weight, '... -> b ...', b = src.size(0))

        # create src embedding
        src = src + self.masks(mask) + self.positions.weight
        src = self.input_to_src(src)

        # encoder src into latents
        for read in self.encoder:
             latents = read(latents, kv = torch.cat([src, latents], dim = 1))
        return latents

class EinDecoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # embeddings
        self.queries = torch.nn.Embedding(2, network.dim_out)
        self.positions = torch.nn.Embedding(world.num_tokens, network.dim_out)
        self.registers = torch.nn.Embedding(default(network.num_cls, 1), network.dim_out)

        # decoder
        self.reader = TransformerBlock(dim = network.dim_out * 2, dim_kv= network.dim, num_heads=network.num_decoder_heads)
        self.writer = torch.nn.ModuleList([
            TransformerBlock(dim=network.dim_out) 
            for _ in range(default(network.num_write_blocks, 1))
            ])
        
        # output mapping
        self.to_fields = FieldDecoder(network, world)

    def forward(self, 
                latents: torch.FloatTensor, 
                mask: torch.BoolTensor,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        B, E = mask.size(0), default(members, 1)

        # create queries from mask
        queries = self.queries(mask.long()) + self.positions.weight

        # add registers
        registers = einops.repeat(self.registers.weight, '... -> b ...', b = B)
        queries = torch.cat([registers, queries], dim = 1)
        
        # decode latents
        queries = self.reader(queries, kv = latents)

        # reparametrization
        mu, sigma = einops.repeat(tgt, 'b n (two d) -> two (b e) n d', e = E, two = 2)
        tgt = mu + torch.randn_like(sigma, generator = rng) * softplus(sigma)

        # update tgt representation
        for write in self.writer:
            tgt = write(tgt)

        # split registers and map back to fields
        _, tgt = tgt.split([registers.size(1), mask.size(1)], dim = 1)
        predicted_fields = self.to_fields(tgt)

        # rearrange to ensemble last
        predicted_fields = einops.rearrange(predicted_fields, "(b e) ... -> b ... e", b = B)

        return predicted_fields

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        self.network = network
        self.world = world

        # I/O
        self.tokenizer = EinMix(
                  f'b {world.field_pattern} -> b ({world.token_pattern}) di',
                  weight_shape= f'v {world.patch_pattern} di',
                  **world.patch_sizes, **world.token_sizes, di = network.dim_in
                  )
        
        # transformer
        self.encoder = EinEncoder(network, world)
        self.decoder = EinDecoder(network, world)

        # weight initialization
        self.apply(self.base_init)

        # maybe compile
        if network.kwargs.get('compile', True):
             self.encoder.compile()
             self.decoder.compile()

    def set_stage(self, stage: int) -> None:
        self.stage = stage
        if stage == 0:
            self.unfreeze_weights(self.tokenizer)
            self.unfreeze_weights(self.decoder)
            self.unfreeze_weights(self.encoder)
        elif stage == 1:
            self.freeze_weights(self.decoder)
            self.unfreeze_weights(self.encoder)
        elif stage == 2:
            self.freeze_weights(self.tokenizer)
            self.freeze_weights(self.encoder)
            self.unfreeze_weights(self.decoder)
        elif stage == 3:
            self.freeze_weights(self.tokenizer)
            self.freeze_weights(self.decoder)
            self.freeze_weights(self.encoder)
        else:
            raise ValueError("Invalid stage configuration.")
        
    @staticmethod
    def freeze_weights(m: torch.nn.Module) -> None:
        for param in m.parameters():
            param.requires_grad = False

    @staticmethod
    def unfreeze_weights(m: torch.nn.Module) -> None:
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
            mask: torch.BoolTensor, 
            members: Optional[int] = None, 
            rng: Optional[torch.Generator] = None,
            ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # deterministic encoder
        latents = self.encode(fields, mask)

        # stochastic decoder
        prediction = self.decode(latents, mask, members, rng)
        return prediction, latents
    
    def encode(self, fields: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        tokens = self.tokenizer(fields)
        src = torch.where(mask[..., None], tokens, 0)
        latents = self.encoder(src, mask)
        return latents
    
    def decode(self, 
                latents: torch.FloatTensor, 
                mask: torch.BoolTensor,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        return self.decoder(latents, mask, members, rng)
    
