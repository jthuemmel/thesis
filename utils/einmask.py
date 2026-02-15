import einops
import torch

from einops.layers.torch import EinMix, Rearrange

from utils.components import *
from utils.config import *
from utils.random_fields import RandomField

    
class LatentTransformer(torch.nn.Module):
    def __init__(self, network: NetworkConfig):
        super().__init__()
        # learnable latents
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)

        # latent transformer
        self.encoder = torch.nn.ModuleList([
            TransformerBlock(network.dim)
            for _ in range(default(network.num_read_blocks, 1))
        ])

        self.processor = torch.nn.ModuleList([
            TransformerBlock(network.dim, drop_path=network.drop_path)
            for _ in range(default(network.num_compute_blocks, 1))
        ])
        
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(network.dim)
            for n in range(default(network.num_write_blocks, 1))
        ])

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None):
        kv = default(kv, q)
        latents = self.latents.weight.expand(q.size(0), -1, -1)
        # map src to latents
        for read in self.encoder:
            latents = read(q = latents, kv = torch.cat([kv, latents], dim = 1))

        # process latents
        for process in self.processor:
            latents = process(q = latents)

        # map latents to tgt
        for write in self.decoder:
            q = write(q = q, kv = latents)

        return q
    
class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # I/O
        self.to_tokens = EinMix(
            pattern=f"b {world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'v {world.patch_pattern} d', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields =EinMix(
            pattern=f"b {world.flat_token_pattern} d -> b {world.field_pattern}", 
            weight_shape=f'v d {world.patch_pattern}', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes 
            )
        
        # Noise
        self.noise_generator = RandomField(network.dim, world)

        # learnable tokens
        self.mask_embedding = torch.nn.Embedding(1, network.dim)
        self.position_embedding = ContinuousPositionalEmbedding(
            dim_per_coord=network.dim_coords, 
            wavelengths=[(1, 2 * k) for k in world.token_shape],
            model_dim=network.dim
        )

        # pre-computed coordinates
        self.register_buffer('indices', torch.arange(world.num_tokens))
        self.register_buffer("coordinates", torch.stack(
            torch.unravel_index(indices = self.indices, shape = world.token_shape), 
            dim = -1)
            )

        # transformer
        if network.backbone == 'natten':
            self.transformer = torch.nn.Sequential(*[
            NattenBlock(network.dim, drop_path= network.drop_path, kernel_size=world.num_tokens)
            for _ in range(network.num_layers)
            ])
        elif network.backbone == 'perceiver':
            self.transformer = LatentTransformer(network)
        else:
            raise NotImplementedError
        
        # Weight initialization
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
                torch.nn.init.trunc_normal_(m.bias, std = 0.02)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        B = fields.size(0)
        E = default(members, 1)

        # expand to ensemble form
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        visible = einops.repeat(visible, 'b ... -> (b e) ... d', d = self.network.dim, e = E, b = B)
        coo = einops.repeat(self.coordinates, '... -> (b e) ...', b = B, e = E)

        # embed full fields as tokens
        tokens = self.to_tokens(fields)

        # apply mask
        tokens = torch.where(visible, tokens, self.mask_embedding.weight)

        # create random field
        noise = self.noise_generator(shape = (B * E,), rng = rng).to(tokens.dtype)

        # add noise and positions
        tokens = tokens + noise + self.position_embedding(coo)      
        
        # apply Natten-transformer
        tokens = self.transformer(tokens)
        
        # map all tokens back to fields
        fields = self.to_fields(tokens)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... -> b ... e", e = E, b = B)
        return fields