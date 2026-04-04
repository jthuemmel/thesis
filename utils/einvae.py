import einops
import torch
import math
from einops.layers.torch import EinMix, Rearrange
from utils.config import *
from utils.components import *

class EinVAE(torch.nn.Module):
    def __init__(self, latent_dim: int, world: WorldConfig):
        super().__init__()
        self.src_encoder = torch.nn.Sequential(
                    EinMix(pattern = f'b {world.field_pattern} -> b ({world.token_pattern}) dp', 
                        weight_shape= f'v {world.patch_pattern} dp',
                        **world.patch_sizes, **world.token_sizes, dp = world.dim_tokens),
                    torch.nn.SiLU(),
                    EinMix(f'b ({world.token_pattern}) dp -> b ({world.token_pattern}) d', 
                            weight_shape = f'v d dp',
                            **world.token_sizes, dp = world.dim_tokens, d = latent_dim),
                )
        
        self.bottleneck = EinMix(f'b ({world.token_pattern}) d1 -> two b ({world.token_pattern}) d2', 
                                weight_shape = 'two v d2 d1', **world.token_sizes,
                                two = 2, d1 = latent_dim, d2 = latent_dim)
        
        self.tgt_decoder = torch.nn.Sequential(
                    EinMix(f'b ({world.token_pattern}) d -> b ({world.token_pattern}) dp', 
                            weight_shape = f'v dp d',
                            **world.token_sizes, d = latent_dim, dp = world.dim_tokens),
                    torch.nn.SiLU(),
                    EinMix(f'b ({world.token_pattern}) dp -> b {world.field_pattern}', 
                            weight_shape = f'v {world.patch_pattern} dp',
                            **world.token_sizes, **world.patch_sizes, dp = world.dim_tokens),
                )
        
        torch.nn.init.trunc_normal_(self.src_encoder[0].weight, std = 1 / math.sqrt(world.dim_tokens))
        torch.nn.init.trunc_normal_(self.src_encoder[2].weight, std = 1 / math.sqrt(world.dim_tokens))
        torch.nn.init.trunc_normal_(self.bottleneck.weight, std = 1 / math.sqrt(latent_dim))
        torch.nn.init.trunc_normal_(self.tgt_decoder[0].weight, std = 1 / math.sqrt(latent_dim))
        torch.nn.init.trunc_normal_(self.tgt_decoder[2].weight, std = 1 / math.sqrt(world.dim_tokens))

    @torch.compile()
    def forward(self, fields: torch.FloatTensor, members: Optional[int] = 1, rng: Optional[torch.Generator] = None):
        x = self.src_encoder(fields)
        mu, sigma = self.bottleneck(x)
        sigma = torch.nn.functional.softplus(sigma).clamp(1e-5)
        kl = -0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2)).sum(dim=-1).mean()
        mu = einops.repeat(mu, 'b ... -> (b e) ...', e = members)
        sigma = einops.repeat(sigma, 'b ... -> (b e) ...', e = members)
        x_hat = mu + torch.randn_like(sigma, generator = rng) * sigma
        x_hat = self.tgt_decoder(x_hat)
        x_hat = einops.rearrange(x_hat, '(b e) ... -> b ... e', e = members)
        return x_hat, kl
