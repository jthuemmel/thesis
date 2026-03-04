import einops
import torch

from einops.layers.torch import EinMix, Rearrange
from utils.config import *

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor.div(keep_prob)

class GatedFFN(torch.nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2, bias: bool = False):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.to_hidden = torch.nn.Linear(dim, 2 * hidden_dim, bias = bias)
        self.to_out = torch.nn.Linear(hidden_dim, dim, bias = bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_hidden(x)
        x1, x2 = x.chunk(2, dim=-1) 
        x = torch.nn.functional.silu(x1) * x2
        x = self.to_out(x)
        return x
    
class AdaptiveRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, dim_ctx: int = None):
        super().__init__()
        self.norm = torch.nn.RMSNorm(dim, elementwise_affine = False)
        self.weight = torch.nn.Parameter(torch.zeros(dim, dim_ctx)) if exists(dim_ctx) else None
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if exists(ctx) and exists(self.weight):
            scale = torch.nn.functional.linear(ctx, self.weight, self.bias)
        else:
            scale = self.bias
        x = (1. + scale) * self.norm(x)
        return x

class EinAttention(torch.nn.Module):
    def __init__(self, dim_q: int, dim_kv: Optional[int] = None, num_heads: Optional[int] = None, dim_heads: int = 64) -> None:
        super().__init__()
        num_heads = default(num_heads, max(dim_q // dim_heads, 1))
        dim_kv = default(dim_kv, dim_q)

        self.norm_qk = torch.nn.RMSNorm(dim_heads)
        self.to_q = EinMix(
            '... nq dq -> ... nh nq dh',
            weight_shape='dq nh dh',
            nh = num_heads, dh = dim_heads, dq = dim_q
        )
        self.to_kv = EinMix(
            '... nk dk -> kv ... nh nk dh',
            weight_shape= 'dk kv nh dh',
            kv = 2, dh = dim_heads, nh = num_heads, dk = dim_kv
        )
        self.to_out = EinMix(
            '... nh nq dh -> ... nq dq',
            weight_shape='nh dh dq',
            nh = num_heads, dh = dim_heads, dq = dim_q
        )

    def forward(self, q: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None):
        kv = default(kv, q)
        K, V = self.to_kv(kv)
        Q = self.to_q(q)
        A = torch.nn.functional.scaled_dot_product_attention(self.norm_qk(Q), self.norm_qk(K), V)
        return self.to_out(A)

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_kv: Optional[int] = None, dim_ctx: Optional[int] = None, num_heads: Optional[int] = None, drop_path: float = 0.) -> None:
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.norm_att = AdaptiveRMSNorm(dim, dim_ctx=dim_ctx)
        self.norm_ffn = AdaptiveRMSNorm(dim, dim_ctx=dim_ctx)
        self.att = EinAttention(dim, num_heads=num_heads, dim_kv= dim_kv)
        self.ffn = GatedFFN(dim=dim)

    def forward(self, x: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None,  ctx: Optional[torch.FloatTensor] = None):
        x = x + self.drop_path(self.att(self.norm_att(x, ctx), kv = kv))
        x = x + self.drop_path(self.ffn(self.norm_ffn(x, ctx)))
        return x
    
class FoldedProjection(torch.nn.Module):
    def __init__(self, dim: int, world: WorldConfig, offset: int = 2):
        super().__init__()
        
        self.project = EinMix(
            f'b ({world.token_pattern}) d -> g b ({world.flat_pattern})',
            weight_shape=f'g {world.patch_pattern} d',
            **world.token_sizes, **{k: s + offset if s > 1 else 1 for k, s in world.patch_sizes.items()}, d=dim, g=2
        )
        self.to_grid = Rearrange(
            f'b ({world.flat_pattern}) -> b {world.field_pattern}',
            **world.token_sizes, **world.patch_sizes
        )
        
        self._build_index(world, offset = offset)

    def _build_index(self, world: WorldConfig, offset: int = 2):
        ts = world.token_sizes
        strides = world.patch_sizes

        field_sizes = world.field_sizes
        stride_vals = {ax: strides[f'{ax}{ax}'] for ax in world.layout}
        us = {f'{ax}{ax}': strides[f'{ax}{ax}'] + offset if strides[f'{ax}{ax}'] > 1 else 1 for ax in world.layout}
        
        grid_strides = {}
        acc = 1
        for ax in reversed(world.layout):
            grid_strides[ax] = acc
            acc *= field_sizes[ax]

        idx = torch.zeros([*ts.values(), *us.values()], dtype=torch.long)

        for ax in world.layout:
            n0 = torch.arange(ts[ax]) * stride_vals[ax]
            dp = torch.arange(us[f'{ax}{ax}'])
            n0 = einops.repeat(n0, f'{ax} -> {world.token_pattern} {world.patch_pattern}', **ts, **us)
            dp = einops.repeat(dp, f'{ax}{ax} -> {world.token_pattern} {world.patch_pattern}', **ts, **us)
            idx += (n0 + dp).clamp(0, field_sizes[ax] - 1) * grid_strides[ax]

        self.register_buffer('idx', idx.flatten())

    def forward(self, out: torch.FloatTensor, tokens: torch.FloatTensor) -> torch.FloatTensor:
        g, x = self.project(tokens)
        x = x * g.sigmoid()
        out = out.scatter_reduce(1, self.idx.expand(out.size(0), -1), x, reduce='mean', include_self=False)
        return self.to_grid(out)
    
class Interface(torch.nn.Module):
    def __init__(self, dim_latents: int, dim_interface: int, num_compute_blocks: int = 0, drop_path: float = 0, dim_heads: int = 64):
        super().__init__()
        self.read = TransformerBlock(dim_latents, dim_kv = dim_interface)
        self.compute = torch.nn.ModuleList([TransformerBlock(dim_latents, drop_path=drop_path) for _ in range(num_compute_blocks)])
        self.write = TransformerBlock(dim_interface, dim_kv=dim_latents, num_heads=dim_latents// dim_heads)

    def forward(self, interface: torch.FloatTensor, latents: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        latents = self.read(latents, kv = interface)
        for block in self.compute:
             latents = block(latents)
        interface = self.write(interface, kv = latents)
        return interface, latents
    
class MaskedModel(torch.nn.Module):
      def __init__(self, network: NetworkConfig):
            super().__init__()
            self.latents = torch.nn.Embedding(network.num_latents, network.dim)
            self.guidance = Interface(network.dim, network.dim_in)
            self.transformer = torch.nn.ModuleList([
            Interface(network.dim, network.dim_in, num_compute_blocks=network.num_compute_blocks) for _ in range(network.num_layers)
            ])
            self.modulate = torch.nn.ModuleList([AdaptiveRMSNorm(network.dim_in, network.dim_in)for _ in range(network.num_layers)])
            
      @torch.compile()
      def forward(self, task: torch.FloatTensor, tokens: torch.FloatTensor, rng: torch.Generator = None) -> torch.FloatTensor:
            latents = einops.repeat(self.latents.weight, 'n d -> b n d', b=task.size(0))
            # create interface
            interface = task + tokens
            # create task representation
            logvar, latents = self.guidance(task, latents)
            # generate task conditioned noise
            ctx = torch.randn_like(logvar, generator = rng) * logvar.exp().sqrt()
            # apply interface network
            for block, mod in zip(self.transformer, self.modulate):
                  interface = mod(interface, ctx)
                  interface, latents = block(interface, latents)
            return interface

class EinMask(torch.nn.Module):
      def __init__(self, network: NetworkConfig, world: WorldConfig):
            super().__init__()
            self.network = network
            self.world = world
            
            # I/O
            self.to_tokens = EinMix(f'b {world.field_pattern} -> b ({world.token_pattern}) d',
                  weight_shape= f'v {world.patch_pattern} d',
                  **world.patch_sizes, **world.token_sizes, d = network.dim_in)
            
            self.to_fields = FoldedProjection(network.dim_in, world = world)
            
            # embeddings
            self.mask_code = torch.nn.Embedding(2, network.dim_in // 2)
            self.positions = torch.nn.Embedding(world.num_tokens, network.dim_in // 2)
            
            # latent transformer
            self.transformer = MaskedModel(network)

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
            # adaRMS
            elif isinstance(m, AdaptiveRMSNorm):
                  torch.nn.init.zeros_(m.bias)
                  if exists(m.weight):
                        torch.nn.init.trunc_normal_(m.weight, std = 0.02)
                        #torch.nn.init.trunc_normal_(m.weight, std = 1e-5)
            # einmix
            elif isinstance(m, EinMix):
                  torch.nn.init.trunc_normal_(m.weight, std = 0.02)
                  if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

      def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
            B = fields.size(0)
            E = default(members, 1)

            # expand to ensemble
            fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
            visible = einops.repeat(visible, 'b ... -> (b e) ...', e = E, b = B)
            positions = einops.repeat(self.positions.weight, '... -> (b e) ...', e = E, b = B)

            # create task representation
            task = torch.cat([positions, self.mask_code(visible.long())], dim = -1)

            # embed tokens and mask
            tokens = self.to_tokens(fields)
            masked_tokens = torch.where(visible[..., None], tokens, 0)
            
            # apply transformer
            predicted_tokens = self.transformer(task, masked_tokens, rng)
            
            # map all interface back to fields
            predicted_fields = self.to_fields(fields.flatten(1), predicted_tokens)
            
            # rearrange to ensemble last
            predicted_fields = einops.rearrange(predicted_fields, "(b e) ... -> b ... e", e = E, b = B)
            return predicted_fields