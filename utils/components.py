from einops import rearrange
from einops.layers.torch import Rearrange

from torch import Tensor
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import normalize, scaled_dot_product_attention, silu

from typing import Optional, Tuple

class TransformerBlock(Module):
    def __init__(self, dim: int, dim_heads: int = 64, dim_ctx: int = 1, **kwargs):
        super().__init__()
        self.att_norm = ConditionalLayerNorm(dim, dim_ctx)
        self.ffn_norm = ConditionalLayerNorm(dim, dim_ctx)
        self.att = Attention(dim, dim_heads)
        self.ffn = GatedFFN(dim)

    def forward(self, q: Tensor, kv: Optional[Tensor] = None, ctx: Optional[Tensor] = None, **attn_kwargs):
        """
        q: query tensor of shape (B, N, D)
        kv: (optional) key-value tensor of shape (B, M, D)
        ctx: (optional) conditioning tensor of shape (*, D) where * must broadcast with q
        returns: tensor of shape (B, N, D)
        """
        skip = q
        q = self.att_norm(q, ctx)
        kv = kv if kv is not None else q
        q = skip + self.att(q, kv, kv, **attn_kwargs)
        q = q + self.ffn(self.ffn_norm(q, ctx))
        return q

class ConditionalLayerNorm(Module):
    def __init__(self, dim: int, dim_ctx: int):
        super().__init__()
        self.dim_ctx = dim_ctx
        self.linear = Linear(dim_ctx, dim * 2, bias=True)

    def forward(self, x: Tensor, ctx: Optional[Tensor] = None) -> Tensor:
        if ctx is None: # default to zero means using learned bias only
            ctx = x.new_zeros(self.dim_ctx)
        scale, shift = self.linear(ctx).chunk(2, dim = -1)
        x = (1. + scale) * normalize(x, dim=-1) + shift
        return x
    
class GatedFFN(Module):
    def __init__(self, dim: int, expansion_factor: int = 2, bias: bool = False):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.to_hidden = Linear(dim, 2 * hidden_dim, bias = bias)
        self.to_out = Linear(hidden_dim, dim, bias = bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.to_hidden(x)
        x, g = x.chunk(2, dim = -1)
        x = x * silu(g)
        x = self.to_out(x)
        return x
    
class Attention(Module):
    def __init__(self, dim: int, dim_heads: int = 64, bias: bool = True):
        super().__init__()
        self.split_heads = Rearrange('... n (h d) -> (...) h n d', d = dim_heads)
        self.to_q = Linear(dim, dim, bias = bias)
        self.to_k = Linear(dim, dim, bias = bias)
        self.to_v = Linear(dim, dim, bias = bias)
        self.to_out = Linear(dim, dim, bias = bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, **attn_kwargs) -> Tensor:
        B = q.size(0) #remember the original batch size
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(self.split_heads, (q, k, v)) # split heads and merge any leading dimensions into the batch
        attn = scaled_dot_product_attention(q, k, v, **attn_kwargs)
        out = rearrange(attn, '(b g) h n d ->  b g n (h d)', b = B).squeeze(1) # if there was no leading dimension, we simply squeeze the empty dimension
        out = self.to_out(out)
        return out

class Interface(Module):
    def __init__(self, dim: int, num_blocks: int = 1, dim_heads: int = 64, dim_ctx: int = 1):
        """
        dim: block dimension
        num_blocks: number of latent transformer blocks
        dim_heads: dimension of heads in attention
        """
        super().__init__()
        self.read = TransformerBlock(dim, dim_heads=dim_heads, dim_ctx=dim_ctx)
        self.compute = ModuleList([TransformerBlock(dim, dim_heads=dim_heads, dim_ctx=dim_ctx) for _ in range(num_blocks)])
        self.write = TransformerBlock(dim, dim_heads=dim_heads, dim_ctx=dim_ctx)

    def forward(self, x: Tensor, z: Tensor, ctx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        x: tensor of shape (B, *, N, D)
        z: tensor of shape (B, *, M, D)
        ctx: (optional) conditioning tensor of shape (B, D)
        returns tuple of (x, z) 
        """
        z = self.read(q = z, kv = x, ctx = ctx)
        for block in self.compute:
            z = block(z, ctx = ctx)
        x = self.write(q = x, kv = z, ctx = ctx)
        return x, z