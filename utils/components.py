import torch
import math

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.functional import scaled_dot_product_attention

from typing import Optional, Tuple, List


def get_weight_std(weight: torch.Tensor):
    return 1 / weight.size(-1)**0.5

class ContinuousPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_per_coord: int, wavelengths: List[Tuple[int, int]] = [(1., 256)], model_dim: Optional[int] = None):
        super().__init__()
        d_half = dim_per_coord // 2

        # Precompute per-coordinate frequency factors
        freqs = torch.stack([
            torch.exp(math.log(2 * math.pi) - math.log(lmin) - torch.linspace(0, 1, d_half) * (math.log(lmax) - math.log(lmin)))
            for lmin, lmax in wavelengths
            ])

        # register buffer and optional projection
        self.register_buffer("freqs", freqs)  # shape (n_coords, dim_per_coord // 2)
        self.embedding_dim = len(wavelengths) * (d_half * 2) #make sure the embedding dim is correct even if d_half rounds
        self.proj = torch.nn.Identity() if model_dim is None else torch.nn.Linear(self.embedding_dim, model_dim)

    def forward(self, positions: torch.Tensor):
        angles = torch.einsum("...i, i d -> ...i d", positions, self.freqs)
        emb = torch.stack((angles.sin(), angles.cos()), dim=-1)
        emb = rearrange(emb, "... n d two -> ... (n d two)")
        return self.proj(emb)

class SwiGLU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.silu(gate)

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_heads: int = 64, dim_ctx: Optional[int] = None, has_skip: bool = True, **kwargs):
        super().__init__()
        self.att_norm = ConditionalLayerNorm(dim, dim_ctx)
        self.ffn_norm = ConditionalLayerNorm(dim, dim_ctx)
        self.att = Attention(dim, dim_heads)
        self.ffn = GatedFFN(dim)
        self.has_skip = has_skip

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None, ctx: Optional[torch.Tensor] = None, **attn_kwargs):
        skip = q if self.has_skip else 0.
        q = self.att_norm(q, ctx)
        kv = kv if kv is not None else q
        q = skip + self.att(q, kv, kv, **attn_kwargs)
        q = q + self.ffn(self.ffn_norm(q, ctx))
        return q

class GatedFFN(torch.nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2, bias: bool = False):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.to_hidden = torch.nn.Linear(dim, 2 * hidden_dim, bias = bias)
        self.swiglu = SwiGLU()
        self.to_out = torch.nn.Linear(hidden_dim, dim, bias = bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_hidden(x)
        x = self.swiglu(x)
        x = self.to_out(x)
        return x
    
class Attention(torch.nn.Module):
    def __init__(self, dim: int, dim_heads: int = 64, bias: bool = True):
        super().__init__()
        self.split_heads = Rearrange('... n (h d) -> (...) h n d', d = dim_heads)
        self.to_q = torch.nn.Linear(dim, dim, bias = bias)
        self.to_k = torch.nn.Linear(dim, dim, bias = bias)
        self.to_v = torch.nn.Linear(dim, dim, bias = bias)
        self.to_out = torch.nn.Linear(dim, dim, bias = bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        B = q.size(0) #remember the original batch size
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(self.split_heads, (q, k, v)) # split heads and merge any leading dimensions into the batch
        attn = scaled_dot_product_attention(q, k, v, **attn_kwargs)
        out = rearrange(attn, '(b g) h n d ->  b g n (h d)', b = B).squeeze(1) # if there was no leading dimension, we simply squeeze the empty dimension
        out = self.to_out(out)
        return out

class ConditionalLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, dim_ctx: Optional[int] = None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine= dim_ctx is None)
        if dim_ctx is None:
            self.linear = None
        else:
            self.linear = torch.nn.Linear(dim_ctx, dim * 2, bias=True)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.linear is None: 
            return self.norm(x)
        scale, shift = self.linear(ctx).chunk(2, dim = -1)
        x = (1. + scale) * self.norm(x) + shift
        return x
    
class SegmentLinear(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_segments: int, bias: bool = True):
        super().__init__()
        self.dim_out = dim_out
        self.weights = torch.nn.Embedding(num_segments, dim_in * dim_out)
        self.bias = torch.nn.Embedding(num_segments, dim_out) if bias else None

    def forward(self, x: torch.Tensor, coords: torch.LongTensor):
        # pre-allocate output tensor
        out = x.new_empty(*x.shape[:-1], self.dim_out, dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else x.dtype)
        # flat views
        xf = rearrange(x, '... d -> (...) d')
        of = rearrange(out, '... d -> (...) d')
        cf = rearrange(coords, '... -> (...)')
        # determine which segments are present
        segments = cf.unique(sorted = False)
        # weights/bias for each segment
        W = rearrange(self.weights(segments), 's (o i) -> s o i', o = self.dim_out)
        b = None if self.bias is None else self.bias(segments)
        # apply linear to segments
        for i, s in enumerate(segments):
            idx = (cf == s).nonzero().squeeze(1) # find all elements of the segment
            lin = torch.nn.functional.linear(xf.index_select(0, idx), W[i], None if b is None else b[i]) # apply corresponding linear layer
            of.index_copy_(0, idx, lin) # write output at index locations
        return out
