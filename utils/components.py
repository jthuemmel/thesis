import torch

from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.functional import scaled_dot_product_attention, silu, linear
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

from typing import Optional, Tuple, List

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_weight_std(weight: torch.Tensor, dim: int = -1):
    return 1 / weight.size(dim)**0.5

class ContinuousPositionalEmbedding(torch.nn.Module):
    def __init__(self, dim_per_coord: int, wavelengths: List[Tuple[int, int]] = [(1., 256)], model_dim: Optional[int] = None):
        super().__init__()
        d_half = dim_per_coord // 2

        # Precompute per-coordinate frequency factors
        freqs = torch.stack([
            torch.exp((2 * torch.pi / lmin).log() - torch.linspace(0, 1, d_half) * (lmax.log() - lmin.log()))
            for lmin, lmax in torch.as_tensor(wavelengths)
            ])

        # register buffer and optional projection
        self.register_buffer("freqs", freqs)  # shape (n_coords, dim_per_coord // 2)
        self.embedding_dim = len(wavelengths) * (d_half * 2) #make sure the embedding dim is correct even if d_half rounds
        self.proj = torch.nn.Linear(self.embedding_dim, model_dim) if exists(model_dim) else torch.nn.Identity()

    def forward(self, coordinates: torch.Tensor):
        with torch.amp.autocast(enabled = False, device_type = coordinates.device.type): # overflows fp16 if not careful
            angles = torch.einsum("...i, i d -> ...i d", coordinates, self.freqs)
            emb = torch.stack((angles.sin(), angles.cos()), dim=-1)
        emb = rearrange(emb, "... n d two -> ... (n d two)")
        return self.proj(emb)
    
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

class SwiGLU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        x1, x2 = x.chunk(2, dim=-1) 
        return silu(x1) * x2

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
    def __init__(self, dim: int, dim_heads: int = 64, bias: bool = False, qk_norm: bool = True):
        super().__init__()
        self.split_heads = Rearrange('... n (h d) -> ... h n d', d = dim_heads)
        self.merge_heads = Rearrange('... h n d -> ... n (h d)')
        self.to_q = torch.nn.Linear(dim, dim, bias = bias)
        self.to_k = torch.nn.Linear(dim, dim, bias = bias)
        self.to_v = torch.nn.Linear(dim, dim, bias = bias)
        self.norm_q = torch.nn.LayerNorm(dim_heads, bias=False) if qk_norm else torch.nn.Identity()
        self.norm_k = torch.nn.LayerNorm(dim_heads, bias=False) if qk_norm else torch.nn.Identity()
        self.to_out = torch.nn.Linear(dim, dim, bias = bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        q, k, v = map(lambda fn, x: fn(x), [self.to_q, self.to_k, self.to_v], [q, k, v])
        q, k, v = map(self.split_heads, [q, k, v]) # split heads and merge any leading dimensions into the batch
        q = self.norm_q(q)
        k = self.norm_k(k)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            attn = scaled_dot_product_attention(q, k, v, **attn_kwargs)
        out = self.merge_heads(attn)
        out = self.to_out(out)
        return out

class ConditionalLayerNorm(torch.nn.Module):
    def __init__(self, dim: int, dim_ctx: int = None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine = False)
        self.weight = torch.nn.Parameter(torch.zeros(dim * 2, dim_ctx)) if exists(dim_ctx) else None
        self.bias = torch.nn.Parameter(torch.zeros(dim * 2))

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if exists(ctx) and exists(self.weight):
            scale, shift = linear(ctx, self.weight, self.bias).chunk(2, dim = -1)
        else:
            scale, shift = self.bias.chunk(2, dim = -1)
        x = (1. + scale) * self.norm(x) + shift
        return x
    
class SelfConditioning(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = GatedFFN(dim)
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine = False)
        self.scale = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, initial: torch.FloatTensor, previous: torch.FloatTensor = None):
        previous = default(previous, torch.zeros_like(initial))
        x = self.scale * self.norm(previous + self.ffn(previous)) + initial
        return x

class TransformerBlock(torch.nn.Module):
    def __init__(self, 
                 dim: int, 
                 dim_heads: int = 64, 
                 dim_ctx: Optional[int] = None, 
                 drop_path: float = 0.,
                 has_skip: bool = True, 
                 qk_norm: bool = True, **kwargs):
        super().__init__()
        self.att_norm = ConditionalLayerNorm(dim, dim_ctx)
        self.ffn_norm = ConditionalLayerNorm(dim, dim_ctx)
        self.drop_path = DropPath(drop_path)
        self.att = Attention(dim, dim_heads, qk_norm = qk_norm)
        self.ffn = GatedFFN(dim)
        self.has_skip = has_skip

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None, ctx: Optional[torch.Tensor] = None, **attn_kwargs):
        skip = q if self.has_skip else 0.
        q = self.att_norm(q, ctx)
        kv = self.att_norm(kv, ctx) if kv is not None else q
        q = skip + self.drop_path(self.att(q, kv, kv, **attn_kwargs))
        q = q + self.drop_path(self.ffn(self.ffn_norm(q, ctx)))
        return q

class InterfaceBlock(torch.nn.Module):
    def __init__(self, dim: int, num_blocks: int = 1, dim_heads: int = 64, dim_ctx: Optional[int] = None, 
                 write_has_skip: bool = True, 
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.read = TransformerBlock(dim, dim_heads=dim_heads, dim_ctx=dim_ctx)
        self.compute = torch.nn.ModuleList([TransformerBlock(dim, dim_heads=dim_heads, dim_ctx=dim_ctx) for _ in range(num_blocks)])
        self.write = TransformerBlock(dim, dim_heads=dim_heads, dim_ctx=dim_ctx, has_skip= write_has_skip)

    def forward(self, 
                x: torch.Tensor,
                z: torch.Tensor, 
                query: Optional[torch.Tensor] = None, 
                ctx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q = query if exists(query) else x
        z = checkpoint(self.read, z, x, ctx, use_reentrant=False) if self.use_checkpoint else self.read(z, x, ctx)
        for block in self.compute:
            z = block(z, ctx = ctx)
        query = checkpoint(self.write, q, z, ctx, use_reentrant=False) if self.use_checkpoint else self.write(q, z, ctx)
        return query, z