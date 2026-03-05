import einops
import torch

from einops.layers.torch import EinMix, Rearrange
from utils.config import *

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_weight_std(weight: torch.Tensor, dim: int = -1):
    return 1 / weight.size(dim)**0.5

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
    def __init__(self, dim: int, dim_kv: Optional[int] = None, dim_ctx: Optional[int] = None, num_heads: Optional[int] = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2 * dim, dim_ctx)) if exists(dim_ctx) else None
        self.bias = torch.nn.Parameter(torch.zeros(2 * dim))
        self.norm = torch.nn.RMSNorm(dim, elementwise_affine = False)
        self.att = EinAttention(dim, num_heads=num_heads, dim_kv= dim_kv)
        self.ffn = GatedFFN(dim=dim)

    def forward(self, x: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None, ctx: Optional[torch.FloatTensor] = None):
        if exists(ctx) and exists(self.weight):
            scale_attn, scale_ffn = torch.nn.functional.linear(ctx, self.weight, self.bias).chunk(2, -1)
        else:
            scale_attn, scale_ffn = self.bias.chunk(2, -1)
        skip = x
        x = self.norm(x) * (1. + scale_attn)
        x = skip + self.att(x, kv = kv) 
        skip = x
        x = self.norm(x) * (1. + scale_ffn)
        x = skip + self.ffn(x) 
        return x
