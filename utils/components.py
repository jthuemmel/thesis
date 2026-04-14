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
    
class EinAttention(torch.nn.Module):
    def __init__(self, dim_q: int, dim_kv: Optional[int] = None, num_heads: Optional[int] = None, dim_heads: int = 64) -> None:
        super().__init__()
        dim_kv = default(dim_kv, dim_q)
        num_heads = default(num_heads, max(dim_q // dim_heads, dim_kv // dim_heads, 1))
        

        self.norm_q = torch.nn.RMSNorm(dim_heads)
        self.norm_k = torch.nn.RMSNorm(dim_heads)
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
        KV = self.to_kv(kv)
        Q = self.to_q(q)
        A = torch.nn.functional.scaled_dot_product_attention(self.norm_q(Q), self.norm_k(KV[0]), KV[1])
        return self.to_out(A)

class AdaTransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_kv: Optional[int] = None, dim_ctx: Optional[int] = None, num_heads: Optional[int] = None) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, elementwise_affine = False)
        self.weight = torch.nn.Parameter(torch.zeros((dim * 6, dim_ctx))) if exists(dim_ctx) else None
        self.bias = torch.nn.Parameter(torch.zeros(dim * 6))
        self.att = EinAttention(dim, num_heads = num_heads, dim_kv = dim_kv)
        self.ffn = GatedFFN(dim = dim)

    def modulate(self, x: torch.FloatTensor, a: torch.FloatTensor, b: torch.FloatTensor):
        return (1. + a) * self.norm(x) + b

    def forward(self, x: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None, ctx: Optional[torch.FloatTensor] = None):
        affine = (torch.nn.functional.linear(ctx, self.weight, self.bias) if exists(self.weight) else self.bias).chunk(6, dim = -1)
        x = x + self.att(self.modulate(x, affine[0], affine[1]), kv = kv) * affine[2]
        x = x + self.ffn(self.modulate(x, affine[3], affine[4])) * affine[5]
        return x
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, dim_kv: Optional[int] = None, num_heads: Optional[int] = None, drop_path: float = 0.) -> None:
        super().__init__()
        self.attn_norm = torch.nn.RMSNorm(dim)
        self.ffn_norm = torch.nn.RMSNorm(dim)
        self.att = EinAttention(dim, num_heads = num_heads, dim_kv = dim_kv)
        self.ffn = GatedFFN(dim = dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.FloatTensor, kv: Optional[torch.FloatTensor] = None):
        x = x + self.drop_path(self.att(self.attn_norm(x), kv = kv))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x

class GaussianSmoothing3D(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        sigma: float = 0.7,
    ):
        super().__init__()
        # gaussian kernels
        kernel_size = 3 * [kernel_size] if isinstance(kernel_size, int) else kernel_size
        gs = []
        for k in kernel_size:
            coords = torch.arange(k, dtype=torch.float32) - k // 2
            g = torch.exp(-0.5 * (coords / sigma) ** 2)
            gs.append(g / g.sum())

        w = torch.einsum('i,j,k->ijk', *gs)

        # dw conv3d
        self.conv = torch.nn.Conv3d(channels, channels, kernel_size=kernel_size, 
                                    padding= 'same', groups= channels, bias = False, padding_mode = 'replicate')
        self.conv.weight = torch.nn.Parameter(w.expand_as(self.conv.weight).clone(), requires_grad = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)