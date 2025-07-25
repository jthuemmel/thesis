from torch.nn import Embedding, Linear, Module, ModuleList
from torch import cat, split, Tensor, LongTensor, einsum
from dataclasses import dataclass
from einops import repeat, rearrange
from utils.components import TransformerBlock, ConditionalLayerNorm, GatedFFN, Interface
from typing import Tuple, Optional

class ViT(Module):
    def __init__(self, cfg: dataclass):
        """
        Args:
            cfg (ViTConfig): Configuration object containing model parameters should contain:
                dim_in: Input dimension.
                dim_out: Output dimension.
                dim: Dimension of the embeddings.
                num_layers: Number of transformer blocks.
                num_tokens: Number of tokens for positional embeddings.
                num_cls: Number of class tokens.
        """
        super().__init__()
        self.positions = Embedding(cfg.num_tokens, cfg.dim)
        self.cls_token = Embedding(cfg.num_cls, cfg.dim)
        self.norm_in = ConditionalLayerNorm(cfg.dim, cfg.dim_noise)
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)
        self.blocks = ModuleList([TransformerBlock(cfg.dim, dim_ctx=cfg.dim_noise) for _ in range(cfg.num_layers)])

    def forward(self, x: Tensor, pos: LongTensor, ctx: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor. Shape [B, N, D_in].
            pos (torch.Tensor): Positional indices. Shape [B, N].
            ctx (torch.Tensor): Context tensor. Shape [B, N, D] or [B, D].
        Returns:
            - x_hat (torch.Tensor): Predicted tensor. Shape [B, N, D_out].
        """
        B, N = x.size(0), x.size(1)
        x = self.proj_in(x) 
        x = self.norm_in(x, ctx) + self.positions(pos)
        cls = repeat(self.cls_token.weight, "n d -> b n d", b = B)
        x = cat([cls, x], dim = 1)
        for block in self.blocks:
            x = block(x, ctx = ctx)
        cls, x = split(x, [cls.size(1), N], dim = 1)
        x = self.proj_out(x)
        return x

class InterfaceNetwork(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        # In/Out
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)

        # Embeddings
        self.latents = Embedding(cfg.num_latents, cfg.dim)
        self.positions = Embedding(cfg.num_tokens, cfg.dim)

        # Interfaces
        self.network = ModuleList([
            Interface(cfg.dim, cfg.num_compute_blocks, dim_ctx= cfg.dim_noise, dim_heads= cfg.dim_heads) for _ in range(cfg.num_layers)
        ])

    def forward(self, x: Tensor, pos: LongTensor, ctx: Optional[Tensor] = None):  
        # initialize interface
        x = self.proj_in(x) + self.positions(pos)
        # initialize latents
        z = repeat(self.latents.weight, "z d -> b z d", b = x.size(0))      
        # update
        for block in self.network:
            x, z = block(x, z, ctx = ctx)
        # project out
        x = self.proj_out(x)
        return x
    
class ModalEncoder(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.in_projection = Embedding(cfg.num_features, cfg.dim * cfg.dim_in)
        self.feature_bias = Embedding(cfg.num_features, cfg.dim)
        self.queries = Embedding(1, cfg.dim)       
        self.kv_norm = ConditionalLayerNorm(cfg.dim, cfg.dim_noise)
        self.cross_attn = TransformerBlock(cfg.dim, dim_ctx=cfg.dim_noise)

    def forward(self, x: Tensor, idx: LongTensor, ctx: Optional[Tensor] = None):
        # ensure correct shapes
        B, N, _, I = x.size()
        # get dynamic weights 
        w = self.in_projection(idx)
        w = rearrange(w, '... f (d i) -> ... f d i', i = I)
        b = self.feature_bias(idx)
        b = rearrange(b, "... f d -> ... () f d")
        # linear projection
        kv = einsum('b n f i, ... f d i -> b n f d', x, w)
        # expand query and context vectors
        q = repeat(self.queries.weight, 'q d -> b n q d', b = B, n = N)
        ctx = rearrange(ctx, 'b 1 d -> b 1 1 d') if ctx is not None else None
        # normalize and add feature-bias
        kv = self.kv_norm(kv, ctx) + b
        # cross attend
        q = self.cross_attn(q = q, kv = kv, ctx = ctx).squeeze(2)
        return q
    
class ModalDecoder(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.norm = ConditionalLayerNorm(cfg.dim, cfg.dim_noise)
        self.ffn = GatedFFN(cfg.dim)
        self.out_projection = Embedding(cfg.num_features, cfg.dim * cfg.dim_out)

    def forward(self, x: Tensor, idx: LongTensor, ctx: Optional[Tensor] = None):
        _, _, D = x.size()
        x = self.ffn(self.norm(x, ctx))
        w = self.out_projection(idx)
        w = rearrange(w, "... f (d o) -> ... f d o", d = D)
        out = einsum("b n d, ... f d o -> b n f o", x, w)
        return out