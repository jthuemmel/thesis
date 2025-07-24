from torch.nn import Embedding, Linear, Module, ModuleList
from torch import cat, split, Tensor, LongTensor
from dataclasses import dataclass
from einops import repeat
from utils.components import TransformerBlock, ConditionalLayerNorm

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
        self.norm_in = ConditionalLayerNorm(cfg.dim, cfg.dim_ctx)
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.proj_out = Linear(cfg.dim, cfg.dim_out)
        self.blocks = ModuleList([TransformerBlock(cfg.dim, dim_ctx=cfg.dim_ctx) for _ in range(cfg.num_layers)])

    def forward(self, x: Tensor, pos: LongTensor, ctx: Tensor = None) -> Tensor:
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