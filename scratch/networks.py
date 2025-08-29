from torch.nn import Embedding, Linear, Module, ModuleList, init
from torch import cat, split, Tensor, LongTensor, zeros_like
from dataclasses import dataclass
from einops import repeat, rearrange, reduce
from utils.components import *
from typing import Tuple, Optional

####
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

####
class InterfaceNetwork(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        # In/Out
        self.proj_in = Linear(cfg.dim_in, cfg.dim)
        self.norm_in = ConditionalLayerNorm(cfg.dim, cfg.dim_noise)
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
        x = self.proj_in(x) 
        x = self.norm_in(x, ctx = ctx) + self.positions(pos)
        # initialize latents
        z = repeat(self.latents.weight, "z d -> b z d", b = x.size(0))      
        # update
        for block in self.network:
            x, z = block(x, z, ctx = ctx)
        # project out
        x = self.proj_out(x)
        return x
    
####
ARCHITECTURES = {
    "interface": InterfaceNetwork,
    "vit": ViT,
}

####
class MTM(Module):
    def __init__(self, decoder_cfg: dataclass, encoder_cfg: Optional[dataclass] = None):
        super().__init__()
        self.queries = Embedding(1, decoder_cfg.dim)
        self.encoder = ARCHITECTURES[encoder_cfg.architecture](encoder_cfg) if encoder_cfg else None
        self.decoder = ARCHITECTURES[decoder_cfg.architecture](decoder_cfg)

        self.apply(self.base_init)
        self.apply(self.zero_init)

    @staticmethod
    def base_init(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std = 0.02)
        if isinstance(m, Embedding):
            init.trunc_normal_(m.weight, std = 0.02)

    @staticmethod
    def zero_init(m):
        if isinstance(m, Attention):
            init.zeros_(m.to_out.weight)
        if isinstance(m, GatedFFN):
            init.zeros_(m.to_out.weight)
        if isinstance(m, ConditionalLayerNorm):
            init.zeros_(m.linear.weight)

    def forward(self, 
                src: Tensor, 
                pos_src: LongTensor, 
                pos_tgt: LongTensor, 
                ctx: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor]:
        if self.encoder is not None:
            src = self.encoder(src, pos_src, ctx = ctx)
        q = self.queries(zeros_like(pos_tgt))
        x_hat = cat([src, q], dim=1)
        pos = cat([pos_src, pos_tgt], dim=1)
        x_hat = self.decoder(x_hat, pos, ctx = ctx)
        src, tgt = x_hat.split([pos_src.size(1), pos_tgt.size(1)], dim=1)
        return src, tgt