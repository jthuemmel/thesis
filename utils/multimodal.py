import torch
from torch.nn import Embedding, Module, Linear
from einops import rearrange, repeat
from dataclasses import dataclass
from utils.masked_models import MTM
from utils.networks import ModalDecoder, ModalEncoder
from utils.components import Attention, GatedFFN, ConditionalLayerNorm

class ModalMTM(Module):
    def __init__(self, modal_cfg: dataclass, mtm_cfg: dataclass):
        super().__init__()
        self.encoder = ModalEncoder(modal_cfg)
        self.decoder = ModalDecoder(modal_cfg)
        self.processor = MTM(mtm_cfg)
        # Init
        self.apply(self.base_init)
        self.apply(self.zero_init)

    @staticmethod
    def base_init(m):
        if isinstance(m, Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        if isinstance(m, Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)

    @staticmethod
    def zero_init(m):
        if isinstance(m, Attention):
            torch.nn.init.zeros_(m.to_out.weight)
        if isinstance(m, GatedFFN):
            torch.nn.init.zeros_(m.to_out.weight)
        if isinstance(m, ConditionalLayerNorm):
            torch.nn.init.zeros_(m.linear.weight)

    def apply_noise(self, *args, ctx=None):
        if ctx is None:
            return args, ctx
        ctx = self.proj_ctx(ctx)
        K = ctx.size(0) // args[0].size(0)
        args = tuple(repeat(x, "b ... -> (b k) ...", k=K) for x in args)
        return args, ctx

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                ):
        #encoder
        src = self.encoder(src, var_src, ctx = None)
        # masked token model
        src, tgt = self.processor(src, pos_src, pos_tgt, ctx = None)
        # decoder
        tgt = self.decoder(tgt, var_tgt, ctx = None)
        return tgt
    
class ModalTailMTM(ModalMTM):
    def __init__(self, modal_cfg, mtm_cfg):
        super().__init__(modal_cfg, mtm_cfg)
        self.proj_ctx = Linear(mtm_cfg.dim_noise, mtm_cfg.dim_noise)
        torch.nn.init.trunc_normal_(self.proj_ctx.weight, std = 0.02)

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                noise: torch.Tensor
                ):
        #encoder
        src = self.encoder(src, var_src, ctx = None)
        # masked token model
        src, tgt = self.processor(src, pos_src, pos_tgt, ctx = None)
        # noise perturbation
        K = noise.size(0) // src.size(0) if noise is not None else 1
        args, ctx = self.apply_noise(tgt, var_tgt, ctx = noise)
        tgt, var_tgt = args
        # decoder
        tgt = self.decoder(tgt, var_tgt, ctx = ctx)
        #split ensemble dim
        tgt = rearrange(tgt, "(b k) ... -> b ... k", k = K)
        return tgt
    
class ModalFuncMTM(ModalMTM):
    def __init__(self, modal_cfg, mtm_cfg):
        super().__init__(modal_cfg, mtm_cfg)
        self.proj_ctx = Linear(mtm_cfg.dim_noise, mtm_cfg.dim_noise)
        torch.nn.init.trunc_normal_(self.proj_ctx.weight, std = 0.02)

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                noise: torch.Tensor
                ):
        # noise perturbation
        K = noise.size(0) // src.size(0) if noise is not None else 1
        args = src, pos_src, pos_tgt, var_src, var_tgt
        args, ctx = self.apply_noise(*args, ctx = noise)
        src, pos_src, pos_tgt, var_src, var_tgt = args
        #encoder
        src = self.encoder(src, var_src, ctx = ctx)
        # masked token model
        src, tgt = self.processor(src, pos_src, pos_tgt, ctx = ctx)
        # decoder
        tgt = self.decoder(tgt, var_tgt, ctx = ctx)
        #split ensemble dim
        tgt = rearrange(tgt, "(b k) ... -> b ... k", k = K)
        return tgt