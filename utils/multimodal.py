import torch
from torch.nn import Embedding, Module, Linear
from einops import rearrange, repeat, reduce
from dataclasses import dataclass
from utils.networks import MTM
from utils.components import Attention, GatedFFN, ConditionalLayerNorm
from typing import Optional

class ModalEncoder(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.in_projection = Embedding(cfg.num_features, cfg.dim * cfg.dim_in)
        self.feature_bias = Embedding(cfg.num_features, cfg.dim)

    def forward(self, x: torch.Tensor, idx: torch.LongTensor):
        _, _, _, I = x.size()
        w = rearrange(self.in_projection(idx), 
                      'b f (d i) -> b f d i', i = I)
        b = rearrange(self.feature_bias(idx), 
                      "b f d -> b () f d")
        kv = torch.einsum('b n f i, b f d i -> b n f d', x, w) + b
        q = reduce(kv, 'b n f d -> b n d', "sum")
        return q
    
class ModalDecoder(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.feature_bias = Embedding(cfg.num_features, cfg.dim)
        self.out_projection = Embedding(cfg.num_features, cfg.dim * cfg.dim_out)

    def forward(self, x: torch.Tensor, idx: torch.LongTensor):
        _, _, D = x.size()
        x = rearrange(x, 'b n d -> b n 1 d')
        w = rearrange(self.out_projection(idx), 
                      "b f (d o) -> b f d o", d = D)
        b = rearrange(self.feature_bias(idx), 
                      'b f d -> b 1 f d')
        x = x + b
        out = torch.einsum("b n f d, b f d o -> b n f o", x, w)
        return out

class ModalTail(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.tail_norm = ConditionalLayerNorm(dim=cfg.dim, dim_ctx=cfg.dim_noise)
        self.tail_network = GatedFFN(dim= cfg.dim)
        self.feature_bias = Embedding(cfg.num_features, cfg.dim)
        self.out_projection = Embedding(cfg.num_features, cfg.dim * cfg.dim_out)

    def forward(self, x: torch.Tensor, idx: torch.LongTensor, ctx: Optional[torch.Tensor] = None):
        _, _, D = x.size()
        x = rearrange(x, 'b n d -> b n 1 d')
        w = rearrange(self.out_projection(idx), 
                      "b f (d o) -> b f d o", d = D)
        b = rearrange(self.feature_bias(idx), 
                      'b f d -> b 1 f d')
        x = x + self.tail_network(self.tail_norm(x + b, ctx))
        out = torch.einsum("b n f d, b f d o -> b n f o", x, w)
        return out

####

class ModalMTM(Module):
    def __init__(self, modal_cfg: dataclass, decoder_cfg: dataclass, encoder_cfg: Optional[dataclass] = None):
        super().__init__()
        self.encoder = ModalEncoder(modal_cfg)
        self.decoder = ModalDecoder(modal_cfg)
        self.processor = MTM(decoder_cfg = decoder_cfg, encoder_cfg = encoder_cfg)
        # shared ctx encoder
        self.proj_ctx = Linear(decoder_cfg.dim_noise, decoder_cfg.dim_noise)
        self.dim_noise = decoder_cfg.dim_noise
        # Init
        self.apply(self.base_init)
        #self.apply(self.zero_init)

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

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                noise: torch.Tensor = None,
                ):
        if noise is None:
            noise = src.new_zeros(self.dim_noise)
        ctx = self.proj_ctx(noise)
        #encoder
        src = self.encoder(src, var_src)
        # masked token model
        src, tgt = self.processor(src, pos_src, pos_tgt, ctx = ctx)
        # decoder
        tgt = self.decoder(tgt, var_tgt)
        return tgt
    
class ModalTailMTM(Module):
    def __init__(self, modal_cfg: dataclass, decoder_cfg: dataclass, encoder_cfg: Optional[dataclass] = None):
        super().__init__()
        self.encoder = ModalEncoder(modal_cfg)
        self.decoder = ModalTail(modal_cfg)
        self.processor = MTM(decoder_cfg = decoder_cfg, encoder_cfg = encoder_cfg)
        self.apply(self.base_init)

    @staticmethod
    def base_init(m):
        if isinstance(m, Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        if isinstance(m, Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                noise: torch.Tensor
                ):
        #encoder
        src = self.encoder(src, var_src)
        # masked token model
        src, tgt = self.processor(src, pos_src, pos_tgt)
        # noise perturbation
        K = noise.size(0) // src.size(0) if noise is not None else 1
        tgt, var_tgt = (repeat(x, 'b ... -> (b k) ...', k = K) 
                        for x in (tgt, var_tgt)
                        )
        # decoder
        tgt = self.decoder(tgt, var_tgt, noise)
        #split ensemble dim
        tgt = rearrange(tgt, "(b k) ... -> b ... k", k = K)
        return tgt
    
class ModalFuncMTM(ModalMTM):
    def __init__(self, modal_cfg: dataclass, decoder_cfg: dataclass, encoder_cfg: Optional[dataclass] = None):
        super().__init__(modal_cfg=modal_cfg, mtm_decoder_cfg = decoder_cfg, encoder_cfg = encoder_cfg)
        self.apply(super().base_init)

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                noise: Optional[torch.Tensor] = None
                ):
        #encoder
        src = self.encoder(src, var_src)

        # noise perturbation
        K = noise.size(0) // src.size(0) if noise is not None else 1        
        if noise is None:
            noise = src.new_zeros(self.dim_noise)
        ctx = self.proj_ctx(noise)
        
        # masked token model
        src, pos_src, pos_tgt, var_tgt = (repeat(x, "b ... -> (b k) ...", k=K) 
                                          for x in (src, pos_src, pos_tgt, var_tgt)
                                          )
        _, tgt = self.processor(src, pos_src, pos_tgt, ctx = ctx)
        
        # decoder
        tgt = self.decoder(tgt, var_tgt)

        #split ensemble dim
        tgt = rearrange(tgt, "(b k) ... -> b ... k", k = K)
        return tgt