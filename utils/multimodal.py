import torch
from torch.nn import Embedding, Module
from einops import rearrange, repeat
from dataclasses import dataclass
from utils.masked_models import MTM, MAE
from utils.networks import ModalDecoder, ModalEncoder

class ModalMTM(Module):
    def __init__(self, modal_cfg: dataclass, encoder_cfg: dataclass, decoder_cfg: dataclass = None):
        super().__init__()
        # Networks
        self.encoder = ModalEncoder(modal_cfg)
        self.processor = MTM(encoder_cfg) if decoder_cfg is None else MAE(encoder_cfg, decoder_cfg)
        self.decoder = ModalDecoder(modal_cfg)
        
        # Mask token
        self.queries = Embedding(1, modal_cfg.dim)

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

    def forward(self, 
                src: torch.Tensor, 
                pos_src: torch.LongTensor, 
                pos_tgt: torch.LongTensor, 
                var_src: torch.LongTensor, 
                var_tgt: torch.LongTensor,
                noise: torch.Tensor
                ):
        K = noise.size(0) // src.size(0)
        #encoder
        src = self.encoder(src, var_src, ctx = None)
        
        # masked token model
        src, tgt = self.processor(src, pos_src, pos_tgt, ctx = None)
        
        # noise perturbation
        tgt = repeat(tgt, "b ... -> (b k) ...", k = K)
        var_tgt = repeat(var_tgt, "b ... -> (b k) ...", k = K)
        
        # decoder
        tgt = self.decoder(tgt, var_tgt, noise)

        #split ensemble dim
        tgt = rearrange(tgt, "(b k) ... -> b ... k", k = K)
        return tgt