import torch
from torch.nn import Linear, Embedding, Module
from dataclasses import dataclass
from utils.networks import ViT, InterfaceNetwork
from utils.components import Attention, GatedFFN, ConditionalLayerNorm
from typing import Tuple

####
ARCHITECTURES = {
    "interface": InterfaceNetwork,
    "vit": ViT,
}

####
class MAE(torch.nn.Module):
    def __init__(self, encoder_cfg: dataclass, decoder_cfg: dataclass):
        super().__init__()
        self.queries = Embedding(1, decoder_cfg.dim)
        self.encoder = ARCHITECTURES[encoder_cfg.architecture](encoder_cfg)
        self.decoder = ARCHITECTURES[decoder_cfg.architecture](decoder_cfg)

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
                ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(src, pos_src, ctx = ctx)
        q = self.queries(torch.zeros_like(pos_tgt))
        x_hat = torch.cat([x, q], dim=1)
        pos = torch.cat([pos_src, pos_tgt], dim=1)
        x_hat = self.decoder(x_hat, pos, ctx = ctx)
        src, tgt = x_hat.split([pos_src.size(1), pos_tgt.size(1)], dim=1)
        return src, tgt

class MTM(Module):
    def __init__(self, cfg: dataclass):
        super().__init__()
        self.processor = ARCHITECTURES[cfg.architecture](cfg)
        self.queries = Embedding(1, cfg.dim)

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
                ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self.queries(torch.zeros_like(pos_tgt))
        x = torch.cat([src, tgt], dim = 1)
        pos = torch.cat([pos_src, pos_tgt], dim = 1)
        x_hat = self.processor(x, pos, ctx = ctx)
        src, tgt = x_hat.split([src.size(1), tgt.size(1)], dim = 1)
        return src, tgt