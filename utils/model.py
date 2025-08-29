import torch
from utils.components import *

class WeatherField(torch.nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        cfg = model_cfg.decoder
        embedding_dim = (len(cfg.wavelengths) + 1) * cfg.dim_coords

        # shared embeddings
        self.position_embedding = ContinuousPositionalEmbedding(cfg.dim_coords, cfg.wavelengths, None)
        self.feature_embedding = torch.nn.Embedding(cfg.num_features, cfg.dim_coords)
        self.latent_embedding = torch.nn.Embedding(cfg.num_latents, cfg.dim)

        # feature-wise linear embedding
        self.proj_src = SegmentLinear(cfg.dim_in, cfg.dim_in, cfg.num_features)

        # linear map into feature-wise source representation
        self.proj_x = SegmentLinear(embedding_dim + cfg.dim_in, cfg.dim, cfg.num_features)
        self.norm_x = torch.nn.LayerNorm(cfg.dim)

        # linear map into feature-wise query representations
        self.proj_q = SegmentLinear(embedding_dim, cfg.dim, cfg.num_features)

        # feature-wise output projection
        self.proj_out = SegmentLinear(cfg.dim, cfg.dim_out, cfg.num_features)

        # Transformer blocks
        self.encoder = torch.nn.ModuleList([TransformerBlock(cfg.dim, dim_heads=cfg.dim_heads) for _ in range(cfg.num_layers)])
        self.decoder = TransformerBlock(cfg.dim, dim_heads=cfg.dim_heads)
        
        # Initialization
        self.apply(self.base_init)

    @staticmethod
    def base_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = get_weight_std(m.weight))

        if isinstance(m, torch.nn.LayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None:
                torch.nn.init.ones_(m.weight)

        if isinstance(m, ConditionalLayerNorm) and m.linear is not None:
            torch.nn.init.trunc_normal_(m.linear.weight, std = 1e-8)

    def forward(self, src_data: torch.Tensor, src_coords: torch.LongTensor, tgt_coords: torch.LongTensor):
        """
        Args:
            src_data: [B, N_src, D_in]
            src_coords: [B, N_src, 4] assuming (V, T, H, W) layout
            tgt_coords: [B, N_tgt, 4] assuming (V, T, H, W) layout
        """
        # slice out variable coordinate, since it is treated separately
        var_src, pos_src = src_coords.split([1, 3], dim = -1)
        var_tgt, pos_tgt = tgt_coords.split([1, 3], dim = -1)

        # get embeddings
        src_data = self.proj_src(src_data, var_src)
        src_positions = self.position_embedding(pos_src)
        src_features = self.feature_embedding(var_src)
        tgt_positions = self.position_embedding(pos_tgt)
        tgt_features = self.feature_embedding(var_tgt)

        # concatenate embeddings
        src = torch.cat([src_data, src_positions, src_features], dim = -1)
        query = torch.cat([tgt_positions, tgt_features], dim = -1)
    
        # linear maps to shared latent space
        src = self.proj_x(src, var_src)
        src = self.norm_x(src)
        query = self.proj_q(query, var_tgt)
        
        # update latents given src
        latents = self.latent_embedding.weight.expand(src.size(0), -1, -1)
        for block in self.encoder:
            kv = torch.cat([latents, src], dim = 1)
            latents = block(latents, kv)

        # decoder
        query = self.decoder(query, latents)
        query = self.proj_out(query, var_tgt)
        return query