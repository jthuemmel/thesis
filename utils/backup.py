import einops
import torch

from einops.layers.torch import EinMix

from utils.components import *
from utils.config import *
from utils.random_fields import *

import einops
import torch

from einops.layers.torch import EinMix, Rearrange

from utils.components import *
from utils.config import *
from utils.random_fields import RandomField

# KUMARASWAMY DISTRIBUTION
class StableKumaraswamy(torch.nn.Module):
    '''Numerically stable methods for Kumaraswamy sampling, courtesy of Wasserman et al 2024'''
    def __init__(self, c1: float = 1., c0: float = 1., epsilon=1e-3):
        super().__init__()
        assert c1 > 0. and c0 > 0., 'invalid concentration'
       
        # Register hyperparameters as buffers for device consistency
        self.register_buffer("c1", torch.as_tensor(c1))
        self.register_buffer("c0", torch.as_tensor(c0))
        self.register_buffer("epsilon", torch.as_tensor(epsilon))
    
    # Kumaraswamy with log1mexp
    @staticmethod
    def log1mexp(t: torch.FloatTensor): #numerically stable log(1 - e**x)
        return torch.where(
        t < -0.69314718, #~ -log2
        torch.log1p(-torch.exp(t)), 
        torch.log(-torch.expm1(t))
    )

    def quantile_dt(self, t: torch.Tensor): # time derivative of the quantile function
        #(1 - t)**((1 - c0) / c0) * (1 - (1 - t)**(1 / c0))**((1 - c1) / c1) / (c1 * c0)
        log_1_minus_t = torch.log1p(-t) # 1 - t
        log_constant = -self.c1.log() - self.c0.log() # 1 / c0 * c1
        log_outer = log_1_minus_t * ((1 - self.c0) / self.c0) # (1 - t)**(1-c0)/c0
        log_inner = ((1 - self.c1) / self.c1) * self.log1mexp(log_1_minus_t / self.c0)
        return torch.exp(log_constant + log_inner + log_outer)

    def quantile(self, t: torch.Tensor): # (1 - (1 - t)**(1 / c0))**(1 / c1)
        return torch.exp(self.log1mexp(torch.log1p(-t) / self.c0) / self.c1)
    
    def cdf(self, t: torch.Tensor): # 1 - (1 - t**c1)**c0
        return -torch.expm1(self.c0 * self.log1mexp(self.c1 * t.log()))
    
    def forward(self, shape: tuple, rng: torch.Generator = None):
        t = torch.rand(shape, device=self.epsilon.device, generator= rng)
        t = t * (1.0 - 2.0 * self.epsilon) + self.epsilon
        return self.quantile(t), self.quantile_dt(t)

class MultinomialMasking(torch.nn.Module):
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()

        self.world = world
        self.objective = objective

        self.register_buffer('epsilon', torch.tensor(self.objective.epsilon))

    @property
    def device(self): return self.epsilon.device

    def uniform_(self, B: int, N: int, rng: torch.Generator):
        if self.objective.stratify:
            U = torch.rand(1, N, device = self.device, generator=rng)
            L = torch.linspace(0, 1, B, device= self.device).view(-1, 1)
            U = (U + L) % 1
        else:
            U = torch.rand(B, N, device = self.device, generator=rng)
        return U.clamp(self.epsilon, 1 - self.epsilon)
    
    def prior_(self, B: int, rng: torch.Generator = None):
        # baseline factors are constant
        F_src, F_tgt = torch.ones(2, B, self.world.num_tokens, device= self.device)

        # for each event dimension
        for dim, alpha in self.objective.event_dims.items():
            # sample uniform variate along the dimension
            U = self.uniform_(B, self.world.token_sizes[dim], rng)

            # broadcast to the size of the full event space
            U = einops.repeat(U, f'b {dim} -> b {self.world.flat_token_pattern}', **self.world.token_sizes)

            # apply quantile function for Kumaraswamy(alpha, 1) and Kumaraswamy(1, alpha) and multiply factors
            F_src *= U.log().div(alpha).exp()
            F_tgt *= (1 - U).log().div(alpha).exp()

        return F_src.clamp(self.epsilon, 1 - self.epsilon), F_tgt.clamp(self.epsilon, 1 - self.epsilon)
    
    def rates_(self, rng: torch.Generator = None):
        step_size = default(self.objective.step_size, 1)
        src_low, src_high = default(self.objective.src_low, 0), default(self.objective.src_high, self.world.num_tokens - 1)
        tgt_low, tgt_high = default(self.objective.tgt_low, 0), default(self.objective.tgt_high, self.world.num_tokens - 1)

        src_low_scaled, src_high_scaled = (src_low + step_size - 1) // step_size, src_high // step_size
        tgt_low_scaled, tgt_high_scaled = (tgt_low + step_size - 1) // step_size, tgt_high // step_size

        K_src_scaled = torch.randint(src_low_scaled, src_high_scaled + 1, (1,), generator=rng, device=self.device).item()
        K_tgt_scaled = torch.randint(tgt_low_scaled, tgt_high_scaled + 1, (1,), generator=rng, device=self.device).item()

        return K_src_scaled * step_size, K_tgt_scaled * step_size
    
    def forward(self, B: int, rng: torch.Generator = None):
        B = B[0] if isinstance(B, tuple) else B

        # sample joint prior
        P_src, P_tgt = self.prior_(B, rng)

        #  sample uniform rates from (low, high)
        K_src, K_tgt = self.rates_(rng)

        # sample multinomial 
        src_indices = torch.multinomial(P_src, K_src, generator=rng) if K_src > 0 else None
        tgt_indices = torch.multinomial(P_tgt, K_tgt, generator=rng) if K_tgt > 0 else None
        
        return src_indices, tgt_indices
    
class LatentTransformer(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()

        # learnable latents
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)

        # latent transformer
        self.encoder = torch.nn.ModuleList([
            TransformerBlock(network.dim)
            for _ in range(default(network.num_read_blocks, 1))
        ])

        self.processor = torch.nn.ModuleList([
            TransformerBlock(network.dim, drop_path=network.drop_path)
            for _ in range(default(network.num_compute_blocks, 1))
        ])
        
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(network.dim)
            for n in range(default(network.num_write_blocks, 1))
        ])

    def forward(self, queries: torch.Tensor):
        latents = einops.repeat(self.latents.weight, 'n d -> b n d', b = queries.size(0))
        # map src to latents
        for read in self.encoder:
            latents = read(q = latents, kv = torch.cat([queries, latents], dim = -2))

        # process latents
        for process in self.processor:
            latents = process(q = latents)

        # map latents to tgt
        for write in self.decoder:
            queries = write(q = queries, kv = latents)

        return queries
    
class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # I/O
        self.to_tokens = EinMix(
            pattern=f"b {world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'v {world.patch_pattern} d', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        self.to_fields =EinMix(
            pattern=f"b {world.flat_token_pattern} d -> b {world.field_pattern}", 
            weight_shape=f'v d {world.patch_pattern}', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes 
            )
        
        # Noise
        self.noise_generator = RandomField(network, world)

        # learnable tokens
        self.mask_embedding = torch.nn.Embedding(1, network.dim)
        self.position_embedding = ContinuousPositionalEmbedding(
            dim_per_coord=network.dim_coords, 
            wavelengths=[(1, 2 * k) for k in world.token_shape],
            model_dim=network.dim
        )

        # pre-computed coordinates
        self.register_buffer(
            "coordinates", 
            torch.stack(
                torch.unravel_index(indices = torch.arange(world.num_tokens), shape = world.token_shape), 
                dim = -1)
                )

        # transformer
        self.transformer = LatentTransformer(network, world)
        
        # Weight initialization
        self.apply(self.base_init)

    @staticmethod
    def base_init(m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # embedding
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # einmix
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.trunc_normal_(m.bias, std = 0.02)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                visible: torch.BoolTensor, 
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        B = fields.size(0)
        E = default(members, 1)

        # expand to ensemble form
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        visible = einops.repeat(visible, 'b ... -> (b e) ... d', d = self.network.dim, e = E, b = B)

        # embed full fields as tokens
        tokens = self.to_tokens(fields)

        # apply mask
        tokens = torch.where(visible, tokens, self.mask_embedding.weight)

        # create random field
        noise = self.noise_generator(shape = (B * E,), rng = rng).to(tokens.dtype)

        # add noise and positions
        tokens = tokens + noise + self.position_embedding(self.coordinates)      
        
        # apply transformer
        tokens = self.transformer(tokens)
        
        # map all tokens back to fields
        fields = self.to_fields(tokens)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... -> b ... e", e = E, b = B)
        return fields

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # noise generator
        s = torch.tensor([500, 1000, 2000, 4000])
        t = torch.tensor([1, 2, 4, 6])
        self.noise_generator = SphericalDiffusionNoise(
            num_channels = len(s) * len(t),
            num_steps = 6,
            num_lat = 45,
            num_lon = 90,
            horizontal_length = einops.repeat(s, 's -> (s t)', t = len(t)),
            temporal_length = einops.repeat(t, 't -> (s t)', s = len(s)),
            lat_slice=slice(14, -15, 1), # 16 latitudes ~ -32 to 32 with 4 deg res
            lon_slice=slice(0, 60, 2) # 30 longitudes ~ 90 to 330 with 4 deg res and 2 step
        )

        # pre-compute positional embedding
        self.positions = ContinuousPositionalEmbedding(network.dim_coords, network.wavelengths, model_dim= None)
        idcs = torch.unravel_index(indices = torch.arange(world.num_tokens), shape = world.token_shape)
        idcs = torch.stack(idcs, dim = -1)
        pos = self.positions(idcs)
        self.register_buffer('coordinates', pos)

        # I/O
        self.token_embedding = EinMix(
            pattern=f"{world.field_pattern} -> b {world.flat_token_pattern} di", 
            weight_shape=f'{world.patch_pattern} v di', 
            di = network.dim_in, **world.patch_sizes, **world.token_sizes
            )
        
        self.token_predictor = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> {world.field_pattern} k", 
            weight_shape=f'd v {world.patch_pattern} k', 
            d = network.dim, k = network.num_tails, 
            **world.patch_sizes, **world.token_sizes 
            )
        
        self.context_embedding = torch.nn.Linear(
                in_features = self.positions.embedding_dim + self.noise_generator.nchannels,
                out_features = network.dim - network.dim_in,
            )

        # learnable parameters (Embedding for convenient initialization)
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)
        self.queries = torch.nn.Embedding(world.num_tokens, network.dim_in)

        # latent transformer
        self.transformer = torch.nn.ModuleList([
            TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                drop_path = network.drop_path,
                ) 
                for _ in range(network.num_layers)
            ])
        self.write = TransformerBlock(
                dim = network.dim, 
                dim_heads = network.dim_heads, 
                dim_ctx = network.dim_noise,
                has_skip = False
                )

        # Weight initialization
        self.apply(self.base_init)
    
    @staticmethod
    def base_init(m):
        '''Explicit weight initialization'''
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # embedding
        if isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # einmix
        if isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # conditional layer norm
        if isinstance(m, ConditionalLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None: # CLN weight close to 0
                torch.nn.init.trunc_normal_(m.weight, std = 1e-7)

    def forward(self, fields: torch.FloatTensor, visible: torch.LongTensor, E: int = 1, rng: torch.Generator = None) -> torch.FloatTensor:
        # visible: (S, B, v) or (B, v)
        if visible.ndim == 2: visible = visible.unsqueeze(0)
        S, B = (visible.size(0), visible.size(1))

        # expand to ensemble form
        xs = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        vs = einops.repeat(visible, 's b ... -> s (b e) ...', e = E, b = B, s = S)

        # iterate
        for s in range(S):
            # sample noise
            eta = self.noise_generator(shape = (B * E,), rng = rng)
            eta = einops.repeat(eta, f"be c t h w -> be {self.world.flat_token_pattern} c", **self.world.token_sizes)
            # step
            xs = self.step(fields=xs, visible=vs[s], noise=eta)
            # detach gradients unless it is the last step
            if s < S - 1:
                xs = xs.detach()
        
        # rearrange to ensemble form
        xs = einops.rearrange(xs, "(b e) ... k -> b ... (e k)", e = E, b = B, k = self.network.num_tails)
        return xs

    def step(self, fields: torch.FloatTensor, visible: torch.LongTensor = None, noise: torch.FloatTensor = None) -> torch.FloatTensor:    
        # expand learnable codes
        latents = einops.repeat(self.latents.weight, 'm d -> b m d', b = fields.size(0))
        queries = einops.repeat(self.queries.weight, 'n di -> b n di', b = fields.size(0))
        positions = einops.repeat(self.coordinates, "n dc -> b n dc", b = fields.size(0))        

        # maybe combine context information
        context = torch.cat([positions, noise], dim = -1) if exists(noise) else positions

        # linear embeddings
        context = self.context_embedding(context)
        tokens = self.token_embedding(fields)

        # concatenate context to tokens and queries
        tokens = torch.cat([tokens, context], dim = -1)
        queries = torch.cat([queries, context], dim = -1)

        # encode only visible tokens
        if exists(visible):
            visible = einops.repeat(visible, 'b n -> b n d', d = tokens.size(-1))
            tokens = tokens.gather(1, visible)
        
        # apply flamingo-style transformer
        for block in self.transformer:
            kv = torch.cat([tokens, latents], dim = 1)
            latents = block(q = latents, kv = kv)
        queries = self.write(q = queries, kv = latents)

        # decode tokens
        queries = self.token_predictor(queries)
        return queries


class EinDecoder(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # config attributes 
        c = network.dim_out
        k = default(network.num_tails, 1)
        vv = world.patch_sizes['vv']
        v = world.token_sizes['v']
        groups = v * k
        self._out_size = tuple(world.field_sizes[ax] for ax in ['t', 'h', 'w'])
        self._mode = 'nearest-exact'

        # project tokens to low dimensional space before upsampling
        self.token_to_grid = EinMix(
            pattern=f"b {world.flat_token_pattern} d -> b (v k c) t h w",
            weight_shape=f"v k c d",
            d = network.dim, c = c, k = k,
            **world.token_sizes
        )

        # small CNN for post-processing
        self.cnn = torch.nn.ModuleList([
            ConvNextBlock(c * groups, (3, 7, 7), groups) 
            for _ in range(default(network.num_cnn_blocks, 0))
            ])

        # pointwise projection to output
        self.grid_to_field = EinMix(
            f'b (v k c) t h w -> b {world.field_pattern} k',
            weight_shape=f"v {world.patch_pattern} k c",
            k = k, c = c, **world.patch_sizes, **world.token_sizes
        )

    def forward(self, x: torch.FloatTensor):
        x = self.token_to_grid(x)
        for block in self.cnn:
            x = block(x)
        x = self.grid_to_field(x)
        return x

class EinMask(torch.nn.Module):
    def __init__(self, network: NetworkConfig, world: WorldConfig):
        super().__init__()
        # store configs
        self.network = network
        self.world = world

        # I/O
        self.to_tokens = EinMix(
            pattern=f"b {world.field_pattern} -> b {world.flat_token_pattern} d", 
            weight_shape=f'{world.patch_pattern} v d', 
            d = network.dim, 
            **world.patch_sizes, **world.token_sizes
            )
        
        if exists(network.dim_out):
            self.to_fields = EinDecoder(network, world)  
        else:
            self.to_fields =EinMix(
                pattern=f"b {world.flat_token_pattern} d -> b {world.field_pattern} k", 
                weight_shape=f'd v {world.patch_pattern} k', 
                d = network.dim, 
                k = default(network.num_tails, 1), 
                **world.patch_sizes, **world.token_sizes 
                )
                
        # noise mapping
        if default(network.num_tails, 1) > 1:
            self.noise_generator = None
        else:
            self.noise_generator = RandomField(network.dim, world, has_ffn=False)
        
        # positional embeddings
        self.src_positions = ContinuousPositionalEmbedding(
            dim_per_coord=network.dim_coords, 
            wavelengths=[(1, 2 * k) for k in world.token_shape],
            model_dim=network.dim
        )

        self.tgt_positions = ContinuousPositionalEmbedding(
            dim_per_coord=network.dim_coords, 
            wavelengths=[(1, 2 * k) for k in world.token_shape],
            model_dim=network.dim
        )
        
        # pre-computed coordinates
        self.register_buffer('indices', torch.arange(world.num_tokens))
        self.register_buffer("coordinates", torch.stack(
            torch.unravel_index(indices = self.indices, shape = world.token_shape), 
            dim = -1)
            )        
        
        # learnable latents
        self.latents = torch.nn.Embedding(network.num_latents, network.dim)

        # latent transformer
        self.encoder = torch.nn.ModuleList([
            TransformerBlock(network.dim, dim_ctx=network.dim_noise)
            for _ in range(default(network.num_read_blocks, 1))
        ])

        self.processor = torch.nn.ModuleList([
            TransformerBlock(network.dim, drop_path=network.drop_path, dim_ctx=network.dim_noise)
            for _ in range(default(network.num_compute_blocks, 1))
        ])
        
        self.decoder = torch.nn.ModuleList([
            TransformerBlock(network.dim, dim_ctx=network.dim_noise)
            for n in range(default(network.num_write_blocks, 1))
        ])

        # Weight initialization
        self.apply(self.base_init)
        self.apply(self.zero_init)

    @staticmethod
    def zero_init(m: torch.nn.Module):
        # residual blocks zero out their last layer 
        if isinstance(m, (TransformerBlock, ConvNextBlock, ConvInterpolate)):
            for name, sm in m.named_modules():
                if "_out" in name and hasattr(sm, 'weight'):
                    torch.nn.init.trunc_normal_(sm.weight, std = 1e-7)

    @staticmethod
    def base_init(m: torch.nn.Module):
        # linear
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # embedding
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
        # einmix
        elif isinstance(m, EinMix):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.trunc_normal_(m.bias, std = 0.02)
        # convolution
        elif isinstance(m, torch.nn.Conv3d):
            torch.nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        # conditional layer norm
        elif isinstance(m, ConditionalLayerNorm):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
            if m.weight is not None: # CLN weight close to 0
                torch.nn.init.trunc_normal_(m.weight, std = 1e-7)
    
    def forward(self, 
                fields: torch.FloatTensor, 
                srcs: List[torch.LongTensor] | torch.LongTensor, 
                tgts: List[torch.LongTensor] | torch.LongTensor = None,
                members: Optional[int] = None, 
                rng: Optional[torch.Generator] = None
                ) -> torch.FloatTensor:
        B = fields.size(0)
        D = self.network.dim
        K = default(self.network.num_tails, 1)
        E = default(members, 1)

        # expand to ensemble form
        fields = einops.repeat(fields, "b ... -> (b e) ...", e = E, b = B)
        latents = einops.repeat(self.latents.weight, '... -> (b e) ...', b = B, e = E)
        coo = einops.repeat(self.coordinates, '... -> (b e) ...', b = B, e = E)
        src_idx = einops.repeat(srcs, 'b ... -> (b e) ... d', d = D, e = E, b = B)
        
        # embed full fields as tokens
        tokens = self.to_tokens(fields).gather(1, src_idx)

        # prepare tgt and src
        tgt = self.tgt_positions(coo)
        src = self.src_positions(coo).scatter_add_(1, src_idx, tokens)

        # maybe add random field
        if exists(self.noise_generator):
            noise = self.noise_generator(shape = (B * E,), rng = rng).to(src.dtype)
            tgt = tgt + noise
            src = src + noise

        # map src to latents
        for read in self.encoder:
            latents = read(q = latents, kv = torch.cat([src, latents], dim = 1))

        # process latents
        for process in self.processor:
            latents = process(q = latents)

        # map latents to tgt
        for write in self.decoder:
            tgt = write(q = tgt, kv = latents)

        # map all tokens back to fields
        fields = self.to_fields(tgt)
        
        # rearrange to ensemble form
        fields = einops.rearrange(fields, "(b e) ... k -> b ... (e k)", e = E, b = B, k = K)
        return fields