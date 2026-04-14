import torch
import einops
from utils.config import *

class MaskingMixture(torch.nn.Module):
    def __init__(self, 
                 world: WorldConfig, 
                 rate_cfg: dict = None,
                 event_cfg: dict = None,
                 ):
        super().__init__()
        self.world = world

        event_cfg = default(event_cfg, {})
        rate_cfg = default(rate_cfg, {})

        self.src = BinaryMasking(world, rate_cfg= rate_cfg.get('src', {}))
        self.tgt = BinaryMasking(world, rate_cfg= rate_cfg.get('tgt', {}))

        self.components = event_cfg.get('components', [{}])
        mixture_weights = event_cfg.get('weights', None)
        self.register_buffer('mixture_weights', torch.as_tensor(default(mixture_weights, torch.ones(len(self.components)))))

    def forward(self, B: int, rng: torch.Generator = None):
        idx = torch.multinomial(self.mixture_weights, 1, generator = rng).item()
        event_cfg = self.components[idx]
        self.src.event_cfg = event_cfg
        self.tgt.event_cfg = event_cfg

        prefix = event_cfg.get("prefix", None)
        if exists(prefix):
            L = torch.linspace(1, 1e-3, self.world.token_sizes['t'], device= self.mixture_weights.device)
            prefix = L.pow(1 / prefix)
            prefix = einops.repeat(prefix, f't -> b ({self.world.token_pattern})', **self.world.token_sizes, b = B)

        src = self.src(B, conditional = prefix, rng=rng)
        tgt = self.tgt(B, conditional = src.logical_not(), rng=rng)
        return src, tgt

class BinaryMasking(torch.nn.Module):
    def __init__(self, 
                 world: WorldConfig, 
                 event_cfg: dict = None, 
                 rate_cfg: dict = None, 
                 epsilon: float = 1e-3):
        super().__init__()
        self.world = world
        
        self.event_cfg = default(event_cfg, {})
        self.rate_cfg = default(rate_cfg, {})

        self.register_buffer('epsilon', torch.tensor(epsilon))

    @property
    def device(self) -> torch.device: return self.epsilon.device

    @staticmethod
    def binary_topk_(P: torch.FloatTensor, K: torch.LongTensor, dim: int = -1) -> torch.BoolTensor:
        return K > P.argsort(descending=True, dim = dim).argsort(dim = dim)

    def uniform_(self, shape: tuple, rng: torch.Generator):
        return torch.rand(*shape, device=self.device, generator=rng).clamp(self.epsilon, 1 - self.epsilon)

    def stratified_uniform_(self, B: int, rng: torch.Generator = None):
        L = torch.linspace(0, 1, B, device=self.device)
        U = torch.rand((1,), device=self.device, generator=rng)
        return (L + U).remainder(1).clamp(self.epsilon, 1 - self.epsilon)

    def event_prior(self, B: int, rng: torch.Generator = None):
        P = self.uniform_((B, self.world.num_tokens), rng).log()
        for dim, alpha in self.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = self.uniform_((B, self.world.token_sizes[dim]), rng)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P

    def rate_prior(self, rng: torch.Generator = None):
        R = torch.empty((1,), device = self.device)
        torch.nn.init.trunc_normal_(
            R, 
            self.rate_cfg.get("mean", 0.5), 
            self.rate_cfg.get("std", 1.),
            self.rate_cfg.get("a", 0.),
            self.rate_cfg.get("b", 1.),
            generator = rng
            )
        return R.mul(self.world.num_tokens).long()

    def forward(self, B: int, conditional: torch.Tensor = None, rng: torch.Generator = None) -> torch.BoolTensor:
        B = B[0] if isinstance(B, tuple) else B
        K = self.rate_prior(rng)
        P = self.event_prior(B, rng)
        if exists(conditional): P += conditional.type_as(P).clamp(min=1e-9).log()
        return self.binary_topk_(P, K)
