import torch
import einops
from itertools import permutations
from utils.config import *

# MASKING STRATEGIES
class ForecastMasking(torch.nn.Module):
    def __init__(self, world: WorldConfig):
        super().__init__()
        self.world = world
        self.event_pattern = 't'
        self.register_buffer("prefix_frames", torch.tensor(world.tau, dtype = torch.long))
        self.register_buffer("total_frames", torch.tensor(world.token_sizes["t"], dtype = torch.long))
    
    def forward(self, shape: tuple, return_indices: bool = True):
        mask = torch.zeros((self.total_frames,), device = self.total_frames.device)
        mask[:self.prefix_frames] = 1
        mask = einops.repeat(mask, f'{self.event_pattern} -> ({self.world.token_pattern})', **self.world.token_sizes)
        if return_indices:
            src = mask.nonzero(as_tuple=True)[0]
            tgt = mask.bool().logical_not().nonzero(as_tuple=True)[0]
            return src.expand(*shape,-1), tgt.expand(*shape,-1)
        else:
            return mask.expand(*shape, -1).bool(), mask.expand(*shape, -1).bool().logical_not()

class MaskingMixture(torch.nn.Module):
    def __init__(self, 
                 world: WorldConfig, 
                 rate_cfg: dict = None,
                 event_cfg: dict = None,
                 ):
        super().__init__()
        self.world = world

        rate_cfg = default(rate_cfg, {})
        self.src = BinaryMasking(world, rate_cfg= rate_cfg.get('src', {}))
        self.tgt = BinaryMasking(world, rate_cfg= rate_cfg.get('tgt', {}))

        components = event_cfg.get('components', None)
        alphas = event_cfg.get('alphas', None)
        mixture_weights = event_cfg.get('weights', None)

        if exists(components):
            components = torch.as_tensor(components)
        elif exists(alphas):
            components = torch.tensor([
                [alphas[v], alphas[t], alphas[hw], alphas[hw]] 
                for (v, t, hw) in permutations(range(len(alphas)), len(world.layout) - 1)
            ])
        else:
            components = torch.ones(len(world.layout))

        components = components.view(-1, len(world.layout)) # ensure (Categories, Dims) shape

        if exists(mixture_weights):
            mixture_weights = torch.as_tensor(mixture_weights)
        else:
            mixture_weights = torch.ones(len(components))
        
        self.register_buffer('components', components)
        self.register_buffer('mixture_weights', mixture_weights)

    def forward(self, B: int, rng: torch.Generator = None):
        idx = torch.multinomial(self.mixture_weights, 1, generator = rng)
        alphas = {dim: self.components[idx, i:i+1] for i, dim in enumerate(self.world.layout)}

        self.src.event_cfg = alphas
        self.tgt.event_cfg = alphas
        src = self.src(B, rng=rng)
        tgt = self.tgt(B, rng=rng)
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
            if dim not in self.world.layout: continue
            U = self.uniform_((B, self.world.token_sizes[dim]), rng)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P

    def rate_prior(self, B: int, rng: torch.Generator = None):
        a_min = self.rate_cfg.get('min', 0)
        a_max = self.rate_cfg.get('max', 1)
        if self.rate_cfg.get('stratify', False):
            U = self.stratified_uniform_(B, rng)
        elif self.rate_cfg.get('randomize', False):
            U = self.uniform_((B,), rng)
        else:
            U = self.uniform_((1,), rng).expand(B,)
        R = U * (a_max - a_min) + a_min
        return R.mul(self.world.num_tokens).long().unsqueeze(-1)

    def forward(self, B: int, conditional: torch.Tensor = None, rng: torch.Generator = None) -> torch.BoolTensor:
        B = B[0] if isinstance(B, tuple) else B
        K = self.rate_prior(B, rng)
        P = self.event_prior(B, rng)
        if exists(conditional): P += conditional.type_as(P).clamp(min=1e-12).log()
        return self.binary_topk_(P, K)
