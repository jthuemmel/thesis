import torch
import einops
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

class BinaryMasking(torch.nn.Module):
    def __init__(self, 
                 world: WorldConfig, 
                 event_cfg: dict = None, 
                 rate_cfg: dict = None, 
                 epsilon: float = 1e-3):
        super().__init__()
        self.world = world
        
        self._event_cfg = default(event_cfg, {})
        self._rate_cfg = default(rate_cfg, {})

        self.register_buffer('epsilon', torch.tensor(epsilon))

    @property
    def device(self) -> torch.device: return self.epsilon.device

    @property
    def event_cfg(self) -> dict:
        return self._event_cfg
    
    @event_cfg.setter
    def event_cfg(self, update: dict):
        self._event_cfg.update(update)

    @property
    def rate_cfg(self) -> dict:
        return self._rate_cfg
    
    @rate_cfg.setter
    def rate_cfg(self, update: dict):
        self._rate_cfg.update(update)

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
        P = self.uniform_((B, self.world.num_tokens), rng).log().div(self.event_cfg.get('base', 1.))
        for dim, alpha in self.event_cfg.items():
            if dim not in self.world.layout:
                continue
            U = self.uniform_((B, self.world.token_sizes[dim]), rng)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P

    def rate_prior(self, B: int, rng: torch.Generator = None):
        a_min = self.rate_cfg.get('min', 0)
        a_max = self.rate_cfg.get('max', 1)
        U = self.stratified_uniform_(B, rng) if self.rate_cfg.get('stratify', False) else self.uniform_((B,), rng)
        R = U * (a_max - a_min) + a_min
        return R.mul(self.world.num_tokens).long().unsqueeze(-1)

    def forward(self, B: int, conditional: torch.Tensor = None, rng: torch.Generator = None) -> torch.BoolTensor:
        B = B[0] if isinstance(B, tuple) else B
        K = self.rate_prior(B, rng)
        P = self.event_prior(B, rng)
        if exists(conditional):
            P += conditional.type_as(P).clamp(min=1e-12).log()
        return self.binary_topk_(P, K)
