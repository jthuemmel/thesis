import torch
import einops
from utils.config import *

class BinaryMasking(torch.nn.Module):
    def __init__(self, 
                 world: WorldConfig, 
                 event_cfg: dict = None, 
                 rate_cfg: dict = None, 
                 epsilon: float = 1e-5):
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

    def sample_weighted_reservoir(self, num_samples: int, rng: torch.Generator = None):
        P = self.uniform_((num_samples, self.world.num_tokens), rng).log()
        for dim, alpha in self.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = self.uniform_((num_samples, self.world.token_sizes[dim]), rng)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P

    def sample_rates(self, rng: torch.Generator = None):
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

    def forward(self, num_samples: int, conditional: torch.Tensor = None, rng: torch.Generator = None) -> torch.BoolTensor:
        K = self.sample_rates(rng)
        P = self.sample_weighted_reservoir(num_samples, rng)
        if exists(conditional): P += conditional.type_as(P).clamp(min=1e-9).log()
        return self.binary_topk_(P, K)