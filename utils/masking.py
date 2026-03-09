import torch
import einops
from utils.config import WorldConfig, ObjectiveConfig, exists, default
from dataclasses import replace

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
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()
        self.world = world
        self.objective = objective
        self.register_buffer('epsilon', torch.tensor(self.objective.epsilon))

    @property
    def device(self) -> torch.device: return self.epsilon.device

    @property
    def event_dims(self) -> Dict[str, float]:
        return self.objective.event_dims

    @event_dims.setter
    def event_dims(self, update: Dict[str, float]):
        self.objective = replace(self.objective, event_dims={**self.event_dims, **update})

    @staticmethod
    def binary_topk_(P: torch.FloatTensor, K: torch.LongTensor) -> torch.BoolTensor:
        return K > P.argsort(descending=True).argsort()
    
    @staticmethod
    def kumaraswamy_quantile(U: torch.FloatTensor, alpha: float = 1.):
        return U.log().div(alpha).exp()
    
    @staticmethod
    def kumaraswamy_quantile_dt(U: torch.FloatTensor, alpha: float = 1.):
        return U.log().mul((1 / alpha) - 1).exp().div(alpha)
    
    def uniform_(self, shape: tuple, rng: torch.Generator):
        return torch.rand(*shape, device = self.device, generator=rng).clamp(self.epsilon, 1 - self.epsilon)

    def stratified_uniform_(self, B: int, rng: torch.Generator = None):
        L = torch.linspace(0, 1, B, device=self.device)
        U = torch.rand((1,), device = self.device, generator=rng)
        return (L + U).remainder(1).clamp(self.epsilon, 1 - self.epsilon)

    def weight_prior(self, B: int, conditional: torch.Tensor = None, rng: torch.Generator = None):
        # baseline per-token noise
        F = self.uniform_((B, self.world.num_tokens), rng).log().div(self.event_dims.get('base', 1.))
        # for each event dimension
        for dim, alpha in self.event_dims.items():
            # skip non-world dimensions
            if dim not in self.world.layout:
                continue
            # sample uniform variate along the dimension
            U = self.uniform_((B, self.world.token_sizes[dim]), rng)
            # broadcast to the size of the full event space
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            # apply quantile function for Kumaraswamy(alpha, 1) and sum log factors
            F += U.log().div(alpha)
        # maybe add conditional factor
        if exists(conditional):
            F += conditional.type_as(F).clamp(min=1e-12).log()
        return F
    
    def rate_prior(self, B: int, kind: str, rng: torch.Generator = None):
        a_min, a_max = self.objective.kwargs.get(f'min_{kind}', 0), self.objective.kwargs.get(f'max_{kind}', 1)
        U = self.stratified_uniform_(B, rng) if self.objective.stratify else self.uniform_((B,), rng)
        R = U * (a_max - a_min) + a_min
        return R.mul(self.world.num_tokens).long().unsqueeze(-1)
    
    def forward(self, B: int, rng: torch.Generator = None):
        B = B[0] if isinstance(B, tuple) else B
        # sample rates
        K_src = self.rate_prior(B, "src", rng)
        K_tgt = self.rate_prior(B, "tgt", rng)
        # sample src
        P_src = self.weight_prior(B, rng = rng)
        src = self.binary_topk_(P_src, K_src)
        # sample tgt w/o src
        P_tgt = self.weight_prior(B, conditional = src, rng = rng)
        tgt = self.binary_topk_(P_tgt, K_tgt)
        return src, tgt, None
