import torch
import einops
from utils.config import WorldConfig, ObjectiveConfig

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

    @staticmethod
    def binary_topk_(P: torch.FloatTensor, K: torch.LongTensor) -> torch.BoolTensor:
        return K > P.argsort(descending=True).argsort()
    
    @staticmethod
    def kumaraswamy_quantile(U: torch.FloatTensor, alpha: float = 1.):
        return U.log().div(alpha).exp()
    
    @staticmethod
    def kumaraswamy_quantile_dt(U: torch.FloatTensor, alpha: float = 1.):
        return U.log().mul((1 / alpha) - 1).exp().div(alpha)
            
    def k_from_rates_(self, R: torch.FloatTensor):
        return R.mul(self.world.num_tokens).long()
    
    def uniform_(self, shape: tuple, rng: torch.Generator, eps: float = 1e-6):
        return torch.rand(*shape, device = self.device, generator=rng).clamp(eps, 1-eps)

    def weight_prior(self, B: int, rng: torch.Generator = None):
        # baseline factors are log-uniform -> Kumaraswamy(1, 1)
        F_src, F_tgt = self.uniform_((2, B, self.world.num_tokens), rng).log().div(1.)
        
        # for each event dimension
        for dim, alpha in self.objective.event_dims.items():
            # sample uniform variate along the dimension
            U = self.uniform_((B, self.world.token_sizes[dim]), rng)

            # broadcast to the size of the full event space
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)

            # apply quantile function for Kumaraswamy(alpha, 1) and Kumaraswamy(1, alpha) and sum log factors
            F_src += U.log().div(alpha)
            F_tgt += (1 - U).log().div(alpha)
        return F_src, F_tgt
    
    def rate_prior(self, B: int, rng: torch.Generator = None):
        if self.objective.stratify:
            # stratification creates a grid of rates across the batch
            L = torch.linspace(self.epsilon, 1 - self.epsilon, B, device=self.device)
            # sample a random offset for the whole grid
            U = self.uniform_((1,), rng)
            # wrap the grid to stay within (0, 1)
            U = (L + U) % 1
        else:
            # sample a random rate for each batch element
            U = self.uniform_((B,), rng).clamp(self.epsilon, 1 - self.epsilon)
        # broadcast to event size
        U = einops.repeat(U, "b -> b n", n = self.world.num_tokens)
        # schedule 
        R_src = self.kumaraswamy_quantile(U, self.objective.src_alpha)
        R_tgt = self.kumaraswamy_quantile(1 - U, self.objective.tgt_alpha)
        dR = self.kumaraswamy_quantile_dt(U, self.objective.src_alpha)
        return R_src, R_tgt, dR
    
    def forward(self, B: int, rng: torch.Generator = None):
        B = B[0] if isinstance(B, tuple) else B

        # sample joint prior
        P_src, P_tgt = self.weight_prior(B, rng)

        # sample rates
        R_src, R_tgt, dR = self.rate_prior(B, rng)
        
        # binary topk selection
        src = self.binary_topk_(P_src, self.k_from_rates_(R_src))
        tgt = self.binary_topk_(P_tgt, self.k_from_rates_(R_tgt))
        return src, tgt, dR
