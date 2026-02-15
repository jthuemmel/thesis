import torch
import einops
from utils.config import WorldConfig, ObjectiveConfig, default
    
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
        mask = einops.repeat(mask, f'{self.event_pattern} -> {self.world.flat_token_pattern}', **self.world.token_sizes)
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
    def device(self): return self.epsilon.device

    @staticmethod
    def binary_topk_(P: torch.FloatTensor, K: torch.LongTensor) -> torch.BoolTensor:
        # sort indices according to prior
        idx = P.argsort(dim=-1, descending=True)
        # rank indices for top-k selection
        rank = idx.argsort(dim=-1)
        return K > rank
    
    @staticmethod
    def sine_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return torch.sin(torch.pi * t * 0.5), torch.cos(torch.pi * t * 0.5) * torch.pi * 0.5
    
    @staticmethod
    def cosine_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return 1 - torch.cos(torch.pi * t * 0.5), torch.sin(torch.pi * t * 0.5) * torch.pi * 0.5
    
    @staticmethod
    def linear_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return t, torch.ones_like(t)
    
    # can add further schedules here
    
    def k_from_rates_(self, rates: torch.FloatTensor):
        return rates.mul(self.world.num_tokens).long()
    
    def uniform_(self, shape: tuple, rng: torch.Generator):
        return torch.rand(*shape, device = self.device, generator=rng).clamp(self.epsilon, 1 - self.epsilon)

    def weight_prior(self, B: int, rng: torch.Generator = None):
        # baseline factors are log-uniform -> Kumaraswamy(1, 1)
        F_src, F_tgt = self.uniform_((2, B, self.world.num_tokens), rng).log().div(1.)
        
        # for each event dimension
        for dim, alpha in self.objective.event_dims.items():
            # sample uniform variate along the dimension
            U = self.uniform_((B, self.world.token_sizes[dim]), rng).sort(dim=-1).values

            # broadcast to the size of the full event space
            U = einops.repeat(U, f'b {dim} -> b {self.world.flat_token_pattern}', **self.world.token_sizes)

            # apply quantile function for Kumaraswamy(alpha, 1) and Kumaraswamy(1, alpha) and sum log factors
            F_src += U.log().div(alpha)
            F_tgt += (1 - U).log().div(alpha)

        return F_src, F_tgt
    
    def rate_prior(self, B: int, rng: torch.Generator = None):
        if self.objective.stratify:
            # stratification creates a grid of rates across the batch
            L = torch.linspace(self.epsilon, 1 - self.epsilon, B, device=self.device).view(1, B)
            # sample a random offset for the whole grid
            U = self.uniform_((2, 1), rng)
            # wrap the grid to stay within (0, 1)
            U = (L + U) % 1
        else:
            # sample a random rate for each batch element
            U = self.uniform_((2, B), rng)
        # broadcast to event size
        U = einops.repeat(U, "two b -> two b n", n = self.world.num_tokens)
        # select schedule fn
        src_schedule = getattr(self, f'{self.objective.src_schedule}_schedule')
        tgt_schedule = getattr(self, f'{self.objective.tgt_schedule}_schedule')
        # apply schedule fn 
        R_src, dR = src_schedule(U[0])
        R_tgt, _ = tgt_schedule(U[1])
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