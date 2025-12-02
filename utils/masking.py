import torch
import einops
from math import prod
from utils.config import WorldConfig, ObjectiveConfig
    
class ForecastMasking(torch.nn.Module):
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()
        self.world = world
        self.objective = objective
        self.event_pattern = 't'
        self.register_buffer("prefix_frames", torch.tensor(objective.tau, dtype = torch.long))
        self.register_buffer("total_frames", torch.tensor(world.token_sizes["t"], dtype = torch.long))
    
    def forward(self, shape: tuple, return_indices: bool = True):
        mask = torch.zeros((self.total_frames,), device = self.total_frames.device)
        mask[:self.prefix_frames] = 1
        mask = einops.repeat(mask, f'{self.event_pattern} -> {self.world.flat_token_pattern}', **self.world.token_sizes)
        if return_indices:
            indices = mask.nonzero(as_tuple=True)[0]
            return indices.expand(*shape,-1)
        else:
            return mask.expand(*shape, -1).bool().logical_not()
    
# KUMARASWAMY PRIOR
class KumaraswamySchedule(torch.nn.Module):
    '''Kumaraswamy Schedule with numerical stable methods courtesy of Wasserman et al 2024'''
    def __init__(self, objective: ObjectiveConfig):
        super().__init__()
        assert objective.c1 > 0. and objective.c0 > 0., 'invalid concentration'
       
        # Register hyperparameters as buffers for device consistency
        self.register_buffer("c1", torch.tensor(objective.c1, requires_grad=False))
        self.register_buffer("c0", torch.tensor(objective.c0, requires_grad=False))
        
        # Epsilon ensures 't' is not exactly 0 or 1, which causes numerical instability
        self.epsilon = objective.epsilon
    
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
    
    def forward(self, t: torch.Tensor):
        # ensure t is in [eps, 1 - eps]
        t = t * (1.0 - 2.0 * self.epsilon) + self.epsilon
        # shared terms
        log_1_minus_t = torch.log1p(-t)
        log_exp_inner = self.log1mexp(log_1_minus_t / self.c0) / self.c1
        # individual terms
        log_constant = -self.c1.log() - self.c0.log()
        log_outer = ((1 - self.c0) / self.c0) * log_1_minus_t
        log_inner = (1 - self.c1) * log_exp_inner
        # combine
        quantile = torch.exp(log_exp_inner)
        quantile_dt = torch.exp(log_constant + log_inner + log_outer)
        return quantile, quantile_dt

# MASKING STRATEGIES
class BernoulliMasking(torch.nn.Module):
    '''BernoulliMasking strategy'''
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()
        # configs
        self.world = world
        self.objective = objective

        # schedule
        self.schedule = KumaraswamySchedule(objective)
        
        # Events
        assert all([d in world.flat_token_pattern for d in objective.event_dims]), 'event dims not in token pattern'
        self.num_events = prod([world.token_sizes[d] for d in objective.event_dims])
        self.event_pattern = f'({" ".join(objective.event_dims)})'
        
    # Masking
    def sample_prior(self, shape: tuple, rng: torch.Generator = None):
        if self.objective.stratify:
            t = torch.rand(shape[:-1] + (1,) + (self.num_events,), device=self.c1.device, generator=rng)
            t = (t + torch.linspace(0, 1, shape[-1], device=self.c1.device).view(-1, 1)) % 1
        else:
            t = torch.rand(shape + (self.num_events,), device=self.c1.device, generator=rng)
        t = t * (1.0 - 2.0 * self.epsilon) + self.epsilon
        return t

    def expand_events(self, *args):
        return einops.repeat(
            [*args],
            f'args ... {self.event_pattern} -> args ... {self.world.flat_token_pattern}', 
            **self.world.token_sizes
            )

    def sample_masks(self, rates: torch.Tensor, rng: torch.Generator = None):
        return rates > torch.rand(rates.shape, device=self.c1.device, generator=rng)

    # Forward
    def forward(self, shape: tuple, rng: torch.Generator = None) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        t = self.sample_prior(shape, rng=rng)
        if self.objective.progressive: t = t.sort(0, descending = True).values
        if self.objective.discretise: t = t.round(decimals=2)
        rates, weights = self.schedule(t)
        rates, weights = self.expand_events(rates, weights)
        masks = self.sample_masks(rates, rng=rng)
        return masks, weights
    
class DirichletMasking(torch.nn.Module):
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()
        # configs
        self.world = world
        self.objective = objective

        #schedule
        self.schedule = KumaraswamySchedule(objective=objective)

        # attributes
        self.k_min = objective.k_min
        self.k_max = objective.k_max

        # Events
        assert all([d in world.flat_token_pattern for d in objective.event_dims]), 'event dims not in token pattern'
        self.num_events = prod([world.token_sizes[d] for d in objective.event_dims])
        self.event_pattern = f'({" ".join(objective.event_dims)})'

        # alpha
        self.register_buffer("alpha", torch.full((self.num_events,), objective.alpha, requires_grad=False))
    

    def sample_dirichlet(self, shape: tuple, rng: torch.Generator = None):
        dirichlet = torch._sample_dirichlet(self.alpha.expand(*shape, -1), generator=rng)
        return einops.repeat(dirichlet, 
                             f'... {self.event_pattern} -> ... {self.world.flat_token_pattern}', 
                             **self.world.token_sizes)
    
    def sample_rate(self, rng: torch.Generator = None):
        t = torch.rand((1,), device= self.alpha.device, generator=rng)
        t, w = self.schedule(t)
        k = self.k_min + (self.k_max - self.k_min) * t
        return k, w
    
    def forward(self, shape: tuple, rng: torch.Generator = None):
        prior = self.sample_dirichlet(shape, rng)
        k, w = self.sample_rate(rng)
        src = torch.multinomial(prior, int(k), generator=rng)
        mask = torch.ones_like(prior, dtype= torch.bool).scatter_(1, src, False)
        return src, mask, w