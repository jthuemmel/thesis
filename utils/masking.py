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
            indices = mask.nonzero(as_tuple=True)[0]
            return indices.expand(*shape,-1)
        else:
            return mask.expand(*shape, -1).bool().logical_not()
        
        
class MultinomialMasking(torch.nn.Module):
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()
        # configs
        self.world = world
        self.objective = objective

        #schedule
        self.src_rates = StableKumaraswamy(c0=objective.c0_src, c1=objective.c1_src)
        self.tgt_rates = StableKumaraswamy(c0=objective.c0_tgt, c1=objective.c1_tgt)
        self.prior = StableKumaraswamy(c0=objective.c0_prior, c1= objective.c1_prior)

        # attributes
        self.k_min = default(objective.k_min, 1)
        self.k_max = default(objective.k_max, world.num_tokens)

        # Events
        assert all([d in world.flat_token_pattern for d in objective.event_dims]), 'event dims not in token pattern'
        self.register_buffer('events', torch.tensor([world.token_sizes[d] for d in objective.event_dims]).prod())
        self.event_pattern = f'({" ".join(objective.event_dims)})'

    def expand_events(self, *args):
        return einops.repeat(
            [*args],
            f'args ... {self.event_pattern} -> args ... {self.world.flat_token_pattern}', 
            **self.world.token_sizes
            )
    
    def k_from_rates(self, r: float):
        return int(self.k_min + (self.k_max - self.k_min) * r)

    def forward(self, shape: tuple, rng: torch.Generator = None):
        # sample prior
        t = torch.rand((*shape, self.events), device=self.events.device, generator= rng)
        t = t * (1.0 - 2.0 * self.prior.epsilon) + self.prior.epsilon
        psrc, ptgt = self.prior.quantile(t), self.prior.quantile(1 - t)
        psrc, ptgt = self.expand_events(psrc, ptgt)
        
        # sample rates
        r_src, _ = self.src_rates((1,), rng)
        r_tgt, _ = self.tgt_rates((1,), rng)

        # sample indices
        src = torch.multinomial(psrc, self.k_from_rates(r_src), generator=rng)
        tgt = torch.multinomial(ptgt, self.k_from_rates(r_tgt), generator = rng)

        binary = torch.zeros_like(psrc, dtype= torch.bool).scatter_(1, tgt, True)
        return src, tgt, binary, None