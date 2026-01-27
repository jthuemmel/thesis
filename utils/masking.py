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
            return mask.expand(*shape, -1).bool().logical_not(), mask.expand(*shape, -1).bool()
        
        
class MultinomialMasking(torch.nn.Module):
    def __init__(self, world: WorldConfig, objective: ObjectiveConfig):
        super().__init__()
        # configs
        self.world = world
        self.objective = objective

        # Events
        assert all([d in world.flat_token_pattern for d in objective.event_dims]), 'event dims not in token pattern'
        self.num_events = torch.tensor([world.token_sizes[d] for d in objective.event_dims]).prod()
        self.event_pattern = f'({" ".join(objective.event_dims)})'
        self.event_sizes = {d: world.token_sizes[d] for d in objective.event_dims}

        #schedule
        self.src_rates = StableKumaraswamy(c0=objective.c0_src, c1=objective.c1_src)
        self.tgt_rates = StableKumaraswamy(c0=objective.c0_tgt, c1=objective.c1_tgt)
        self.prior = StableKumaraswamy(c0= objective.c0_prior, c1= objective.c1_prior, epsilon=1e-2)
        
        # attributes
        self.k_min = default(objective.k_min, 2)
        self.k_max = default(objective.k_max, world.num_tokens)

    def expand_events(self, *args):
        return einops.repeat(
            [*args],
            f'args ... {self.event_pattern} -> args ... {self.world.flat_token_pattern}', 
            **self.world.token_sizes
            )
    
    def k_from_rates(self, r: float):
        return int(self.k_min + (self.k_max - self.k_min) * r)

    def _compute_event_weights(self, shape: tuple, rng: torch.Generator = None) -> torch.Tensor:
            """Generates factorized probability weights for source and target masking."""
            device = self.prior.c1.device
            
            # Generate independent factors for each event dimension
            dim_factors = [
                einops.repeat(
                    torch.rand((*shape, s), device=device, generator=rng),
                    f"... {d} -> ... {self.event_pattern}",
                    **self.event_sizes
                )
                for d, s in self.event_sizes.items()
            ]
            
            # Multiply factors to get joint probabilities
            # Source uses the factors directly; target uses the complements
            stacked_factors = torch.stack(dim_factors)
            src_weights = stacked_factors.prod(dim=0)
            tgt_weights = (1 - stacked_factors).prod(dim=0)
            
            return src_weights, tgt_weights

    def forward(self, shape: tuple, rng: torch.Generator = None):
        assert len(shape) == 1, "Currently only supports 1D batch shapes."
        
        src_w, tgt_w = self._compute_event_weights((1,) if self.objective.stratify else shape, rng)

        if self.objective.stratify:
            # Shift the base weights across the batch to ensure uniform coverage
            offsets = torch.linspace(0, 1, shape[0], device=src_w.device).view(-1, 1)
            src_w = (src_w + offsets) % 1
            tgt_w = (tgt_w + offsets) % 1

        src_probs, tgt_probs = self.expand_events(
             self.prior.quantile(src_w), self.prior.quantile(tgt_w)
             )

        k_src = self.k_from_rates(self.src_rates((1,), rng)[0])
        k_tgt = self.k_from_rates(self.tgt_rates((1,), rng)[0])

        src_indices = torch.multinomial(src_probs, k_src, generator=rng)
        tgt_indices = torch.multinomial(tgt_probs, k_tgt, generator=rng)

        src_binary = torch.zeros_like(src_probs, dtype= torch.bool).scatter_(1, src_indices, True)
        tgt_binary = torch.zeros_like(src_probs, dtype= torch.bool).scatter_(1, tgt_indices, True)

        return src_indices, None, tgt_binary, None