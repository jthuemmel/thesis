import torch
import einops

# BASE INTERFACE
class MaskingStrategy(torch.nn.Module):
    def __init__(self, world):
        super().__init__()
        self.world = world

    def sample_prior(self, S: int, device: torch.device) -> torch.FloatTensor:
        raise NotImplementedError
    
    def sample_timesteps(self, S: int, device: torch.device) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor) -> torch.BoolTensor:
        raise NotImplementedError

    def forward(self, S: int, device: torch.device, rng: torch.Generator = None) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        prior = self.sample_prior(S, device, rng = rng)
        rates, weights = self.sample_timesteps(S, device, rng = rng)
        masks = self.sample_masks(prior, rates)
        return masks, weights

    # SELECT
    @staticmethod
    def binary_topk(prior: torch.FloatTensor, rates: torch.FloatTensor) -> torch.BoolTensor:
        ks = (prior.size(-1) * rates).long() # rate -> 1 == #masked -> N
        idx = prior.argsort(dim=-1, descending=False)
        rank = torch.arange(prior.size(-1), device=prior.device).expand_as(prior)
        mask = torch.zeros_like(prior, dtype=torch.bool, device=prior.device)
        mask.scatter_(dim = -1, index = idx, src= ks > rank) # True for top-k False otherwise
        return mask
    
    # SAMPLING
    @staticmethod
    def log_dirichlet(shape: tuple, alpha: float, device: torch.device,  eps: float = 1e-7, rng: torch.Generator = None) -> torch.FloatTensor:
        return torch.log(torch._sample_dirichlet(torch.full(shape, alpha, device=device), generator=rng) + eps)
    
    @staticmethod
    def gumbel(shape: tuple, device: torch.device, eps: float = 1e-7, rng: torch.Generator = None) -> torch.FloatTensor:
        return -torch.log(-torch.log(torch.rand(shape, device=device, generator=rng) + eps) + eps)
    
    # SCHEDULES
    @staticmethod
    def arcsine_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        rates = torch.sin(torch.pi * t * 0.5)**2
        weights = torch.sin(torch.pi * t * 0.5) * torch.cos(torch.pi * t * 0.5) * torch.pi
        return rates, weights
    
    @staticmethod
    def sine_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        rates = torch.sin(torch.pi * t * 0.5)
        weights = torch.cos(torch.pi * t * 0.5) * torch.pi * 0.5
        return rates, weights

    @staticmethod
    def linear_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return t, torch.ones_like(t)
    
# FRAME-WISE DIRICHLET PRIOR    
class DirichletMasking(MaskingStrategy):
    def __init__(
            self,
            world, 
            alpha: float = 1.0, 
            schedule: str = "sine", 
            stratify: bool = True, 
            progressive: bool = True, 
            tmin: float = 0.01, 
            tmax: float = 0.99
            ):
        super().__init__(world)
        self.schedule = getattr(self, f"{schedule}_schedule")
        self.alpha = alpha
        self.tmin = tmin
        self.tmax = tmax
        self.stratify = stratify
        self.progressive = progressive

    def sample_prior(self, S: int, device: torch.device, rng: torch.Generator = None) -> torch.FloatTensor:
        B, N, T = self.world.batch_size, self.world.num_tokens, self.world.token_sizes["t"]
        log_dirichlet = self.log_dirichlet((B, T), self.alpha, device, generator=rng)
        gumbel_noise = self.gumbel((B, N), device, generator=rng)
        log_dirichlet = einops.repeat(log_dirichlet, f"b t -> b {self.world.flat_token_pattern}", **self.world.token_sizes)
        prior = log_dirichlet + gumbel_noise
        prior = einops.repeat(prior, 'b n -> s b n', s = S)
        return prior

    def sample_timesteps(self, S: int, device: torch.device, rng: torch.Generator = None) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        B = self.world.batch_size
        # add batch-level stratification for low-discrepancy sampling
        if self.stratify:
            t = torch.rand((S, 1), device=device, generator=rng)
            t = (t + torch.linspace(0, 1, B, device = device).view(1, -1)) % 1
        else:
            t = torch.rand((S, B), device=device, generator=rng)
        # repeat for all elements in a frame
        t = einops.repeat(t, f"s b -> s b {self.world.flat_token_pattern}", **self.world.token_sizes)
        # sort timesteps for progressive masking
        if self.progressive: 
            t = t.sort(dim=0, descending = True).values  
        t_adj = (self.tmax - self.tmin) * t + self.tmin  # maps t ∈ [0,1] → [a, b]
        rates, weights = self.schedule(t_adj)
        return rates, weights
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor) -> torch.BoolTensor:
        return self.binary_topk(prior, rates)

# FORECAST MASKING STRATEGIES    
class ForecastMasking(MaskingStrategy):
    def __init__(self, world, tau: int, schedule: str = "linear"):
        super().__init__(world)
        self.schedule = getattr(self, f"{schedule}_schedule")
        T = self.world.token_sizes["t"]
        N = self.world.num_tokens

        self.prefix_frames = tau
        self.frcst_frames = T - tau
        self.tokens_per_frame = N // T
        self.prefix_length = tau * self.tokens_per_frame
        self.frcst_length = N - self.prefix_length

    def sample_prior(self, S: int,device: torch.device, rng: torch.Generator = None) -> torch.FloatTensor:
        # base frame order repeated per token
        B = self.world.batch_size
        frames = torch.arange(self.frcst_frames, 0, -1, device=device).float()
        frames = einops.repeat(frames, 'f -> s b (f n)', s=S, b=B, n=self.tokens_per_frame)
        return frames

    def sample_timesteps(self, S: int,device: torch.device) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        t = torch.linspace(1.0, 1 / S, steps=S, device=device)
        t = einops.repeat(t, "s -> s b f", b=self.world.batch_size, f = self.frcst_length)
        rates, weights = self.schedule(t)
        return rates, weights

    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor) -> torch.BoolTensor:
        frcst_mask = self.binary_topk(prior, rates)
        prefix_mask = frcst_mask.new_zeros(frcst_mask.size(0), frcst_mask.size(1), self.prefix_length)
        return torch.cat([prefix_mask, frcst_mask], dim=-1)

# FRAME-WISE BERNOULLI PRIOR
class KumaraswamyMasking(torch.nn.Module):
    '''Kumaraswamy Masking strategy with numerical stable methods courtesy of Wasserman et al 2024'''
    def __init__(self, world: WorldConfig, c1: float, c0: float, event_dims: tuple, discretise: bool = False, progressive: bool = False, stratify: bool = False):
        super().__init__()
        
        assert c1 > 0. and c0 > 0., 'invalid concentration'
        assert all([d in world.flat_token_pattern for d in event_dims]), 'event dims not in token pattern'

        self.num_events = prod([world.token_sizes[d] for d in event_dims])
        self.event_pattern = f'({' '.join(event_dims)})'
        self.event_dims = event_dims
        self.world = world
        
        self.register_buffer("c1", torch.tensor(c1))
        self.register_buffer("c0", torch.tensor(c0))
        
        self.epsilon = 5e-3
        self.stratify = stratify
        self.discretise = discretise
        self.progressive = progressive
    
    @staticmethod
    def log1mexp(t: torch.FloatTensor):
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

    def quantile(self, t: torch.Tensor):
        # (1 - (1 - t)**(1 / c0))**(1 / c1)
        return torch.exp(self.log1mexp(torch.log1p(-t) / self.c0) / self.c1)
    
    def cdf(self, t: torch.Tensor):
        # 1 - (1 - t**c1)**c0
        return -torch.expm1(self.c0 * self.log1mexp(self.c1 * t.log()))

    def expand_events(self, t: torch.FloatTensor):
        return einops.repeat(t,
                             f'... {self.event_pattern} -> ... {self.world.flat_token_pattern}', 
                             **self.world.token_sizes
                             )
        

    def sample_prior(self, shape: tuple, rng: torch.Generator = None):
        if self.stratify:
            t = torch.rand(shape[:-1] + (1,) + (self.num_events,), device=self.c1.device, generator=rng)
            t = (t + torch.linspace(0, 1, shape[-1]).view(1, -1, 1)) % 1
        else:
            t = torch.rand(shape + (self.num_events,), device=self.c1.device, generator=rng)
        t = t * (1.0 - 2.0 * self.epsilon) + self.epsilon
        return t

    def sample_masks(self, rates: torch.Tensor, rng: torch.Generator = None):
        if self.discretise: rates = rates.round(decimals = 2)
        return torch.bernoulli(rates, generator = rng).bool()

    def forward(self, shape: tuple, rng: torch.Generator = None) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        t = self.sample_prior(shape, rng=rng)
        if self.progressive: t = t.sort(0, descending = True).values
        rates = self.expand_events(self.quantile(t))
        weights = self.expand_events(self.quantile_dt(t))
        masks = self.sample_masks(rates, rng=rng)
        return masks, weights
