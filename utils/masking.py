import torch
import einops

# BASE INTERFACE
class MaskingStrategy(torch.nn.Module):
    def __init__(self, world, schedule: str = 'linear'):
        super().__init__()
        self.world = world
        self.schedule = getattr(self, f"{schedule}_schedule")

    def sample_prior(self, S: int, device: torch.device, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError
    
    def sample_timesteps(self, S: int, device: torch.device, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        raise NotImplementedError

    def forward(self, S: int, device: torch.device, rng: torch.Generator = None, **kwargs) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        prior = self.sample_prior(S, device, rng = rng, **kwargs)
        rates, weights = self.sample_timesteps(S, device, rng = rng, **kwargs)
        masks = self.sample_masks(prior, rates, **kwargs)
        return masks, weights

    # SELECT
    @staticmethod
    def binary_topk(prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        ks = (prior.size(-1) * rates).long() # rate -> 1 == #masked -> N
        idx = prior.argsort(dim=-1, descending=False) # prior -> -inf = high likelihood of masking
        rank = torch.arange(prior.size(-1), device=prior.device).expand_as(prior)
        mask = torch.zeros_like(prior, dtype=torch.bool, device=prior.device)
        mask.scatter_(dim = -1, index = idx, src= ks > rank) # True for top-k False otherwise
        return mask
    
    # SAMPLING
    @staticmethod
    def log_dirichlet(shape: tuple, alpha: float, device: torch.device,  eps: float = 1e-7, rng: torch.Generator = None, **kwargs) -> torch.FloatTensor:
        return torch.log(torch._sample_dirichlet(torch.full(shape, alpha, device=device), generator=rng) + eps)
    
    @staticmethod
    def gumbel(shape: tuple, device: torch.device, eps: float = 1e-7, rng: torch.Generator = None, **kwargs) -> torch.FloatTensor:
        return -torch.log(-torch.log(torch.rand(shape, device=device, generator=rng) + eps) + eps)
    
    # SCHEDULES
    @staticmethod
    def cosine_schedule(t: torch.FloatTensor, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # t -> 1 = high rate, t -> 0 = low rate
        rates = torch.cos(torch.pi * (1 - t) / 2)
        weights = torch.sin(torch.pi * (1 - t) / 2) * torch.pi / 2
        return rates, weights

    @staticmethod
    def linear_schedule(t: torch.FloatTensor, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return t, torch.ones_like(t)
    
# FRAME-WISE DIRICHLET PRIOR    
class DirichletMasking(MaskingStrategy):
    def __init__(self, world, alpha: float = 1.0, schedule: str = "cosine", stratify: bool = True, progressive: bool = True):
        super().__init__(world, schedule = schedule)
        self.alpha = alpha
        self.stratify = stratify
        self.progressive = progressive

    def sample_prior(self, S: int, device: torch.device, rng: torch.Generator = None, **kwargs) -> torch.FloatTensor:
        B, N, T = self.world.batch_size, self.world.num_tokens, self.world.token_sizes["t"]
        log_dirichlet = self.log_dirichlet((1, B, T), self.alpha, device, generator=rng)
        gumbel_noise = self.gumbel((1, B, N), device, generator=rng)
        log_dirichlet = einops.repeat(log_dirichlet, f"1 b t -> 1 b {self.world.flat_token_pattern}", **self.world.token_sizes)
        prior = log_dirichlet + gumbel_noise
        return prior.expand(S, -1, -1)

    def sample_timesteps(self, S: int, device: torch.device, rng: torch.Generator = None, eps: float = 1e-2, **kwargs
                         ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        B, N = self.world.batch_size, self.world.num_tokens
        t = torch.rand((S, B, 1), device=device, generator=rng).expand(-1, -1, N)
        # add batch-level stratification for low-discrepancy sampling
        if self.stratify: 
            bs = torch.linspace(0, 1, B, device = device).view(1, -1, 1)
            t = (t + bs) % 1
        # sort timesteps for progressive masking
        if self.progressive: 
            t = t.sort(dim=0, descending = True).values  
        t_adj = t * (1 - 2*eps) + eps  # maps t ∈ [0,1] → [eps, 1-eps]
        rates, weights = self.schedule(t_adj)
        return rates, weights
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return self.binary_topk(prior, rates)

# FORECAST MASKING STRATEGIES    
class ForecastMasking(MaskingStrategy):
    def __init__(self, world, tau: int, schedule: str = "linear", noise_scale: float = 0.0):
        super().__init__(world, schedule=schedule)
        T = self.world.token_sizes["t"]
        N = self.world.num_tokens

        self.prefix_frames = tau
        self.frcst_frames = T - tau
        self.tokens_per_frame = N // T
        self.prefix_length = tau * self.tokens_per_frame
        self.frcst_length = N - self.prefix_length
        self.noise_scale = noise_scale

    def sample_prior(self, S: int,device: torch.device, rng: torch.Generator = None, **kwargs) -> torch.FloatTensor:
        # base frame order repeated per token
        B = self.world.batch_size
        frames = torch.arange(self.frcst_frames, 0, -1, device=device).float()
        frames = einops.repeat(frames, 'f -> s b (f n)', s=S, b=B, n=self.tokens_per_frame)

        if self.noise_scale > 0.0:
            noise = self.gumbel((S, B, self.frcst_length), device=device, generator=rng)
            frames = frames + self.noise_scale * noise

        return frames

    def sample_timesteps(self, S: int,device: torch.device, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        t = torch.linspace(1.0, 1 / S, steps=S, device=device)
        t = einops.repeat(t, "s -> s b f", b=self.world.batch_size, f = self.frcst_length)
        rates, weights = self.schedule(t)
        return rates, weights

    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        frcst_mask = self.binary_topk(prior, rates)
        prefix_mask = frcst_mask.new_zeros(frcst_mask.size(0), frcst_mask.size(1), self.prefix_length)
        return torch.cat([prefix_mask, frcst_mask], dim=-1)
