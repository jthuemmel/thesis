import torch
import einops

class MaskingStrategy(torch.nn.Module):
    def __init__(self, world, schedule: str):
        super().__init__()
        self.world = world
        self.schedule = getattr(self, f"{schedule}_schedule", default = self.linear_schedule)

    def sample_prior(self, S: int, B: int, device: torch.device, **kwargs) -> torch.FloatTensor:
        raise NotImplementedError
    
    def sample_timesteps(self, S: int, B: int, device: torch.device, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        raise NotImplementedError

    def forward(self, S: int, B: int, device: torch.device, rng: torch.Generator = None, **kwargs) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        prior = self.sample_prior(S, B, device, rng = rng, **kwargs)
        rates, weights = self.sample_timesteps(S, B, device, rng = rng, **kwargs)
        masks = self.sample_masks(prior, rates, **kwargs)
        return masks, weights

    # SELECT
    @staticmethod
    def binary_topk(prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        ks = (prior.size(-1) * rates).long()
        idx = prior.argsort(dim=-1, descending=True)
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
        rates = 1 - torch.cos(torch.pi * t / 2)
        weights = 0.5 * torch.pi * torch.sin(torch.pi * t / 2)
        return rates, weights

    @staticmethod
    def linear_schedule(t: torch.FloatTensor, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        return t, torch.ones_like(t)
    
class DirichletMasking(MaskingStrategy):
    def __init__(self, world, alpha: float = 1.0, schedule: str = "cosine"):
        super().__init__(world, schedule = schedule)
        self.alpha = alpha

    def sample_prior(self, S: int, B: int, device: torch.device, rng: torch.Generator = None, **kwargs) -> torch.FloatTensor:
        N, T = self.world.num_tokens, self.world.token_sizes["t"]
        log_dirichlet = self.log_dirichlet((S, B, T), self.alpha, device, generator=rng)
        gumbel_noise = self.gumbel((S, B, N), device, generator=rng)
        prior = einops.repeat(log_dirichlet, f"s b t -> s b {self.world.flat_token_pattern}", n=N) + gumbel_noise
        return prior

    def sample_timesteps(self, S: int, B: int, device: torch.device, rng: torch.Generator = None, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        t = torch.rand((S, B, 1), device=device, generator=rng)
        t = t.sort(dim=0).values  # sort timesteps for progressive masking
        rates, weights = self.schedule(t)
        return rates, weights
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return self.binary_topk(prior, rates)
    

class ForecastMasking(MaskingStrategy):
    def __init__(self, world, schedule: str = "linear"):
        super().__init__(world, schedule = schedule)
        self.tau = self.world.tau

    def sample_prior(self, S: int, B: int, device: torch.device, **kwargs) -> torch.FloatTensor:
        prior = torch.arange(self.world.num_tokens, device=device).float()
        prior = einops.repeat(prior, "n -> s b n", s=S, b=B)
        return prior

    def sample_timesteps(self, S: int, B: int, device: torch.device, **kwargs) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        T = self.world.token_sizes["t"]
        start = (self.tau / T).float()
        t = torch.linspace(start, 1.0, steps=S, device=device)
        t = einops.repeat(t, "s -> s b ()", b=B)
        rates, weights = self.schedule(t) 
        return rates, weights
    
    def sample_masks(self, prior: torch.FloatTensor, rates: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        return self.binary_topk(prior, rates)
    