import torch
import einops
from utils.model import MaskedPredictor

class AnyOrder_RIN(torch.nn.Module):
    def __init__(self, model, world, device: torch.device, generator: torch.Generator = None):
        super().__init__()
        # Generator and device
        self.generator = generator
        self.device = device
        
        # Denoiser
        self.model = MaskedPredictor(model, world)

        # World
        self.world = world

    # SAMPLING 
    def log_dirichlet(self, shape: tuple, alpha: float, eps: float = 1e-7) -> torch.FloatTensor:
        alpha_tensor = torch.full(shape, alpha, device=self.device)
        return torch.log(torch._sample_dirichlet(alpha_tensor, generator=self.generator) + eps)
    
    def gumbel(self, shape: tuple, eps: float = 1e-7) -> torch.FloatTensor:
        return -torch.log(-torch.log(self.uniform(shape) + eps) + eps)
    
    def uniform(self, shape: tuple, eps: float = 0.) -> torch.FloatTensor:
        return torch.rand(shape, device=self.device, generator=self.generator) * (1 - 2 * eps) + eps

    # CATEGORICAL
    def binary_topk(self, weights: torch.FloatTensor, ks: torch.LongTensor) -> torch.BoolTensor:
        idx = weights.argsort(dim=-1, descending=True)
        rank = torch.arange(weights.size(-1), device=self.device).expand_as(weights)
        mask = torch.zeros_like(weights, dtype=torch.bool, device=self.device)
        mask.scatter_(dim = -1, index = idx, src= ks > rank) # True for top-k False otherwise
        return mask

    # SCHEDULES
    @staticmethod
    def cosine_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        rates = 1 - torch.cos(torch.pi * t / 2)
        weights = 0.5 * torch.pi * torch.sin(torch.pi * t / 2)
        return rates, weights

    @staticmethod
    def linear_schedule(t: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        rates = t
        weights = torch.ones_like(t)
        return rates, weights
    
    @staticmethod
    def minmax_schedule(t: torch.FloatTensor, a: float = 0., b: float = 1.) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        rates = a + (b - a) * t
        weights = torch.ones_like(t)
        return rates, weights

    # Forward
    def sample_permutation(self, prior: str) -> torch.FloatTensor:
        if prior == 'dirichlet': # weight frames via dirichlet prior
            W = self.log_dirichlet((self.world.batch_size, self.world.token_sizes['t']), alpha=0.5)
            lhs = "b t"
        elif prior == 'prefix': # ensure initial tau frames are always masked last
            W = torch.zeros((self.world.batch_size, self.world.token_sizes['t']), device=self.device)
            W[:, :self.world.tau] = float('-inf')
            lhs = "b t"
        else: # uniform prior over all permutations
            W = torch.zeros((self.world.batch_size, self.world.token_sizes['t']), device=self.device)
            lhs = "b t"
        W = einops.repeat(W, f'{lhs} -> b {self.world.flat_token_pattern}', **self.world.token_sizes, b = self.world.batch_size)
        G = self.gumbel((self.world.batch_size, self.world.num_tokens))
        return G + W
    
    def sample_timesteps(self, num_steps: int, schedule: str, stratify: bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # sample uniform random timesteps for all steps and batch elements
        t = self.uniform((num_steps, self.world.batch_size, 1))
        # optionally, we can apply batch stratification here
        if stratify:
            t = (torch.linspace(0, 1, self.world.batch_size)[None, :, None] + t) % 1
        # sort the sequence dimension s.t. earlier steps have higher masking rates
        t = t.sort(dim = 0).values
        # calculate rs and ws based on schedule
        if schedule == "cosine":
            rs, ws = self.cosine_schedule(t)
        elif schedule == 'minmax':
            rs, ws = self.minmax_schedule(t)
        else:
            rs, ws = self.linear_schedule(t)
        return rs, ws

    def sample_masks(self, num_steps: int, prior: str, schedule: str, stratify: bool = False) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        # determine permutation order
        prior = self.sample_permutation(prior = prior)
        # share prior across steps
        prior = einops.repeat(prior, 'b n -> s b n', s = num_steps)
        # sample random masking rates
        rates, weights = self.sample_timesteps(num_steps, schedule = schedule, stratify = stratify)
        # determine masks via gumbel-topk
        ks = (rates * self.world.num_tokens).long()
        masks = self.binary_topk(prior, ks)
        return masks, weights

    def sample_latent_noise(self, num_samples: int) -> torch.FloatTensor: # sample standard normal noise for latents
        return torch.randn((num_samples, 1, self.model.dim_noise), device = self.device, generator = self.generator)

    def forward(self, 
                tokens: torch.FloatTensor, 
                num_steps: int = 1, 
                num_ensemble: int = 1, 
                prior: str = 'dirichlet', 
                schedule: str = 'cosine'
                ) -> tuple[torch.FloatTensor, torch.BoolTensor, torch.FloatTensor]:
        # sample sequence of masks
        masks, weights = self.sample_masks(num_steps, prior = prior, schedule = schedule)

        # repeat for functional risk minimization
        interface = einops.repeat(tokens, "b n d -> (b e) n d", e = num_ensemble)
        ms = einops.repeat(masks, 's b n -> s (b e) n 1', e = num_ensemble)
    
        # iterate without gradient for self-conditioning
        latents = None
        with torch.no_grad():
            for s in range(num_steps - 1):
                noise = self.sample_latent_noise(interface.size(0))
                interface, latents = self.model(tokens = interface, mask = ms[s], latents = latents, noise = noise)
                
        # last step with gradient
        noise = self.sample_latent_noise(interface.size(0))
        interface, latents = self.model(tokens = interface, mask = ms[-1], latents = latents, noise = noise)

        # rearrange to ensemble form
        tokens = einops.rearrange(interface, "(b e) n d -> b n d e", e = num_ensemble)
        
        return tokens, masks[-1], weights[-1]