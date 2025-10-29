import torch
import einops
from utils.model import MaskedPredictor

class AnyOrder_RIN(torch.nn.Module):
    def __init__(self, model, world, device: torch.Device, generator: torch.Generator = None):
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

    @staticmethod
    def rate_to_k(r: torch.FloatTensor, N: int) -> torch.LongTensor:
        return (r * N).long()

    # SCHEDULES
    @staticmethod
    def cosine_schedule(t: torch.FloatTensor) -> torch.FloatTensor:
        return 1 - torch.cos(torch.pi * t / 2)

    @staticmethod
    def cosine_dt(t: torch.FloatTensor) -> torch.FloatTensor:
        return 0.5 * torch.pi * torch.sin(torch.pi * t / 2)

    # Forward
    def sample_permutation(self, prior: str):
        if prior == 'dirichlet':
            W = self.log_dirichlet((self.world.batch_size, self.world.token_sizes['t']), alpha=0.5)
            lhs = "b t"
        elif prior == 'history':
            W = torch.zeros((self.world.batch_size, self.world.token_sizes['t']), device=self.device)
            W[:, :self.world.tau] = float('-inf') # ensure these are always masked last
            lhs = "b t"
        else:
            W = torch.zeros((self.world.batch_size, self.world.token_sizes['t']), device=self.device)
            lhs = "b t"
        W = einops.repeat(W, f'{lhs} -> b {self.world.flat_token_pattern}', **self.world.token_sizes, b = self.world.batch_size)
        G = self.gumbel((self.world.batch_size, self.world.num_tokens))
        return G + W
    
    def sample_masks(self, num_steps: int, prior: int, schedule: str):
        # determine permutation order
        prior = self.sample_permutation(prior = prior)
        prior = einops.repeat(prior, 'b n -> s b n', s = num_steps)
        # sample random timesteps for the whole sequence and all batch elements
        t = torch.rand((num_steps, self.world.batch_size, 1), generator = self.generator, device = self.device)
        # optionally, we can apply stratification here without breaking the prior
        # sort the sequence dimension s.t. earlier steps have higher masking rates
        t = t.sort(dim = 0).values
        # calculate k and w
        ks = (self.cosine_schedule(t) * self.world.num_tokens).long() if schedule == "cosine" else (t * self.world.num_tokens).long()
        ws = self.cosine_dt(t) if schedule == "cosine" else torch.ones_like(t)
        # determine masks
        masks = self.binary_topk(prior, ks)
        return masks, ws

    def sample_functional(self):
        return torch.randn((self.world.batch_size, self.world.num_tokens, self.model.dim_noise), device = self.device, generator = self.generator)

    def forward(self, tokens, num_steps: int = 1, num_ensemble: int = 1) -> torch.FloatTensor:
        # sample sequence of masks
        masks, weights = self.sample_masks(num_steps)

        # repeat for functional risk minimization
        tokens = einops.repeat(tokens, "b n d -> (b e) n d", e = num_ensemble)
        masks = einops.repeat(masks, 's b n -> s (b e) n d', e = num_ensemble, d = self.world.dim_tokens)
    
        # iterate without gradient for self-conditioning
        latents = None
        with torch.no_grad():
            for s in range(num_steps - 1):
                noise = self.sample_functional()
                tokens, latents = self.model(tokens = tokens, mask = masks[s], latents = latents, noise = noise)
                
        # last step with gradient
        noise = self.sample_functional()
        tokens, latents = self.model(tokens = tokens, mask = masks[-1], latents = latents, noise = noise)

        # rearrange to ensemble form
        tokens = einops.rearrange(tokens, "(b e) n d -> b n d e", e = num_ensemble)
        masks = einops.rearrange(masks, 's (b e) n d -> s b n d e', e = num_ensemble)
        return tokens, masks, weights