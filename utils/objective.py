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

    def normal(self, shape: tuple) -> torch.FloatTensor:
        return torch.randn(shape, device = self.device, generator = self.generator)
    
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

    # MASKING
    def sample_prior(self, S: int, prior: str) -> torch.FloatTensor:
        B, N, T = self.world.batch_size, self.world.num_tokens, self.world.token_sizes['t']
        if prior == 'dirichlet': # weight frames via dirichlet prior
            log_prob = self.log_dirichlet((B, T), alpha=0.5)
            lhs = "b t"
        elif prior == 'prefix': # ensure initial tau frames are always masked last
            log_prob = torch.zeros((B, T), device=self.device)
            log_prob[:, :self.world.tau] = float('-inf')
            lhs = "b t"
        else: # uniform prior over all permutations
            log_prob = torch.zeros((B, T), device=self.device)
            lhs = "b t"
        # expand to tokens and share across steps
        log_prob = einops.repeat(log_prob, f'{lhs} -> s b {self.world.flat_token_pattern}', **self.world.token_sizes, b = B, s = S)
        # add gumbel noise
        g = self.gumbel((1, B, N)) + log_prob
        return g
    
    def sample_timestep(self, S: int, stratify: bool = False) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        B = self.world.batch_size
        # sample uniform random timesteps for all steps and batch elements
        t = self.uniform((S, B, 1))
        # optionally, we can apply batch stratification here
        if stratify:
            t = (torch.linspace(0, 1, B)[None, :, None] + t) % 1
        # sort the sequence dimension s.t. earlier steps have higher masking rates
        t = t.sort(dim = 0).values
        return t
    
    def apply_schedule(self, t: torch.FloatTensor, schedule: str) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # calculate rs and ws based on schedule
        if schedule == "cosine":
            rs, ws = self.cosine_schedule(t)
        elif schedule == 'minmax':
            rs, ws = self.minmax_schedule(t)
        else:
            rs, ws = self.linear_schedule(t)
        # convert rates to number of tokens to mask
        ks = (rs * self.world.num_tokens).long()
        return ks, ws

    def get_masks(self, S: int, prior: str, schedule: str, stratify: bool = False) -> tuple[torch.BoolTensor, torch.FloatTensor]:
        # determine permutation order
        ps = self.sample_prior(S = S, prior = prior)
        # sample random denoising timesteps
        ts = self.sample_timestep(S = S, stratify = stratify)
        # get masking rates and weights
        ks, ws = self.apply_schedule(ts, schedule = schedule)
        # determine masks via topk
        ms = self.binary_topk(ps, ks)
        return ms, ws
    
    # FORWARD
    def forward(self, tokens: torch.FloatTensor, masks: torch.BoolTensor, S: int, E: int) -> torch.FloatTensor:
        # parallelise ensemble processing
        fs = self.normal(shape = (S, self.world.batch_size * E, 1, self.model.dim_noise)) # s (b e) 1 d
        xs = einops.repeat(tokens, "b n c -> (b e) n c", e = E)
        ms = einops.repeat(masks, 's b n -> s (b e) n 1', e = E)

        # iterate without gradient for self-conditioning
        zs = None
        with torch.no_grad():
            for s in range(S - 1):
                xs, zs = self.model(tokens = xs, mask = ms[s], latents = zs, noise = fs[s])
                
        # last step with gradient
        xs, zs = self.model(tokens = xs, mask = ms[-1], latents = zs, noise = fs[-1])

        # rearrange to ensemble form
        xs = einops.rearrange(xs, "(b e) n c -> b n c e", e = E)
        return xs