import torch
import einops
from utils.model import MaskedPredictor

class MaskedDiffusion(torch.nn.Module):
    def __init__(self, model, world, device: torch.Device, generator: torch.Generator = None):
        super().__init__()
        # Generator and device
        self.generator = generator
        self.device = device
        
        # Denoiser
        self.model = MaskedPredictor(model, world)

        # World
        self.world = world

    # CATEGORICAL SAMPLING 
    def dirichlet(self, shape: torch.Size, alpha: float) -> torch.FloatTensor:
        alpha_tensor = torch.full(shape, alpha, device=self.device)
        return torch._sample_dirichlet(alpha_tensor, generator=self.generator)

    def gumbel_noise(self, shape: torch.Size, eps: float = 1e-7) -> torch.FloatTensor:
        U = torch.rand(shape, device=self.device, generator=self.generator)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def binary_topk(self, weights: torch.FloatTensor, ks: torch.LongTensor) -> torch.BoolTensor:
        idx = weights.argsort(dim=-1, descending=True)
        rank = torch.arange(weights.size(-1), device=self.device).expand_as(weights)
        mask = torch.zeros_like(weights, dtype=torch.bool, device=self.device)
        mask.scatter_(dim = -1, index = idx, src= ks > rank)
        return mask

    def ks_from_rates(self, rates: torch.FloatTensor) -> torch.LongTensor:
        return (self.world.num_tokens * rates).long()

    # SCHEDULES
    @staticmethod
    def linear_schedule(t: torch.Tensor):
        return t
    
    @staticmethod
    def linear_weight(t: torch.Tensor):
        return torch.ones_like(t)

    @staticmethod
    def arcsine_schedule(t: torch.Tensor):
        return 0.5 - 0.5 * torch.cos(torch.pi * t)

    @staticmethod
    def arcsine_weight(t: torch.Tensor, eps: float = 1e-3):
        t_adj = t * (1 - 2*eps) + eps  # maps t ∈ [0,1] → [eps, 1-eps]
        return 0.5 * torch.pi * torch.sin(torch.pi * t_adj)

    @staticmethod
    def cosine_schedule(t: torch.Tensor):
        return 1 - torch.cos(torch.pi * t / 2)

    @staticmethod
    def cosine_weight(t: torch.Tensor, eps: float = 1e-3):
        t_adj = t * (1 - 2*eps) + eps  # maps t ∈ [0,1] → [eps, 1-eps]
        return 0.5 * torch.pi * torch.sin(torch.pi * t_adj / 2)
            
    # Forward
    def forward(self, batch):
        B, N = batch.size(0), batch.size(1)
        t = torch.rand(B, device=self.device)
        r, w = self.cosine_schedule(t), self.cosine_weight(t)
        G = self.gumbel_noise((B, N))
        D = self.dirichlet((B, T), alpha=0.5).log()
        ks = self.ks_from_rates(r).unsqueeze(-1)
        mask = self.binary_topk(G + D, ks)
        out = self.model(batch, mask)
        return out, w, mask