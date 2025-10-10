import torch
import einops

from utils.components import exists
from utils.model import MaskedPredictor

class MaskedDiffusion(torch.nn.Module):
    def __init__(self, model, world, generator: torch.Generator = None):
        super().__init__()

        # Generator
        self.generator = generator
        
        # Denoiser
        self.model = MaskedPredictor(model, world)

        # 

    def noise_like(self, x):
         return torch.randn((x.size(0), 1, self.model.dim_noise), device = x.device, generator = self.generator)
    
    @staticmethod
    def repeat_as_ensemble(arg, E: int):
        return einops.repeat(arg, 'b ... -> (b e) ...', e = E)
    
    @staticmethod
    def batch_to_ensemble(arg, E: int):
         return einops.rearrange(arg, '(b e) ... -> b ... e', e = E)


    def forward(self, x0, S: int, E: int):


        return             
