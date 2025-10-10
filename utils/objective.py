import torch
import einops
from utils.model import MaskedPredictor

class MaskedDiffusion(torch.nn.Module):
    def __init__(self, model, world, generator: torch.Generator = None):
        super().__init__()
        # Generator
        self.generator = generator
        
        # Denoiser
        self.model = MaskedPredictor(model, world)
            
    def forward(self, x0: torch.FloatTensor, schedule: list, E: int, lsm: torch.BoolTensor):
        out = []
        z = None
        x = einops.repeat(x0, 'b ... -> (b e) ...', e = E)
        lsm = einops.repeat(lsm, 'b ... -> (b e) ...', e = E)

        for s in schedule:
            noise = torch.randn((x.size(0), 1, self.model.dim_noise), device = x.device, generator = self.generator)

            m = torch.bernoulli(1 - s, generator=self.generator).bool() # s is visibility rate
            m = einops.repeat(m, 'b ... -> (b e) ...', e = E)

            xs, zs = self.model(tokens = x, mask = m, latents = z, noise = noise)
            
            fm = torch.logical_and(m, lsm) #masked and sea
            xs = torch.where(fm, xs, x) # copy unmasked and land

            out.append((
                einops.rearrange(xs, '(b e) ... -> b ... e', e = E),
                einops.rearrange(fm, '(b e) ... -> b ... e', e = E)
                ))

            z = zs.detach()
            x = xs.detach()
        
        return out
