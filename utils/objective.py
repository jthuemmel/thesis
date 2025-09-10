import torch
import einops

from utils.model import MaskedPredictor
from utils.loss_fn import f_kernel_crps

class DiscreteDiffusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device('cpu')
        self.world = cfg.world
        self.network = MaskedPredictor(cfg.network, cfg.world)

    ### MASKING
    def gumbel_noise(self, shape: tuple):
        u = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(u))
            
    def dirichlet_marginal(self, ax: str):
        concentration = torch.full((self.world.batch_size, self.world.token_sizes[ax]), self.world.alphas[ax], device=self.device)
        log_probs = torch._sample_dirichlet(concentration).log()
        return einops.repeat(log_probs, 
                             f'b {ax} -> b {self.world.flat_token_pattern}',
                             **self.world.token_sizes)
    
    def k_from_rates(self, rates):
        return (self.world.num_tokens * rates).long().clamp(1, self.world.num_tokens - 1)
            
    def binary_topk(self, weights, ks):
        index = weights.argsort(dim=-1, descending=True)
        ks = einops.rearrange(ks, 'b -> b ()')
        pos = einops.rearrange(torch.arange(weights.size(-1), device=weights.device), 'n -> () n')
        topk = ks > pos
        binary = torch.zeros_like(topk, dtype=torch.bool).scatter(1, index, topk)
        return binary
    
    ### PRIORS
    def get_visible_ws(self):
        G = [self.gumbel_noise((self.world.batch_size, self.world.num_tokens))]
        D = [self.dirichlet_marginal(ax) for ax in self.world.alphas.keys()]
        return einops.reduce(G + D, 'factors ... -> ...', 'sum')

    def get_history_ws(self):
        step = torch.zeros((self.world.token_sizes['t'],), device=self.device)
        step[:self.world.tau] = float('inf')
        return einops.repeat(step,
                             f't -> b {self.world.flat_token_pattern}',
                             **self.world.token_sizes,b=self.world.batch_size)
    
    def get_visible_ks(self):
        linear_grid = torch.linspace(0, 1, self.world.batch_size, device=self.device)
        u = torch.rand((1,), device=self.device)
        rates = (u + linear_grid) % 1
        return self.k_from_rates(rates)
    
    def get_history_ks(self):
        rates = torch.full((self.world.batch_size,), self.world.tau / self.world.token_sizes['t'],device=self.device)
        return self.k_from_rates(rates)

    ### FORWARD
    def forward(self, tokens: torch.Tensor, land_sea_mask: torch.BoolTensor = None, mode: str = 'prior'):
        # masks
        ws = self.get_visible_ws() if mode == 'prior' else self.get_history_ws()  
        ks = self.get_visible_ks() if mode == 'prior' else self.get_history_ks() 
        visible = self.binary_topk(ws, ks)[..., None]  # add singleton D dimension
        # predict
        prediction = self.network(tokens, visible)
        # scoring rule
        ensemble = einops.rearrange(prediction, '(b n) ... (d e) -> b ... d (n e)', b = tokens.size(0), d = tokens.size(-1))
        score = f_kernel_crps(tokens, ensemble)
        # masked loss
        mask = ~visible.expand_as(tokens) if land_sea_mask is None else torch.logical_and(land_sea_mask, ~visible)
        rate_correction = einops.reduce(mask, 'b n d -> b 1 1', 'sum') / self.world.num_elements
        loss = (score * mask / rate_correction).mean()
        return loss