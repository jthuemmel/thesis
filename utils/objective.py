import torch
import einops
import math

from utils.model import MaskedPredictor
from utils.loss_fn import f_kernel_crps

class DiscreteDiffusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device('cpu')
        self.network = MaskedPredictor(cfg.network)

        # Config shapes
        self.batch_size = cfg.world.batch_size
        self.field_sizes = cfg.world.field_sizes
        self.patch_sizes = cfg.world.patch_sizes
        self.field_layout = tuple(self.field_sizes.keys())
        self.patch_layout = tuple(self.patch_sizes.keys())

        # derived sizes and shapes
        self.token_sizes = {ax: (self.field_sizes[ax] // self.patch_sizes[f'{ax*2}'])
                            for ax in self.field_layout}
        self.token_shape = tuple(self.token_sizes[ax] for ax in self.field_layout)
        self.num_tokens = math.prod(self.token_sizes[ax] for ax in self.field_layout)
        self.num_elements = math.prod(self.field_sizes[ax] for ax in self.field_layout)
        self.dim_tokens = math.prod(self.patch_sizes[ax] for ax in self.patch_layout)

        #einops patterns
        field = " ".join([f"({f} {p})" for f, p in zip(self.field_layout, self.patch_layout)])
        self.field_pattern = f"b {field}"
        self.flat_token_pattern = f"({' '.join(self.field_layout)})"
        self.flat_patch_pattern = f"({' '.join(self.patch_layout)})"
        self.flatland_pattern = f"b {self.flat_token_pattern} {self.flat_patch_pattern}"

        # Index tensors
        flatland_index = torch.arange(self.num_tokens, device=self.device).expand(self.batch_size, -1)
        self.register_buffer("flatland_index", flatland_index)
        token_index = torch.stack(torch.unravel_index(flatland_index, self.token_shape), dim=-1)
        self.register_buffer("token_index", token_index)

        # additional config attributes
        self.alphas = cfg.world.alphas
        self.tau = cfg.world.tau

        #check
        assert self.num_tokens * self.dim_tokens == self.num_elements, 'sus'
        assert token_index.shape == (self.batch_size, self.num_tokens, len(self.field_layout)), 'sus'

    ### TOKENIZATION
    def field_to_tokens(self, field):
        return einops.rearrange(field, f'{self.field_pattern} -> {self.flatland_pattern}', **self.patch_sizes)
    
    def tokens_to_field(self, patch):
        return einops.rearrange(patch, f"{self.flatland_pattern} ... -> {self.field_pattern} ...", **self.token_sizes, **self.patch_sizes)

    ### MASKING
    def gumbel_noise(self, shape: tuple):
        u = torch.rand(shape, device = self.device)
        return -torch.log(-torch.log(u))
            
    def dirichlet_marginal(self, ax: str):
        concentration = torch.full((self.batch_size, self.token_sizes[ax]), self.alphas[ax], device= self.device)
        probs = torch._sample_dirichlet(concentration)
        probs = einops.repeat(probs, f'b {ax} -> b {self.flat_token_pattern}', **self.token_sizes)
        return probs.log()
    
    def k_from_rates(self, rates):
        return (self.num_tokens * rates).long().clamp(1, self.num_tokens - 1).view(-1, 1)
            
    def binary_topk(self, weights, ks):
        index = weights.argsort(dim = -1, descending=True)
        topk = self.flatland_index < ks
        binary = torch.zeros_like(topk, dtype=torch.bool).scatter(1, index, topk)
        return binary
    
    ### PRIORS
    def get_visible_ws(self):
        G = [self.gumbel_noise((self.batch_size, self.num_tokens))]
        D = [self.dirichlet_marginal(ax) for ax in self.alphas.keys()]
        return einops.reduce(G + D, 'factors ... -> ...', 'sum')

    def get_history_ws(self):
        step = torch.zeros((self.token_sizes['t'],), device=self.device)
        step[:self._cfg.tau] = float('inf')
        return einops.repeat(step, f't -> b {self.flat_token_pattern}', **self.token_sizes, b=self.batch_size)
    
    def get_visible_ks(self):
        linear_grid = torch.linspace(0, 1, self.batch_size, device= self.device)
        u = torch.rand((1,), device = self.device)
        rates = (u + linear_grid) % 1 
        return self.k_from_rates(rates)
    
    def get_history_ks(self):
        rates = torch.full((self.batch_size,), self.tau / self.token_sizes['t'], device = self.device)
        return self.k_from_rates(rates)

    ### FORWARD
    def forward(self, data, land_sea_mask: torch.BoolTensor = None, mode: str = 'prior'):
        # tokens
        data = data.to(self.device)
        tokens = self.field_to_tokens(data)
        # masks
        ws = self.get_visible_ws() if mode == 'prior' else self.get_history_ws()  
        ks = self.get_visible_ks() if mode == 'prior' else self.get_history_ks() 
        visible = self.binary_topk(ws, ks)[..., None] # add singleton D dimension
        # predict
        pred = self.network(tokens, visible, self.token_index)
        # scoring rule
        ensemble = einops.rearrange(pred, '(b n) ... (d e) -> b ... d (n e)', d = tokens.size(-1), b = self.batch_size)
        score = f_kernel_crps(tokens, ensemble)
        # masked loss
        mask = ~visible if land_sea_mask is None else torch.logical_and(land_sea_mask, ~visible)
        rate_correction = einops.reduce(mask, 'b n d -> b 1 1', 'sum') / self.num_elements
        loss = (score * mask / rate_correction).mean()
        return loss
