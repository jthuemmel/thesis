import torch
import einops

class Masking:
    def __init__(self, world_cfg, optim_cfg, device, generator=None):
        self.world = world_cfg
        self.optim_cfg = optim_cfg
        self.device = device
        self.generator = generator

    def get_timestep(self, timestep: str):
        if timestep == "uniform":
            t = self.single_timestep()
            lhs = 'b 1'
        elif timestep == "framewise":
            t = self.framewise_timestep()
            lhs = 'b t'
        elif timestep == 'history':
            t = self.history_timestep()
            lhs = 't'
        else:
            raise ValueError(f"Unknown timestep: {timestep}")
        return einops.repeat(t, f"{lhs} -> b {self.world.flat_token_pattern} ()", **self.world.token_sizes,b=self.optim_cfg.batch_size,)

    def get_schedule(self, t, schedule: str):
        if schedule == "cosine":
            rate = self.cosine_schedule(t)
            weight = self.cosine_weight(t)
        elif schedule == "arcsine":
            rate = self.arcsine_schedule(t)
            weight = self.arcsine_weight(t)
        elif schedule == "uniform":
            rate = t
            weight = 1.
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        return rate, weight
    
    def get_mask(self, rate, mask: str):
        if mask == "bernoulli":
            visible = torch.bernoulli(rate, generator=self.generator).bool()
        elif mask == "dirichlet_topk":
            permute = self.dirichlet_permute()
            ks = self.k_from_rates(rate)
            visible = self.binary_topk(permute, ks=ks)
        elif mask == "uniform_topk":
            permute = self.gumbel_noise((self.optim_cfg.batch_size, self.world.num_tokens))
            ks = self.k_from_rates(rate)
            visible = self.binary_topk(permute, ks=ks)
        else:
            raise ValueError(f"Unknown mask: {mask}")
        return visible

    def __call__(self,
                timestep: str = "uniform", 
                schedule: str = "cosine", 
                mask: str = "bernoulli"
                ):
        
        # pick timestep Float: (B, N, 1)
        t = self.get_timestep(timestep) 

        # pick schedule (elementwise) Float: (B, N, 1)
        rate, weight = self.get_schedule(t, schedule)

        # pick mask generator Boolean: (B, N, 1)
        visible = self.get_mask(rate, mask)

        return visible, weight

    # SAMPLING
    def uniform(self, shape: tuple):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def gumbel_noise(self, shape: tuple):
        return -torch.log(-torch.log(self.uniform(shape)))

    def dirichlet_marginal(self, ax: str):
        concentration = torch.full(
            (self.optim_cfg.batch_size, self.world.token_sizes[ax]),
            self.world.alphas[ax],
            device=self.device,
        )
        probs = torch._sample_dirichlet(concentration, generator=self.generator)
        return einops.repeat(
            probs,
            f"b {ax} -> b {self.world.flat_token_pattern}",
            **self.world.token_sizes,
            b=self.optim_cfg.batch_size,
        )

    # TIMESTEPS
    def stratification(self, t):
        return (t + torch.linspace(0, 1, self.optim_cfg.batch_size, device=self.device).view(-1, 1)) % 1

    def framewise_timestep(self):
        t = self.uniform((1, self.world.token_sizes["t"]))
        t = self.stratification(t)
        return t

    def single_timestep(self):
        t = self.uniform((1, 1))
        t = self.stratification(t)
        return t
    
    def history_timestep(self):
        t = torch.zeros((self.world.token_sizes["t"],), device=self.device)
        t[: self.world.tau] = 1.0
        return t
    
    # TOPK 
    def k_from_rates(self, rates):
        return (self.world.num_tokens * rates).long().clamp(1, self.world.num_tokens - 1)

    def binary_topk(self, weights, ks):
        index = weights.argsort(dim=-1, descending=True)
        pos = torch.arange(weights.size(-1), device=weights.device)
        index = einops.rearrange(index, "b n -> b n 1")
        ks = einops.rearrange(ks, "b n 1 -> b 1 1")
        pos = einops.rearrange(pos, "n -> 1 n 1")
        binary = torch.zeros_like(index, dtype=torch.bool, device=self.device).scatter(
            1, index, ks > pos
        )
        return binary

    # DIRICHLET
    def dirichlet_permute(self):
        G = self.gumbel_noise((self.optim_cfg.batch_size, self.world.num_tokens))
        D = einops.reduce(
            [self.dirichlet_marginal(ax).log() for ax in self.world.alphas.keys()],
            "factors ... -> ...",
            "sum",
        )
        return G + D

    # SCHEDULES
    @staticmethod
    def arcsine_schedule(t: torch.Tensor):
        return 0.5 - 0.5 * torch.cos(torch.pi * t)

    @staticmethod
    def arcsine_weight(t: torch.Tensor):
        return 0.5 * torch.pi * torch.sin(torch.pi * t)

    @staticmethod
    def cosine_schedule(t: torch.Tensor):
        return 1 - torch.cos(torch.pi * t / 2)

    @staticmethod
    def cosine_weight(t: torch.Tensor):
        return 0.5 * torch.pi * torch.sin(torch.pi * t / 2)


