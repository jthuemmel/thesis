
import torch
import os
import argparse
import math

from dataclasses import replace
from typing import Callable
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import utils.config as cfg

from utils.loss_fn import f_kernel_crps, f_gaussian_ignorance
from utils.field_network import StochasticWeatherField

from utils.dataset import NinoData, MultifileNinoDataset
from utils.trainer import DistributedTrainer

### HELPER FUNCTIONS
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#### MASKING
class MaskingMixin:
    @property
    def frcst_src_prior(self):
        tau = self.world_cfg.tau
        prior = torch.ones(self.token_sizes["t"], device = self.device)
        prior[tau:] = 0
        return {"t": lambda: prior}
    
    @property
    def frcst_tgt_prior(self):
        tau = self.world_cfg.tau
        prior = torch.ones(self.token_sizes["t"], device = self.device)
        prior[:tau] = 0
        return {"t": lambda: prior}

    @property
    def src_priors(self):
        '''Creates custom priors for any combination of dimensions. Keys must reflect the correct dimensions.'''
        return {
            "bt": lambda: self.sample_dirichlet((self.token_sizes['b'], self.token_sizes["t"]), sharpness=self.world_cfg.alpha),
            "bv": lambda: self.sample_dirichlet((self.token_sizes['b'], self.token_sizes["v"])),
        }
    
    @property
    def tgt_priors(self):
        return {
            "bt": lambda: self.sample_dirichlet((self.token_sizes['b'], self.token_sizes["t"]), sharpness=self.world_cfg.alpha),
            "bv": lambda: self.sample_dirichlet((self.token_sizes['b'], self.token_sizes["v"]), weights= self.per_variable_weights),
        }

    @property
    def uniform_priors(self):
        '''Creates a uniform prior for each dimension in the token set.'''
        return {k: lambda K= K: torch.ones((K,), device=self.device) for k, K in self.token_sizes.items()}

    def get_masks(self, task: str):
        return
    
    def get_src_mask(self):
        rate = self.sample_masking_rates(self.world_cfg.num_tokens, **self.world_cfg.mask_rates_src)
        prior = self.src_priors()
        uniform = self.uniform_priors()
        prior.update(uniform)
        joint = self.compose_einsum_prior(prior, self.flatland_token_layout)
        return self.sample_multinomial(joint, rate)

    def get_masking_rate(self, N: int, **kwargs):
        rate = self.sample_trunc_normal(**kwargs)
        return int(N * rate.item())

    @staticmethod    
    def sample_topk(weights: torch.Tensor, k: int):
        return weights.topk(k, sorted= False).indices

    def sample_dirichlet(self, shape: tuple, weights: torch.Tensor | float = 1.0, sharpness: float = 1.0) -> torch.Tensor:
        w = torch.ones(shape, device= self.device) * torch.as_tensor(weights, device= self.device)
        alpha = w.softmax(-1) * sharpness
        return torch._sample_dirichlet(alpha, generator=self.generator)

    def sample_multinomial(self, weight: torch.Tensor, rate: int):
        '''Samples from a multinomial distribution parameterized by the outer product of priors and rate'''
        return torch.multinomial(weight, rate, replacement=False, generator=self.generator)

    def sample_trunc_normal(self, mean, std, a, b):
        rate = torch.empty((1,), device = self.device)
        if std > 0:
            return torch.nn.init.trunc_normal_(rate, mean=mean,std=std,a=a, b=b, generator=self.generator)
        return rate.fill_(mean)
        
    @staticmethod
    def compose_einsum_prior(prior_registry: dict[Callable], layout: list) -> torch.Tensor:
        '''Computes the outer product over a set of priors obtained from the registry and contracts it to the layout.'''
        # read the registry
        dims, einsum_args = [], []
        for d, fn in prior_registry.items():
            dims.append(d)
            einsum_args.append(fn())

        # dynamically create patterns
        einsum_lhs = ",".join(dims)
        layout_str = " ".join(layout) # the whitespace is important here
        has_batch = any("b" in x for x in einsum_lhs)
        einsum_rhs = "b"+" "+ layout_str if has_batch else layout_str # the whitespace is important here
        flat_shape = f"b ({layout_str})" if has_batch else f"({layout_str})"
        
        #compute outer product and flatten
        prior = torch.einsum(f"{einsum_lhs}->{einsum_rhs}", *einsum_args)
        flat = rearrange(prior, f"{einsum_rhs} -> {flat_shape}")
        return flat

    def apply_masks(self, tokens, src_mask, tgt_mask):
        _, _, D = tokens.size()
        # src masks
        expanded_src_mask = repeat(src_mask, 'b n -> b n d', d = D)
        src = tokens.gather(1, expanded_src_mask)
        # tgt masks
        expanded_tgt_mask = repeat(tgt_mask, 'b n -> b n d', d = D)        
        tgt = tokens.gather(1, expanded_tgt_mask)
        lsm = self.land_sea_mask.gather(1, expanded_tgt_mask)
        return src, tgt, lsm

### TOKEN PROPERTIES
class TokenMixin:
    @property
    def flatland_pattern(self):
        return 'b (t v h w) (tt hh ww) ...'
    
    @property
    def lsm_pattern(self):
        return '1 (h hh) (w ww)'

    @property
    def field_pattern(self):
        return 'b (v vv) (t tt) (h hh) (w ww) ...'    

    @property
    def spatial_only_pattern(self):
        return '(... b v vv t tt) (h hh) (w ww)'

    @property
    def flatland_token_layout(self):
        # returns the characters the first group in the flatland pattern (e.g. (t h w) -> ['t', 'h', 'w'])
        return self.parse_pattern_groups(self.flatland_pattern)[0]
    
    @property
    def flatland_patch_layout(self):
        return self.parse_pattern_groups(self.flatland_pattern)[1]

    @property
    def field_sizes(self):
        return {
            'b': self.batch_size,
            'v': len(self.data_cfg.variables),
            't': self.data_cfg.sequence_length,
            'h': self.data_cfg.grid_size["lat"],
            'w': self.data_cfg.grid_size["lon"]
        }
    
    @property
    def patch_sizes(self):
        # Return patch sizes per axis (default to 1)
        return {
            k: self.world_cfg.patch_size.get(k, 1)
            for k in ["tt", "vv", "hh", "ww"]
        }
    
    @property
    def token_sizes(self):
        # sizes after patching
        return {
            k: self.field_sizes[k] // self.patch_sizes.get(f"{k*2}", 1)
            for k in ["b", "t", "v", "h", "w"]
        }
    
    @staticmethod
    def parse_pattern_groups(pattern):
        """
        Parses all parenthesized groups in the flatland pattern.
        E.g., '(t v h w) (tt vv hh ww)' → [['t', 'v', 'h', 'w'], ['tt', 'vv', 'hh', 'ww']]
        """
        from re import findall
        groups = findall(r"\(([^)]+)\)", pattern)
        return [group.strip().split() for group in groups]

    # SHAPES
    def get_flatland_shape(self):
        B = self.batch_size
        N = math.prod([self.token_sizes[t] for t in self.flatland_token_layout])
        D = math.prod([self.patch_sizes[p] for p in self.flatland_patch_layout])
        return B, N, D
    
    def get_flatland_index(self):
        B, N, _ = self.get_flatland_shape()
        return repeat(torch.arange(N, device = self.device), "n -> b n", b = B)
    
    # TOKENIZERS
    def field_to_tokens(self, field):
        return rearrange(field, f'{self.field_pattern} -> {self.flatland_pattern}', **self.field_sizes, **self.patch_sizes)
    
    def tokens_to_field(self, tokens):
        return rearrange(tokens, f'{self.flatland_pattern} -> {self.field_pattern}', **self.field_sizes, **self.patch_sizes)

    def tokens_to_spatial(self, tokens):
        return rearrange(tokens, f"{self.field_pattern} -> {self.spatial_only_pattern}", **self.patch_sizes)

    # COORDINATE TRANSFORMS
    @staticmethod
    def compute_strides(sizes: dict, layout: list):
        strides = {}
        stride = 1
        for ax in reversed(layout):
            strides[ax] = stride
            stride *= sizes[ax]
        return strides
    
    def index_to_coords(self, idx_flat: torch.LongTensor, sizes: dict = None, layout: list = None):
        sizes = sizes if exists(sizes) else self.token_sizes
        layout = layout if exists(layout) else self.flatland_token_layout
        strides = self.compute_strides(sizes, layout)
        rem = idx_flat
        coords = []
        for ax in layout:
            val = rem.div(strides[ax], rounding_mode="floor")
            rem = rem.fmod(strides[ax])
            coords.append(val)
        return torch.stack(coords, dim=-1) 

    def coords_to_index(self, coords: torch.LongTensor, sizes: dict = None, layout: list = None):
        sizes = sizes if exists(sizes) else self.token_sizes
        layout = layout if exists(layout) else self.flatland_token_layout
        strides = self.compute_strides(sizes, layout)
        # Compute dot product over coordinate axis
        idx = sum(coords[..., i] * strides[layout[i]] for i in range(len(layout)))
        return idx 
    
### ENSO DATA
class OceanMixin:
    @property
    def land_sea_mask(self):
        lsm = self._train_lsm if self.mode == "train" else self._val_lsm
        return repeat(lsm, f"{self.lsm_pattern} -> {self.flatland_pattern}", 
                      **self.field_sizes, **self.patch_sizes, b = self.batch_size)
    
    def lens_data(self):
        if not hasattr(self, "_lens_data"):
            lens_config = replace(self.data_cfg, stats = default(self.data_cfg.stats, cfg.LENS_STATS))
            self._lens_data = MultifileNinoDataset(self.cfg.lens_path, lens_config, self.rank, self.world_size)
        return self._lens_data       

    def godas_data(self):
        if not hasattr(self, "_godas_data"):
            self._godas_data = NinoData(self.cfg.godas_path, self.data_cfg)
        return self._godas_data

    def picontrol_data(self):
        if not hasattr(self, "_picontrol_data"):
            picontrol_config = replace(self.data_cfg, time_slice = {"start": "0700", "stop": "0900", "step": None})
            self._picontrol_data = NinoData(self.cfg.picontrol_path, picontrol_config)
        return self._picontrol_data

    def oras5_data(self):
        if not hasattr(self, "_oras5_data"):
            self._oras5_data = NinoData(self.cfg.oras5_path, self.data_cfg)
        return self._oras5_data

### METRICS
class MetricsMixin:
    def compute_rapsd(self, tokens: torch.Tensor):
        x = self.tokens_to_spatial(tokens)
        with torch.autocast(device_type=self.device_type, enabled=False):
            rapsd = self.rapsd(x.float())
            rapsd = torch.clamp(rapsd, min=1e-8).log10()
        return rapsd

    @staticmethod
    def compute_acc(pred: torch.Tensor, obs: torch.Tensor):
        numerator = (pred * obs).nansum()
        denominator = (pred.pow(2).nansum().sqrt() * obs.pow(2).nansum().sqrt())
        return numerator / denominator

    @staticmethod
    def compute_rmse(pred: torch.Tensor, obs: torch.Tensor):
        return (pred - obs).pow(2).nanmean().sqrt()

    def compute_crps(self, pred: torch.Tensor, obs: torch.Tensor):
        crps = f_kernel_crps(observation=obs, ensemble=pred, fair=True)
        return crps.nanmean()

    def compute_ign(self, pred: torch.Tensor, obs: torch.Tensor):
        ign = f_gaussian_ignorance(observation=obs, mu=pred.mean(-1), sigma=pred.std(-1))
        return ign.nanmean()
    
    def compute_spectral_crps(self, pred: torch.Tensor, obs: torch.Tensor, fair: bool = True):
        with torch.autocast(device_type=self.device_type, enabled=False):
            pred_spectrum = torch.fft.rfft2(self.tokens_to_spatial(pred.float())).view_as_real()
            obs_spectrum = torch.fft.rfft2(self.tokens_to_spatial(obs.float())).view_as_real()
        pred_spectrum = rearrange(pred_spectrum, "(e bb) ... -> bb ... e", e = pred.size(-1))
        return f_kernel_crps(observation= obs_spectrum, ensemble= pred_spectrum, fair= fair)

    @staticmethod
    def compute_spread(ens_pred: torch.Tensor):
        return ens_pred.var(-1).mean().sqrt()

    @staticmethod
    def compute_spread_skill(pred: torch.Tensor, obs: torch.Tensor):
        K = pred.shape[-1]
        correction = math.sqrt((K + 1) / K)
        mean = pred.nanmean(-1)
        spread = pred.var(-1).nanmean().sqrt()
        skill = (obs - mean).pow(2).nanmean().sqrt()
        return correction * (spread / skill.clamp(1e-8))

    @staticmethod
    def rapsd(x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.size()
        device = x.device

        spectrum = torch.fft.fft2(x)
        psd = spectrum.abs().pow(2)

        ky = torch.fft.fftfreq(H, d=1.0).to(device)
        kx = torch.fft.fftfreq(W, d=2.0).to(device)

        kx = rearrange(kx, 'w -> 1 w')
        ky = rearrange(ky, 'h -> h 1')

        radial_freq = torch.sqrt(kx ** 2 + ky ** 2).round()
        radial_freq = radial_freq / radial_freq.max()

        num_bins = min(H, W)
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)
        bin_indices = torch.bucketize(radial_freq.flatten(), bin_edges, right=False) - 1
        bin_indices = bin_indices.clamp(min=0, max=num_bins - 1)

        bin_counts = torch.bincount(bin_indices, minlength=num_bins).clamp(min=1)

        binned = torch.zeros(B, num_bins, device=device)
        binned.scatter_add_(dim=1, index=bin_indices.expand(B, -1), src=psd.view(B, -1))
        binned = binned / bin_counts

        return binned

    def ensemble_metrics(self, ens_pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        E = ens_pred.size(-1)
        crps = self.compute_crps(pred=ens_pred, obs=obs)
        spectral_crps = self.compute_spectral_crps(pred= ens_pred, obs = obs, fair = True)
        ssr = self.compute_spread_skill(pred=ens_pred, obs=obs)
        ign = self.compute_ign(pred=ens_pred, obs=obs)
        spread = self.compute_spread(ens_pred=ens_pred)
        member_acc = torch.as_tensor([self.compute_acc(ens_pred[..., e], obs=obs) for e in range(E)]).mean()
        member_rmse = torch.as_tensor([self.compute_rmse(ens_pred[..., e], obs) for e in range(E)]).mean()

        self.current_metrics.log_metric(f"{label}_crps" if self.exists(label) else "crps", crps.item())
        self.current_metrics.log_metric(f"{label}_spect_crps" if self.exists(label) else "spect_crps", spectral_crps.item())
        self.current_metrics.log_metric(f"{label}_ssr" if self.exists(label) else "ssr", ssr.item())
        self.current_metrics.log_metric(f"{label}_ign" if self.exists(label) else "ign", ign.item())
        self.current_metrics.log_metric(f"{label}_spread" if self.exists(label) else "spread", spread.item())
        self.current_metrics.log_metric(f"{label}_member_rmse" if self.exists(label) else "member_rmse", member_rmse.item())
        self.current_metrics.log_metric(f"{label}_member_acc" if self.exists(label) else "member_acc", member_acc.item())

    def deterministic_metrics(self, pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        acc = self.compute_acc(pred=pred, obs=obs)
        rmse = self.compute_rmse(pred=pred, obs=obs)
        self.current_metrics.log_metric(f"{label}_acc" if self.exists(label) else "acc", acc.item())
        self.current_metrics.log_metric(f"{label}_rmse" if self.exists(label) else "rmse", rmse.item())

    def compute_metrics(self, pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor, label: str = None):
        pred = pred[lsm]
        obs = obs[lsm]

        if pred.ndim == 2:
            self.ensemble_metrics(ens_pred=pred, obs=obs, label=label)
            self.deterministic_metrics(pred.mean(-1), obs, label)
        else:
            self.deterministic_metrics(pred, obs, label)

    @staticmethod
    def _slice_from_cfg(s):
        if isinstance(s, dict):
            return slice(s["start"], s["stop"], s["step"])
        elif isinstance(s, slice):
            return s
        else:
            return slice(None)
        
    def get_frcst_metrics(self, pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor):
        pred, obs, lsm = (self.tokens_to_field(x) for x in (pred, obs, lsm))
        frcst_tasks = {"frcst": (slice(None), slice(None))} # default task
        if exists(self.data_cfg.frcst_tasks): # additional tasks from config
            frcst_tasks.update(self.data_cfg.frcst_tasks)
        for label, (vi, ti) in frcst_tasks.items(): # for all tasks
            vi, ti = (self._slice_from_cfg(x) for x in (vi, ti))
            p, o, l = (x[:, vi, ti] for x in (pred, obs, lsm)) # select the task-specific subset
            self.compute_metrics(p, o, l, label= label) # compute all metrics and label them according to the task

### Trainer
class MTMTrainer(DistributedTrainer, MetricsMixin, OceanMixin, TokenMixin, MaskingMixin):
    # PROPERTIES
    @property
    def cfg(self):
        return self._cfg.trainer
    
    @property
    def optim_cfg(self):
        return self._cfg.optim

    @property
    def data_cfg(self):
        return self._cfg.data

    @property
    def model_cfg(self):
        return self._cfg.model.decoder
    
    @property
    def world_cfg(self):
        return self._cfg.world
    
    @property
    def batch_size(self):
        return self.optim_cfg.batch_size

    @property
    def backend(self):
         return SDPBackend.FLASH_ATTENTION if torch.cuda.get_device_capability()[0] >= 8 else SDPBackend.EFFICIENT_ATTENTION
    
    @property
    def use_fair_crps(self):
        return self.world_cfg.num_ens > 1
    
    @property
    def per_variable_weights(self):
        w = self.data_cfg.weights if hasattr(self.data_cfg, "weights") and exists(self.data_cfg.weights) else 1.
        w = torch.tensor(w, device = self.device)
        return w
    
    # DATA
    def create_dataset(self):
        # instantiate datasets
        self.train_dataset = self.lens_data()
        self.val_dataset = self.godas_data() if self.data_cfg.eval_data == "godas" else self.picontrol_data()
        
        # create land-sea mask
        self._val_lsm = torch.logical_not(self.val_dataset.land_sea_mask.to(device=self.device, dtype=torch.bool))
        self._train_lsm = torch.logical_not(self.train_dataset.land_sea_mask.to(device= self.device, dtype=torch.bool))

        # dataloaders
        train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_dl, val_dl

    # SETUP
    def create_job_name(self):
        if exists(self.cfg.job_name):
            base_name = str(self.cfg.job_name).replace('/', '_')
        else:
            base_name = self.slurm_id
        self.job_name = f"{base_name}"
        self.cfg.job_name = self.job_name # enables resuming from config by using the job name
    
    # OPTIMIZATION
    def create_optimizer(self, named_params):
        return torch.optim.AdamW(
            named_params,
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
            betas=(self.optim_cfg.beta1, self.optim_cfg.beta2),
            #decoupled_weight_decay=True,
        )

    def create_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optim_cfg.num_epochs,
            eta_min=self.optim_cfg.eta_min,
            last_epoch=-1,
        )
 
    # LOSS
    def create_loss(self):
        def loss(pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor):
            # spectral_crps
            spectral_crps = self.compute_spectral_crps(pred = pred, obs = obs, fair = self.use_fair_crps)
            # pointwise crps
            point_crps = f_kernel_crps(observation=obs, ensemble=pred, fair= self.use_fair_crps)
            # apply land-sea mask
            point_crps = point_crps[lsm]
            # reduce to scalar
            return point_crps.nanmean() + 1e-3 * spectral_crps.nanmean()
        return loss

    # MODEL
    def create_model(self):
        model = StochasticWeatherField(self.model_cfg)      
        if hasattr(model, "generator"):
            model.generator = self.generator
        count = count_parameters(model)
        print(f'Created model with {count:,} parameters')
        self.misc_metrics.log_python_object("num_params", count)
        return model
    
    ### FORWARD
    def forward_step(self, batch_idx, batch):
        if self.mode == "train":
            return self.step(batch_idx, batch)
        else:
            _ = self.step(batch_idx, batch)
            _ = self.step(batch_idx, batch, task = "frcst")
            return None

    def step(self, batch_idx, batch, task: str = "train"):
        """
        Performs a forward pass and returns the loss.
        """
        # to device and tokenize
        tokens = self.field_to_tokens(batch.to(self.device))
        
        # masking
        src_mask, tgt_mask = self.get_masks(task)
        src, tgt, lsm = self.apply_masks(tokens, src_mask, tgt_mask)
        
        # convert flat indices to coordinates
        src_mask = self.index_to_coords(src_mask)
        tgt_mask = self.index_to_coords(tgt_mask)
        
        # expand src, tgt, lsm for ensemble
        src, src_mask, tgt_mask = (repeat(x, "b ... -> (b k) ...", k = self.world_cfg.num_ens) for x in (src, src_mask, tgt_mask))
        
        # forward
        with sdpa_kernel(self.backend):
            tgt_pred = self.model(src, src_mask, tgt_mask, num_steps=1)

        # split out ensemble dimension(s) if necessary
        tgt_pred = rearrange(tgt_pred, '(b k) ... (c e) -> b ... c (k e)', c = tokens.size(-1), b = tokens.size(0))
            
        # compute loss
        loss = self.loss_fn(pred = tgt_pred, obs = tgt, lsm = lsm)


        # metrics
        if task == "train":
            self.current_metrics.log_metric(f"loss", loss.item())
            self.compute_metrics(pred = tgt_pred, obs = tgt, lsm = lsm, label = None)
        elif task == "frcst":
            self.get_frcst_metrics(pred = tgt_pred, obs = tgt, lsm = lsm)
        
        return loss

def main():
    parser = argparse.ArgumentParser(description="Train a MIN model")
    parser.add_argument("--id", type=str, default=None, help="alias for the task id")
    parser.add_argument("--config", type=str, default="mae.yaml", help="path to the config file")
    args = parser.parse_args() 

    # task_id for selecting the config overrides
    task_id = args.id if exists(args.id) else os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    # Create a config object
    cfg_file = OmegaConf.load(args.config)
    merged_cfg = OmegaConf.merge(cfg_file.get("defaults", {}), cfg_file.get(task_id, {})) # order matters here!
    config = cfg.MTMConfig.from_omegaconf(merged_cfg)

    #Run the trainer
    trainer = MTMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()