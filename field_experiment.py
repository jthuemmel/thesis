# MIN Trainer

import torch
import os
import argparse
import math

from dataclasses import replace
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

### Trainer
class MTMTrainer(DistributedTrainer):
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

    # DATA
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

    @property
    def land_sea_mask(self):
        lsm = self._train_lsm if self.mode == "train" else self._val_lsm
        B, T, V, S = self.get_flatland_shape()
        return repeat(lsm, "1 (h hh) (w ww) -> b (t v h w) (tt hh ww)", b = B, t = T, v = V, **self.world_cfg.patch_size)

    @property
    def per_variable_weights(self):
        w = self.data_cfg.weights if hasattr(self.data_cfg, "weights") and exists(self.data_cfg.weights) else 1.
        w = torch.tensor(w, device = self.device)
        return w

    # SETUP
    def create_job_name(self):
        if exists(self.cfg.job_name):
            base_name = str(self.cfg.job_name).replace('/', '_')
        else:
            base_name = self.slurm_id
        self.job_name = f"{base_name}"
        self.cfg.job_name = self.job_name # enables resuming from config by using the job name

    # MODEL
    def create_model(self):
        model = StochasticWeatherField(self.model_cfg)      
        if hasattr(model, "generator"):
            model.generator = self.generator
        count = count_parameters(model)
        print(f'Created model with {count:,} parameters')
        self.misc_metrics.log_python_object("num_params", count)
        return model
    
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
    @property
    def use_fair_crps(self):
        return self.world_cfg.num_ens > 1
    
    def create_loss(self):
        def loss(pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor):
            # spectral_crps
            obs_psd= self.compute_rapsd(obs)
            pred_psd = self.compute_rapsd(pred)
            pred_psd = rearrange(pred_psd, "(e bb) ... -> bb ... e", e = pred.size(-1))
            spectral_crps = f_kernel_crps(observation=obs_psd, ensemble=pred_psd, fair= self.use_fair_crps)
            # pointwise crps
            point_crps = f_kernel_crps(observation=obs, ensemble=pred, fair= self.use_fair_crps)
            # apply land-sea mask
            point_crps = point_crps[lsm]
            # reduce to scalar
            return point_crps.nanmean() + 1e-3 * spectral_crps.nanmean()
        return loss

    # METRICS
    def compute_rapsd(self, x: torch.Tensor):
            x = self.tokens_to_field(x)
            x = rearrange(x, "b v t h w ... -> (... b v t) h w")
            with torch.autocast(device_type = self.device_type, enabled = False):
                rapsd = self.rapsd(x.float())
                rapsd = torch.clamp(rapsd, min=1e-8).log10()
            return rapsd
    
    @staticmethod
    def compute_acc(pred, obs):
        return (pred * obs).nansum() / (pred.pow(2).nansum().sqrt() * obs.pow(2).nansum().sqrt())
    
    @staticmethod
    def compute_rmse(pred, obs):
        return (pred - obs).pow(2).nanmean().sqrt()
    
    @staticmethod
    def compute_crps(pred, obs):
        crps = f_kernel_crps(observation=obs, ensemble=pred, fair = True)
        return crps.nanmean()
    
    @staticmethod
    def compute_ign(pred, obs):
        ign = f_gaussian_ignorance(observation=obs, mu=pred.mean(-1), sigma=pred.std(-1))
        return ign.nanmean()
    
    @staticmethod
    def compute_spread(ens_pred):
        return ens_pred.var(-1).nanmean().sqrt()

    @staticmethod
    def compute_spread_skill(pred, obs):
        K = pred.shape[-1]
        correction = math.sqrt((K + 1) / K)
        mean = pred.nanmean(-1)
        spread = pred.var(-1).nanmean().sqrt()
        skill = (obs - mean).pow(2).nanmean().sqrt()
        return correction * (spread / skill.clamp(1e-8))

    @staticmethod
    def rapsd(x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the radial average of a batched 2D tensor.

        Args:
            psd (torch.Tensor): Tensor of shape (B, H, W)
        Returns:
            Tensor of shape (B, L) — where L is the number of radial bins
        '''
        B, H, W = x.size()
        device = x.device

         # compute spectrum using FFT
        spectrum = torch.fft.fft2(x)

        # take absolute value and square to get power spectral density
        psd = spectrum.abs().pow(2)

        # Compute frequency grid (accounting for 1° spacing in H, 2° in W)
        ky = torch.fft.fftfreq(H, d=1.0).to(device)   # shape: (H,)
        kx = torch.fft.fftfreq(W, d=2.0).to(device)   # shape: (W,)

        # Get 2D radial wavenumber magnitude grid
        kx = rearrange(kx, 'w -> 1 w')  # (1, W)
        ky = rearrange(ky, 'h -> h 1')  # (H, 1)
        radial_freq = torch.sqrt(kx ** 2 + ky ** 2).round()  # Integer radial frequencies
        radial_freq = radial_freq / radial_freq.max()

        # Bin the radial frequencies
        num_bins = min(H, W)
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=device)
        bin_indices = torch.bucketize(radial_freq.flatten(), bin_edges, right=False) - 1  # [H*W]
        bin_indices = bin_indices.clamp(min=0, max=num_bins - 1)

         # Count elements per bin
        bin_counts = torch.bincount(bin_indices, minlength=num_bins).clamp(min=1)

        # Bin average using scatter_add
        binned = torch.zeros(B, num_bins, device=device)
        binned.scatter_add_(dim=1, index=bin_indices.expand(B, -1), src=psd.view(B, -1))
        binned = binned / bin_counts

        return binned
    
    ### METRICS LOGGING
    def ensemble_metrics(self, ens_pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        crps = self.compute_crps(pred = ens_pred, obs = obs)
        ssr = self.compute_spread_skill(pred = ens_pred, obs = obs)
        ign = self.compute_ign(pred = ens_pred, obs = obs)
        spread = self.compute_spread(ens_pred = ens_pred)
        self.current_metrics.log_metric(f"{label}_crps" if exists(label) else "crps", crps.item())
        self.current_metrics.log_metric(f"{label}_ssr" if exists(label) else "ssr", ssr.item())
        self.current_metrics.log_metric(f"{label}_ign" if exists(label) else "ign", ign.item())
        self.current_metrics.log_metric(f"{label}_spread" if exists(label) else "spread", spread.item())
    
    def deterministic_metrics(self, pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        acc = self.compute_acc(pred = pred, obs = obs)
        rmse = self.compute_rmse(pred = pred, obs = obs)
        self.current_metrics.log_metric(f"{label}_acc" if exists(label) else "acc", acc.item())
        self.current_metrics.log_metric(f"{label}_rmse" if exists(label) else "rmse", rmse.item())

    def compute_metrics(self, pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor, label: str = None):
        # apply land-sea mask
        pred = pred[lsm]
        obs = obs[lsm]

        if pred.ndim == 2:
            self.ensemble_metrics(ens_pred=pred, obs= obs,label= label)
            # deterministic metrics are computed on the mean of the ensemble
            mean_pred = pred.mean(-1)
            self.deterministic_metrics(mean_pred, obs, label)
        else:
            self.deterministic_metrics(pred, obs, label)

    def get_spectral_metrics(self, pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        obs_psd= self.compute_rapsd(obs)
        pred_psd = self.compute_rapsd(pred)
        label = f"{label}_spectral" if exists(label) else "spectral"

        if obs.ndim < pred.ndim:
            pred_psd = rearrange(pred_psd, "(e bb) ... -> bb ... e", e = pred.size(-1))
            self.ensemble_metrics(ens_pred= pred_psd, obs= obs_psd, label = label)
            self.deterministic_metrics(pred = pred_psd.mean(-1), obs= obs_psd, label = label)
        else:
            self.deterministic_metrics(pred=pred_psd, obs= obs_psd)

    @staticmethod
    def _slice_from_cfg(s):
        if isinstance(s, dict):
            return slice(s["start"], s["stop"], s["step"])
        elif isinstance(s, slice):
            return s
        else:
            return slice(None)

    def select_field_subset(self, x: torch.Tensor, var_idx: slice | int, time_idx: slice | int):
        _, _ , V, S = self.get_flatland_shape()
        # extract time from flatland format and variables from token dimension
        x = rearrange(x, "b (t v s) (tt hh ww) ... -> v (t tt) b s (hh ww) ...", **self.world_cfg.patch_size, s = S, v = V)
        return x[var_idx, time_idx]
    
    def get_frcst_metrics(self, pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor):
        frcst_tasks = {"frcst": (slice(None), slice(None))} # default task
        if exists(self.data_cfg.frcst_tasks): # additional tasks from config
            frcst_tasks.update(self.data_cfg.frcst_tasks)
        for label, (vi, ti) in frcst_tasks.items(): # for all tasks
            vi, ti = map(lambda x: self._slice_from_cfg(x), (vi, ti))
            p, o, l = (self.select_field_subset(x, vi, ti) for x in (pred, obs, lsm)) # select the task-specific subset
            self.compute_metrics(p, o, l, label= label) # compute all metrics and label them according to the task

    ### TOKENS
    def tokens_to_field(self, tokens):
        V = len(self.data_cfg.variables)
        H = self.data_cfg.grid_size["lat"] // self.world_cfg.patch_size["hh"]
        W = self.data_cfg.grid_size["lon"] // self.world_cfg.patch_size["ww"]
        return rearrange(tokens, "b (t v h w) (tt hh ww) ... -> b v (t tt) (h hh) (w ww) ...", 
                        v = V, h = H, w = W, **self.world_cfg.patch_size)

    def field_to_tokens(self, field):
        return rearrange(field, "b v (t tt) (h hh) (w ww) -> b (t v h w) (tt hh ww)", **self.world_cfg.patch_size)

    ### MASKING
    def get_full_index(self):
        B, T, V, S = self.get_flatland_shape()
        return repeat(torch.arange(T * S * V, device = self.device), "i -> b i", b = B)

    def get_flatland_shape(self):
        # flatland shape: (batch, variables, time, space) 
        B = self.batch_size
        V = len(self.data_cfg.variables)
        T = self.data_cfg.sequence_length // self.world_cfg.patch_size["tt"]
        H = self.data_cfg.grid_size["lat"] // self.world_cfg.patch_size["hh"]
        W = self.data_cfg.grid_size["lon"] // self.world_cfg.patch_size["ww"]
        S = H * W
        return B, T, V, S

    def forecast_mask(self):
        _, _, V, S = self.get_flatland_shape()
        tau = self.world_cfg.tau
        idx = self.get_full_index()
        src_mask = idx[:, :S * V * tau]
        tgt_mask = idx[:, S * V * tau:]
        return src_mask, tgt_mask

    def sample_dirichlet(self, shape: tuple, weights: torch.Tensor | float = 1.0, sharpness: float = 1.0) -> torch.Tensor:
        w = torch.ones(shape, device= self.device) * torch.as_tensor(weights, device= self.device)
        probs = w / w.sum(-1, keepdim= True)  # normalize to probability simplex
        alpha = probs * sharpness
        return torch._sample_dirichlet(alpha, generator=self.generator)

    def sample_masking_rates(self, rate_cfg: dict):
        N = self.world_cfg.num_tokens 
        rate = torch.empty((1,), device = self.device)
        if rate_cfg["std"] > 0:
            rate = torch.nn.init.trunc_normal_(rate, **rate_cfg, generator=self.generator)
        else:
            rate = rate.fill_(rate_cfg["mean"])
        return int(N * rate.item())
    
    def sample_joint_masks(self):
        B, T, V, S = self.get_flatland_shape()
        tgt_mask = self.get_full_index()

        #src masking rate
        src_rate = self.sample_masking_rates(self.world_cfg.mask_rates_src)

        # temporal dirichlet
        time_prior = self.sample_dirichlet((B, T), sharpness= self.world_cfg.alpha)
        
        # variate dirichlet
        var_prior = self.sample_dirichlet((B, V), self.per_variable_weights, sharpness= self.world_cfg.alpha)
        
        # joint prior distribution repeated over space
        joint = torch.einsum('b t, b v -> b t v', time_prior, var_prior)
        joint = repeat(joint, 'b t v -> b (t v s)', s = S)

        # multinomial
        src_mask = torch.multinomial(joint, src_rate, replacement = False, generator = self.generator)
        return src_mask, tgt_mask
    
    def sample_individual_masks(self):
        B, T, V, S = self.get_flatland_shape()
        tgt_mask = self.get_full_index()

        # shared dirichlet
        prior = self.sample_dirichlet((B, T), sharpness= self.world_cfg.alpha)
        prior = repeat(prior, "b t -> b (t s)", s = S)

        # per variable multinomial
        masks = []
        for v in range(V):
            idx = torch.multinomial(prior, 
                                    num_samples = self.sample_masking_rates(self.world_cfg.mask_rates_src),
                                    replacement=False, generator=self.generator)  # (B, num_samples)
            full_idx = (idx // S) * V * S + v * S + (idx % S) # correct to T V S shape
            masks.append(full_idx)
            
        src_mask = torch.cat(masks, dim = -1)
        return src_mask, tgt_mask

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
    
    def index_to_coords(self, idx_flat: torch.LongTensor):
        """
        Converts a flat index to coordinates in the grid.
        Args:
            idx_flat (torch.LongTensor): Flat index tensor of shape (B, N).
        Returns:
            torch.Tensor: Coordinates tensor of shape (B, N, C), where C is the number of coordinates.
        """
        # get shape
        B, T, V, S = self.get_flatland_shape()

        # default to full index
        if idx_flat is None:
            idx_flat = self.get_full_index()

        # stride for t
        tvs = V * S

        # stride for v
        vs = S

        # gridsize for h, w
        W = self.data_cfg.grid_size["lon"] // self.world_cfg.patch_size["ww"] 
        H = self.data_cfg.grid_size["lat"] // self.world_cfg.patch_size["hh"]
        
        # now peel off dims
        # time coordinate
        t = idx_flat.div(tvs, rounding_mode='floor')              # (B, N)
        rem  = idx_flat.fmod(tvs)                                 # (B, N)

        # variable coordinate
        v = rem.div(vs, rounding_mode='floor')                    # (B, N)
        rem  = rem.fmod(vs)                                       # (B, N)

        # spatial h, w
        h = rem.div(W, rounding_mode='floor')                     # (B, N)
        w = rem.fmod(W)                                           # (B, N)

        # stack into (B, N, 4)
        coords = torch.stack([t, v, h, w], dim=-1)
        
        return coords
    
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
        src_mask, tgt_mask = self.sample_individual_masks() if task == "train" else self.forecast_mask()
        src, tgt, lsm = self.apply_masks(tokens, src_mask, tgt_mask)
        
        # convert flat indices to coordinates
        src_mask = self.index_to_coords(src_mask)
        tgt_mask = self.index_to_coords(tgt_mask)
        
        # expand src, tgt, lsm for ensemble
        src, src_mask, tgt_mask = (repeat(x, "b ... -> (b k) ...", k = self.world_cfg.num_ens) 
                                    for x in (src, src_mask, tgt_mask))
        
        # forward
        with sdpa_kernel(self.backend):
            if task == "frcst":
                 num_steps = 2
            else:
                 num_steps = 1 if torch.rand(1, device=self.device, generator = self.generator) < 0.2 else 2
            tgt_pred = self.model(src, src_mask, tgt_mask, num_steps=num_steps)

        # split out ensemble dimension(s) if necessary
        tgt_pred = rearrange(tgt_pred, '(b k) ... (c e) -> b ... c (k e)', c = tokens.size(-1), b = tokens.size(0))
            
        # compute loss
        loss = self.loss_fn(pred = tgt_pred, obs = tgt, lsm = lsm)


        # metrics
        if task == "train":
            self.current_metrics.log_metric(f"loss", loss.item())
            self.compute_metrics(pred = tgt_pred, obs = tgt, lsm = lsm, label = None)
            self.get_spectral_metrics(pred= tgt_pred, obs = tgt, label = None)
        elif task == "frcst":
            self.get_frcst_metrics(pred = tgt_pred, obs = tgt, lsm = lsm)
            self.get_spectral_metrics(pred= tgt_pred, obs = tgt, label = task)         
        
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