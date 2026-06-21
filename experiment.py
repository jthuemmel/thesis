import torch
import os
import argparse
import math
import einops

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from scipy.special import erf
from matplotlib.lines import Line2D

from pathlib import Path
from dataclasses import replace
from omegaconf import OmegaConf

from utils.config import *
from utils.dataset import *
from utils.trainer import *
from utils.einmask import *
from utils.masking import *
from utils.loss_fn import *

### HELPER FUNCTIONS
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Trainer
class Experiment(DistributedTrainer):
    # PROPERTIES
    @property
    def cfg(self) -> TrainerConfig:
        return self._cfg.trainer
    
    @property
    def data_cfg(self) -> DatasetConfig:
        return self._cfg.data

    @property
    def model_cfg(self) -> NetworkConfig:
        return self._cfg.model
    
    @property
    def world(self) -> WorldConfig:
        return self._cfg.world
    
    @property
    def objective(self) -> ObjectiveConfig:
        return self._cfg.objective

    @property
    def use_ens(self) -> bool:
        return self.world.kwargs.get('ensemble', False)

    @property
    def use_fair_crps(self) -> bool:
        return default(self.world.ens_size, 1) > 1
    
    # DATA
    def lens_data(self) -> NinoData:
        if not hasattr(self, "_lens_data"):
            lens_config = self.data_cfg
            lens_config = replace(lens_config, 
                                  time_slice = {"start": "1850", "stop": "2000", "step": None},
                                  stats = default(self.data_cfg.stats, LENS_STATS)
                                  )
            self._lens_data = MultifileNinoDataset(self.cfg.lens_path, lens_config, self.rank, self.world_size)
        return self._lens_data       

    def godas_data(self) -> NinoData:
        if not hasattr(self, "_godas_data"):
            godas_config = self.data_cfg
            godas_config = replace(godas_config, 
                                   time_slice = {"start": "1980", "stop": "2020", "step": None},
                                   stats = default(self.data_cfg.stats, GODAS_STATS)
                                   )
            self._godas_data = NinoData(self.cfg.godas_path, godas_config)
        return self._godas_data

    def picontrol_data(self) -> NinoData:
        if not hasattr(self, "_picontrol_data"):
            picontrol_config = self.data_cfg
            picontrol_config = replace(picontrol_config, 
                                       time_slice = {"start": "1900", "stop": "2000", "step": None},
                                       stats = default(self.data_cfg.stats, PICONTROL_STATS))
            self._picontrol_data = NinoData(self.cfg.picontrol_path, picontrol_config)
        return self._picontrol_data

    def oras5_data(self) -> NinoData:
        if not hasattr(self, "_oras5_data"):
            oras5_config = self.data_cfg
            oras5_config = replace(oras5_config, time_slice = {"start": "1980", "stop": "2020", "step": None})
            self._oras5_data = NinoData(self.cfg.oras5_path, oras5_config)
        return self._oras5_data

    def create_dataset(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        # instantiate datasets
        self.train_dataset = self.lens_data()
        self.val_dataset = self.picontrol_data()

        # create land-sea masks
        val_lsm = torch.logical_not(self.val_dataset.land_sea_mask.to(device=self.device, dtype=torch.bool))
        self._val_lsm = einops.repeat(val_lsm, f"1 (h hh) (w ww) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)
        
        train_lsm = torch.logical_not(self.train_dataset.land_sea_mask.to(device= self.device, dtype=torch.bool))
        self._train_lsm = einops.repeat(train_lsm, f"1 (h hh) (w ww) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)

        # dataloaders
        train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.world.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

        val_dl = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.world.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            )
        return train_dl, val_dl

    # SETUP
    def setup_misc(self) -> None:
        self.step_counter = 0

    def create_job_name(self) -> None:
        if exists(self.cfg.job_name):
            base_name = str(self.cfg.job_name).replace('/', '_')
        else:
            base_name = self.slurm_id
        self.job_name = f"{base_name}"
        self.cfg.job_name = self.job_name # enables resuming from config by using the job name
    
    def create_optimizer(self, named_params):
        return torch.optim.AdamW(
                named_params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=(self.cfg.beta1, self.cfg.beta2),
                fused=True
            )

    def create_scheduler(self, optimizer):
        schedulers = []
        milestones = []
        total = 0

        for sch_cfg in self.cfg.schedulers:  # list of dicts
            typ = sch_cfg["type"].lower()
            steps = sch_cfg["steps"]
            total += steps
            milestones.append(total)

            if typ == "linear":
                sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=sch_cfg.get("start_factor", 1.0),
                    end_factor=sch_cfg.get("end_factor", 1.0),
                    total_iters=steps
                )
            elif typ == "constant":
                sched = torch.optim.lr_scheduler.ConstantLR(
                    optimizer,
                    factor=sch_cfg.get("factor", 1.0),
                    total_iters=steps
                )
            elif typ == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=steps,
                    eta_min=sch_cfg.get("eta_min", 0.0)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {typ}")

            schedulers.append(sched)
        # remember total of scheduled steps
        self.total_steps = total
        # milestones exclude final stage
        milestones = milestones[:-1]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=schedulers,
            milestones=milestones
        )
        return scheduler

    def create_model(self) -> torch.nn.Module:
        if self.world.kwargs.get('ensemble', True):
            return EinMask_ENS(network=self.model_cfg, world=self.world)
        return EinMask(network=self.model_cfg, world=self.world)

    @property
    def land_sea_mask(self) -> torch.BoolTensor:
        return self._train_lsm if self.mode == "train" else self._val_lsm

    @property
    def total_epochs(self) -> int:
        return max(1, self.total_steps // len(self.train_dl))
    
    @property
    def per_variable_weights(self) -> torch.FloatTensor:
        weights = {
            # 'temp_ocn_0a': 1.,
            # 'temp_ocn_1a': 0.1,
            # 'temp_ocn_3a': 0.1,
            # 'temp_ocn_5a': 0.1,
            # 'temp_ocn_8a': 0.1,
            # 'temp_ocn_11a': 0.1,
            # 'temp_ocn_14a': 0.1,
            # 'tauxa': 0.01,
            # 'tauya': 0.01,
        }
        w = torch.as_tensor([weights.get(var, 1.) for var in self.data_cfg.variables], device = self.device)
        return einops.repeat(w, 
                             f"(v vv) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)
    
    # MASKING
    @property
    def frcst_prefix(self) -> None:
        prefix = torch.zeros((self.world.token_sizes["t"],), device = self.device, dtype = torch.bool)
        prefix[:self.world.tau] = True
        return einops.repeat(prefix, f't -> ({self.world.token_pattern})', **self.world.token_sizes)
        
    def sample_normal_rates_(self, mean: float, std: float, a: float = 0., b: float = 1.):
        return torch.nn.init.trunc_normal_(
            torch.empty((1,), device = self.device), 
            mean = mean, std = std, a = a, b = b, generator = self.generator
            ).mul(self.world.num_tokens).long()

    def sample_weighted_reservoir(self, num_samples: int):
        P = torch.rand((num_samples, self.world.num_tokens), device=self.device, generator=self.generator).log()
        for dim, alpha in self.objective.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = torch.rand((num_samples, self.world.token_sizes[dim]), device=self.device, generator=self.generator)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P
    
    def sample_block_noise(self, K: int, num_samples: int):
        block_weight = self.objective.kwargs.get('block_weight', 1.)
        d = torch.arange(1, K + 1, device = self.device)
        d = d[K % d == 0]
        idx = torch.multinomial(1 / d.pow(block_weight), 1, generator= self.generator)
        KK = d[idx]
        U = torch.rand((num_samples, K // KK), device= self.device, generator= self.generator)
        U = einops.repeat(U, f'... k -> ... (k kk)', kk = KK, k = K // KK)
        return U
    
    def sample_weighted_reservoir_blocks(self, num_samples: int):
        P = torch.rand((num_samples, self.world.num_tokens), device=self.device, generator=self.generator).log()
        for dim, alpha in self.objective.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = self.sample_block_noise(self.world.token_sizes[dim], num_samples)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P
    
    def sample_antithetic_weighted_reservoir(self, num_samples: int):
        P1, P2 = torch.rand((2, num_samples, self.world.num_tokens), device=self.device, generator=self.generator).log()
        for dim, alpha in self.objective.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = torch.rand((num_samples, self.world.token_sizes[dim]), device=self.device, generator=self.generator)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P1 += U.log().div(alpha)
            P2 += (1 - U).log().div(alpha)
        return P1, P2

    def sample_masks(self, num_samples: int):
        # get config options
        reservoir_mode = self.objective.kwargs.get('reservoir_mode', 'shared')  # shared | antithetic | independent
        src_rate_cfg = self.objective.rate_cfg.src
        tgt_rate_cfg = self.objective.rate_cfg.get('tgt', None)

        # get reservoir weights
        if reservoir_mode == 'antithetic':
            src_weights, tgt_weights = self.sample_antithetic_weighted_reservoir(num_samples)
        elif reservoir_mode == 'independent':
            src_weights = self.sample_weighted_reservoir(num_samples)
            tgt_weights = self.sample_weighted_reservoir(num_samples)
        elif reservoir_mode == 'blocks':
            src_weights = self.sample_weighted_reservoir_blocks(num_samples)
            tgt_weights = src_weights
        else:  # shared
            src_weights = self.sample_weighted_reservoir(num_samples)
            tgt_weights = src_weights

        # sample src rate
        K_src = self.sample_normal_rates_(**src_rate_cfg)

        # choose K elements from src reservoir
        src_reservoir = src_weights.argsort(descending=True)
        src = src_reservoir.argsort(descending=False).lt(K_src)

        # boolean complement if no tgt rate is specified
        if tgt_rate_cfg is None:
            return src, ~src

        # else sample tgt rate
        K_tgt = self.sample_normal_rates_(**tgt_rate_cfg)

        # maybe condition on src mask
        if self.objective.kwargs.get('condition_on_src', False):
            tgt_weights = tgt_weights + src.float().clamp(1e-9).log()

        # select from tgt reservoir
        tgt_reservoir = tgt_weights.argsort(descending= self.objective.kwargs.get('tgt_descending', True))
        tgt = tgt_reservoir.argsort(descending=False).lt(K_tgt)

        return src, tgt

    # FORWARD METHODS
    def forward_step(self, batch_idx, batch, step: str = 'train'):
        if step == 'frcst':
            loss, samples, visible = self.frcst_step(batch_idx, batch)
        else:
            loss, samples, visible = self.masked_step(batch_idx, batch)
        if step == 'train':
            return loss
        return samples, visible

    def masked_step(self, batch_idx, batch):
        # sample masks
        visible, masked = self.sample_masks(batch.size(0))

        # forward model
        samples = self.model(batch, visible, rng=self.generator)

        # sea-only mask
        mask = einops.repeat(masked, f"b ({self.world.token_pattern}) -> b {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)
        mask = torch.logical_and(mask, self.land_sea_mask)

        if self.use_ens:
            # samples: (B, V, T, H, W, E)
            samples = samples * self.land_sea_mask[..., None]
            loss = f_kernel_crps(observation=batch, ensemble=samples, fair=self.use_fair_crps
                                 ).mul(self.per_variable_weights)[mask].mean()
            metrics = {'loss' : loss.item(), **self.compute_metrics_torch_ens(samples, batch, mask)}
        else:
            # samples: (2, B, V, T, H, W) — unpack mu/sigma heads
            mu, sigma = samples
            mu = mu * self.land_sea_mask
            sigma = torch.nn.functional.softplus(sigma)
            loss = f_gaussian_crps(batch, mu, sigma).mul(self.per_variable_weights)[mask].mean()
            metrics = {'loss' : loss.item(), **self.compute_metrics_torch_mve(mu, sigma, batch, mask)}
            samples = torch.stack([mu, sigma])

        self.log_metrics(metrics)
        self.step_counter = self.step_counter + 1 if self.mode == 'train' else self.step_counter
        return loss, samples, visible

    def frcst_step(self, batch_idx, batch):
        visible = self.frcst_prefix.expand(batch.size(0), -1)

        # forward model — rng is accepted by both EinMask and EinMask_ENS
        samples = self.model(batch, visible, rng=self.generator)

        # sea-only mask (forecast region only)
        mask = einops.repeat(visible.logical_not(), f"b ({self.world.token_pattern}) -> b {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)
        mask = torch.logical_and(mask, self.land_sea_mask)

        if self.use_ens:
            # samples: (B, V, T, H, W, E)
            samples = samples * self.land_sea_mask[..., None]
            loss = f_kernel_crps(observation=batch, ensemble=samples, fair=self.use_fair_crps)[mask].mean()
            step_metrics = self.compute_metrics_torch_ens(samples, batch, mask)
        else:
            # samples: (2, B, V, T, H, W) — unpack mu/sigma heads
            mu, sigma = samples
            mu = mu * self.land_sea_mask
            sigma = torch.nn.functional.softplus(sigma)
            loss = f_gaussian_crps(batch, mu, sigma)[mask].mean()
            step_metrics = self.compute_metrics_torch_mve(mu, sigma, batch, mask)
            samples = torch.stack([mu, sigma])

        metrics = {'frcst_loss' : loss.item(), **{f'frcst_{k}' : v for k, v in step_metrics.items()}}
        self.log_metrics(metrics)
        return loss, samples, visible

    #EVAL
    def evaluate_epoch(self):
        super().evaluate_epoch()
        self.evaluate_step('frcst')
        if self.world.kwargs.get('eval_masked', False):
            self.evaluate_step('masked')

    def evaluate_step(self, step: str = 'frcst'):
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        results = []
        for batch_idx, batch in enumerate(self.val_dl):
            batch = batch.to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
                    samples, visible = self.forward_step(batch_idx, batch, step)
                    if self.use_ens:
                        results.append(self.get_xarray_dataset_ens(
                            batch_idx, obs=batch.cpu(), samples=samples.cpu(), visible=visible.cpu()))
                    else:
                        mu, sigma = samples
                        results.append(self.get_xarray_dataset_mve(
                            batch_idx, obs=batch.cpu(), mu=mu.cpu(), sigma=sigma.cpu(), visible=visible.cpu()))

        ds = xr.concat(results, dim='time')
        ds['lsm'] = xr.DataArray(
            np.bool_(self.val_dataset.land_sea_mask[0]),
            coords={'lat' : self.val_dataset.dataset.lat, 'lon' : self.val_dataset.dataset.lon}
        )
        ds = ds.sel(lat=slice(-20., 20.), lon=slice(90, 270))

        if self.use_ens:
            self.get_nino_metrics_ens(ds)
            self.get_field_metrics_ens(ds)
            if self.is_root:
                self.make_eval_plots_ens(ds)
        else:
            self.get_nino_metrics_mve(ds)
            self.get_field_metrics_mve(ds)
            if self.is_root:
                self.make_eval_plots_mve(ds)

        if self.is_root and self.current_epoch == self.total_epochs and self.cfg.save_eval:
            self.write_to_disk(ds)

    def make_eval_plots_ens(self, ds: xr.Dataset):
        history = self.world.tau * self.world.patch_sizes['tt']
        valid = ~ds['temp_ocn_0a_visible'].astype(bool) & ~ds['lsm']
        pred = ds['temp_ocn_0a_pred'].where(valid).isel(step=slice(history, None))
        obs = ds['temp_ocn_0a_obs'].where(valid).isel(step=slice(history, None))
        pred_mean = pred.mean('ens', skipna=True)
        self.plot_sample_ens(pred, obs, 20)
        self.plot_rank_hist(pred, obs, [1, 7, 13, 19])
        self.plot_skill(pred_mean, obs)
        self.plot_monthly_init(pred, obs)
        self.plot_info_noise_ens(pred, obs)

    def make_eval_plots_mve(self, ds: xr.Dataset):
        history = self.world.tau * self.world.patch_sizes['tt']
        valid = ~ds['temp_ocn_0a_visible'].astype(bool) & ~ds['lsm']
        mu = ds['temp_ocn_0a_pred_mu'].where(valid).isel(step=slice(history, None))
        sigma = ds['temp_ocn_0a_pred_sigma'].where(valid).isel(step=slice(history, None))
        obs = ds['temp_ocn_0a_obs'].where(valid).isel(step=slice(history, None))
        self.plot_sample_mve(mu, sigma, 20)
        self.plot_skill(mu, obs)
        self.plot_monthly_init(mu, obs, sigma=sigma)
        self.plot_info_noise_mve(mu, obs)

    def plot_sample_ens(self, pred: xr.DataArray, obs: xr.DataArray, step_fc: int):
        E = pred.sizes['ens']
        E_plot = min(4, E)
        n_panels = 2 + E_plot
        ncols = 2
        nrows = math.ceil(n_panels / ncols)
        kw = dict(vmin=-2, vmax=2, cmap='bwr')
        plt.figure(figsize=(12, 4 * nrows))
        plt.subplot(nrows, ncols, 1)
        pred.isel(time=0, step=step_fc).mean('ens', skipna=True).plot(**kw)
        plt.title('ens mean')
        plt.subplot(nrows, ncols, 2)
        obs.isel(time=0, step=step_fc).plot(**kw)
        plt.title('obs')
        for i in range(E_plot):
            plt.subplot(nrows, ncols, 3 + i)
            pred.isel(time=0, step=step_fc, ens=i).plot(**kw)
            plt.title(f'member {i}')
        plt.tight_layout()
        plt.savefig(self.model_dir / "test_sample.png")
        plt.close()

    def plot_rank_hist(self, pred: xr.DataArray, obs: xr.DataArray, step_lags: list):
        E = pred.sizes['ens']
        n = len(step_lags)
        width = 0.8 / n
        bins = np.arange(E + 1)
        lags = pred['step'].isel(step=step_lags).values - pred['step'].values[0] + 1
        colors = plt.colormaps['viridis'](np.linspace(0.2, 0.85, n))
        nino4_pred = self.get_nino4(pred)
        nino4_obs = self.get_nino4(obs)
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        for ax, (p, o) in zip(axes, [(pred, obs), (nino4_pred, nino4_obs)]):
            for i, sl in enumerate(step_lags):
                ens_v = p.isel(step=sl).values.reshape(-1, E)
                obs_v = o.isel(step=sl).values.reshape(-1, 1)
                valid = ~np.isnan(ens_v).any(-1) & ~np.isnan(obs_v[:, 0])
                ens_v, obs_v = ens_v[valid], obs_v[valid]
                ranks = np.bincount(np.sum(ens_v < obs_v, axis=-1), minlength=E + 1) / len(ens_v)
                ax.bar(bins + i * width, ranks, width=width, alpha=0.8,
                       color=colors[i], edgecolor='k', lw=0.4, label=f'Lag {lags[i]}')
            ax.axhline(1 / (E + 1), color='red', linestyle='dashed', lw=1)
            ax.set_xticks(bins + width * (n - 1) / 2)
            ax.set_xticklabels(bins)
            ax.set_ylabel('Frequency')
        axes[0].set_title('SSTA Field')
        axes[1].set_title('Nino4')
        axes[1].set_xlabel('Rank')
        axes[0].legend(ncol=n, loc='upper center', bbox_to_anchor=(0.5, -0.05), frameon=True)
        plt.tight_layout()
        plt.savefig(self.model_dir / "rank_hist.png")
        plt.close()

    def plot_monthly_init(self, pred: xr.DataArray, obs: xr.DataArray, sigma: xr.DataArray = None):
        space = ('lat', 'lon')
        mu = pred.mean('ens', skipna=True) if self.use_ens else pred
        n_steps = obs.sizes['step']
        step_ax = obs['step'].values - obs['step'].values[0] + 1
        nino4_pred = self.get_nino4(pred)
        nino4_obs = self.get_nino4(obs)
        nino4_sigma = self.get_nino4(sigma) if sigma is not None else None
        acc, info, crps_ss, ssr = (np.full((12, n_steps), np.nan) for _ in range(4))
        acc_n, info_n, crps_ss_n, ssr_n = (np.full((12, n_steps), np.nan) for _ in range(4))
        for m in range(1, 13):
            mask = obs['time.month'] == m
            if not bool(mask.any()):
                continue
            pred_m = pred.sel(time=mask)
            mu_m = pred_m.mean('ens', skipna=True) if self.use_ens else pred_m
            obs_m = obs.sel(time=mask)
            nino4_pred_m = nino4_pred.sel(time=mask)
            nino4_mu_m = nino4_pred_m.mean('ens', skipna=True) if self.use_ens else nino4_pred_m
            nino4_obs_m = nino4_obs.sel(time=mask)
            clim_mean = obs_m.mean('time', skipna=True)
            clim_std = obs_m.std('time', skipna=True).clip(min=1e-6)
            clim_mean_n = nino4_obs_m.mean('time', skipna=True)
            clim_std_n = nino4_obs_m.std('time', skipna=True).clip(min=1e-6)
            acc[m-1] = self.xr_acc(mu_m, obs_m, space).mean('time', skipna=True).values
            info[m-1] = self.xr_ie(mu_m, obs_m, space).mean('time', skipna=True).values
            acc_n[m-1] = self.xr_acc(nino4_mu_m, nino4_obs_m, ('time',)).values
            info_n[m-1] = self.xr_ie(nino4_mu_m, nino4_obs_m, ('time',)).values
            crps_clim = self.xr_gaussian_crps(clim_mean, clim_std, obs_m).mean(space + ('time',), skipna=True).values
            crps_clim_n = self.xr_gaussian_crps(clim_mean_n, clim_std_n, nino4_obs_m).mean('time', skipna=True).values
            if self.use_ens:
                crps = self.xr_kernel_crps(pred_m, obs_m, fair=True).mean(space + ('time',), skipna=True).values
                crps_n = self.xr_kernel_crps(nino4_pred_m, nino4_obs_m, fair=True).mean('time', skipna=True).values
                ssr[m-1] = self.xr_spread_skill_ens(pred_m, obs_m, space).mean('time', skipna=True).values
                ssr_n[m-1] = self.xr_spread_skill_ens(nino4_pred_m, nino4_obs_m, ('time',)).values
            else:
                sigma_m = sigma.sel(time=mask)
                nino4_sigma_m = nino4_sigma.sel(time=mask)
                crps = self.xr_gaussian_crps(mu_m, sigma_m, obs_m).mean(space + ('time',), skipna=True).values
                crps_n = self.xr_gaussian_crps(nino4_mu_m, nino4_sigma_m, nino4_obs_m).mean('time', skipna=True).values
                ssr[m-1] = self.xr_spread_skill_mve(mu_m, sigma_m, obs_m, space).mean('time', skipna=True).values
                ssr_n[m-1] = self.xr_spread_skill_mve(nino4_mu_m, nino4_sigma_m, nino4_obs_m, ('time',)).values
            crps_ss[m-1] = 1 - crps / crps_clim
            crps_ss_n[m-1] = 1 - crps_n / crps_clim_n
        months_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        lev = np.linspace(0, 1, 11)
        ssr_lev = np.linspace(0.5, 1.5, 11)
        panels = [('ACC', lev, acc, acc_n),
                  ('Information Error', lev, info, info_n),
                  ('CRPS-SS', lev, crps_ss, crps_ss_n),
                  ('SSR', ssr_lev, ssr, ssr_n)]
        fig, axes = plt.subplots(4, 2, figsize=(15, 15))
        for row, (label, levels, field, nino4) in enumerate(panels):
            for col, (data, region) in enumerate(zip([field, nino4], ['SSTa', 'Nino4'])):
                ax = axes[row, col]
                filled = ax.contourf(data, cmap='coolwarm', levels=levels)
                contours = ax.contour(data, levels=levels, colors='black')
                ax.clabel(contours, inline=True, fontsize=9)
                ax.set_ylabel('Init Month')
                ax.set_xlabel('Lead Time (Months)')
                ax.set_xticks(np.arange(0, n_steps, 3))
                ax.set_xticklabels(step_ax[::3])
                ax.set_yticks(np.arange(0, 12))
                ax.set_yticklabels(months_str)
                fig.colorbar(filled, ax=ax, label=f'{label} [{region}]')
        plt.tight_layout()
        plt.savefig(self.model_dir / "monthly_init.png")
        plt.close()

    def plot_skill(self, pred: xr.DataArray, obs: xr.DataArray):
        lags_axis = pred['step'].values - pred['step'].values[0] + 1
        nino34_pcc = self.xr_pcc(self.get_nino34(pred), self.get_nino34(obs), ('time',))
        nino4_pcc = self.xr_pcc(self.get_nino4(pred), self.get_nino4(obs), ('time',))
        pcc = self.xr_pcc(pred, obs, ('lat', 'lon')).mean('time', skipna=True)
        plt.figure(figsize=(12, 4))
        plt.plot(lags_axis, nino34_pcc.values, label='nino3.4')
        plt.plot(lags_axis, nino4_pcc.values, label='nino4')
        plt.plot(lags_axis, pcc.values, label='SSTa')
        plt.ylim(0, 1)
        plt.hlines(0.5, lags_axis[0], lags_axis[-1], colors='r', linestyles='dashed')
        plt.legend()
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.tight_layout()
        plt.savefig(self.model_dir / "skill.png")
        plt.close()

    def plot_sample_mve(self, mu: xr.DataArray, sigma: xr.DataArray, step_fc: int):
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        mu.isel(time=0, step=step_fc).plot(vmin=-2, vmax=2, cmap='bwr')
        plt.title('pred mu')
        plt.subplot(122)
        sigma.isel(time=0, step=step_fc).plot(vmin=0, cmap='viridis')
        plt.title('pred sigma')
        plt.tight_layout()
        plt.savefig(self.model_dir / "mu_sigma.png")
        plt.close()

    def _info_noise_vals(self, p: xr.DataArray, o: xr.DataArray, dim: tuple) -> tuple:
        info = self.xr_info(p, o, dim)
        ne = self.xr_ne(p, o, dim)
        if 'time' in info.dims:
            info = info.mean('time', skipna=True)
            ne = ne.mean('time', skipna=True)
        return info.values, ne.values

    def _draw_info_noise_bg(self, ax, sdav: float, rmax: float):
        th = np.linspace(0, np.pi / 2, 200)
        for acc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            a = np.arccos(acc)
            ax.plot([0, rmax * np.sin(a)], [0, rmax * np.cos(a)], ':', color="skyblue", lw=0.6, zorder=0)
            ax.text(rmax * np.sin(a), rmax * np.cos(a), f"{acc:g}", fontsize=7, color="dimgray")
        for r in np.linspace(rmax / 6, rmax, 6):
            ax.plot(r * np.sin(th), r * np.cos(th), ':', color="lightgray", lw=0.5, zorder=0)
        ax.plot(sdav * np.sin(th), sdav * np.cos(th), 'k--', lw=1, zorder=1)
        semi = np.linspace(0, np.pi, 200)
        ax.plot((sdav / 2) * np.sin(semi), sdav / 2 + (sdav / 2) * np.cos(semi), 'k--', lw=1, zorder=1)
        ax.plot(0, sdav, 'k^', ms=9, zorder=3)
        ax.set_xlim(0, rmax)
        ax.set_ylim(0, rmax)
        ax.set_aspect("equal")
        ax.set_xlabel("Noise")
        ax.set_ylabel("Information")

    def plot_info_noise_mve(self, mu: xr.DataArray, obs: xr.DataArray):
        space = ('lat', 'lon')
        lags = obs['step'].values - obs['step'].values[0] + 1
        nino4_mu = self.get_nino4(mu)
        nino4_obs = self.get_nino4(obs)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        for ax, (p, o, dim, title) in zip(axes, [
            (mu, obs, space, 'SSTA Field'),
            (nino4_mu, nino4_obs, ('time',), 'Nino4'),
        ]):
            info, ne = self._info_noise_vals(p, o, dim)
            sdav_da = self.xr_sdav(o, dim)
            sdav = np.nanmean(sdav_da.mean('time', skipna=True).values if 'time' in sdav_da.dims else sdav_da.values)
            rmax = np.nanmax([sdav, np.nanmax(info), np.nanmax(ne)]) * 1.1
            self._draw_info_noise_bg(ax, sdav, rmax)
            colors = plt.colormaps["viridis"](np.linspace(0, 1, len(lags)))
            ax.plot(ne, info, '-', color="0.4", lw=0.8, zorder=2)
            for x, y, lg, c in zip(ne, info, lags, colors):
                ax.scatter(x, y, color=c, s=45, zorder=4, edgecolor="k", lw=0.4)
                ax.annotate(f"{lg}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.model_dir / "info_noise.png")
        plt.close()

    def plot_info_noise_ens(self, pred: xr.DataArray, obs: xr.DataArray):
        space = ('lat', 'lon')
        E_plot = min(4, pred.sizes['ens'])
        cmaps = ['Reds', 'Blues', 'Greens', 'YlOrBr']
        lags = obs['step'].values - obs['step'].values[0] + 1
        nino4_pred = self.get_nino4(pred)
        nino4_obs = self.get_nino4(obs)
        pred_mean = pred.mean('ens')
        nino4_pred_mean = nino4_pred.mean('ens')
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        for ax, (p, p_mean, o, dim, title) in zip(axes, [
            (pred.isel(ens=slice(E_plot)), pred_mean, obs, space, 'SSTA Field'),
            (nino4_pred.isel(ens=slice(E_plot)), nino4_pred_mean, nino4_obs, ('time',), 'Nino4'),
        ]):
            info, ne = self._info_noise_vals(p, o, dim)
            info_mean, ne_mean = self._info_noise_vals(p_mean, o, dim)
            sdav_da = self.xr_sdav(o, dim)
            sdav = np.nanmean(sdav_da.mean('time', skipna=True).values if 'time' in sdav_da.dims else sdav_da.values)
            rmax = np.nanmax([sdav, np.nanmax(info), np.nanmax(ne),
                              np.nanmax(info_mean), np.nanmax(ne_mean)]) * 1.1
            self._draw_info_noise_bg(ax, sdav, rmax)
            for i in range(E_plot):
                colors = plt.colormaps[cmaps[i]](np.linspace(0.8, 0.3, len(lags)))
                ax.plot(ne[:, i], info[:, i], '-', color='0.4', lw=0.8, zorder=2)
                for x, y, lg, c in zip(ne[:, i], info[:, i], lags, colors):
                    ax.scatter(x, y, color=c, s=45, zorder=4, edgecolor='k', lw=0.4)
                    ax.annotate(f'{lg}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=7)
            mean_colors = plt.colormaps['viridis'](np.linspace(0, 1, len(lags)))
            ax.plot(ne_mean, info_mean, '-', color='0.4', lw=0.8, zorder=2)
            for x, y, lg, c in zip(ne_mean, info_mean, lags, mean_colors):
                ax.scatter(x, y, color=c, s=60, zorder=5, edgecolor='k', lw=0.6, marker='D')
                ax.annotate(f'{lg}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=7)
            handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.colormaps[cmaps[i]](0.6), markersize=8, label=f'Member {i + 1}')
                for i in range(E_plot)
            ]
            handles.append(Line2D([0], [0], marker='D', color='w',
                                  markerfacecolor=plt.colormaps['viridis'](0.5),
                                  markersize=9, label='Ensemble Mean'))
            ax.legend(handles=handles, loc='upper right', fontsize=8)
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.model_dir / "info_noise.png")
        plt.close()

    def write_to_disk(self, data: xr.Dataset):
        path = self.model_dir / f"{self.data_cfg.eval_data}_eval.zarr"
        data.to_zarr(path, mode='w')

    def get_xarray_dataset_mve(self, batch_idx, obs, mu, sigma, visible):
        meta_data = self.val_dataset.dataset
        time, lat, lon = meta_data.time, meta_data.lat, meta_data.lon
        T, tt = self.world.token_sizes['t'], self.world.patch_sizes['tt']
        step = np.arange(T * tt)
        vis_field = einops.repeat(
            visible,
            f'b ({self.world.token_pattern}) -> b {self.world.field_pattern}',
            **self.world.token_sizes, **self.world.patch_sizes
        )
        arrays = []
        for v, var in enumerate(self.data_cfg.variables):
            if var not in self.data_cfg.eval_variables:
                continue
            std = self.val_dataset._stds.sel(variable=var).values.astype(np.float32)
            arrays.append(xr.Dataset(
                data_vars={
                    f'{var}_obs' :        (['time', 'step', 'lat', 'lon'], obs[:, v].numpy() * std),
                    f'{var}_pred_mu' :    (['time', 'step', 'lat', 'lon'], mu[:, v].numpy() * std),
                    f'{var}_pred_sigma' : (['time', 'step', 'lat', 'lon'], sigma[:, v].numpy() * std),
                    f'{var}_visible' :    (['time', 'step', 'lat', 'lon'], vis_field[:, v].numpy()),
                },
                coords={
                    'time' : time[batch_idx * self.world.batch_size : (batch_idx + 1) * self.world.batch_size],
                    'step' : step,
                    'lat' :  lat,
                    'lon' :  lon,
                }
            ))
        return xr.merge(arrays, compat='no_conflicts')

    def get_xarray_dataset_ens(self, batch_idx, obs, samples, visible):
        meta_data = self.val_dataset.dataset
        time, lat, lon = meta_data.time, meta_data.lat, meta_data.lon
        T, tt = self.world.token_sizes['t'], self.world.patch_sizes['tt']
        step = np.arange(T * tt)
        ens = np.arange(samples.shape[-1])
        vis_field = einops.repeat(
            visible,
            f'b ({self.world.token_pattern}) -> b {self.world.field_pattern}',
            **self.world.token_sizes, **self.world.patch_sizes
        )
        arrays = []
        for v, var in enumerate(self.data_cfg.variables):
            if var not in self.data_cfg.eval_variables:
                continue
            std = self.val_dataset._stds.sel(variable=var).values.astype(np.float32)
            arrays.append(xr.Dataset(
                data_vars={
                    f'{var}_obs' :     (['time', 'step', 'lat', 'lon'],       obs[:, v].numpy() * std),
                    f'{var}_pred' :    (['time', 'step', 'lat', 'lon', 'ens'], samples[:, v].numpy() * std),
                    f'{var}_visible' : (['time', 'step', 'lat', 'lon'],        vis_field[:, v].numpy()),
                },
                coords={
                    'time' : time[batch_idx * self.world.batch_size : (batch_idx + 1) * self.world.batch_size],
                    'step' : step,
                    'lat' :  lat,
                    'lon' :  lon,
                    'ens' :  ens,
                }
            ))
        return xr.merge(arrays, compat='no_conflicts')

    def get_field_metrics_ens(self, eval_data: xr.Dataset):
        for var in self.data_cfg.variables:
            if var not in self.data_cfg.eval_variables:
                continue
            valid = ~eval_data[f'{var}_visible'].astype(bool) & ~eval_data['lsm']
            pred = eval_data[f'{var}_pred'].where(valid)
            obs = eval_data[f'{var}_obs'].where(valid)
            pred_mean = pred.mean('ens', skipna=True)
            pcc = self.xr_pcc(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            rmse = self.xr_rmse(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            ssr = self.xr_spread_skill_ens(pred, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            crps_ss = self.xr_crps_ss_ens(pred, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            acc = self.xr_acc(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            fi = self.xr_fi(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            info = self.xr_info(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            ie = self.xr_ie(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            ne = self.xr_ne(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            stde = self.xr_stde(pred_mean, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            self.current_metrics.log_metric(f'{var}_pcc', pcc.item())
            self.current_metrics.log_metric(f'{var}_ssr', ssr.item())
            self.current_metrics.log_metric(f'{var}_rmse', rmse.item())
            self.current_metrics.log_metric(f'{var}_crps_ss', crps_ss.item())
            self.current_metrics.log_metric(f'{var}_acc', acc.item())
            self.current_metrics.log_metric(f'{var}_fi', fi.item())
            self.current_metrics.log_metric(f'{var}_info', info.item())
            self.current_metrics.log_metric(f'{var}_ie', ie.item())
            self.current_metrics.log_metric(f'{var}_ne', ne.item())
            self.current_metrics.log_metric(f'{var}_stde', stde.item())

    def get_field_metrics_mve(self, eval_data: xr.Dataset):
        for var in self.data_cfg.variables:
            if var not in self.data_cfg.eval_variables:
                continue
            valid = ~eval_data[f'{var}_visible'].astype(bool) & ~eval_data['lsm']
            mu = eval_data[f'{var}_pred_mu'].where(valid)
            sigma = eval_data[f'{var}_pred_sigma'].where(valid)
            obs = eval_data[f'{var}_obs'].where(valid)
            pcc = self.xr_pcc(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            rmse = self.xr_rmse(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            ssr = self.xr_spread_skill_mve(mu, sigma, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            crps = self.xr_gaussian_crps(mu, sigma, obs).mean(('lat', 'lon', 'time', 'step'), skipna=True)
            crps_ss = self.xr_crps_ss(mu, sigma, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            acc = self.xr_acc(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            fi = self.xr_fi(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            info = self.xr_info(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            ie = self.xr_ie(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            ne = self.xr_ne(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            stde = self.xr_stde(mu, obs, ('lat', 'lon')).mean(('time', 'step'), skipna=True)
            self.current_metrics.log_metric(f'{var}_pcc', pcc.item())
            self.current_metrics.log_metric(f'{var}_ssr', ssr.item())
            self.current_metrics.log_metric(f'{var}_rmse', rmse.item())
            self.current_metrics.log_metric(f'{var}_crps', crps.item())
            self.current_metrics.log_metric(f'{var}_crps_ss', crps_ss.item())
            self.current_metrics.log_metric(f'{var}_acc', acc.item())
            self.current_metrics.log_metric(f'{var}_fi', fi.item())
            self.current_metrics.log_metric(f'{var}_info', info.item())
            self.current_metrics.log_metric(f'{var}_ie', ie.item())
            self.current_metrics.log_metric(f'{var}_ne', ne.item())
            self.current_metrics.log_metric(f'{var}_stde', stde.item())

    def get_nino_metrics_ens(self, eval_data: xr.Dataset):
        history = self.world.tau * self.world.patch_sizes['tt']
        valid = ~eval_data['temp_ocn_0a_visible'].astype(bool) & ~eval_data['lsm']
        pred = eval_data['temp_ocn_0a_pred'].where(valid).mean('ens', skipna=True)
        obs = eval_data['temp_ocn_0a_obs'].where(valid)
        nino34_pred = self.get_nino34(pred)
        nino34_obs = self.get_nino34(obs)
        nino4_pred = self.get_nino4(pred)
        nino4_obs = self.get_nino4(obs)
        nino34_pcc = self.xr_pcc(nino34_pred, nino34_obs, ('time',)).isel(step=slice(history, None))
        nino4_pcc = self.xr_pcc(nino4_pred, nino4_obs, ('time',)).isel(step=slice(history, None))
        nino34_rmse = self.xr_rmse(nino34_pred, nino34_obs, ('time',)).isel(step=slice(history, None))
        nino4_rmse = self.xr_rmse(nino4_pred, nino4_obs, ('time',)).isel(step=slice(history, None))
        nino4_thresh_month = 1 + np.argwhere(nino4_pcc.values > 0.5).max(initial=0)
        nino34_thresh_month = 1 + np.argwhere(nino34_pcc.values > 0.5).max(initial=0)
        self.current_metrics.log_metric('nino4_pcc_month', float(nino4_thresh_month))
        self.current_metrics.log_metric('nino34_pcc_month', float(nino34_thresh_month))
        for lag in [3, 9, 15, 18, 21]:
            self.current_metrics.log_metric(f'nino34_pcc_{lag}', nino34_pcc.isel(step=lag - 1).item())
            self.current_metrics.log_metric(f'nino34_rmse_{lag}', nino34_rmse.isel(step=lag - 1).item())
            self.current_metrics.log_metric(f'nino4_pcc_{lag}', nino4_pcc.isel(step=lag - 1).item())
            self.current_metrics.log_metric(f'nino4_rmse_{lag}', nino4_rmse.isel(step=lag - 1).item())

    def get_nino_metrics_mve(self, eval_data: xr.Dataset):
        history = self.world.tau * self.world.patch_sizes['tt']
        valid = ~eval_data['temp_ocn_0a_visible'].astype(bool) & ~eval_data['lsm']
        mu = eval_data['temp_ocn_0a_pred_mu'].where(valid)
        obs = eval_data['temp_ocn_0a_obs'].where(valid)
        nino34_mu = self.get_nino34(mu)
        nino34_obs = self.get_nino34(obs)
        nino4_mu = self.get_nino4(mu)
        nino4_obs = self.get_nino4(obs)
        nino34_pcc = self.xr_pcc(nino34_mu, nino34_obs, ('time',)).isel(step=slice(history, None))
        nino4_pcc = self.xr_pcc(nino4_mu, nino4_obs, ('time',)).isel(step=slice(history, None))
        nino34_rmse = self.xr_rmse(nino34_mu, nino34_obs, ('time',)).isel(step=slice(history, None))
        nino4_rmse = self.xr_rmse(nino4_mu, nino4_obs, ('time',)).isel(step=slice(history, None))
        nino4_thresh_month = 1 + np.argwhere(nino4_pcc.values > 0.5).max(initial=0)
        nino34_thresh_month = 1 + np.argwhere(nino34_pcc.values > 0.5).max(initial=0)
        self.current_metrics.log_metric('nino4_pcc_month', float(nino4_thresh_month))
        self.current_metrics.log_metric('nino34_pcc_month', float(nino34_thresh_month))
        for lag in [3, 9, 15, 18, 21]:
            self.current_metrics.log_metric(f'nino34_pcc_{lag}', nino34_pcc.isel(step=lag - 1).item())
            self.current_metrics.log_metric(f'nino34_rmse_{lag}', nino34_rmse.isel(step=lag - 1).item())
            self.current_metrics.log_metric(f'nino4_pcc_{lag}', nino4_pcc.isel(step=lag - 1).item())
            self.current_metrics.log_metric(f'nino4_rmse_{lag}', nino4_rmse.isel(step=lag - 1).item())
        
    def log_metrics(self, metrics: dict, task: str = None):
        for key, val in metrics.items():
            name = f"{task}_{key}" if exists(task) and task != 'prior' else key
            self.current_metrics.log_metric(name, val)
    
    def compute_metrics_torch_ens(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor) -> dict:
        ens = ens[mask]     # (N, E)
        obs = obs[mask]     # (N,)
        return {
            'crps' :   self.compute_crps_ens(ens, obs, fair=self.use_fair_crps),
            'ssr' :    self.compute_spread_skill_ens(ens, obs),
            'ign' :    self.compute_ign_ens(ens, obs),
            'spread' : self.compute_spread_ens(ens),
            'acc' :    self.compute_acc(ens.mean(-1), obs),
            'rmse' :   self.compute_rmse(ens.mean(-1), obs),
        }

    def compute_metrics_torch_mve(self, mu: torch.Tensor, sigma: torch.Tensor,
                                  obs: torch.Tensor, mask: torch.BoolTensor) -> dict:
        mu = mu[mask]
        sigma = sigma[mask]
        obs = obs[mask]
        spread = sigma.pow(2).mean().sqrt()
        return {
            'crps' :   f_gaussian_crps(obs, mu, sigma).nanmean().item(),
            'ign' :    f_gaussian_ignorance(obs, mu, sigma).nanmean().item(),
            'spread' : spread.item(),
            'ssr' :    (spread / (mu - obs).pow(2).mean().sqrt()).item(),
            'acc' :    self.compute_acc(mu, obs),
            'rmse' :   self.compute_rmse(mu, obs),
        }

    @staticmethod
    def get_nino4(da: xr.DataArray) -> xr.DataArray:
        return da.sel(lon=slice(160, 210), lat=slice(-5, 5)).mean(dim=['lon', 'lat'], skipna=True)

    @staticmethod
    def get_nino34(da: xr.DataArray) -> xr.DataArray:
        return da.sel(lon=slice(190, 240), lat=slice(-5, 5)).mean(dim=['lon', 'lat'], skipna=True)

    @staticmethod
    def xr_pcc(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        num = (pred * obs).sum(dim, skipna=True)
        denom = np.sqrt((pred**2).sum(dim, skipna=True)) * np.sqrt((obs**2).sum(dim, skipna=True))
        return num / denom

    @staticmethod
    def xr_rmse(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        return np.sqrt(((pred - obs)**2).mean(dim, skipna=True))

    @staticmethod
    def _debias(x: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        return x - x.mean(dim, skipna=True)

    @staticmethod
    def _winner(x: xr.DataArray, y: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        return (x * y).mean(dim, skipna=True)

    def xr_acc(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
               debias: bool = True) -> xr.DataArray:
        af = self._debias(pred, dim) if debias else pred
        at = self._debias(obs, dim) if debias else obs
        var_f = self._winner(af, af, dim)
        var_t = self._winner(at, at, dim)
        cov = self._winner(af, at, dim)
        return cov / (np.sqrt(var_f) * np.sqrt(var_t))

    def xr_fi(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
              debias: bool = True) -> xr.DataArray:
        af = self._debias(pred, dim) if debias else pred
        at = self._debias(obs, dim) if debias else obs
        return self._winner(af, at, dim) / self._winner(at, at, dim)

    def xr_info(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
                debias: bool = True) -> xr.DataArray:
        fi = self.xr_fi(pred, obs, dim, debias)
        at = self._debias(obs, dim) if debias else obs
        return fi * np.sqrt(self._winner(at, at, dim))

    def xr_ie(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
              debias: bool = True) -> xr.DataArray:
        fi = self.xr_fi(pred, obs, dim, debias)
        at = self._debias(obs, dim) if debias else obs
        return np.abs(1 - fi) * np.sqrt(self._winner(at, at, dim))

    def xr_ne(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
              debias: bool = True) -> xr.DataArray:
        af = self._debias(pred, dim) if debias else pred
        at = self._debias(obs, dim) if debias else obs
        var_f = self._winner(af, af, dim)
        var_t = self._winner(at, at, dim)
        cov = self._winner(af, at, dim)
        fi = cov / var_t
        return np.sqrt(var_f - 2 * fi * cov + fi**2 * var_t)

    def xr_stde(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
                debias: bool = True) -> xr.DataArray:
        af = self._debias(pred, dim) if debias else pred
        at = self._debias(obs, dim) if debias else obs
        var_f = self._winner(af, af, dim)
        var_t = self._winner(at, at, dim)
        acc = self._winner(af, at, dim) / (np.sqrt(var_f) * np.sqrt(var_t))
        return np.sqrt(var_t + var_f - 2 * np.sqrt(var_t) * np.sqrt(var_f) * acc)

    def xr_sdav(self, obs: xr.DataArray, dim: tuple[str], debias: bool = True) -> xr.DataArray:
        at = self._debias(obs, dim) if debias else obs
        return np.sqrt(self._winner(at, at, dim))

    @staticmethod
    def xr_spread_skill_ens(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        K = pred.sizes['ens']
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean('ens', skipna=True)
        spread = np.sqrt(pred.var('ens', skipna=True).mean(dim, skipna=True))
        skill = np.sqrt(((obs - mean)**2).mean(dim, skipna=True))
        return correction * (spread / skill)

    @staticmethod
    def xr_spread_skill_mve(mu: xr.DataArray, sigma: xr.DataArray,
                            obs: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        var = (sigma**2).mean(dim, skipna=True)
        mse = ((obs - mu)**2).mean(dim, skipna=True)
        return np.sqrt(var / mse)

    @staticmethod
    def xr_gaussian_crps(mu: xr.DataArray, sigma: xr.DataArray,
                         obs: xr.DataArray) -> xr.DataArray:
        sqrtPi, sqrtTwo = math.sqrt(math.pi), math.sqrt(2)
        sigma = sigma.clip(min=1e-6)
        z = (obs - mu) / sigma
        phi = np.exp(-z**2 / 2) / (sqrtTwo * sqrtPi)
        return sigma * (z * xr.apply_ufunc(erf, z / sqrtTwo) + 2 * phi - 1 / sqrtPi)

    @staticmethod
    def xr_gaussian_ign(mu: xr.DataArray, sigma: xr.DataArray,
                        obs: xr.DataArray) -> xr.DataArray:
        sigma = sigma.clip(min=1e-6)
        z = (obs - mu) / sigma
        return 0.5 * math.log(2 * math.pi) + np.log(sigma) + 0.5 * z**2

    def xr_crps_ss(self, mu: xr.DataArray, sigma: xr.DataArray,
                   obs: xr.DataArray, dim: tuple[str]) -> xr.DataArray:
        crps = self.xr_gaussian_crps(mu, sigma, obs).mean(dim, skipna=True)
        clim = self.xr_gaussian_crps(xr.zeros_like(obs), xr.ones_like(obs), obs).mean(dim, skipna=True)
        return 1 - crps / clim

    @staticmethod
    def xr_kernel_crps(pred: xr.DataArray, obs: xr.DataArray, fair: bool = False) -> xr.DataArray:
        E = pred.sizes['ens']
        coef = -1 / (E * (E - 1)) if fair else -1 / (E**2)
        mae = np.abs(pred - obs).mean('ens', skipna=True)
        def _pairwise(p):
            total = np.zeros(p.shape[:-1], dtype=np.float32)
            for i in range(E):
                total = total + np.sum(np.abs(p[..., i:i + 1] - p[..., i + 1:]), axis=-1)
            return total
        ens_var = xr.apply_ufunc(_pairwise, pred, input_core_dims=[['ens']])
        return mae + coef * ens_var

    def xr_crps_ss_ens(self, pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str],
                       fair: bool = False) -> xr.DataArray:
        crps = self.xr_kernel_crps(pred, obs, fair=fair).mean(dim, skipna=True)
        clim = self.xr_gaussian_crps(xr.zeros_like(obs), xr.ones_like(obs), obs).mean(dim, skipna=True)
        return 1 - crps / clim

    @staticmethod
    def compute_acc(pred, obs, eps: float = 1e-5) -> float:
        return (pred * obs).nansum().div(
            pred.pow(2).nansum().sqrt() * obs.pow(2).nansum().sqrt() + eps).item()

    @staticmethod
    def compute_rmse(pred, obs) -> float:
        return (pred - obs).pow(2).nanmean().sqrt().item()

    @staticmethod
    def compute_crps_ens(pred, obs, fair: bool = True) -> float:
        return f_kernel_crps(observation=obs, ensemble=pred, fair=fair).nanmean().item()

    @staticmethod
    def compute_ign_ens(pred, obs, eps: float = 1e-5) -> float:
        return f_gaussian_ignorance(observation=obs, mu=pred.mean(-1),
                                    sigma=pred.std(-1) + eps).nanmean().item()

    @staticmethod
    def compute_spread_ens(pred) -> float:
        return pred.var(-1).mean().sqrt().item()

    @staticmethod
    def compute_spread_skill_ens(pred, obs, eps: float = 1e-5) -> float:
        K = pred.shape[-1]
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean(-1)
        spread = pred.var(-1).mean().sqrt()
        skill = (obs - mean).pow(2).mean().sqrt() + eps
        return (spread / skill).mul(correction).item()

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
    config = MTMConfig.from_omegaconf(merged_cfg)

    #Run the trainer
    trainer = Experiment(config)
    trainer.train()

if __name__ == "__main__":
    main()
