import torch
import os
import argparse
import math
import einops

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

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
    
    def create_testset(self) -> torch.utils.data.DataLoader:
        # instantiate dataset
        self.test_dataset = self.godas_data()

        # create land-sea mask
        test_lsm = torch.logical_not(self.test_dataset.land_sea_mask.to(device=self.device, dtype=torch.bool))
        self._test_lsm = einops.repeat(test_lsm, f"1 (h hh) (w ww) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)

        # dataloader
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.world.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            shuffle=False,
            pin_memory=True,
            )

    # SETUP
    def setup_misc(self) -> None:
        self.step_counter = 0
        self.create_prior()

    def create_prior(self) -> None:
        prefix = torch.zeros((self.world.token_sizes["t"],), device = self.device, dtype = torch.bool)
        prefix[:self.world.tau] = True
        self.frcst_prefix = einops.repeat(prefix, f't -> ({self.world.token_pattern})', **self.world.token_sizes)
        
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
        model = EinMask(network=self.model_cfg, world=self.world)
        return model

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
    def forward_step(self, batch_idx, batch):        
        # sample masks
        visible, masked = self.sample_masks(batch.size(0))
        
        # foward model
        prediction = self.model(batch, visible)
        
        # sea-only mask
        mask = einops.repeat(masked, f"b ({self.world.token_pattern}) -> b {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)
        mask = torch.logical_and(mask, self.land_sea_mask)

        # loss
        mu, sigma = prediction
        sigma = torch.nn.functional.softplus(sigma)
        loss = f_gaussian_crps(batch, mu, sigma).mul(self.per_variable_weights)[mask].mean()

        #track metrics
        metrics = {'loss' : loss.item(),
                   'acc': self.compute_acc(mu[mask], batch[mask]),
                   'rmse': self.compute_rmse(mu[mask], batch[mask]),
                   'ssr': (sigma[mask].pow(2).mean().sqrt() / (mu[mask] - batch[mask]).pow(2).mean().sqrt()).item(),
                   }
        self.log_metrics(metrics)

        # update step counter if training
        self.step_counter = self.step_counter + 1 if self.mode == 'train' else self.step_counter
        return loss
    
    def frcst_step(self, batch_idx, batch):
        visible = self.frcst_prefix.expand(batch.size(0), -1)
        prediction = self.model(batch, visible)
        
        # sea-only mask
        mask = einops.repeat(visible.logical_not(), f"b ({self.world.token_pattern}) -> b {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)
        mask = torch.logical_and(mask, self.land_sea_mask)

        # gaussian loss
        mu, sigma = prediction
        mu = mu * self.land_sea_mask
        sigma = torch.nn.functional.softplus(sigma)
        loss = f_gaussian_crps(batch, mu, sigma)[mask].mean()

        metrics = {'frcst_loss' : loss.item(),
                   'frcst_acc': self.compute_acc(mu[mask], batch[mask]),
                   'frcst_rmse': self.compute_rmse(mu[mask], batch[mask]),
                   'frcst_ssr': (sigma[mask].pow(2).mean().sqrt() / (mu[mask] - batch[mask]).pow(2).mean().sqrt()).item(),
                   }
        self.log_metrics(metrics)
        return mu, sigma

    #EVAL
    def evaluate_epoch(self):
        super().evaluate_epoch()
        self.evaluate_frcst()

    def evaluate_frcst(self):
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        samples = []
        for batch_idx, batch in enumerate(self.val_dl):
            batch = batch.to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type = self.device.type, enabled=self.cfg.mixed_precision):
                    mu, sigma = self.frcst_step(batch_idx, batch)
                    samples.append(self.get_xarray_dataset(batch_idx, obs = batch.cpu(), pred = mu[..., None].cpu()))

        ds = xr.concat(samples, dim = "time")
        ds = ds.sel(lat = slice(-20., 20.), lon = slice(90, 270))
        self.get_nino_metrics(ds)
        self.get_field_metrics(ds)

        if self.is_root:
            self.make_eval_plots(ds)

        if self.is_root and self.current_epoch == self.total_epochs and self.cfg.save_eval:
            self.write_to_disk(ds)

    def make_eval_plots(self, ds: xr.Dataset):
        # SAMPLES
        plt.figure(figsize=(12,12))
        plt.subplot(321)
        ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20).mean('ens').plot(vmin=-2, vmax = 2, cmap= 'bwr')
        plt.subplot(322)
        ds[f"temp_ocn_0a_tgt"].isel(time = 0, lag = 20).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        # plt.subplot(323)
        # ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 0).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        # plt.subplot(324)
        # ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 1).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        # plt.subplot(325)
        # ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 2).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        # plt.subplot(326)
        # ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 3).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        plt.savefig(self.model_dir / "test_sample.png")
        plt.close()

        #RANK HIST
        # plt.figure(figsize=(12,4))
        # E = len(ds.ens)
        # ens = ds[f"temp_ocn_0a_pred"].sel(lag = [1, 7, 13, 19]).values.reshape(-1, E)
        # obs = ds[f"temp_ocn_0a_tgt"].sel(lag = [1, 7, 13, 19]).values.reshape(-1, 1)
        # rank_counts = np.bincount(np.sum(ens < obs, axis= -1), minlength= E + 1) / ens.shape[0]
        # plt.bar(np.arange(E + 1), rank_counts, alpha = 0.5)
        # plt.hlines(1 / (E + 1), 0, E, color="red", linestyle="dashed", linewidth=1)
        # plt.ylabel('Frequency')
        # plt.xlabel("Rank")
        # plt.savefig(self.model_dir / "rank_hist.png")
        # plt.close()

        # ACC vs LAG
        plt.figure(figsize=(12,4))
        nino34_tgt, nino34_pred = self.get_nino34(ds["temp_ocn_0a_tgt"]), self.get_nino34(ds["temp_ocn_0a_pred"].mean('ens'))
        nino4_tgt, nino4_pred = self.get_nino4(ds["temp_ocn_0a_tgt"]), self.get_nino4(ds["temp_ocn_0a_pred"].mean('ens'))
        nino34_pcc = self.xr_pcc(nino34_pred, nino34_tgt, ("time",))
        nino4_pcc = self.xr_pcc(nino4_pred, nino4_tgt, ("time",))
        pcc = self.xr_pcc(ds["temp_ocn_0a_pred"].mean('ens'), ds["temp_ocn_0a_tgt"], ('lat', 'lon')).mean(('time'))
        plt.plot(ds.lag, nino34_pcc, label = 'nino3.4')
        plt.plot(ds.lag, nino4_pcc, label = 'nino4')
        plt.plot(ds.lag, pcc, label = 'SSTa')
        plt.ylim(0, 1)
        plt.hlines(0.5, ds.lag[0], ds.lag[-1], colors='r', linestyles='dashed')
        plt.legend()
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.tight_layout()
        plt.savefig(self.model_dir / "skill.png")
        plt.close()

    def write_to_disk(self, data: xr.Dataset):
        path = self.model_dir / f"{self.data_cfg.eval_data}_eval.zarr"
        data.to_zarr(path, mode = "w")

    def get_xarray_dataset(self, batch_idx, pred, obs):
        #meta data
        meta_data = self.val_dataset.dataset
        time, lat, lon = meta_data.time, meta_data.lat, meta_data.lon
        ens = np.arange(pred.shape[-1])
        tau = self.world.tau
        T, tt = self.world.token_sizes["t"], self.world.patch_sizes["tt"]
        lag = np.arange(1, 1 + ((T - tau) * tt))
        history = tau * tt

        # variables
        arrays = []
        for v, var in enumerate(self.data_cfg.variables):
            if var not in self.data_cfg.eval_variables:
                continue
            
            std = self.val_dataset._stds.sel(variable = var).values
            p = pred[:, v, history:].float().detach().cpu().numpy() * std
            o = obs[:, v, history:].float().detach().cpu().numpy() * std

            #create xarray
            data_array = xr.Dataset(
                data_vars = {
                    f"{var}_pred": (["time", "lag", "lat", "lon", "ens"], p),
                    f"{var}_tgt": (["time", "lag", "lat", "lon"], o),
                },
                coords = {
                    "time": time[batch_idx * self.world.batch_size: (batch_idx + 1) * self.world.batch_size],
                    "lag": lag,
                    "lat": lat,
                    "lon": lon,
                    "ens": ens
                },
            )
            arrays.append(data_array)
        ds = xr.merge(arrays)
        return ds

    def get_xr_lsm(self, data: xr.Dataset):
        if "sftlf" in data:
            lsm = data["sftlf"]
        else:
            lsm = data[self.data_cfg.variables[0]].isel(time=0).isnull()
            lsm = lsm.drop_vars(["time", "month"], errors="ignore")
        return lsm

    def get_field_metrics(self, eval_data: xr.Dataset):
        for var in self.data_cfg.variables:
            if var not in self.data_cfg.eval_variables:
                continue
            tgt, pred = eval_data[f"{var}_tgt"], eval_data[f'{var}_pred']
            pcc = self.xr_pcc(pred.mean('ens'), tgt, ('lat', 'lon')).mean(('time', 'lag'))
            rmse = self.xr_rmse(pred.mean('ens'), tgt, ('lat', 'lon')).mean(('time', 'lag'))
            ssr = self.xr_spread_skill_ens(pred, tgt, ('lat', 'lon')).mean(('time', 'lag'))
            
            self.current_metrics.log_metric(f"{var}_pcc", pcc.item())
            self.current_metrics.log_metric(f"{var}_ssr", ssr.item())
            self.current_metrics.log_metric(f"{var}_rmse", rmse.item())

    def get_nino_metrics(self, eval_data: xr.Dataset):
        nino34_tgt, nino34_pred = self.get_nino34(eval_data["temp_ocn_0a_tgt"]), self.get_nino34(eval_data["temp_ocn_0a_pred"]).mean("ens")
        nino4_tgt, nino4_pred = self.get_nino4(eval_data["temp_ocn_0a_tgt"]), self.get_nino4(eval_data["temp_ocn_0a_pred"]).mean("ens")
        
        nino34_pcc = self.xr_pcc(nino34_pred, nino34_tgt, ("time",))
        nino4_pcc = self.xr_pcc(nino4_pred, nino4_tgt, ("time",))

        nino34_rmse = self.xr_rmse(nino34_pred, nino34_tgt, ("time",))
        nino4_rmse = self.xr_rmse(nino4_pred, nino4_tgt, ("time",))

        nino4_thresh_month =  1 + np.argwhere(nino4_pcc.values > 0.5).max(initial=0)
        nino34_thresh_month = 1 + np.argwhere(nino34_pcc.values > 0.5).max(initial=0)

        self.current_metrics.log_metric('nino4_pcc_month', float(nino4_thresh_month))
        self.current_metrics.log_metric('nino34_pcc_month', float(nino34_thresh_month))

        for lag in [3, 9, 15, 18, 21]:
            self.current_metrics.log_metric(f"nino34_pcc_{lag}", nino34_pcc.sel(lag = lag).item())
            self.current_metrics.log_metric(f"nino34_rmse_{lag}", nino34_rmse.sel(lag = lag).item())
            self.current_metrics.log_metric(f"nino4_pcc_{lag}", nino4_pcc.sel(lag = lag).item())
            self.current_metrics.log_metric(f"nino4_rmse_{lag}", nino4_rmse.sel(lag = lag).item())
        
    def log_metrics(self, metrics: dict, task: str = None):
        for key, val in metrics.items():
            name = f"{task}_{key}" if exists(task) and task != 'prior' else key
            self.current_metrics.log_metric(name, val)
    
    def compute_metrics_torch(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor):
        ens = ens[mask]
        obs = obs[mask]
        metrics = {
            "crps": self.compute_crps(pred=ens, obs=obs, fair =  self.use_fair_crps).item(),
            "ssr": self.compute_spread_skill(pred=ens, obs=obs).item(),
            "ign": self.compute_ign(pred=ens, obs=obs).item(),
            "spread": self.compute_spread(pred=ens).item(),
            "acc": self.compute_acc(pred=ens.mean(-1), obs=obs).item(),
            "rmse": self.compute_rmse(pred=ens.mean(-1), obs=obs).item(),
        }
        return metrics

    @staticmethod
    def get_nino4(da: xr.DataArray):
        return da.sel(lon=slice(160, 210), lat=slice(-5, 5)).mean(dim=['lon', 'lat'])
    
    @staticmethod
    def get_nino34(da: xr.DataArray):
        return da.sel(lon=slice(190, 240), lat=slice(-5, 5)).mean(dim=['lon', 'lat'])

    @staticmethod
    def xr_pcc(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]):
        num = (pred * obs).sum(dim)
        denom = np.sqrt((pred**2).sum(dim)) * np.sqrt((obs**2).sum(dim))
        return num / denom

    @staticmethod
    def xr_rmse(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]):
        return np.sqrt(((pred - obs) ** 2).mean(dim))

    @staticmethod
    def xr_spread_skill_ens(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]):
        K = pred.sizes["ens"]
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean("ens")
        spread = np.sqrt(pred.var("ens").mean(dim))
        skill = np.sqrt(((obs - mean) ** 2).mean(dim))
        return correction * (spread / skill)
    
    @staticmethod
    def compute_acc(pred, obs, eps = 1e-5)-> float:
        return (pred * obs).nansum().div(pred.pow(2).nansum().sqrt() * obs.pow(2).nansum().sqrt() + eps).item()
    
    @staticmethod
    def compute_rmse(pred, obs)-> float:
        return (pred - obs).pow(2).nanmean().sqrt().item()
    
    @staticmethod
    def compute_crps_ens(pred, obs, fair: bool = True)-> float:
        crps = f_kernel_crps(observation=obs, ensemble=pred, fair = fair)
        return crps.nanmean().item()
    
    @staticmethod
    def compute_ign_ens(pred, obs, eps = 1e-5)-> float:
        ign = f_gaussian_ignorance(observation=obs, mu=pred.mean(-1), sigma=pred.std(-1) + eps)
        return ign.nanmean().item()
    
    @staticmethod
    def compute_spread_ens(pred)-> float:
        return pred.var(-1).mean().sqrt().item()

    @staticmethod
    def compute_spread_skill_ens(pred, obs, eps = 1e-5) -> float:
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
