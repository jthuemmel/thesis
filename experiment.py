import torch
import os
import argparse
import math
import einops

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from dataclasses import replace
from omegaconf import OmegaConf
from torch.distributed.optim import ZeroRedundancyOptimizer

import utils.config as cfg
from utils.dataset import NinoData, MultifileNinoDataset
from utils.trainer import DistributedTrainer
from utils.einmask import EinMask
from utils.einvae import EinVAE
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
    def cfg(self):
        return self._cfg.trainer
    
    @property
    def data_cfg(self):
        return self._cfg.data

    @property
    def model_cfg(self):
        return self._cfg.model
    
    @property
    def world(self):
        return self._cfg.world
    
    @property
    def use_fair_crps(self):
        return default(self.world.ens_size, 1) > 1
    
    # DATA
    def lens_data(self):
        if not hasattr(self, "_lens_data"):
            lens_config = self.data_cfg
            lens_config = replace(lens_config, 
                                  time_slice = {"start": "1850", "stop": "2000", "step": None},
                                  stats = default(self.data_cfg.stats, cfg.LENS_STATS)
                                  )
            self._lens_data = MultifileNinoDataset(self.cfg.lens_path, lens_config, self.rank, self.world_size)
        return self._lens_data       

    def godas_data(self):
        if not hasattr(self, "_godas_data"):
            godas_config = self.data_cfg
            godas_config = replace(godas_config, 
                                   time_slice = {"start": "1980", "stop": "2020", "step": None},
                                   stats = default(self.data_cfg.stats, cfg.GODAS_STATS)
                                   )
            self._godas_data = NinoData(self.cfg.godas_path, godas_config)
        return self._godas_data

    def picontrol_data(self):
        if not hasattr(self, "_picontrol_data"):
            picontrol_config = self.data_cfg
            picontrol_config = replace(picontrol_config, 
                                       time_slice = {"start": "1900", "stop": "2000", "step": None},
                                       stats = default(self.data_cfg.stats, cfg.PICONTROL_STATS))
            self._picontrol_data = NinoData(self.cfg.picontrol_path, picontrol_config)
        return self._picontrol_data

    def oras5_data(self):
        if not hasattr(self, "_oras5_data"):
            oras5_config = self.data_cfg
            oras5_config = replace(oras5_config, time_slice = {"start": "1980", "stop": "2020", "step": None})
            self._oras5_data = NinoData(self.cfg.oras5_path, oras5_config)
        return self._oras5_data

    def create_dataset(self):
        # instantiate datasets
        self.train_dataset = self.lens_data()
        self.val_dataset = self.godas_data() if self.data_cfg.eval_data == "godas" else self.picontrol_data()

        # create land-sea mask
        self._val_lsm = torch.logical_not(self.val_dataset.land_sea_mask.to(device=self.device, dtype=torch.bool))
        self._train_lsm = torch.logical_not(self.train_dataset.land_sea_mask.to(device= self.device, dtype=torch.bool))

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
    def setup_misc(self):
        self.step_counter = 0
        self.create_prior()

    def create_prior(self):
        prefix = torch.zeros((self.world.token_sizes["t"],), device = self.device, dtype = torch.bool)
        prefix[:self.world.tau] = True
        self.frcst_prefix = einops.repeat(prefix, f't -> ({self.world.token_pattern})', **self.world.token_sizes)
        
        self.prior = MaskingMixture(
            world= self.world,
            event_cfg = self._cfg.objective.event_cfg,
            rate_cfg = self._cfg.objective.rate_cfg,
            ).to(self.device)
        
    def create_job_name(self):
        if exists(self.cfg.job_name):
            base_name = str(self.cfg.job_name).replace('/', '_')
        else:
            base_name = self.slurm_id
        self.job_name = f"{base_name}"
        self.cfg.job_name = self.job_name # enables resuming from config by using the job name
    
    def create_optimizer(self, named_params):
        if self.cfg.use_zero:
            optimizer = ZeroRedundancyOptimizer(
                [{'params': p} for n, p in named_params],
                torch.optim.AdamW,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=(self.cfg.beta1, self.cfg.beta2)
            )
        else:
            optimizer = torch.optim.AdamW(
                named_params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                betas=(self.cfg.beta1, self.cfg.beta2),
                fused=True
            )
        return optimizer

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

    def create_model(self):
        #model = EinMask(network=self.model_cfg, world=self.world)
        model = EinVAE(self.model_cfg.dim_in, self.world)
        return model

    # LOSS
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
            'tauxa': 0.1,
            'tauya': 0.1,
        }
        w = torch.as_tensor([weights.get(var, 1.) for var in self.data_cfg.variables], device = self.device)
        return einops.repeat(w, 
                             f"(v vv) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)
    
    @property
    def land_sea_mask(self) -> torch.BoolTensor:
        lsm = self._train_lsm if self.mode == "train" else self._val_lsm
        return einops.repeat(lsm, 
                             f"1 (h hh) (w ww) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes)
    
    def node_crps(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor):
        score = f_kernel_crps(observation=obs, ensemble=ens, fair = self.use_fair_crps)
        loss = (score * self.per_variable_weights)[mask].mean()
        return loss

    def spectral_crps(self, ens: torch.Tensor, obs: torch.Tensor, **kwargs):
        w_v = self.per_variable_weights.mean(dim = (-1, -2, -3))
        with torch.amp.autocast(enabled = True, device_type = self.device.type, dtype = torch.float32):
            e_fft = torch.fft.rfftn(ens.float(), dim = (-2, -3, -4)) #[B, V, ft, fh, fw, E]
            o_fft = torch.fft.rfftn(obs.float(), dim = (-1, -2, -3))
        score = f_kernel_crps(observation=o_fft, ensemble= e_fft, fair = self.use_fair_crps).mean(dim=(-1, -2, -3)) #[B, V]
        return score.mul(w_v).mean()
    
    def loss_fn(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor):
        w_spectral = self.cfg.spectral_loss_weight
        if w_spectral > 0:
            spectral_loss = self.spectral_crps(ens = ens, obs = obs, mask = mask)
            node_loss = self.node_crps(ens = ens, obs = obs, mask = mask)
            loss = w_spectral * spectral_loss + node_loss  
        else:
            loss = self.node_crps(ens = ens, obs = obs, mask = mask)
        return loss
    
    #FORWARD
    @property
    def total_epochs(self):
        return max(1, self.total_steps // len(self.train_dl))

    def forward_step(self, batch_idx, batch):
        x_hat, kl = self.model(batch, members = self.world.ens_size, rng = self.generator)

        with torch.amp.autocast(enabled = True, device_type = self.device.type, dtype = torch.float32):
            e_fft = torch.fft.rfftn(x_hat.float(), dim = (-2, -3, -4)) #[B, V, ft, fh, fw, E]
            o_fft = torch.fft.rfftn(batch.float(), dim = (-1, -2, -3))
        spectral = f_kernel_crps(observation=o_fft, ensemble= e_fft, fair = x_hat.size(-1) > 1).mean()
        crps = f_kernel_crps(observation = batch, ensemble = x_hat, fair = x_hat.size(-1) > 1)[:, self.land_sea_mask].mean()
        loss = crps + self.model_cfg.kwargs.get('beta', 1.) * kl + spectral * self.cfg.spectral_loss_weight

        metrics = {}
        metrics['spectral_crps'] = spectral.item()
        metrics['rmse'] = self.compute_rmse(x_hat.detach().mean(-1)[:, self.land_sea_mask], batch[:, self.land_sea_mask]).item()
        metrics['acc'] = self.compute_acc(x_hat.detach().mean(-1)[:, self.land_sea_mask], batch[:, self.land_sea_mask]).item()
        metrics['loss'] = loss.item()
        metrics['crps'] = crps.item()
        metrics['kl'] = kl.item()
        self.log_metrics(metrics)
        # update step counter if training
        self.step_counter = self.step_counter + 1 if self.mode == 'train' else self.step_counter
        return loss

    def mtm_step(self, batch_idx, batch):        
        # sample visible and masked
        visible, masked = self.prior(batch.size(0), rng = self.generator)

        # foward model and zero out land
        prediction = self.model(batch, visible, members = self.world.ens_size, rng = self.generator)
        prediction = prediction * self.land_sea_mask[..., None]
        
        # loss only on masked & sea
        mask = einops.repeat(masked, 
                             f"b ({self.world.token_pattern}) -> b {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes
                             )
        mask = torch.logical_and(mask, self.land_sea_mask)
        loss = self.loss_fn(ens = prediction, obs = batch, mask = mask)

        # track metrics
        metrics = self.compute_metrics(ens = prediction.float().detach(), obs = batch.float().detach(), mask = mask)
        metrics['loss'] = loss.item()
        self.log_metrics(metrics)

        # update step counter if training
        self.step_counter = self.step_counter + 1 if self.mode == 'train' else self.step_counter
        return loss
    
    def frcst_step(self, batch_idx, batch):
        visible = self.frcst_prefix.expand(batch.size(0), -1)
        prediction = self.model(batch, visible, members = self.world.ens_size, rng = self.generator)
        prediction = prediction * self.land_sea_mask[..., None]
        return prediction

    #EVAL
    def evaluate_epoch(self):
        super().evaluate_epoch()
        self.evaluate_vae()
        #self.evaluate_frcst()

    def evaluate_vae(self):
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        batch = next(iter(self.val_dl)).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type = self.device.type, enabled=self.cfg.mixed_precision):
                x_hat, _ = self.model(batch, members=self.world.ens_size, rng = self.generator)
        sample = x_hat.mean(-1).mul(self.land_sea_mask).cpu().float().numpy()
        batch = batch.cpu().float().numpy()
        
        if self.is_root:
            plt.figure(figsize=(12,6))
            plt.subplot(221)
            plt.pcolormesh(batch[0, 0, 5, :, :], vmin=-2, vmax = 2, cmap= 'bwr')
            plt.subplot(222)
            plt.pcolormesh(batch[0, 0, 6, :, :], vmin=-2, vmax = 2, cmap= 'bwr')
            plt.subplot(223)
            plt.pcolormesh(sample[0, 0, 5, :, :], vmin=-2, vmax = 2, cmap= 'bwr')
            plt.subplot(224)
            plt.pcolormesh(sample[0, 0, 6, :, :], vmin=-2, vmax = 2, cmap= 'bwr')
            plt.tight_layout()
            plt.savefig(self.model_dir / 'reconstruction.png')
            plt.close()

    def evaluate_frcst(self):
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        samples = []
        for batch_idx, batch in enumerate(self.val_dl):
            batch = batch.to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type = self.device.type, enabled=self.cfg.mixed_precision):
                    prediction = self.frcst_step(batch_idx, batch).cpu()
                    samples.append(self.get_xarray_dataset(batch_idx, obs = batch, pred = prediction))

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
        plt.subplot(323)
        ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 0).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        plt.subplot(324)
        ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 1).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        plt.subplot(325)
        ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 2).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        plt.subplot(326)
        ds[f"temp_ocn_0a_pred"].isel(time = 0, lag = 20, ens = 3).plot(vmin=-2, vmax = 2, cmap= 'bwr')
        plt.savefig(self.model_dir / "test_sample.png")
        plt.close()

        #RANK HIST
        plt.figure(figsize=(12,4))
        E = len(ds.ens)
        ens = ds[f"temp_ocn_0a_pred"].sel(lag = [1, 7, 13, 19]).values.reshape(-1, E)
        obs = ds[f"temp_ocn_0a_tgt"].sel(lag = [1, 7, 13, 19]).values.reshape(-1, 1)
        rank_counts = np.bincount(np.sum(ens < obs, axis= -1), minlength= E + 1) / ens.shape[0]
        plt.bar(np.arange(E + 1), rank_counts, alpha = 0.5)
        plt.hlines(1 / (E + 1), 0, E, color="red", linestyle="dashed", linewidth=1)
        plt.ylabel('Frequency')
        plt.xlabel("Rank")
        plt.savefig(self.model_dir / "rank_hist.png")
        plt.close()

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
            ssr = self.xr_spread_skill(pred, tgt, ('lat', 'lon')).mean(('time', 'lag'))
            
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
        
    def log_metrics(self, metrics: dict, task: str = None):
        for key, val in metrics.items():
            name = f"{task}_{key}" if exists(task) and task != 'prior' else key
            self.current_metrics.log_metric(name, val)

    def compute_metrics(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor):
        ens = ens[mask]
        obs = obs[mask]
        metrics = {
            "crps": self.compute_crps(pred=ens, obs=obs).item(),
            "ssr": self.compute_spread_skill(pred=ens, obs=obs).item(),
            "ign": self.compute_ign(pred=ens, obs=obs).item(),
            "spread": self.compute_spread(pred=ens).item(),
            "acc": self.compute_acc(pred=ens.mean(-1), obs=obs).item(),
            "rmse": self.compute_rmse(pred=ens.mean(-1), obs=obs).item(),
        }
        return metrics


    @staticmethod
    def get_nino4(da: xr.DataArray):
        return da.sel(lon=slice(160, 210), lat=slice(-5, 5)).mean(dim=['lon', 'lat'])#.rolling(time = 3).mean()
    
    @staticmethod
    def get_nino34(da: xr.DataArray):
        return da.sel(lon=slice(190, 240), lat=slice(-5, 5)).mean(dim=['lon', 'lat'])#.rolling(time = 3).mean()

    @staticmethod
    def xr_pcc(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]):
        num = (pred * obs).sum(dim)
        denom = np.sqrt((pred**2).sum(dim)) * np.sqrt((obs**2).sum(dim))
        return num / denom

    @staticmethod
    def xr_rmse(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]):
        return np.sqrt(((pred - obs) ** 2).mean(dim))

    @staticmethod
    def xr_spread_skill(pred: xr.DataArray, obs: xr.DataArray, dim: tuple[str]):
        K = pred.sizes["ens"]
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean("ens")
        spread = np.sqrt(pred.var("ens").mean(dim))
        skill = np.sqrt(((obs - mean) ** 2).mean(dim))
        return correction * (spread / skill)
    
    @staticmethod
    def compute_acc(pred, obs, eps = 1e-5):
        return (pred * obs).nansum() / (pred.pow(2).nansum().sqrt() * obs.pow(2).nansum().sqrt() + eps)
    
    @staticmethod
    def compute_rmse(pred, obs):
        return (pred - obs).pow(2).nanmean().sqrt()
    
    @staticmethod
    def compute_crps(pred, obs):
        crps = f_kernel_crps(observation=obs, ensemble=pred, fair = True)
        return crps.nanmean()
    
    @staticmethod
    def compute_ign(pred, obs, eps = 1e-5):
        ign = f_gaussian_ignorance(observation=obs, mu=pred.mean(-1), sigma=pred.std(-1) + eps)
        return ign.nanmean()
    
    @staticmethod
    def compute_spread(pred):
        return pred.var(-1).mean().sqrt()

    @staticmethod
    def compute_spread_skill(pred, obs, eps = 1e-5):
        K = pred.shape[-1]
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean(-1)
        spread = pred.var(-1).mean().sqrt()
        skill = (obs - mean).pow(2).mean().sqrt() + eps
        return correction * (spread / skill)

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
    trainer = Experiment(config)
    trainer.train()

if __name__ == "__main__":
    main()
