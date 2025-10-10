import torch
import os
import argparse
import math
import einops

import xarray as xr
import numpy as np

from dataclasses import replace
from omegaconf import OmegaConf
from torch.distributed.optim import ZeroRedundancyOptimizer

import utils.config as cfg
from utils.dataset import NinoData, MultifileNinoDataset
from utils.trainer import DistributedTrainer
from utils.model import MaskedPredictor
from utils.masking import Masking
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
    def optim_cfg(self):
        return self._cfg.optim

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
    def masking_cfg(self):
        return self._cfg.masking       

    @property
    def use_fair_crps(self):
        return self.world.num_ens > 1 
    
    # DATA
    def lens_data(self):
        if not hasattr(self, "_lens_data"):
            lens_config = self.data_cfg
            lens_config = replace(lens_config, stats = default(self.data_cfg.stats, cfg.LENS_STATS))
            lens_config = replace(lens_config, time_slice = {"start": "1850", "stop": "2000", "step": None})
            self._lens_data = MultifileNinoDataset(self.cfg.lens_path, lens_config, self.rank, self.world_size)
        return self._lens_data       

    def godas_data(self):
        if not hasattr(self, "_godas_data"):
            godas_config = self.data_cfg
            godas_config = replace(godas_config, stats = default(self.data_cfg.stats, cfg.GODAS_STATS))
            godas_config = replace(godas_config, time_slice = {"start": "1980", "stop": "2020", "step": None})
            self._godas_data = NinoData(self.cfg.godas_path, godas_config)
        return self._godas_data

    def picontrol_data(self):
        if not hasattr(self, "_picontrol_data"):
            picontrol_config = self.data_cfg
            picontrol_config = replace(picontrol_config, stats = default(self.data_cfg.stats, cfg.PICONTROL_STATS))
            picontrol_config = replace(picontrol_config, time_slice = {"start": "1900", "stop": "2000", "step": None})
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
            batch_size=self.optim_cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        val_dl = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.optim_cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False
        )
        return train_dl, val_dl
    
    # SETUP
    def setup_misc(self):
        self.mask_generator = Masking(
            masking_cfg = self.masking_cfg, world_cfg= self.world, optim_cfg= self.optim_cfg, 
            device= self.device, generator= self.generator
            )
        
    def create_job_name(self):
        if exists(self.cfg.job_name):
            base_name = str(self.cfg.job_name).replace('/', '_')
        else:
            base_name = self.slurm_id
        self.job_name = f"{base_name}"
        self.cfg.job_name = self.job_name # enables resuming from config by using the job name
    
    def create_optimizer(self, named_params):
        if self.cfg.use_zero:
            return ZeroRedundancyOptimizer(
                [{'params': p} for n, p in named_params],
                torch.optim.AdamW,
                lr=self.optim_cfg.lr,
                weight_decay=self.optim_cfg.weight_decay,
                betas=(self.optim_cfg.beta1, self.optim_cfg.beta2)
            )
        else:
            return torch.optim.AdamW(
                named_params,
                lr=self.optim_cfg.lr,
                weight_decay=self.optim_cfg.weight_decay,
                betas=(self.optim_cfg.beta1, self.optim_cfg.beta2),
            )

    def create_scheduler(self, optimizer):
        if self.cfg.scheduler_step == 'batch':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer = optimizer,
                max_lr = self.optim_cfg.lr,
                total_steps = self.optim_cfg.total_steps,
                pct_start = self.optim_cfg.warmup_steps / self.optim_cfg.total_steps,
                cycle_momentum = False,
                div_factor = self.optim_cfg.div_factor,
                final_div_factor = self.optim_cfg.final_div_factor,
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.optim_cfg.num_epochs,
                eta_min=self.optim_cfg.eta_min,
                last_epoch=-1,
            )
    
    def create_model(self):
        model = MaskedPredictor(
            self.model_cfg, self.world, 
            generator= self.generator
            )
        return model
    
    # LOSS
    def create_loss(self):
        def loss_fn(ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor, mask_weight: torch.Tensor = 1.):
            w_spectral = self.world.spectral_loss_weight
            if w_spectral > 0:
                spectral_loss = self.spectral_crps(ens, obs, mask, mask_weight)
                node_loss = self.node_crps(ens, obs, mask, mask_weight)
                loss = w_spectral * spectral_loss + node_loss  
            else:
                loss = self.node_crps(ens, obs, mask, mask_weight)
            return loss 
        return loss_fn

    @property
    def per_variable_weights(self):
        weights = {
            'temp_ocn_0a': 1.,
            'temp_ocn_1a': 0.1,
            'temp_ocn_3a': 0.1,
            'temp_ocn_5a': 0.1,
            'temp_ocn_8a': 0.1,
            'temp_ocn_11a': 0.1,
            'temp_ocn_14a': 0.1,
            'tauxa': 0.01,
            'tauya': 0.01,
        }
        w = torch.as_tensor([weights.get(var, 1.) for var in self.data_cfg.variables], device = self.device)
        return einops.repeat(w, 
                             f"(v vv) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes, b = self.optim_cfg.batch_size)
    
    @property
    def land_sea_mask(self):
        lsm = self._train_lsm if self.mode == "train" else self._val_lsm
        return einops.repeat(lsm, 
                             f"1 (h hh) (w ww) -> {self.world.field_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes, b = self.optim_cfg.batch_size)
    
    def node_crps(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor, mask_weight: torch.Tensor = 1.):
        score = f_kernel_crps(obs, ens, fair = self.use_fair_crps)
        score = score * self.per_variable_weights
        loss = (score * mask_weight)[mask].mean()
        return loss

    def spectral_crps(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor, mask_weight: torch.Tensor = 1.):
        m, w_m, w_v = map(lambda x: x.float().mean(dim = (-1, -2)), (mask, mask_weight, self.per_variable_weights))
        with torch.amp.autocast(enabled = True, device_type = self.device.type, dtype = torch.float32):
            e_fft = torch.fft.rfftn(ens.float(), dim = (-2, -3)) #[B, V, T, fh, fw, E]
            o_fft = torch.fft.rfftn(obs.float(), dim = (-1, -2))
        score = f_kernel_crps(o_fft, e_fft, fair = self.use_fair_crps).mean(dim=(-1, -2)) #[B, V, T]
        correction = w_v * w_m / (1.0 - m.clamp(max = 1 / self.world.num_elements))
        return (score * correction).mean()

    #FORWARD       
    def frcst_step(self, batch_idx, batch):
        batch = batch.to(self.device)
        tokens = self.field_to_tokens(batch)
        model = self.model if (self.mode == 'train' or not self.cfg.use_ema) else self.ema_model

        visible, _ = self.mask_generator(
            timestep= 'history',
            schedule= 'uniform',
            mask= 'bernoulli',
        )

        E = self.world.num_ens
        if E > 1:
            tokens = einops.repeat(tokens, 'b ... -> (b e) ...', e = E)
            vis = einops.repeat(visible, 'b ... -> (b e) ...', e = E)
            prediction = model(tokens, vis)
            prediction = einops.rearrange(prediction, '(b e) ... k -> b ... (e k)', e = E)
        else:
            prediction = model(tokens, visible)

        prediction = self.tokens_to_field(prediction) * self.land_sea_mask[..., None]
        return prediction

    def forward_step(self, batch_idx, batch):
        batch = batch.to(self.device)
        tokens = self.field_to_tokens(batch)
        model = self.model if (self.mode == 'train' or not self.cfg.use_ema) else self.ema_model

        visible, weight = self.mask_generator(
            timestep= self.masking_cfg.timestep,
            schedule= self.masking_cfg.schedule,
            mask= self.masking_cfg.mask,
        )

        E = self.world.num_ens
        if E > 1:
            tokens = einops.repeat(tokens, 'b ... -> (b e) ...', e = E)
            vis = einops.repeat(visible, 'b ... -> (b e) ...', e = E)
            prediction = model(tokens, vis)
            prediction = einops.rearrange(prediction, '(b e) ... k -> b ... (e k)', e = E)
        else:
            prediction = model(tokens, visible)

        prediction = self.tokens_to_field(prediction) * self.land_sea_mask[..., None]
        vis, weight = self.mask_to_field(visible), self.mask_to_field(weight)

        mask = torch.logical_and(self.land_sea_mask, ~vis)
        loss = self.loss_fn(prediction, batch, mask, weight)

        metrics = self.compute_metrics(ens = prediction.detach(), obs = batch, mask= mask)
        metrics['loss'] = loss.item()
        self.log_metrics(metrics)
        return loss
    
    # Tokenizers
    def field_to_tokens(self, field):
        return einops.rearrange(field, 
                                f'{self.world.field_pattern} -> {self.world.flatland_pattern}',
                                **self.world.patch_sizes)

    def tokens_to_field(self, patch):
        return einops.rearrange(patch, 
                                f"{self.world.flatland_pattern} ... -> {self.world.field_pattern} ...",
                                **self.world.token_sizes, **self.world.patch_sizes)
    
    def mask_to_field(self, mask):
        return einops.repeat(mask,
                             f"{self.world.flat_mask_pattern} -> {self.world.field_pattern}",
                                **self.world.token_sizes, **self.world.patch_sizes)

    # EVAL
    def evaluate_epoch(self):
        super().evaluate_epoch()
        self.evaluate_frcst()

    def evaluate_frcst(self):
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        samples = []
        for batch_idx, batch in enumerate(self.val_dl):
            with torch.no_grad():
                with torch.amp.autocast(device_type = self.device.type, enabled=self.cfg.mixed_precision):
                    prediction = self.frcst_step(batch_idx, batch)
                    samples.append(self.get_xarray_dataset(batch_idx, obs = batch, pred = prediction))

        ds = xr.concat(samples, dim = "time")
        ds = ds.sel(lat = slice(-20., 20.), lon = slice(90, 270))
        self.get_nino_metrics(ds)
        self.get_field_metrics(ds)

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
            mean = self.val_dataset._means.sel(variable = var).values
            p = pred[:, v, history:].float().detach().cpu().numpy() * std + mean
            o = obs[:, v, history:].float().detach().cpu().numpy() * std + mean

            #create xarray
            data_array = xr.Dataset(
                data_vars = {
                    f"{var}_pred": (["time", "lag", "lat", "lon", "ens"], p),
                    f"{var}_tgt": (["time", "lag", "lat", "lon"], o),
                },
                coords = {
                    "time": time[batch_idx * self.optim_cfg.batch_size: (batch_idx + 1) * self.optim_cfg.batch_size],
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
        nino34_pcc = self.xr_pcc(nino34_pred, nino34_tgt, ("time",))
        nino34_rmse = self.xr_rmse(nino34_pred, nino34_tgt, ("time",))        
        for lag in [3, 9, 15, 21]:
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
        return da.sel(lon=slice(160, 210), lat=slice(-5, 5)).mean(dim=['lon', 'lat']).rolling(time = 3).mean()
    
    @staticmethod
    def get_nino34(da: xr.DataArray):
        return da.sel(lon=slice(190, 240), lat=slice(-5, 5)).mean(dim=['lon', 'lat']).rolling(time = 3).mean()

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
    def compute_spread(pred):
        return pred.var(-1).mean().sqrt()

    @staticmethod
    def compute_spread_skill(pred, obs):
        K = pred.shape[-1]
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean(-1)
        spread = pred.var(-1).mean().sqrt()
        skill = (obs - mean).pow(2).mean().sqrt()
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