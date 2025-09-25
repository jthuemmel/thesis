# MIN Trainer

import torch
import os
import argparse
import math

from dataclasses import replace
from einops import rearrange, repeat, reduce
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import utils.config as cfg
from utils.loss_fn import *
from utils.dirichlet_multinomial import DirichletMultinomial
from utils.masked_models import *
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
class MAETrainer(DistributedTrainer):
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
        B, T, V = self.batch_size, self.data_cfg.sequence_length // self.world_cfg.patch_size["tt"], len(self.data_cfg.variables)
        return repeat(lsm, "1 (h hh) (w ww) -> b (t h w) (v tt hh ww)", b = B, t = T, v = V, **self.world_cfg.patch_size)

    @property
    def per_variable_weights(self):
        if not hasattr(self.data_cfg, "weights") or not exists(self.data_cfg.weights):
            return 1.
        w = self.data_cfg.weights
        w = torch.tensor(w, device = self.device)
        return repeat(w, "v -> (v tt hh ww)", **self.world_cfg.patch_size)

    # SETUP
    def create_job_name(self):
        if exists(self.cfg.job_name):
            base_name = str(self.cfg.job_name).replace('/', '_')
        else:
            base_name = self.slurm_id
        self.job_name = f"{base_name}"
        self.cfg.job_name = self.job_name # enables resuming from config by using the job name

    def create_model(self):
        if exists(self.model_cfg.encoder) and exists(self.model_cfg.decoder):
            if exists(self.world_cfg.num_ens):
                model = EnsembleMAE(self.model_cfg.encoder, self.model_cfg.decoder)
                model.noise.generator = self.generator
            else:
                model = MAE(self.model_cfg.encoder, self.model_cfg.decoder)
        elif exists(self.model_cfg.mtm):
            model = MTM(self.model_cfg.mtm)
        else:
            raise ValueError("No valid model configuration found.")
        
        self.misc_metrics.log_python_object("num_params", sum(p.numel() for p in model.parameters()))
        return model
    
    def create_loss(self):
        def nino_loss(pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor):
            # pointwise crps -> [B, N, D]
            if self.world_cfg.crps_alpha == 0:
                loss = f_kernel_crps(observation=obs, ensemble=pred, fair= False)
            elif self.world_cfg.crps_alpha == 1:
                loss = f_kernel_crps(observation=obs, ensemble=pred, fair= True)
            else:
                loss = f_almost_fair_kernel_crps(observation=obs, ensemble=pred, alpha=self.world_cfg.crps_alpha)
            # apply weights along D
            loss = loss * self.per_variable_weights
            # apply land-sea mask -> [B * N * D]
            loss = loss[lsm]
            # reduce to scalar
            return loss.nanmean()
        return nino_loss

    def create_optimizer(self, named_params):
        return torch.optim.AdamW(
            named_params,
            lr=self.optim_cfg.lr,
            weight_decay=self.optim_cfg.weight_decay,
            betas=(self.optim_cfg.beta1, self.optim_cfg.beta2),
        )

    def create_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.optim_cfg.num_epochs,
            eta_min=self.optim_cfg.eta_min,
        )
    
    # METRICS
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
        return ens_pred.var(-1).mean().sqrt()

    @staticmethod
    def compute_spread_skill(pred, obs):
        K = pred.shape[-1]
        correction = math.sqrt((K + 1) / K)
        mean = pred.mean(-1)
        spread = pred.var(-1).mean().sqrt()
        skill = (obs - mean).pow(2).mean().sqrt()
        return correction * (spread / skill)
    
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

    @staticmethod
    def _slice_from_cfg(s):
        if isinstance(s, dict):
            return slice(s["start"], s["stop"], s["step"])
        elif isinstance(s, slice):
            return s
        else:
            return slice(None)

    def select_field_subset(self, x: torch.Tensor, var_idx: slice | int, time_idx: slice | int):
        _, _, S = self.get_flatland_shape()
        # extract time from flatland format and variables from token dimension
        x = rearrange(x, "b (t s) (v tt hh ww) ... -> v (t tt) b s (hh ww) ...", **self.world_cfg.patch_size, s = S)
        return x[var_idx, time_idx]
    
    def get_frcst_metrics(self, pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor):
        frcst_tasks = {"frcst": (slice(None), slice(None))} # default task
        if exists(self.data_cfg.frcst_tasks): # additional tasks from config
            frcst_tasks.update(self.data_cfg.frcst_tasks)
        for label, (vi, ti) in frcst_tasks.items(): # for all tasks
            vi, ti = map(lambda x: self._slice_from_cfg(x), (vi, ti))
            p, o, l = (self.select_field_subset(x, vi, ti) for x in (pred, obs, lsm)) # select the task-specific subset
            self.compute_metrics(p, o, l, label= label) # compute all metrics and label them according to the task

    ### MASKING
    def setup_misc(self):
        self.masking_prior = DirichletMultinomial(self.world_cfg.alpha, device=self.device, generator=self.generator)

    def sample_masking_rates(self):
        N = self.world_cfg.num_tokens
        R_src, R_tgt = torch.empty((2,), device = self.device)
        if self.world_cfg.mask_rates_src["std"] > 0:
            torch.nn.init.trunc_normal_(R_src, **self.world_cfg.mask_rates_src, generator = self.generator)
        else:
            R_src.fill_(self.world_cfg.mask_rates_src["mean"])
        if self.world_cfg.mask_rates_tgt["std"] > 0:
            torch.nn.init.trunc_normal_(R_tgt, **self.world_cfg.mask_rates_tgt, generator = self.generator)
        else:
            R_tgt.fill_(self.world_cfg.mask_rates_tgt["mean"])
        
        M_src, M_tgt = int(N * R_src.item()), int(N * R_tgt.item())
        return M_src, M_tgt

    def get_flatland_shape(self):
        # flatland shape: (batch, time, space) 
        B = self.batch_size
        T = self.data_cfg.sequence_length // self.world_cfg.patch_size["tt"]
        S = self.world_cfg.num_tokens // T
        return B, T, S

    def forecast_mask(self):
        B, T, S = self.get_flatland_shape()
        tau = self.world_cfg.tau
        indices = repeat(torch.arange(T * S, device = self.device), "i -> b i", b = B)
        # (batch, (time space)) -> (batch, (src_time, space), (tgt_time, space))
        src_mask, tgt_mask = indices.split([S * tau, (T - tau) * S], dim = 1) 
        return src_mask, tgt_mask

    def sample_masks(self):
        M_src, M_tgt = self.sample_masking_rates()
        B, T, S = self.get_flatland_shape()
        src_mask, tgt_mask = self.masking_prior((B, T, S), M_src, M_tgt) #(batch, time, space) -> (batch, M)
        return src_mask, tgt_mask
    
    def apply_masks(self, tokens, src_mask, tgt_mask): 
        src = self.masking_prior.apply_masking(tokens, src_mask)
        tgt = self.masking_prior.apply_masking(tokens, tgt_mask)
        lsm_src = self.masking_prior.apply_masking(self.land_sea_mask, src_mask)
        lsm_tgt = self.masking_prior.apply_masking(self.land_sea_mask, tgt_mask)
        return src, tgt, lsm_src, lsm_tgt

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
        tokens = rearrange(batch.to(self.device), 
                           "b v (t tt) (h hh) (w ww) -> b (t h w) (v tt hh ww)", 
                           **self.world_cfg.patch_size
                           )
        
        # masking
        src_mask, tgt_mask = self.sample_masks() if task == "train" else self.forecast_mask()
        src, tgt, _, lsm_tgt = self.apply_masks(tokens, src_mask, tgt_mask)

        # forward
        with sdpa_kernel(self.backend):
            if exists(self.world_cfg.num_ens):
                src_pred, tgt_pred = self.model(src, src_mask, tgt_mask, K = self.world_cfg.num_ens)
            else: 
                src_pred, tgt_pred = self.model(src, src_mask, tgt_mask)
                D = tgt.size(-1)
                src_pred, tgt_pred = map(lambda x: rearrange(x, "... (d ens) -> ... d ens", d = D), (src_pred, tgt_pred))
            
        # compute loss
        loss = self.loss_fn(pred = tgt_pred, obs = tgt, lsm = lsm_tgt)
        
        # metrics
        if task == "train":
            self.current_metrics.log_metric(f"loss", loss.item())
            self.compute_metrics(pred = tgt_pred, obs = tgt, lsm = lsm_tgt, label = None)
        elif task == "frcst":
            self.get_frcst_metrics(pred = tgt_pred, obs = tgt, lsm = lsm_tgt)                   

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
    config = cfg.MAEConfig.from_omegaconf(merged_cfg)

    #Run the trainer
    trainer = MAETrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()