import torch
import os
import argparse
import math
import einops

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
    
    # SETUP
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
                div_factor = self.optim_cfg.lr / self.optim_cfg.eta_min
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.optim_cfg.num_epochs,
                eta_min=self.optim_cfg.eta_min,
                last_epoch=-1,
            )
    
    def create_model(self):
        model = MaskedPredictor(self.model_cfg, self.world)
        return model
    
    def create_loss(self):
        def loss_fn(ens: torch.Tensor, obs: torch.Tensor, visible: torch.BoolTensor, weight: torch.Tensor = 1.):
            mask = torch.logical_and(self.land_sea_mask, ~visible)
            score = f_kernel_crps(obs, ens)
            loss = (score * weight)[mask].mean()
            return loss 
        return loss_fn
    
    def setup_misc(self):
        self.mask_generator = Masking(world_cfg= self.world, optim_cfg= self.optim_cfg, device= self.device, generator= self.generator)

    #FORWARD       
    def frcst_step(self, batch_idx, batch):
        batch = batch.to(self.device)
        tokens = self.field_to_tokens(batch)
        visible, _ = self.mask_generator(
            timestep= 'history',
            schedule= 'uniform',
            mask= 'bernoulli',
        )
        prediction = self.model(tokens, visible) if self.mode == 'train' or not self.cfg.use_ema else self.ema_model(tokens, visible)
        ensemble = einops.rearrange(prediction, '(b k) ... (d e) -> b ... d (k e)', b = tokens.size(0), d = tokens.size(-1))
        ensemble = self.tokens_to_field(ensemble)
        return batch, ensemble

    def forward_step(self, batch_idx, batch):
        # tokenize
        tokens = self.field_to_tokens(batch.to(self.device))
        visible, weight = self.mask_generator(
            timestep= self.world.timestep,
            schedule= self.world.schedule,
            mask= self.world.mask,
        )
        prediction = self.model(tokens, visible) if self.mode == 'train' or not self.cfg.use_ema else self.ema_model(tokens, visible)
        ensemble = einops.rearrange(prediction, '(b k) ... (d e) -> b ... d (k e)', b = tokens.size(0), d = tokens.size(-1))
        loss = self.loss_fn(ensemble, tokens, visible, weight)

        # metrics
        metrics = self.compute_metrics(ens = ensemble, obs = tokens, vis = visible)
        metrics['loss'] = loss.item()
        self.log_metrics(metrics)
        return loss
    
    # Tokenizer
    def field_to_tokens(self, field):
        return einops.rearrange(field, 
                                f'{self.world.field_pattern} -> {self.world.flatland_pattern}',
                                **self.world.patch_sizes)

    def tokens_to_field(self, patch):
        return einops.rearrange(patch, 
                                f"{self.world.flatland_pattern} ... -> {self.world.field_pattern} ...",
                                **self.world.token_sizes, **self.world.patch_sizes)
    
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
    
    def log_metrics(self, metrics: dict, task: str = None):
        for key, val in metrics.items():
            name = f"{task}_{key}" if exists(task) and task != 'prior' else key
            self.current_metrics.log_metric(name, val)

    def compute_metrics(self, ens: torch.Tensor, obs: torch.Tensor, vis: torch.Tensor):
        mask = torch.logical_and(self.land_sea_mask, ~vis)
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
            godas_config = replace(self.data_cfg, time_slice = {"start": "1980", "stop": "2015", "step": None})
            self._godas_data = NinoData(self.cfg.godas_path, time_slice= godas_config)
        return self._godas_data

    def picontrol_data(self):
        if not hasattr(self, "_picontrol_data"):
            picontrol_config = replace(self.data_cfg, time_slice = {"start": "1800", "stop": "2000", "step": None})
            self._picontrol_data = NinoData(self.cfg.picontrol_path, picontrol_config)
        return self._picontrol_data

    def oras5_data(self):
        if not hasattr(self, "_oras5_data"):
            oras5_config = replace(self.data_cfg, time_slice = {"start": "1980", "stop": "2015", "step": None})
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
        )
        return train_dl, val_dl

    @property
    def land_sea_mask(self):
        lsm = self._train_lsm if self.mode == "train" else self._val_lsm
        return einops.repeat(lsm, 
                             f"1 (h hh) (w ww) -> {self.world.flatland_pattern}", 
                             **self.world.token_sizes, **self.world.patch_sizes, b = self.optim_cfg.batch_size)
    
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