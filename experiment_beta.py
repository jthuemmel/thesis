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
        count = count_parameters(model)
        print(f'Created model with {count:,} parameters')
        self.misc_metrics.log_python_object("num_params", count)
        return model
    
    def create_loss(self):
        def loss_fn(ens: torch.Tensor, obs: torch.Tensor, visible: torch.BoolTensor, weight: torch.Tensor = 1.):
            mask = torch.logical_and(self.land_sea_mask, ~visible)
            score = f_kernel_crps(obs, ens)
            loss = (score * mask * weight).sum() / mask.sum().clamp(1.)
            self.current_metrics.log_metric(f"loss", loss.item())
            return loss 
        return loss_fn
    
    #FORWARD
    def forward_step(self, batch_idx, batch):
        if self.mode == "train":
            return self.step(batch, task = 'prior')
        else:
            _ = self.step(batch, task = 'prior')
            _ = self.step(batch, task = "frcst")
            if hasattr(self, 'ema_model'):
                _ = self.step(batch, task = "ema") 
            return None
        
    def step(self, batch: torch.Tensor, task: str = 'prior'):
        # tokenize
        tokens = self.field_to_tokens(batch.to(self.device))

        # mask
        t = self.timestep() if task == 'prior' else self.beta_history()
        weight = 1.
        visible = self.beta_schedule(t).bernoulli(generator=self.generator).bool()

        # predict
        prediction = self.ema_model(tokens, visible) if task == 'ema' else self.model(tokens, visible)

        # score
        ensemble = einops.rearrange(prediction, '(b n) ... (d e) -> b ... d (n e)', b = tokens.size(0), d = tokens.size(-1))
        loss = self.loss_fn(ensemble, tokens, visible, weight)

        # metrics
        metrics = self.compute_metrics(ens = ensemble, obs = tokens, vis = visible)
        self.log_metrics(metrics, task=task)
        return loss
    
    
    ### SAMPLING
    def uniform(self, shape: tuple):
        return torch.rand(shape, device=self.device, generator = self.generator)
            
    def beta_history(self):
        step = torch.zeros((self.world.token_sizes['t'],), device=self.device)
        step[:self.world.tau] = 1.
        return einops.repeat(step,
                             f't -> b {self.world.flat_token_pattern} ()',
                             **self.world.token_sizes, b=self.optim_cfg.batch_size)
    
    @staticmethod
    def beta_schedule(t: torch.Tensor):
        return 0.5 - 0.5 * torch.cos(torch.pi * t)

    @staticmethod
    def beta_dt(t: torch.Tensor):
        return 0.5 * torch.pi * torch.sin(torch.pi * t)
    
    def timestep(self):
        stratification = torch.linspace(0, 1, self.optim_cfg.batch_size, device=self.device).view(-1, 1)
        t = self.uniform((1, self.world.token_sizes['t'],))
        t = (t + stratification) % 1
        t = einops.repeat(t, f'b t -> b {self.world.flat_token_pattern} ()', **self.world.token_sizes)
        return t
    
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
            name = f"{task}_{key}" if task is not None else key
            self.current_metrics.log_metric(name, val.item())

    def compute_metrics(self, ens: torch.Tensor, obs: torch.Tensor, vis: torch.Tensor):
        mask = torch.logical_and(self.land_sea_mask, ~vis)
        ens = ens[mask]
        obs = obs[mask]
        metrics = {
            "crps": self.compute_crps(pred=ens, obs=obs),
            "ssr": self.compute_spread_skill(pred=ens, obs=obs),
            "ign": self.compute_ign(pred=ens, obs=obs),
            "spread": self.compute_spread(pred=ens),
            "acc": self.compute_acc(pred=ens.mean(-1), obs=obs),
            "rmse": self.compute_rmse(pred=ens.mean(-1), obs=obs),
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
            self._godas_data = NinoData(self.cfg.godas_path, self.data_cfg)
        return self._godas_data

    def picontrol_data(self):
        if not hasattr(self, "_picontrol_data"):
            picontrol_config = replace(self.data_cfg, time_slice = {"start": "1900", "stop": "2000", "step": None})
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