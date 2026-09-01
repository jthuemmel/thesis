import torch
import os
import math
import argparse
import einops

import xarray as xr
import numpy as np

from dataclasses import replace
from omegaconf import OmegaConf

from utils.config import *
from utils.dataset import *
from utils.trainer import *
from utils.einmask import *
from utils.masking import *
from utils.loss_fn import *

from analysis.pieces import Evaluation

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
    def eval_cfg(self) -> EvaluationConfig:
        return self._cfg.evaluation

    @property
    def evaluation(self) -> Evaluation:
        if not hasattr(self, "_evaluation"):
            self._evaluation = Evaluation(self.eval_cfg)
        return self._evaluation

    @property
    def use_ens(self) -> bool:
        return self.world.kwargs.get('ensemble', False)

    @property
    def use_fair_crps(self) -> bool:
        return default(self.world.ens_size, 1) > 1

    @property
    def land_sea_mask(self) -> torch.BoolTensor:
        return self._train_lsm if self.mode == "train" else self._val_lsm

    @property
    def total_epochs(self) -> int:
        return max(1, self.total_steps // len(self.train_dl))

    @property
    def per_variable_weights(self) -> torch.FloatTensor:
        weights = {}
        w = torch.as_tensor([weights.get(var, 1.) for var in self.data_cfg.variables], device = self.device)
        return einops.repeat(w,
                             f"(v vv) -> {self.world.field_pattern}",
                             **self.world.token_sizes, **self.world.patch_sizes)

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

    def get_test_data(self, name: str) -> NinoData:
        if name == 'godas':
            return self.godas_data()
        if name == 'oras5':
            return self.oras5_data()
        return self.picontrol_data()

    def create_dataset(self):
        # instantiate datasets
        self.train_dataset = self.lens_data()
        self.val_dataset = self.get_test_data(self.data_cfg.eval_data.lower())

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
        return EinMask_ENS(network=self.model_cfg, world=self.world)

    # MASKING
    @property
    def frcst_prefix(self) -> torch.BoolTensor:
        return self.prefix_mask()

    def prefix_mask(self, reverse: bool = False) -> torch.BoolTensor:
        prefix = torch.zeros((self.world.token_sizes["t"],), device = self.device, dtype = torch.bool)
        if reverse:
            prefix[-self.world.tau:] = True
        else:
            prefix[:self.world.tau] = True
        return einops.repeat(prefix, f't -> ({self.world.token_pattern})', **self.world.token_sizes)

    def sample_normal_rates_(self, mean: float, std: float, a: float = 0., b: float = 1.):
        return torch.nn.init.trunc_normal_(
            torch.empty((1,), device = self.device),
            mean = mean, std = std, a = a, b = b, generator = self.generator
            ).mul(self.world.num_tokens).long()

    def sample_noise(self, K: int, num_samples: int):
        block_weight = self.objective.kwargs.get('block_weight', None)
        if exists(block_weight):
            d = torch.arange(1, K + 1, device = self.device)
            d = d[K % d == 0]
            idx = torch.multinomial(1 / d.pow(block_weight), 1, generator= self.generator)
            KK = d[idx]
            U = torch.rand((num_samples, K // KK), device= self.device, generator= self.generator)
            U = einops.repeat(U, f'... k -> ... (k kk)', kk = KK, k = K // KK)
        else:
            U = torch.rand((num_samples, K), device=self.device, generator=self.generator)
        return U

    def sample_weighted_reservoir(self, num_samples: int):
        P = torch.rand((num_samples, self.world.num_tokens), device=self.device, generator=self.generator).log()
        for dim, alpha in self.objective.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = self.sample_noise(self.world.token_sizes[dim], num_samples)
            U = einops.repeat(U, f'b {dim} -> b ({self.world.token_pattern})', **self.world.token_sizes)
            P += U.log().div(alpha)
        return P

    def sample_antithetic_weighted_reservoir(self, num_samples: int):
        P1, P2 = torch.rand((2, num_samples, self.world.num_tokens), device=self.device, generator=self.generator).log()
        for dim, alpha in self.objective.event_cfg.items():
            if not (exists(alpha) and dim in self.world.layout): continue
            U = self.sample_noise(self.world.token_sizes[dim], num_samples)
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
            crps = f_kernel_crps(observation=batch, ensemble=samples, fair=self.use_fair_crps)
            loss = crps.mul(self.per_variable_weights)[mask].mean()

            # maybe spectral loss
            spectral_weight = self.cfg.loss_kwargs.get('spectral_weight', 0.)
            if default(spectral_weight, 0.) > 0.:
                with torch.amp.autocast(enabled = True, device_type = self.device.type, dtype = torch.float32):
                    e_fft = torch.fft.rfftn(samples.float(), dim = (-2, -3, -4)) #[B, V, ft, fh, fw, E]
                    o_fft = torch.fft.rfftn(batch.float(), dim = (-1, -2, -3))
                spectral_crps = f_kernel_crps(o_fft, e_fft, self.use_fair_crps).mean()
                loss = loss + spectral_crps * spectral_weight

            metrics = {'loss' : loss.item(), **self.compute_metrics_torch_ens(samples, batch, mask)}
        else:
            # samples: (B, V, T, H, W, 2) — unpack mu/sigma heads
            mu = samples[..., 0] * self.land_sea_mask
            sigma = torch.nn.functional.softplus(samples[..., 1])
            crps = f_gaussian_crps(batch, mu, sigma)
            loss = crps.mul(self.per_variable_weights)[mask].mean()
            metrics = {'loss' : loss.item(), **self.compute_metrics_torch_mve(mu, sigma, batch, mask)}
            samples = torch.stack([mu, sigma], dim=-1)

        self.log_metrics(metrics, task = None)
        self.step_counter = self.step_counter + 1 if self.mode == 'train' else self.step_counter
        return loss, samples, visible

    def frcst_step(self, batch_idx, batch):
        visible = self.frcst_prefix.expand(batch.size(0), -1)

        # forward model
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
            # samples: (B, V, T, H, W, 2) — unpack mu/sigma heads
            mu = samples[..., 0] * self.land_sea_mask
            sigma = torch.nn.functional.softplus(samples[..., 1])
            loss = f_gaussian_crps(batch, mu, sigma)[mask].mean()
            step_metrics = self.compute_metrics_torch_mve(mu, sigma, batch, mask)
            samples = torch.stack([mu, sigma], dim=-1)

        metrics = {'loss' : loss.item(), **step_metrics}
        self.log_metrics(metrics, task = 'frcst')
        return loss, samples, visible

    # VALIDATION
    def evaluate_epoch(self):
        super().evaluate_epoch()
        self.evaluate_step('frcst')

    def evaluate_step(self, step: str = 'frcst'):
        # lean validation pass: scalar metrics only, drives val csv and early stopping
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        for batch_idx, batch in enumerate(self.val_dl):
            batch = batch.to(self.device)
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
                    _ = self.forward_step(batch_idx, batch, step)

    # TEST DISPATCH
    @property
    def test_tasks(self) -> dict:
        # {name : (mask_fn(num_samples, rng) -> visible, tier)}
        # placeholder battery — the objective refactor will build these from ObjectiveConfig
        if not hasattr(self, '_test_tasks'):
            self._test_tasks = {
                'frcst' : (lambda B, rng: self.prefix_mask().expand(B, -1), 'full'),
                'bcast' : (lambda B, rng: self.prefix_mask(reverse=True).expand(B, -1), 'standard'),
            }
        return self._test_tasks

    def post_batch(self):
        if not exists(self.eval_cfg.test_every) or self.mode != 'train':
            return
        if self.step_counter > 0 and self.step_counter % self.eval_cfg.test_every == 0:
            self.dispatch_tests()

    def dispatch_tests(self):
        try:
            self.save_weights()
            self.run_test_battery()
        except Exception as err:
            print(f'Rank {self.rank}: test dispatch failed at step {self.step_counter}: {err!r}')
        finally:
            self.switch_mode(train=True)

    def save_weights(self):
        if not (self.is_root and self.eval_cfg.save_weights):
            return
        weight_dir = self.model_dir / 'weights'
        weight_dir.mkdir(exist_ok=True)
        state = {k : v.half() if v.is_floating_point() else v for k, v in self.model.module.state_dict().items()}
        torch.save(state, weight_dir / f'step_{self.step_counter}.pt')

    def run_test_battery(self):
        self.switch_mode(train=False)
        datasets = default(self.eval_cfg.test_data, [self.data_cfg.eval_data])
        battery = [(name, task) for name in datasets for task in self.test_tasks]

        # shard cells across ranks — pieces are additive, each rank computes whole cells
        results = []
        for i, (name, task) in enumerate(battery):
            if i % self.world_size != self.rank: continue
            results.append(self.run_test_cell(name, task, seed=i))

        gathered = [None] * self.world_size if self.is_root else None
        dist.gather_object(results, gathered)

        if self.is_root:
            path = self.model_dir / 'eval.zarr'
            attrs = {'job_name' : self.job_name, 'seed' : self.cfg.seed}
            for group, static, pieces in (cell for cells in gathered for cell in cells):
                if exists(static):
                    self.evaluation.write_static(path, static, group)
                self.evaluation.write(path, pieces, group, self.step_counter, attrs=attrs)

    def run_test_cell(self, name: str, task: str, seed: int):
        data = self.get_test_data(name)
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.world.kwargs.get('test_batch_size', self.world.batch_size), # forward-only, can exceed the train batch
            num_workers=0,
            drop_last=True,
            shuffle=False,
            )
        mask_fn, tier = self.test_tasks[task]
        # fixed per-task seed keeps masks dispatch-invariant, so the static group stays valid
        rng = torch.Generator(device=self.device).manual_seed(default(self.cfg.seed, 0) + seed)
        group = f'{name}/{task}'
        need_static = not self.evaluation.has_group(self.model_dir / 'eval.zarr', f'{group}/static')

        # maybe enlarge the ensemble at test time — forward reads world.ens_size per call
        # only meaningful with a noise generator (ens_size > 1 at init); tails-only models would emit duplicates
        # memory scales with test_batch_size * test_ens_size; keep the schema fixed across dispatches
        ens_size = self.world.ens_size
        if self.use_fair_crps:
            self.world.ens_size = self.world.kwargs.get('test_ens_size', ens_size)
        E_test = default(self.world.ens_size, 1)
        stds = {var : float(data._stds.sel(variable=var)) for var in self.data_cfg.eval_variables}

        chunks, static_chunks = [], []
        try:
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                visible = mask_fn(batch.size(0), rng)
                with torch.no_grad():
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.cfg.mixed_precision):
                        samples = self.model(batch, visible, rng=self.generator)
                if not self.use_ens:
                    samples = torch.stack([samples[..., 0], torch.nn.functional.softplus(samples[..., 1])], dim=-1)
                ds = self.get_eval_dataset(data, batch_idx, batch, samples, visible)
                chunks.append(self.evaluation.compute(ds, tier=tier, stds=stds))
                if need_static:
                    static_chunks.append(self.evaluation.static(ds, stds=stds))
        finally:
            self.world.ens_size = ens_size

        pieces = self.evaluation.combine(chunks)
        pieces.attrs = {'tier' : tier, 'head' : 'ens' if self.use_ens else 'mve', 'ens_size' : E_test}
        static = self.evaluation.combine(static_chunks) if need_static else None
        if exists(static):
            static.attrs = {'stds' : stds}
        return group, static, pieces

    def get_eval_dataset(self, data: NinoData, batch_idx, batch, samples, visible) -> xr.Dataset:
        meta_data = data.dataset
        time, lat, lon = meta_data.time, meta_data.lat, meta_data.lon
        T, tt = self.world.token_sizes['t'], self.world.patch_sizes['tt']
        step = np.arange(T * tt)
        vis_field = einops.repeat(
            visible,
            f'b ({self.world.token_pattern}) -> b {self.world.field_pattern}',
            **self.world.token_sizes, **self.world.patch_sizes
        )
        B = batch.size(0)
        coords = {
            'time' : time[batch_idx * B : (batch_idx + 1) * B],
            'step' : step,
            'lat' :  lat,
            'lon' :  lon,
        }
        arrays = []
        for v, var in enumerate(self.data_cfg.variables):
            if var not in self.data_cfg.eval_variables:
                continue
            std = data._stds.sel(variable=var).values.astype(np.float32)
            fields = {
                f'{var}_obs' :     (['time', 'step', 'lat', 'lon'], batch[:, v].float().cpu().numpy() * std),
                f'{var}_visible' : (['time', 'step', 'lat', 'lon'], vis_field[:, v].cpu().numpy()),
            }
            if self.use_ens:
                fields[f'{var}_pred'] = (['time', 'step', 'lat', 'lon', 'ens'], samples[:, v].float().cpu().numpy() * std)
            else:
                fields[f'{var}_pred_mu'] =    (['time', 'step', 'lat', 'lon'], samples[:, v, ..., 0].float().cpu().numpy() * std)
                fields[f'{var}_pred_sigma'] = (['time', 'step', 'lat', 'lon'], samples[:, v, ..., 1].float().cpu().numpy() * std)
            arrays.append(xr.Dataset(data_vars=fields, coords=coords))
        ds = xr.merge(arrays, compat='no_conflicts')
        ds['lsm'] = xr.DataArray(
            np.bool_(data.land_sea_mask[0]),
            coords={'lat' : lat, 'lon' : lon}
        )
        return ds.sel(lat=slice(-20., 20.), lon=slice(90, 270))

    # METRICS
    def log_metrics(self, metrics: dict, task: str = None):
        for key, val in metrics.items():
            name = f"{task}_{key}" if exists(task) and task != 'prior' else key
            self.current_metrics.log_metric(name, val)

    def compute_metrics_torch_ens(self, ens: torch.Tensor, obs: torch.Tensor, mask: torch.BoolTensor) -> dict:
        ens = ens[mask]     # (N, E)
        obs = obs[mask]     # (N,)
        spread = ens.var(-1).mean().sqrt()
        K = ens.shape[-1]
        return {
            'crps' :   f_kernel_crps(observation=obs, ensemble=ens, fair=self.use_fair_crps).nanmean().item(),
            'ssr' :    (spread / (ens.mean(-1) - obs).pow(2).mean().sqrt().add(1e-5)).mul(math.sqrt((K + 1) / K)).item(),
            'ign' :    f_gaussian_ignorance(observation=obs, mu=ens.mean(-1), sigma=ens.std(-1) + 1e-5).nanmean().item(),
            'spread' : spread.item(),
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
    def compute_acc(pred, obs, eps: float = 1e-5) -> float:
        return (pred * obs).nansum().div(
            pred.pow(2).nansum().sqrt() * obs.pow(2).nansum().sqrt() + eps).item()

    @staticmethod
    def compute_rmse(pred, obs) -> float:
        return (pred - obs).pow(2).nanmean().sqrt().item()


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
