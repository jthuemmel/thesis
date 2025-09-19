
import torch
import os
import argparse
import math
import einops
import xarray as xr
import numpy as np

from dataclasses import replace
from functools import cached_property
from omegaconf import OmegaConf
from typing import Tuple, Iterable
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import one_hot

import utils.config as cfg
from utils.loss_fn import f_kernel_crps, f_gaussian_ignorance
from utils.field_network import StochasticWeatherField
from utils.model import WeatherField
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
class TrainerMixin(DistributedTrainer):
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
    
    @property
    def use_fair_crps(self):
        return self.world_cfg.num_ens > 1
    
    # DATA
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
            batch_size=self.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
        )

        val_dl = torch.utils.data.DataLoader(
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
            

    # MODEL
    def create_model(self):
        model = WeatherField(self.model_cfg)      
        if hasattr(model, "generator"):
            model.generator = self.generator
        count = count_parameters(model)
        print(f'Created model with {count:,} parameters')
        self.misc_metrics.log_python_object("num_params", count)
        return model
    
    # LOSS
    def create_loss(self):
        def loss(pred: torch.Tensor, obs: torch.Tensor, lsm: torch.Tensor, var: torch.Tensor = None):
            # pointwise crps
            point_crps = f_kernel_crps(observation=obs, ensemble=pred, fair= self.use_fair_crps)
            # apply channel weights
            point_crps = point_crps * var
            # apply land-sea mask
            point_crps = point_crps[lsm]
            # reduce to scalar
            return point_crps.nanmean()
        return loss
    
    ### FORWARD
    def forward_step(self, batch_idx, batch):
        # step
        tgt_pred, tgt, lsm, var = self.step(batch_idx, batch, None)        
        # compute loss
        loss = self.loss_fn(pred = tgt_pred, obs = tgt, lsm = lsm, var = var)
        # metrics
        self.current_metrics.log_metric(f"loss", loss.item())
        self.compute_metrics(pred = tgt_pred, obs = tgt, lsm = lsm, label = None)
        return loss
    
    def frcst_step(self, batch_idx, batch):
        # step
        tgt_pred, tgt, lsm, var = self.step(batch_idx, batch, "frcst")        
        self.get_frcst_metrics(pred = tgt_pred, obs = tgt, lsm = lsm)
        return tgt_pred, tgt
    
    def evaluate_epoch(self):
        self.switch_mode(train=False)
        if not exists(self.val_dl):
            return
        
        samples = []
        for batch_idx, batch in enumerate(self.val_dl):
            #no gradients needed for evaluation
            with torch.no_grad():
                with torch.amp.autocast(device_type = self.device_type, enabled=self.cfg.mixed_precision):
                    _ = self.forward_step(batch_idx, batch)
                    tgt_pred, tgt = self.frcst_step(batch_idx, batch)
                    samples.append(self.get_arrays(batch_idx, tgt_pred, tgt))

        ds = xr.concat(samples, dim = "time")
        self.get_nino_metrics(ds)

    @staticmethod
    def get_nino4(da: xr.DataArray):
        return da.sel(lon=slice(160, 210), lat=slice(-5, 5)).mean(dim=['lon', 'lat']).rolling(time = 3).mean()
    
    @staticmethod
    def get_nino34(da: xr.DataArray):
        return da.sel(lon=slice(190, 240), lat=slice(-5, 5)).mean(dim=['lon', 'lat']).rolling(time = 3).mean()

    def get_nino_metrics(self, eval_data: xr.Dataset):
        # {var: T, Lag, Lat, Lon, (Ens)}
        eval_data = eval_data.sel(lat = slice(-20., 20.), lon = slice(90, 270))
        # T, Lag, (Ens)
        nino4_tgt, nino4_pred = self.get_nino4(eval_data["temp_ocn_0a_tgt"]), self.get_nino4(eval_data["temp_ocn_0a_pred"]).mean("ens")
        nino34_tgt, nino34_pred = self.get_nino34(eval_data["temp_ocn_0a_tgt"]), self.get_nino34(eval_data["temp_ocn_0a_pred"]).mean("ens")
        # PCC [Lag]
        nino4_pcc = (nino4_tgt * nino4_pred).sum("time") / (((nino4_tgt ** 2).sum("time") ** 0.5) * (nino4_pred ** 2).sum("time") ** 0.5)
        nino34_pcc = (nino34_tgt * nino34_pred).sum("time") / (((nino34_tgt ** 2).sum("time") ** 0.5) * (nino34_pred ** 2).sum("time") ** 0.5)
        # RMSE [Lag]
        nino4_rmse = np.sqrt(((nino4_tgt - nino4_pred)**2).mean(["time"]))
        nino34_rmse = np.sqrt(((nino34_tgt - nino34_pred)**2).mean(["time"]))
        # Log
        for lag in [3, 9, 15, 21]:
            self.current_metrics.log_metric(f"nino4_pcc_{lag}", nino4_pcc.sel(lag = lag).values)
            self.current_metrics.log_metric(f"nino34_pcc_{lag}", nino34_pcc.sel(lag = lag).values)
            self.current_metrics.log_metric(f"nino4_rmse_{lag}", nino4_rmse.sel(lag = lag).values)
            self.current_metrics.log_metric(f"nino34_rmse_{lag}", nino34_rmse.sel(lag = lag).values)

    @property
    def xr_ds_eval(self):
        return self.val_dataset.dataset
    
    def get_arrays(self, batch_idx, pred, obs):
        #meta data
        time, lat, lon = self.xr_ds_eval.time, self.xr_ds_eval.lat, self.xr_ds_eval.lon
        ens = np.arange(pred.shape[-1])
        tau = self.world_cfg.tau
        T, tt = self.token_sizes["t"], self.patch_sizes["tt"]
        lag = np.arange(1, 1 + ((T - tau) * tt))
        # to field and numpy
        pred, obs = self.frcst_to_field(pred).float().cpu().numpy(), self.frcst_to_field(obs).float().cpu().numpy()
        # variables
        arrays = []
        for v, var in enumerate(self.data_cfg.variables):
            #create xarray
            data_array = xr.Dataset(
                data_vars = {
                    f"{var}_pred": (["time", "lag", "lat", "lon", "ens"], np.sort(pred[:, v], axis = -1)),
                    f"{var}_tgt": (["time", "lag", "lat", "lon"], obs[:, v]),
                },
                coords = {
                    "time": time[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size],
                    "lag": lag,
                    "lat": lat,
                    "lon": lon,
                    "ens": ens
                },
            )
            arrays.append(data_array)
        ds = xr.merge(arrays)
        return ds

### TOKENIZATION
class ShapeMixin:
    # HELPERS
    @staticmethod
    def _as_factors(factors):
        if isinstance(factors, torch.Tensor):
            return (factors,)
        # allow generator/iterable
        return tuple(factors)

    @staticmethod
    def _as_axes(axes) -> Tuple[str, ...]:
        return (axes,) if isinstance(axes, str) else tuple(axes)
    
    def sizes_of(self, axes: Iterable[str] | str) -> dict:
        A = self._as_axes(axes)
        return {a: self.sizes.get(a, 1) for a in A}

    def shape_of(self, axes: Iterable[str] | str) -> Tuple[int]:
        A = self._as_axes(axes)
        return tuple(self.sizes.get(a, 1) for a in A)
    
    # LAYOUT
    @cached_property
    def token_layout(self):
        return self._as_axes(self.world_cfg.token_layout)
    
    @cached_property
    def plate_layout(self):
        return self._as_axes(self.world_cfg.plate_layout)
    
    @cached_property
    def patch_layout(self):
        return self._as_axes(self.world_cfg.patch_layout)
       
    @cached_property
    def patch_sizes(self):
        # Return patch sizes per axis (default to 1)
        return {
            k: self.world_cfg.size_cfg.get(k, 1)
            for k in self.patch_layout
        }
    
    @cached_property
    def token_sizes(self):
        # sizes after patching
        return {
            k: self.world_cfg.size_cfg[k] // self.world_cfg.size_cfg.get(f"{k*2}", 1)
            for k in self.token_layout
        }
    
    @cached_property
    def plate_sizes(self):
        return {k: self.world_cfg.size_cfg[k] for k in self.plate_layout}

    @cached_property
    def sizes(self):
        return {**self.plate_sizes, **self.token_sizes, **self.patch_sizes}
    # PATTERNS
    @property
    def lsm_pattern(self):
        return '1 (h hh) (w ww) ...'

    @property
    def field_pattern(self):
        return 'b (v vv) (t tt) (h hh) (w ww) ...'    

    @property
    def flatland_pattern(self) -> str:
        plates = ' '.join(self.plate_layout)
        tokens = ' '.join(self.token_layout)
        patches = ' '.join(self.patch_layout)
        return f'({plates}) ({tokens}) ({patches}) ...'    

    @property
    def var_pattern(self):
        return '(v vv) ...'

    # SHAPES
    def get_flatland_shape(self):
        B = self.batch_size
        N = math.prod([self.token_sizes[t] for t in self.token_layout])
        D = math.prod([self.patch_sizes[p] for p in self.patch_layout])
        return B, N, D
    
    def get_flatland_index(self):
        B, N, _ = self.get_flatland_shape()
        return einops.repeat(torch.arange(N, device = self.device), "n -> b n", b = B)
    
    # TOKENIZERS
    def field_to_tokens(self, field):
        return einops.rearrange(field.rename(None), 
                                f'{self.field_pattern} -> {self.flatland_pattern}', 
                                **self.token_sizes, **self.patch_sizes)
    
    def frcst_to_field(self, tokens):
        valid_axes = ('h', 'w', 'v') + self.patch_layout # need to ensure that we only query spatial and variable sizes
        return einops.rearrange(tokens.rename(None), f'{self.flatland_pattern} -> {self.field_pattern}', **{ax: self.sizes[ax] for ax in valid_axes})

    def masked_to_spatial(self, tokens):
        lhs, rhs = 'b (m h w) (d hh ww) ...', '(... b m d) (h hh) (w ww)' # use dummy names for the masked parts
        valid_axes = ('h', 'w', 'hh','ww') # need to ensure that we only query spatial sizes
        return einops.rearrange(tokens.rename(None), f'{lhs} -> {rhs}', **{ax: self.sizes[ax] for ax in valid_axes})
    
    # COORDINATE TRANSFORMS
    def compute_strides(self, layout: tuple):
        strides, sizes = {}, self.sizes_of(layout)
        stride = 1
        for ax in reversed(layout):
            strides[ax] = stride
            stride *= sizes[ax]
        return strides
    
    def index_to_coords(self, idx_flat: torch.LongTensor, layout: tuple = None):
        layout = layout if exists(layout) else self.token_layout
        strides = self.compute_strides(layout)
        rem = idx_flat.rename(None)
        coords = []
        for ax in layout:
            val = rem.div(strides[ax], rounding_mode="floor")
            rem = rem.fmod(strides[ax])
            coords.append(val)
        return torch.stack(coords, dim=-1) 

    def coords_to_index(self, coords: torch.LongTensor, layout: tuple = None):
        layout = layout if exists(layout) else self.token_layout
        strides = self.compute_strides(layout)
        # Compute dot product over coordinate axis
        idx = sum(coords[..., i] * strides[ax] for i, ax in enumerate(layout))
        return idx 

#### MASKING
class SamplingMixin:
    def measure(self, axes: Iterable[str]) -> int:
        A = self._as_axes(axes)
        return math.prod(self.shape_of(A))

    def k_from_rate(self, rate: float | torch.Tensor, axes: Iterable[str]) -> int:
        m = self.measure(axes)
        r = torch.as_tensor(rate, device = self.device)
        return (r * m).long().clamp(1, m).item()

    # generators
    def constant(self, axes: Iterable[str], val=1.):
        A = self._as_axes(axes)
        return torch.as_tensor(val, device=self.device, dtype = torch.float).broadcast_to(self.shape_of(A)).rename(*A)

    def dirichlet(self, axes: Iterable[str], logits=1.0, sharpness=1.0):
        A = self._as_axes(axes)
        alpha = sharpness * self.constant(A, logits)
        return torch._sample_dirichlet(alpha.rename(None), generator=self.generator).rename(*A)

    def trunc_normal(self, axes: Iterable[str], mean=0., std=0., a=0., b=1.):
        A = self._as_axes(axes)
        w = torch.empty(self.shape_of(A), device=self.device)
        w = torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=a, b=b, generator=self.generator) if std > 0 else w.fill_(mean)
        return w.rename(*A)

    def uniform(self, axes: Iterable[str]):
        A = self._as_axes(axes)
        return torch.rand(self.shape_of(A), device=self.device, generator=self.generator).rename(*A)

    # product/contract
    def marginalize(self, factors: Iterable[torch.Tensor], axes: Iterable[str]):
        F = self._as_factors(factors)
        names = {n for f in F for n in f.names if exists(n)}
        A = self._as_axes(axes)
        plates = tuple(self.constant(ax) for ax in A if ax not in names)
        inputs = F + plates
        args = tuple(f.rename(None) for f in inputs)
        lhs  = ",".join(" ".join(f.names) for f in inputs)
        rhs  = " ".join(A)
        return einops.einsum(*args, f"{lhs} -> {rhs}").rename(*A)

    # packing/unpacking events
    def pack(self, factor: torch.Tensor, axes: Iterable[str]):
        A = self._as_axes(axes)
        plate = tuple(n for n in factor.names if n not in A)
        x = factor.align_to(*(plate + A)).rename(None)
        pattern = (" ".join(plate) + " *").strip()
        flat, _ = einops.pack([x], pattern)
        return flat.rename(*(plate + ("event",)))

    def unpack(self, factor: torch.Tensor, axes: Iterable[str]):
        A = self._as_axes(axes)
        plate = tuple(n for n in factor.names if n != "event")
        pattern = (" ".join(plate) + " *").strip()
        [restored] = einops.unpack(factor.rename(None), pattern=pattern, packed_shapes=[self.shape_of(A)])
        return restored.rename(*(plate + A))

    # selection and conditioning
    def select(self, factor: torch.Tensor, k: int, axis: str = "event", method: str = "topk"):
        plate = tuple(n for n in factor.names if n != axis)
        x = factor.align_to(*(plate + self._as_axes(axis))).rename(None)
        flat = einops.rearrange(x, "... ax -> (...) ax")
        if method == "topk":
            idcs = flat.topk(k, sorted=False, dim=-1).indices
        else:
            idcs = torch.multinomial(flat, k, replacement=False, generator=self.generator)
        front = " ".join(plate)
        idcs = einops.rearrange(idcs, f"({front}) ax -> {front} ax", **self.sizes_of(plate))
        return idcs.rename(*(plate + ("event",)))

    def take(self, factor: torch.Tensor, idx: torch.LongTensor, axis: str = "event"):
        dim = factor.names.index(axis)
        out_names = tuple("event" if n == axis else n for n in factor.names)
        out_shape = tuple(idx.size("event") if n == axis else factor.size(n) for n in factor.names)
        idx = idx.align_to(*out_names).rename(None).expand(out_shape)
        gathered = torch.gather(factor.rename(None), dim=dim, index=idx)
        return gathered.rename(*out_names)
    
### ENSO DATA
class OceanMixin:
    @property
    def land_sea_mask(self):
        lsm = self._train_lsm if self.mode == "train" else self._val_lsm
        return einops.repeat(lsm, f"{self.lsm_pattern} -> {self.flatland_pattern}", 
                      **self.token_sizes, **self.patch_sizes, b = self.batch_size)
    
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
            picontrol_config = replace(self.data_cfg, time_slice = {"start": "1850", "stop": "2000", "step": None})
            self._picontrol_data = NinoData(self.cfg.picontrol_path, picontrol_config)
        return self._picontrol_data

    def oras5_data(self):
        if not hasattr(self, "_oras5_data"):
            self._oras5_data = NinoData(self.cfg.oras5_path, self.data_cfg)
        return self._oras5_data
        
### METRICS
class MetricsMixin:
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

    @staticmethod
    def compute_spread(ens_pred: torch.Tensor):
        return ens_pred.var(-1).mean().sqrt()

    def compute_spread_skill(self, pred: torch.Tensor, obs: torch.Tensor):
        E = pred.shape[-1]
        correction = (E + 1) / E
        spread = self.compute_spread(ens_pred=pred)
        skill = self.compute_rmse(pred.mean(-1), obs)
        return correction * (spread / skill)

    ### SPECTRAL
    def compute_rapsd(self, tokens: torch.Tensor):
        x = self.masked_to_spatial(tokens)
        with torch.autocast(device_type=self.device_type, enabled=False):
            rapsd = self.rapsd(x.float())
            rapsd = torch.clamp(rapsd, min=1e-8).log10()
        return rapsd
    
    def compute_spectral_crps(self, pred: torch.Tensor, obs: torch.Tensor, fair: bool = True):
        with torch.autocast(device_type=self.device_type, enabled=False):
            pred_spectrum = torch.fft.rfft2(self.masked_to_spatial(pred.float()))#.view_as_real()
            obs_spectrum = torch.fft.rfft2(self.masked_to_spatial(obs.float()))#.view_as_real()
        pred_spectrum = einops.rearrange(pred_spectrum, "(e bb) ... -> bb ... e", e = pred.size(-1))
        return f_kernel_crps(observation= obs_spectrum, ensemble= pred_spectrum, fair= fair)
    
    @staticmethod
    def rapsd(x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.size()
        device = x.device

        spectrum = torch.fft.fft2(x)
        psd = spectrum.abs().pow(2)

        ky = torch.fft.fftfreq(H, d=1.0).to(device)
        kx = torch.fft.fftfreq(W, d=2.0).to(device)

        kx = einops.rearrange(kx, 'w -> 1 w')
        ky = einops.rearrange(ky, 'h -> h 1')

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

    ### HELPERS
    def ensemble_metrics(self, ens_pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        E = ens_pred.size(-1)
        crps = self.compute_crps(pred=ens_pred, obs=obs)
        ssr = self.compute_spread_skill(pred=ens_pred, obs=obs)
        ign = self.compute_ign(pred=ens_pred, obs=obs)
        spread = self.compute_spread(ens_pred=ens_pred)
        member_acc = torch.as_tensor([self.compute_acc(ens_pred[..., e], obs=obs) for e in range(E)]).mean()
        member_rmse = torch.as_tensor([self.compute_rmse(ens_pred[..., e], obs) for e in range(E)]).mean()

        self.current_metrics.log_metric(f"{label}_crps" if exists(label) else "crps", crps.item())
        self.current_metrics.log_metric(f"{label}_ssr" if exists(label) else "ssr", ssr.item())
        self.current_metrics.log_metric(f"{label}_ign" if exists(label) else "ign", ign.item())
        self.current_metrics.log_metric(f"{label}_spread" if exists(label) else "spread", spread.item())
        self.current_metrics.log_metric(f"{label}_member_rmse" if exists(label) else "member_rmse", member_rmse.item())
        self.current_metrics.log_metric(f"{label}_member_acc" if exists(label) else "member_acc", member_acc.item())

    def deterministic_metrics(self, pred: torch.Tensor, obs: torch.Tensor, label: str = None):
        acc = self.compute_acc(pred=pred, obs=obs)
        rmse = self.compute_rmse(pred=pred, obs=obs)
        self.current_metrics.log_metric(f"{label}_acc" if exists(label) else "acc", acc.item())
        self.current_metrics.log_metric(f"{label}_rmse" if exists(label) else "rmse", rmse.item())

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
        pred, obs, lsm = (self.frcst_to_field(x) for x in (pred, obs, lsm))
        frcst_tasks = {"frcst": (slice(0, 1), slice(None))} # default task, sst only
        if exists(self.data_cfg.frcst_tasks): # additional tasks from config
            frcst_tasks.update(self.data_cfg.frcst_tasks)
        for label, (vi, ti) in frcst_tasks.items(): # for all tasks
            vi, ti = (self._slice_from_cfg(x) for x in (vi, ti))
            p, o, l = (x[:, vi, ti] for x in (pred, obs, lsm)) # select the task-specific subset
            self.compute_metrics(p, o, l, label= label) # compute all metrics and label them according to the task

### MODEL
class MTMTrainer(TrainerMixin, MetricsMixin, OceanMixin, SamplingMixin, ShapeMixin):
    ### Masking
    @cached_property
    def per_variable_weights(self):
        weights = {'temp_ocn_0a': 1.,
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
        return einops.repeat(w, f"{self.var_pattern} -> {self.flatland_pattern}", **self.token_sizes, **self.patch_sizes, b = self.batch_size)
    
    # HELPERS
    def get_index(self, factors: Iterable[torch.Tensor], axes: Iterable[str], rate: float, method: str = "multinomial"):
        A = self._as_axes(axes)
        F = self._as_factors(factors)
        K = self.k_from_rate(rate, A)
        joint = self.marginalize(F, self.plate_layout + A)
        flat = self.pack(joint, A)  
        return self.select(flat, K, method=method)

    def get_multi_index(self, factors: Iterable[torch.Tensor], axes: Iterable[str], rates: torch.Tensor, method: str = "multinomial"):
        A = self._as_axes(axes)
        F = self._as_factors(factors)
        ax = rates.names
        M = tuple(a for a in A if a not in ax)
        idx = self.arange_axis(ax)
        idcs = []
        for i in range(self.measure(ax)):
            K = self.k_from_rate(rates[i], M)
            I = self._as_factors(self.indicator(idx, i))
            joint = self.marginalize(F + I, self.plate_layout + A)
            flat = self.pack(joint, A)
            idcs.append(self.select(flat, K, method=method).rename(None))
        idx = torch.cat(idcs, dim = -1)
        return idx.rename(*flat.names)
        
    def indicator(self, factor: torch.Tensor, idx: torch.LongTensor):
        if not isinstance(idx, torch.LongTensor): idx = torch.as_tensor(idx, device = factor.device, dtype = torch.long)
        return torch.zeros_like(factor.rename(None)).scatter_(-1, idx.rename(None), 1).rename(*factor.names)

    def dirichlet_Nd(self, axes: Iterable[str]):
        #combine multiple 1d dirichlet priors via outer product
        A = self._as_axes(axes)
        return self.marginalize([self.dirichlet_batched(ax) for ax in A], self.plate_layout + A)

    def dirichlet_batched(self, d: str):
        A = self.plate_layout + self._as_axes(d)
        return self.dirichlet(A, sharpness= self.world_cfg.alphas.get(d, 1.), logits= self.world_cfg.weights.get(d, 1.)).refine_names(*A)

    def arange_axis(self, d: str):
        A = self._as_axes(d)
        return torch.arange(self.measure(A), device = self.device).rename(*A)

    def split_index(self, d: str, split_at: int):
        idx = self.arange_axis(d)
        front = self.indicator(idx, idx[:split_at])
        tail = self.indicator(idx, idx[split_at:])
        return front, tail
    
    ### PRIORS
    @staticmethod
    def gumbel_topk(phi, k):
        return (phi - torch.log(-torch.log(torch.rand_like(phi)))).topk(k).indices

    @staticmethod
    def as_logit(p):
        return torch.log(p) - torch.log1p(-p)

    def src_rate(self):
        cfg = self.world_cfg.mask_rates_src
        return self.trunc_normal('rate', **cfg) if exists(cfg) else 1.
    
    def tgt_rate(self):
        cfg = self.world_cfg.mask_rates_tgt
        return self.trunc_normal('rate', **cfg) if exists(cfg) else 1.

    def frcst(self):
        tau = self.world_cfg.tau 
        r = tau / self.token_sizes['t']
        W_src, W_tgt = self.split_index('t', tau)
        src_mask = self.get_index(W_src, self.token_layout, rate = r, method='topk')
        tgt_mask = self.get_index(W_tgt, self.token_layout, rate = 1 - r, method='topk')
        return src_mask, tgt_mask
    
    def prior(self):
        A = self._as_axes(self.token_layout)
        K_src = self.k_from_rate(self.src_rate(), A)
        K_tgt = self.k_from_rate(self.tgt_rate(), A)
        T = self.dirichlet_batched('t')
        V = self.dirichlet_batched('v')
        packed = self.pack(self.marginalize((T, V), self.plate_layout + A), A)
        prior = self.as_logit(packed / packed.sum(-1, True)).rename(None)
        src_mask = self.gumbel_topk(prior, K_src).rename(*packed.names)
        tgt_mask = self.gumbel_topk(1 - prior, K_tgt).rename(*packed.names)
        return src_mask, tgt_mask
    
    def apply_masks(self, tokens, src_mask, tgt_mask):
        tokens = tokens.rename('b', 'event','dim')
        src = self.take(tokens, src_mask)
        tgt = self.take(tokens, tgt_mask)
        return src.rename(None), tgt.rename(None)

    def get_loss_masks(self, tgt_mask):
        lsm = self.land_sea_mask.rename('b', 'event', 'dim')
        var = self.per_variable_weights.rename('b', 'event', 'dim')
        lsm = self.take(lsm, tgt_mask)
        var = self.take(var, tgt_mask)
        return lsm.rename(None), var.rename(None)

    ### STEP
    def step(self, batch_idx, batch, task: str = None):
        # to device and tokenize
        tokens = self.field_to_tokens(batch.to(self.device))
        
        # masking
        src_mask, tgt_mask = self.prior() if task != "frcst" else self.frcst() 
        src, tgt= self.apply_masks(tokens, src_mask, tgt_mask)
        lsm, var = self.get_loss_masks(tgt_mask)
        
        # convert flat indices to coordinates
        src_coords = self.index_to_coords(src_mask)
        tgt_coords = self.index_to_coords(tgt_mask)
        #src_coords, tgt_coords = src_mask.rename(None), tgt_mask.rename(None)
        
        # expand src, tgt, lsm for ensemble
        src, src_coords, tgt_coords = (einops.repeat(x, "b ... -> (b k) ...", k = self.world_cfg.num_ens) for x in (src, src_coords, tgt_coords))
        
        # forward
        with sdpa_kernel(self.backend):
            tgt_pred = self.model(src, src_coords, tgt_coords)

        # split out ensemble dimension(s) if necessary
        tgt_pred = einops.rearrange(tgt_pred, '(b k) ... (c e) -> b ... c (k e)', c = tokens.size(-1), b = tokens.size(0))
            
        return tgt_pred, tgt, lsm, var

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