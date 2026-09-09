import os

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.special import erf
from omegaconf import OmegaConf

from utils.config import MTMConfig
from utils.dataset import NinoData
from utils.einmask import EinMask, EinMask_ENS
from backcast import DEVICE, OUT, TASKS, task_mask, to_leads, compute_metrics, get_nino34

MODELS = {'mve': ('2498967', 'EinMask (MVE)'), 'vv1': ('2580181', 'EinMask (variable tokens)'), 'ens': ('2631290', 'EinMask (ensemble)')}
DSETS = ('picontrol', 'godas')
LAGS = np.arange(1, 25)


def load_cfg(run):
    return MTMConfig.from_omegaconf(OmegaConf.load(f'runs/{run}/config.yaml'))


def load_any(cfg, run):
    cls = EinMask_ENS if (cfg.world.ens_size or 1) > 1 else EinMask
    model = cls(network=cfg.model, world=cfg.world)
    model.load_state_dict(torch.load(f'runs/{run}/best.pth', map_location='cpu', weights_only=False)['model_state'])
    return model.to(DEVICE).eval()


def get_data(cfg, name):
    if name == 'picontrol':
        return NinoData(cfg.trainer.picontrol_path, cfg.data.__class__(**{
            **cfg.data.__dict__, 'time_slice': {'start': '1900', 'stop': '2000', 'step': None}}))
    return NinoData(cfg.trainer.godas_path, cfg.data.__class__(**{**cfg.data.__dict__, 'time_slice': None}))


def zarr_path(ds_name, tag, task):
    if ds_name == 'picontrol' and tag == 'mve':
        return f'{OUT}/picontrol_{task}_eval.zarr'
    if ds_name == 'picontrol' and tag == 'vv1' and task == 'frcst':
        return f'{OUT}/picontrol_abl_base_eval.zarr'
    return f'{OUT}/{ds_name}_{tag}_{task}_eval.zarr'


def run_eval(cfg, model, data, task):
    world = cfg.world
    var = cfg.data.eval_variables[0]
    v = cfg.data.variables.index(var)
    std = data._stds.sel(variable=var).values.astype(np.float32)
    meta = data.dataset
    visible = task_mask(world, task)
    ens = (world.ens_size or 1) > 1

    dl = torch.utils.data.DataLoader(data, batch_size=world.batch_size, shuffle=False, num_workers=0)
    chunks = []
    for batch_idx, batch in enumerate(dl):
        batch = batch.to(DEVICE)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True):
            out = model(batch, visible.expand(batch.size(0), -1))
        if ens:
            members = out[:, v].float()
            mu = members.mean(-1).cpu().numpy() * std
            sigma = members.std(-1).cpu().numpy() * std
        else:
            mu, sigma = out
            sigma = torch.nn.functional.softplus(sigma[:, v].float()).cpu().numpy() * std
            mu = mu[:, v].float().cpu().numpy() * std
        obs = batch[:, v].cpu().numpy() * std
        t0 = batch_idx * world.batch_size
        ds = xr.Dataset(
            data_vars={
                f'{var}_obs':        (['time', 'step', 'lat', 'lon'], obs),
                f'{var}_pred_mu':    (['time', 'step', 'lat', 'lon'], mu),
                f'{var}_pred_sigma': (['time', 'step', 'lat', 'lon'], sigma),
            },
            coords={
                'time': meta.time[t0 : t0 + obs.shape[0]],
                'step': np.arange(obs.shape[1]),
                'lat':  meta.lat,
                'lon':  meta.lon,
            },
        )
        chunks.append(ds.sel(lat=slice(-20., 20.), lon=slice(90, 270)))

    ds = xr.concat(chunks, dim='time')
    lsm = xr.DataArray(np.bool_(data.land_sea_mask[0]), coords={'lat': meta.lat, 'lon': meta.lon})
    ds['lsm'] = lsm.sel(lat=slice(-20., 20.), lon=slice(90, 270))
    return ds


def linear_skill(data, var, n_train, gap=40, n_pc=300, lam=1e3):
    std = float(data._stds.sel(variable=var))
    X = data.tensor_data.numpy()[:, :, ::4, ::4]
    T = X.shape[1]
    nino = get_nino34(data.dataset[var] * std).values
    field = (data.dataset[var] * std).sel(lat=slice(-20., 20.), lon=slice(90, 270)).values
    sea = ~np.isnan(field[0])
    fieldv = np.nan_to_num(field)[:, sea]
    starts = np.arange(T - 36)
    train, test = starts[:n_train], starts[n_train + gap :]

    res = {}
    for task, w0 in [('frcst', 0), ('bcast', 24)]:
        Ftr = np.stack([X[:, i + w0 : i + w0 + 12].reshape(-1) for i in train])
        Fte = np.stack([X[:, i + w0 : i + w0 + 12].reshape(-1) for i in test])
        mu_f = Ftr.mean(0)
        _, _, Vt = np.linalg.svd(Ftr - mu_f, full_matrices=False)
        P = Vt[: min(n_pc, len(train) - 1)].T
        Ztr, Zte = (Ftr - mu_f) @ P, (Fte - mu_f) @ P
        G = np.linalg.inv(Ztr.T @ Ztr + lam * np.eye(P.shape[1])) @ Ztr.T
        nr, fr = [], []
        for L in LAGS:
            tgt = 11 + L if task == 'frcst' else 24 - L
            pn, pf = Zte @ (G @ nino[train + tgt]), Zte @ (G @ fieldv[train + tgt])
            on, of = nino[test + tgt], fieldv[test + tgt]
            nr.append((pn * on).sum() / np.sqrt((pn ** 2).sum() * (on ** 2).sum()))
            fr.append(((pf * of).sum(1) / (np.sqrt((pf ** 2).sum(1)) * np.sqrt((of ** 2).sum(1)))).mean())
        res[task] = {'nino34_pcc': np.array(nr), 'pcc': np.array(fr)}
    return res


def plot_probabilistic(leads, var, path, lags=(1, 12, 24), bins=10):
    edges = np.linspace(0, 1, bins + 1)
    width = 0.85 / (bins * len(lags) * 2)
    colors = plt.colormaps['viridis'](np.linspace(0.25, 0.8, len(lags)))
    fig, axes = plt.subplots(2, len(DSETS), figsize=(12, 7), layout='constrained')
    for col, ds_name in enumerate(DSETS):
        ax = axes[0, col]
        for i, lag in enumerate(lags):
            for j, task in enumerate(TASKS):
                d = leads[('mve', ds_name)][task].sel(lead=lag)
                z = ((d[f'{var}_obs'] - d[f'{var}_pred_mu']) / d[f'{var}_pred_sigma']).where(~d['lsm']).values.ravel()
                pit = 0.5 * (1 + erf(z[~np.isnan(z)] / np.sqrt(2)))
                hist, _ = np.histogram(pit, bins=edges)
                ax.bar(edges[:-1] + (i * 2 + j) * width, hist / hist.sum(), width=width, align='edge',
                       color=colors[i], edgecolor='k', lw=0.3, hatch=None if task == 'frcst' else '///',
                       label=f'{task} lag {lag}')
        ax.axhline(1 / bins, color='red', linestyle='dashed', lw=1)
        ax.set_title(f'SSTA PIT, {ds_name}')
        ax.set_xlabel('PIT')
        ax = axes[1, col]
        for c, (key, name) in enumerate([('spread', 'SSTA'), ('nino34_spread', 'Nino3.4')]):
            for task, ls in [('frcst', '-'), ('bcast', '--')]:
                m = metrics[('mve', ds_name)][task][key]
                ax.plot(m['lead'], m.values, ls, color=f'C{c}', label=f'{name} ({task})')
        ax.set_title(f'Spread growth, {ds_name}')
        ax.set_xlabel('Lead / Lag (Months)')
        ax.set_ylabel('Spread')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend(fontsize=6, ncol=len(lags))
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f'EinMask (MVE, {MODELS["mve"][0]}): probabilistic calibration, forecast vs backcast')
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_grid(metrics, linear, path):
    rows = [*MODELS] + ['linear']
    fig, axes = plt.subplots(len(rows), len(DSETS), figsize=(11, 3.2 * len(rows)),
                             layout='constrained', sharex=True, sharey=True)
    for col, ds_name in enumerate(DSETS):
        for r, tag in enumerate(rows):
            ax = axes[r, col]
            for c, (key, name) in enumerate([('pcc', 'SSTA'), ('nino34_pcc', 'Nino3.4')]):
                for task, ls in [('frcst', '-'), ('bcast', '--')]:
                    m = linear[ds_name][task][key] if tag == 'linear' else metrics[(tag, ds_name)][task][key].values
                    ax.plot(LAGS, m, ls, color=f'C{c}', label=f'{name} ({task})')
            ax.axhline(0.5, color='r', linestyle='dashed', lw=0.8)
            ax.set_ylim(0, 1)
            title = 'Linear (ridge)' if tag == 'linear' else f'{MODELS[tag][1]} {MODELS[tag][0]}'
            ax.set_title(f'{title} — {ds_name}', fontsize=10)
            if r == len(rows) - 1:
                ax.set_xlabel('Lead / Lag (Months)')
            if col == 0:
                ax.set_ylabel('ACC')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Forecast (solid) vs Backcast (dashed) ACC by model and dataset')
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == '__main__':
    leads, metrics = {}, {}
    for tag, (run, name) in MODELS.items():
        cfg = load_cfg(run)
        var = cfg.data.eval_variables[0]
        model = None
        for ds_name in DSETS:
            data = None
            for task in TASKS:
                path = zarr_path(ds_name, tag, task)
                if not os.path.exists(path):
                    model = model or load_any(cfg, run)
                    data = data or get_data(cfg, ds_name)
                    print('running', tag, ds_name, task)
                    run_eval(cfg, model, data, task).to_zarr(path, mode='w')
                ds = to_leads(xr.open_zarr(path), cfg.world, task)
                leads.setdefault((tag, ds_name), {})[task] = ds
                metrics.setdefault((tag, ds_name), {})[task] = compute_metrics(ds, var)
            for task in TASKS:
                m = metrics[(tag, ds_name)][task]
                print(tag, ds_name, task, {k: round(float(m[k].mean()), 3) for k in ('pcc', 'nino34_pcc')})
        del model
        torch.cuda.empty_cache()

    cfg = load_cfg(MODELS['mve'][0])
    var = cfg.data.eval_variables[0]
    linear = {'picontrol': linear_skill(get_data(cfg, 'picontrol'), var, n_train=760),
              'godas': linear_skill(get_data(cfg, 'godas'), var, n_train=260)}
    for ds_name, res in linear.items():
        for task in TASKS:
            print('linear', ds_name, task, {k: round(float(v.mean()), 3) for k, v in res[task].items()})

    plot_probabilistic(leads, var, f'{OUT}/backcast_probabilistic.png')
    plot_model_grid(metrics, linear, f'{OUT}/backcast_model_grid.png')
