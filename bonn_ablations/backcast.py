import torch
import einops

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.special import erf

from omegaconf import OmegaConf

from utils.config import MTMConfig
from utils.dataset import NinoData
from utils.einmask import EinMask

RUN = '2498967'
TASKS = ('frcst', 'bcast')
DEVICE = torch.device('cuda')
OUT = 'bonn_ablations'


def load_config():
    return MTMConfig.from_omegaconf(OmegaConf.load(f'runs/{RUN}/config.yaml'))


def load_model(cfg):
    model = EinMask(network=cfg.model, world=cfg.world)
    ckpt = torch.load(f'runs/{RUN}/best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model.to(DEVICE).eval()


def task_mask(world, task):
    m = torch.zeros(world.token_sizes['t'], dtype=torch.bool, device=DEVICE)
    if task == 'frcst':
        m[:world.tau] = True
    else:
        m[-world.tau:] = True
    return einops.repeat(m, f't -> ({world.token_pattern})', **world.token_sizes)


def run_task(cfg, model, data, task):
    world = cfg.world
    var = cfg.data.eval_variables[0]
    v = cfg.data.variables.index(var)
    std = data._stds.sel(variable=var).values.astype(np.float32)
    meta = data.dataset
    visible = task_mask(world, task)

    dl = torch.utils.data.DataLoader(data, batch_size=world.batch_size, shuffle=False, num_workers=0)
    chunks = []
    for batch_idx, batch in enumerate(dl):
        batch = batch.to(DEVICE)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=True):
            mu, sigma = model(batch, visible.expand(batch.size(0), -1))
        mu = mu[:, v].float().cpu().numpy() * std
        sigma = torch.nn.functional.softplus(sigma[:, v].float()).cpu().numpy() * std
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


def to_leads(ds, world, task):
    history = world.tau * world.patch_sizes['tt']
    T = ds.sizes['step']
    if task == 'frcst':
        ds = ds.isel(step=slice(history, None))
        lead = ds['step'].values - history + 1
    else:
        ds = ds.isel(step=slice(None, T - history))
        lead = T - history - ds['step'].values
    ds = ds.assign_coords(lead=('step', lead)).swap_dims(step='lead').sortby('lead')
    return ds


def get_nino4(da):
    return da.sel(lon=slice(160, 210), lat=slice(-5, 5)).mean(dim=['lon', 'lat'], skipna=True)


def get_nino34(da):
    return da.sel(lon=slice(190, 240), lat=slice(-5, 5)).mean(dim=['lon', 'lat'], skipna=True)


def xr_pcc(pred, obs, dim):
    num = (pred * obs).sum(dim, skipna=True)
    denom = np.sqrt((pred ** 2).sum(dim, skipna=True)) * np.sqrt((obs ** 2).sum(dim, skipna=True))
    return num / denom


def xr_rmse(pred, obs, dim):
    return np.sqrt(((pred - obs) ** 2).mean(dim, skipna=True))


def xr_ssr_mve(mu, sigma, obs, dim):
    var = (sigma ** 2).mean(dim, skipna=True)
    mse = ((obs - mu) ** 2).mean(dim, skipna=True)
    return np.sqrt(var / mse)


def compute_metrics(ds, var):
    valid = ~ds['lsm']
    obs = ds[f'{var}_obs'].where(valid)
    mu = ds[f'{var}_pred_mu'].where(valid)
    sigma = ds[f'{var}_pred_sigma'].where(valid)
    space = ('lat', 'lon')
    m = {
        'pcc':  xr_pcc(mu, obs, space).mean('time', skipna=True),
        'rmse': xr_rmse(mu, obs, space).mean('time', skipna=True),
        'ssr':  xr_ssr_mve(mu, sigma, obs, space).mean('time', skipna=True),
        'spread': np.sqrt((sigma ** 2).mean(space, skipna=True)).mean('time', skipna=True),
    }
    for name, fn in [('nino34', get_nino34), ('nino4', get_nino4)]:
        o, p, s = fn(obs), fn(mu), fn(sigma)
        m[f'{name}_pcc'] = xr_pcc(p, o, ('time',))
        m[f'{name}_rmse'] = xr_rmse(p, o, ('time',))
        m[f'{name}_ssr'] = xr_ssr_mve(p, s, o, ('time',))
        m[f'{name}_spread'] = np.sqrt((s ** 2).mean('time', skipna=True))
    return m


def plot_metrics(metrics, path, title=None):
    styles = {'frcst': '-', 'bcast': '--'}
    labels = [('', 'SSTA'), ('nino34_', 'Nino3.4'), ('nino4_', 'Nino4')]
    plt.figure(figsize=(16, 4))
    for i, (key, title) in enumerate([('pcc', 'PCC'), ('rmse', 'RMSE'), ('ssr', 'Spread Skill Ratio'), ('spread', 'Spread')]):
        plt.subplot(1, 4, 1 + i)
        for c, (prefix, name) in enumerate(labels):
            for task, ls in styles.items():
                m = metrics[task][f'{prefix}{key}']
                plt.plot(m['lead'], m.values, ls, color=f'C{c}', label=f'{name} ({task})')
        if key == 'pcc':
            plt.hlines(0.5, 1, 24, colors='r', linestyles='dashed')
            plt.ylim(0, 1)
        if key == 'ssr':
            plt.ylim(0.65, 1.35)
        plt.title(title)
        plt.xlabel('Lead / Lag (Months)')
        if i == 0:
            plt.legend(fontsize=7)
    plt.suptitle(title or f'Forecast vs Backcast ({RUN}, piControl)')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_samples(datasets, var, lead, time_idx, path):
    plt.figure(figsize=(15, 6))
    for row, (task, ds) in enumerate(datasets.items()):
        valid = ~ds['lsm']
        sel = dict(time=time_idx, lead=lead)
        for col, (name, da, kw) in enumerate([
            ('pred mu', ds[f'{var}_pred_mu'], dict(vmin=-2, vmax=2, cmap='bwr')),
            ('obs', ds[f'{var}_obs'], dict(vmin=-2, vmax=2, cmap='bwr')),
            ('pred sigma', ds[f'{var}_pred_sigma'], dict(vmin=0, cmap='viridis')),
        ]):
            plt.subplot(2, 3, 1 + row * 3 + col)
            da.where(valid).isel(**sel).plot(**kw)
            plt.title(f'{task} {name} (lead {lead})')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roundtrip(raw, var, time_idx, path, lead=18):
    # the bcast instance sits 13 months earlier, so its context start coincides with the frcst
    # context end: one shared initial frame at t with symmetric targets at t -/+ lead
    fr = raw['frcst'].isel(time=time_idx)
    bc = raw['bcast'].isel(time=time_idx - 13)
    valid = ~raw['frcst']['lsm']
    date = str(raw['frcst']['time'].values[time_idx + 11])[:7]
    kw = dict(vmin=-2, vmax=2, cmap='bwr', add_colorbar=False)
    panels = [
        [(bc[f'{var}_obs'].isel(step=24 - lead), f'Ground truth: obs (t-{lead})'),
         (fr[f'{var}_obs'].isel(step=11), 'Initial: obs (t)'),
         (fr[f'{var}_pred_mu'].isel(step=11 + lead), f'Forecast mu (t+{lead})')],
        [(bc[f'{var}_pred_mu'].isel(step=24 - lead), f'Backcast mu (t-{lead})'),
         (bc[f'{var}_obs'].isel(step=24), 'Initial: obs (t)'),
         (fr[f'{var}_obs'].isel(step=11 + lead), f'Ground truth: obs (t+{lead})')],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.5), layout='constrained')
    fig.get_layout_engine().set(wspace=0.08)
    for row, row_panels in enumerate(panels):
        for col, (da, title) in enumerate(row_panels):
            im = da.where(valid).plot(ax=axes[row, col], **kw)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('')
            if col > 0:
                axes[row, col].set_ylabel('')
                axes[row, col].set_yticklabels([])
    fig.suptitle(f'Round trip ({RUN}, piControl, t = {date}, lead {lead})')
    fig.colorbar(im, ax=axes, label=var, shrink=0.8)
    fig.canvas.draw()
    for row, col, arrow in [(0, 1, r'$\longrightarrow$'), (1, 0, r'$\longleftarrow$')]:
        p0, p1 = axes[row, col].get_position(), axes[row, col + 1].get_position()
        fig.text((p0.x1 + p1.x0) / 2, (p0.y0 + p0.y1) / 2, arrow, fontsize=24, ha='center', va='center')
    plt.savefig(path)
    plt.close()


def plot_pit_hist(datasets, var, path, leads=(1, 6, 12, 18, 24), bins=10, title=None):
    edges = np.linspace(0, 1, bins + 1)
    width = 0.8 / (bins * len(leads))
    colors = plt.colormaps['viridis'](np.linspace(0.2, 0.85, len(leads)))
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), layout='constrained')
    for ax, (task, ds) in zip(axes, datasets.items()):
        valid = ~ds['lsm']
        for i, lead in enumerate(leads):
            d = ds.sel(lead=lead)
            z = (d[f'{var}_obs'] - d[f'{var}_pred_mu']) / d[f'{var}_pred_sigma']
            z = z.where(valid).values.ravel()
            pit = 0.5 * (1 + erf(z[~np.isnan(z)] / np.sqrt(2)))
            hist, _ = np.histogram(pit, bins=edges)
            ax.bar(edges[:-1] + i * width, hist / hist.sum(), width=width, align='edge',
                   color=colors[i], edgecolor='k', lw=0.4, label=f'Lag {lead}')
        ax.axhline(1 / bins, color='red', linestyle='dashed', lw=1)
        ax.set_title(task)
        ax.set_xlabel('PIT')
    axes[0].set_ylabel('Frequency')
    axes[0].legend(ncol=len(leads), loc='upper center')
    fig.suptitle(title or f'SSTA PIT rank histograms ({RUN}, piControl)')
    plt.savefig(path)
    plt.close()


def main():
    cfg = load_config()
    model = load_model(cfg)
    data = NinoData(cfg.trainer.picontrol_path, cfg.data.__class__(**{
        **cfg.data.__dict__, 'time_slice': {'start': '1900', 'stop': '2000', 'step': None}}))
    var = cfg.data.eval_variables[0]

    raw, datasets, metrics = {}, {}, {}
    for task in TASKS:
        ds = run_task(cfg, model, data, task)
        ds.to_zarr(f'{OUT}/picontrol_{task}_eval.zarr', mode='w')
        raw[task] = ds
        ds = to_leads(ds, cfg.world, task)
        datasets[task] = ds
        metrics[task] = compute_metrics(ds, var)
        print(task, {k: float(v.mean()) for k, v in metrics[task].items()})

    plot_metrics(metrics, f'{OUT}/backcast_metrics.png')
    plot_samples(datasets, var, lead=20, time_idx=100, path=f'{OUT}/backcast_samples.png')
    plot_roundtrip(raw, var, time_idx=100, path=f'{OUT}/backcast_roundtrip.png')
    plot_pit_hist(datasets, var, path=f'{OUT}/backcast_pit_hist.png')


if __name__ == '__main__':
    main()
