import torch
import einops

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from utils.config import MTMConfig
from utils.dataset import NinoData
from utils.einmask import EinMask

from backcast import DEVICE, OUT, to_leads, compute_metrics, plot_pit_hist

RUN = '2580181'
CHANNELS = {'tauxa': 'tauxa', 't150a': 'temp_ocn_14a'}
SETTINGS = ('masked', 'nudged', 'zeroed', 'partial')
COLORS = {'base': 'k', 'masked': 'C0', 'nudged': 'C2', 'zeroed': 'C1', 'partial': 'C3'}
LABELS = {'base': 'base', 'masked': '0% observed', 'nudged': '100% observed', 'zeroed': 'zeroed', 'partial': '5% observed'}


def load_config():
    return MTMConfig.from_omegaconf(OmegaConf.load(f'runs/{RUN}/config.yaml'))


def load_model(cfg):
    model = EinMask(network=cfg.model, world=cfg.world)
    ckpt = torch.load(f'runs/{RUN}/best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    return model.to(DEVICE).eval()


def setting_mask(world, v_idx, setting, rate=0.05, seed=0):
    s = world.token_sizes
    m = torch.zeros(s['v'], s['t'], s['h'], s['w'], dtype=torch.bool, device=DEVICE)
    m[:, :world.tau] = True
    if setting == 'masked':
        m[v_idx] = False
    elif setting == 'nudged':
        m[v_idx] = True
    elif setting == 'partial':
        g = torch.Generator().manual_seed(seed)
        flat = torch.zeros((s['t'] - world.tau) * s['h'] * s['w'], dtype=torch.bool)
        flat[torch.randperm(flat.numel(), generator=g)[: int(rate * flat.numel())]] = True
        m[v_idx, world.tau:] = flat.view(s['t'] - world.tau, s['h'], s['w']).to(DEVICE)
    return einops.rearrange(m, f'{world.token_pattern} -> ({world.token_pattern})')


def run_setting(cfg, model, data, visible, zero_channel=None):
    world = cfg.world
    var = cfg.data.eval_variables[0]
    v = cfg.data.variables.index(var)
    std = data._stds.sel(variable=var).values.astype(np.float32)
    meta = data.dataset
    history = world.tau * world.patch_sizes['tt']

    dl = torch.utils.data.DataLoader(data, batch_size=world.batch_size, shuffle=False, num_workers=0)
    chunks = []
    for batch_idx, batch in enumerate(dl):
        batch = batch.to(DEVICE)
        if zero_channel is not None:
            batch[:, zero_channel, :history] = 0
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


def plot_ablation_metrics(metrics, label, path):
    rows = [('', 'SSTA'), ('nino34_', 'Nino3.4'), ('nino4_', 'Nino4')]
    cols = [('pcc', 'ACC'), ('rmse', 'RMSE')]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(10, 10), layout='constrained', sharex=True)
    for r, (prefix, name) in enumerate(rows):
        for c, (key, title) in enumerate(cols):
            ax = axes[r, c]
            for setting, m in metrics.items():
                v = m[f'{prefix}{key}']
                ax.plot(v['lead'], v.values, color=COLORS[setting], label=LABELS[setting])
            if key == 'pcc':
                ax.axhline(0.5, color='r', linestyle='dashed', lw=1)
                ax.set_ylim(0, 1)
            ax.set_title(f'{name} {title}')
            if r == len(rows) - 1:
                ax.set_xlabel('Lead (Months)')
    axes[0, 0].legend()
    fig.suptitle(f'{label} ablation ({RUN}, piControl)')
    plt.savefig(path)
    plt.close()


def plot_acc_comparison(metrics, path):
    styles = {'tauxa': '-', 't150a': '--'}
    plt.figure(figsize=(6.5, 4.5))
    m = metrics['base']['pcc']
    plt.plot(m['lead'], m.values, color='k', label='base')
    for label, ls in styles.items():
        for s in SETTINGS:
            m = metrics[f'{label}_{s}']['pcc']
            plt.plot(m['lead'], m.values, ls, color=COLORS[s], label=f'{label} {LABELS[s]}')
    plt.axhline(0.5, color='r', linestyle='dashed', lw=1)
    plt.ylim(0, 1)
    plt.xlabel('Lead (Months)')
    plt.ylabel('ACC')
    plt.title(f'SSTA ACC, channel ablations ({RUN}, piControl)')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_settings_sample(raw, data, cfg, label, channel, time_idx, path, months=(1, 6, 11, 17, 23, 29, 35)):
    var = cfg.data.eval_variables[0]
    v_idx = cfg.data.variables.index(channel)
    history = cfg.world.tau * cfg.world.patch_sizes['tt']
    meta = data.dataset
    x = data[time_idx]
    ch = xr.DataArray(
        x[v_idx].numpy() * float(data._stds.sel(variable=channel)),
        coords={'step': np.arange(x.shape[1]), 'lat': meta.lat, 'lon': meta.lon},
    ).sel(lat=slice(-20., 20.), lon=slice(90, 270))
    valid = ~raw['base']['lsm']
    date = str(raw['base']['time'].values[time_idx])[:7]
    vmax_ch = 0.6 if channel == 'tauxa' else 2.0
    kw_ssta = dict(vmin=-2, vmax=2, cmap='bwr')
    kw_ch = dict(vmin=-vmax_ch, vmax=vmax_ch, cmap='PuOr_r')

    p = cfg.world.patch_sizes
    pix = einops.rearrange(setting_mask(cfg.world, v_idx, 'partial').cpu(), '(v t h w) -> v t h w', **cfg.world.token_sizes)[v_idx]
    pix = einops.repeat(pix, 't h w -> (t tt) (h hh) (w ww)', tt=p['tt'], hh=p['hh'], ww=p['ww'])
    pix = xr.DataArray(
        pix.numpy(), coords={'step': np.arange(x.shape[1]), 'lat': meta.lat, 'lon': meta.lon},
    ).sel(lat=slice(-20., 20.), lon=slice(90, 270))

    def channel_input(setting, step):
        c = ch.sel(step=step)
        if setting == 'masked':
            return c * np.nan
        if step < history:
            return c * 0 if setting == 'zeroed' else c
        if setting == 'nudged':
            return c
        if setting == 'partial':
            return c.where(pix.sel(step=step))
        return c * np.nan

    settings = ['base'] + list(SETTINGS)
    n_rows = 2 + 2 * len(settings)
    fig, axes = plt.subplots(n_rows, len(months), figsize=(2.6 * len(months), 1.35 * n_rows),
                             layout='constrained', sharex=True, sharey=True)
    ims = {}
    for col, step in enumerate(months):
        obs = raw['base'][f'{var}_obs'].isel(time=time_idx, step=step).where(valid)
        ims['ssta'] = axes[0, col].pcolormesh(obs['lon'], obs['lat'], obs, **kw_ssta)
        axes[0, col].set_title(f't={step}' + (' (ctx)' if step < history else ''), fontsize=10)
        axes[1, col].pcolormesh(obs['lon'], obs['lat'], obs if step < history else obs * np.nan, **kw_ssta)
        for r, setting in enumerate(settings):
            tag = 'base' if setting == 'base' else f'{label}_{setting}'
            ci = channel_input(setting, step).where(valid)
            ims['ch'] = axes[2 + 2 * r, col].pcolormesh(ci['lon'], ci['lat'], ci, **kw_ch)
            mu = raw[tag][f'{var}_pred_mu'].isel(time=time_idx, step=step).where(valid)
            axes[3 + 2 * r, col].pcolormesh(mu['lon'], mu['lat'], mu, **kw_ssta)
    axes[0, 0].set_ylabel('SSTA obs\n(reference)', fontsize=9)
    axes[1, 0].set_ylabel('SSTA in\n(all settings)', fontsize=9)
    for r, setting in enumerate(settings):
        axes[2 + 2 * r, 0].set_ylabel(f'{label} in\n({setting})', fontsize=9)
        axes[3 + 2 * r, 0].set_ylabel(f'SSTA mu\n({setting})', fontsize=9)
    fig.colorbar(ims['ssta'], ax=axes[:, -1], label=var, shrink=0.4)
    fig.colorbar(ims['ch'], ax=axes[:, -1], label=channel, shrink=0.4)
    fig.suptitle(f'{label} ablation sample ({RUN}, piControl, init {date}); blank = masked')
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    cfg = load_config()
    model = load_model(cfg)
    data = NinoData(cfg.trainer.picontrol_path, cfg.data.__class__(**{
        **cfg.data.__dict__, 'time_slice': {'start': '1900', 'stop': '2000', 'step': None}}))
    var = cfg.data.eval_variables[0]

    raw, datasets, metrics = {}, {}, {}
    jobs = [('base', None, 'base')]
    for label, channel in CHANNELS.items():
        jobs += [(f'{label}_{s}', channel, s) for s in SETTINGS]
    for tag, channel, setting in jobs:
        v_idx = cfg.data.variables.index(channel) if channel else None
        visible = setting_mask(cfg.world, v_idx, setting)
        ds = run_setting(cfg, model, data, visible, zero_channel=v_idx if setting == 'zeroed' else None)
        ds.to_zarr(f'{OUT}/picontrol_abl_{tag}_eval.zarr', mode='w')
        raw[tag] = ds
        datasets[tag] = to_leads(ds, cfg.world, 'frcst')
        metrics[tag] = compute_metrics(datasets[tag], var)
        print(tag, {k: float(v.mean()) for k, v in metrics[tag].items() if k in ('pcc', 'rmse', 'ssr', 'nino34_pcc')})

    for label, channel in CHANNELS.items():
        m = {'base': metrics['base'], **{s: metrics[f'{label}_{s}'] for s in SETTINGS}}
        plot_ablation_metrics(m, label, f'{OUT}/ablation_{label}_metrics.png')
        d = {'base': datasets['base'], **{s: datasets[f'{label}_{s}'] for s in SETTINGS}}
        plot_pit_hist(d, var, f'{OUT}/ablation_{label}_pit.png', title=f'SSTA PIT rank histograms, {label} ablation ({RUN}, piControl)')
        plot_settings_sample(raw, data, cfg, label, channel, time_idx=100, path=f'{OUT}/ablation_{label}_samples.png')
    plot_acc_comparison(metrics, f'{OUT}/ablation_ssta_acc.png')


if __name__ == '__main__':
    main()
