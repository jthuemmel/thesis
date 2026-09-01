import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pathlib import Path

from analysis.metrics import (
    open_pieces,
    p_pcc, p_rmse, p_acc, p_fi, p_info, p_ie, p_ne, p_stde, p_sdav, p_activ,
    p_crps, p_ign, p_spread, p_ssr, p_crps_ss, p_rmse_ss,
)

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SEASONS = {'DJF' : (12, 1, 2), 'MAM' : (3, 4, 5), 'JJA' : (6, 7, 8), 'SON' : (9, 10, 11)}

### HELPERS

def to_leads(da: xr.DataArray, history: int) -> xr.DataArray:
    # forecast steps only, indexed by lead time in months
    lead = da['step'] - history + 1
    return da.assign_coords(step=lead).rename(step='lead').sel(lead=slice(1, None))

def index_stats(m: xr.Dataset, name: str):
    # point forecast, spread and obs for a stored index series
    obs = m[f'{name}_obs']
    if f'{name}_member' in m:
        member = m[f'{name}_member'].astype('float32')
        return member.mean('ens'), member.std('ens', ddof=1), member, obs
    return m[f'{name}_mu'], m[f'{name}_sigma'], None, obs

def index_pcc(m: xr.Dataset, name: str) -> xr.DataArray:
    mu, _, _, obs = index_stats(m, name)
    return (mu * obs).sum('time') / (np.sqrt((mu**2).sum('time')) * np.sqrt((obs**2).sum('time')))

def index_rmse(m: xr.Dataset, name: str) -> xr.DataArray:
    mu, _, _, obs = index_stats(m, name)
    return np.sqrt(((mu - obs)**2).mean('time'))

def index_acc(m: xr.Dataset, name: str) -> xr.DataArray:
    mu, _, _, obs = index_stats(m, name)
    af, at = mu - mu.mean('time'), obs - obs.mean('time')
    return (af * at).mean('time') / (np.sqrt((af**2).mean('time')) * np.sqrt((at**2).mean('time')))

def field_mean(metric_fn, m: xr.Dataset, prefix: str) -> xr.DataArray:
    # spatial metric per (time, step), averaged over inits
    return metric_fn(m, prefix).mean('time', skipna=True)

def season_sums(m: xr.Dataset, prefix: str, season: tuple) -> xr.Dataset:
    # rebin monthly map moments into one season
    keys = [k for k in m.data_vars if k.startswith(f'{prefix}mon_')]
    out = m[keys].sel(month=list(season)).sum('month')
    return out.rename({k : k.replace('mon_', '') for k in keys})

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

### PER-RUN DIAGNOSTICS

def plot_skill(m: xr.Dataset, prefix: str, history: int, path: Path):
    pcc = to_leads(field_mean(p_pcc, m, prefix), history)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(pcc['lead'], to_leads(index_pcc(m, 'nino34'), history), label='nino3.4')
    plt.plot(pcc['lead'], to_leads(index_pcc(m, 'nino4'), history), label='nino4')
    plt.plot(pcc['lead'], pcc, label='field')
    plt.axhline(0.5, color='r', linestyle='dashed')
    plt.ylim(0, 1)
    plt.legend()
    plt.xlabel('Lead')
    plt.ylabel('Correlation')
    savefig(fig, path)

def plot_monthly_init(m: xr.Dataset, prefix: str, history: int, path: Path, E: int = None):
    month = m['time.month']
    panels = [
        ('ACC', p_acc),
        ('IE', p_ie),
        ('CRPS-SS', lambda mm, q: p_crps_ss(mm, q, E=E)),
        ('SSR', lambda mm, q: p_ssr(mm, q, E=E)),
    ]
    lev = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 4 * len(panels)))
    for ax, (label, field_fn) in zip(np.atleast_1d(axes), panels):
        field = to_leads(field_fn(m, prefix).groupby(month).mean('time', skipna=True), history)
        field = field.transpose('month', 'lead')
        levels = np.linspace(0.5, 1.5, 11) if label == 'SSR' else lev
        filled = ax.contourf(field['lead'], field['month'], field, cmap='coolwarm', levels=levels, extend='both')
        ax.set_yticks(np.arange(1, 13))
        ax.set_yticklabels(MONTHS)
        ax.set_xlabel('Lead')
        ax.set_ylabel('Init Month')
        fig.colorbar(filled, ax=ax, label=label)
    savefig(fig, path)

def plot_histograms(m: xr.Dataset, prefix: str, history: int, path: Path):
    # rank (ens) or pit (mve) frequencies over all leads, normalized by their own bin sums
    key = f'{prefix}rank' if f'{prefix}rank' in m else f'{prefix}pit'
    counts = m[key]
    if 'time' in counts.dims:
        counts = counts.sum('time')
    counts = to_leads(counts, history)
    freq = (counts / counts.sum('bin')).transpose('lead', 'bin')
    fig = plt.figure(figsize=(8, 4))
    pcm = plt.pcolormesh(freq['bin'], freq['lead'], freq, cmap='coolwarm',
                         vmin=0, vmax=2 / counts.sizes['bin'])
    plt.colorbar(pcm, label='Frequency')
    plt.xlabel('Bin')
    plt.ylabel('Lead')
    savefig(fig, path)

def member_view(m: xr.Dataset, prefix: str) -> xr.Dataset:
    # alias the per-member sums onto the standard moment names so the p_* recovery applies per member
    q = prefix
    return xr.Dataset({
        f'{q}n' :   m[f'{q}n'],
        f'{q}so' :  m[f'{q}so'],
        f'{q}soo' : m[f'{q}soo'],
        f'{q}sf' :  m[f'{q}msf'],
        f'{q}sff' : m[f'{q}msff'],
        f'{q}sfo' : m[f'{q}msfo'],
    })

def plot_info_noise(m: xr.Dataset, prefix: str, history: int, path: Path):
    info = to_leads(field_mean(p_info, m, prefix), history)
    ne = to_leads(field_mean(p_ne, m, prefix), history)
    sdav = float(field_mean(p_sdav, m, prefix).mean('step', skipna=True))
    lead = info['lead'].values
    rmax = np.nanmax([sdav, np.nanmax(info), np.nanmax(ne)]) * 1.1
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    th = np.linspace(0, np.pi / 2, 200)
    for acc in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        a = np.arccos(acc)
        ax.plot([0, rmax * np.sin(a)], [0, rmax * np.cos(a)], ':', color='skyblue', lw=2.5 if acc == 0.5 else 0.6, zorder=0)
        ax.text(rmax * np.sin(a), rmax * np.cos(a), f'{acc:g}', fontsize=7, color='dimgray')
    ax.plot(sdav * np.sin(th), sdav * np.cos(th), 'k--', lw=1, zorder=1)
    ax.plot(0, sdav, 'k^', ms=9, zorder=3)
    # maybe per-member trajectories
    if f'{prefix}msf' in m:
        mv = member_view(m, prefix)
        info_m = to_leads(field_mean(p_info, mv, prefix), history)
        ne_m = to_leads(field_mean(p_ne, mv, prefix), history)
        for e in mv[f'{prefix}sf']['ens'].values:
            ax.plot(ne_m.sel(ens=e), info_m.sel(ens=e), '-', color='0.7', lw=0.6, zorder=1)
    colors = plt.colormaps['viridis'](np.linspace(0, 1, len(lead)))
    ax.plot(ne, info, '-', color='0.4', lw=0.8, zorder=2)
    for x, y, lg, c in zip(ne.values, info.values, lead, colors):
        ax.scatter(x, y, color=c, s=45, zorder=4, edgecolor='k', lw=0.4)
        ax.annotate(f'{lg}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=7)
    ax.set_xlim(0, rmax)
    ax.set_ylim(0, rmax)
    ax.set_aspect('equal')
    ax.set_xlabel('Noise')
    ax.set_ylabel('Information')
    savefig(fig, path)

def plot_calibration(m: xr.Dataset, prefix: str, history: int, path: Path, E: int = None):
    ssr = to_leads(field_mean(lambda mm, q: p_ssr(mm, q, E=E), m, prefix), history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ssr['lead'], ssr)
    axes[0].axhline(1., color='r', linestyle='dashed')
    axes[0].set_xlabel('Lead')
    axes[0].set_ylabel('SSR')
    if f'{prefix}sign' in m:
        ign = to_leads(field_mean(p_ign, m, prefix), history)
        axes[1].plot(ign['lead'], ign)
    axes[1].set_xlabel('Lead')
    axes[1].set_ylabel('IGN')
    # spread-reliability curve from the binned (σ², err²) sums — RMS vs RMS sits on the 1:1 diagonal
    n, sig2, err2 = m[f'{prefix}rel_n'], m[f'{prefix}rel_sig2'], m[f'{prefix}rel_err2']
    if 'step' in n.dims:
        n, sig2, err2 = n.sum('step'), sig2.sum('step'), err2.sum('step')
    keep = n > 0
    axes[2].plot(np.sqrt(sig2 / n).where(keep), np.sqrt(err2 / n).where(keep), 'o-', ms=3)
    lim = float(np.nanmax(np.sqrt(sig2 / n).where(keep))) * 1.1
    axes[2].plot([0, lim], [0, lim], 'k--', lw=1)
    axes[2].set_xlabel('Spread')
    axes[2].set_ylabel('Error')
    savefig(fig, path)

def plot_seasonal_maps(m: xr.Dataset, prefix: str, path: Path):
    fig, axes = plt.subplots(len(SEASONS), 2, figsize=(12, 3 * len(SEASONS)))
    for row, (label, season) in enumerate(SEASONS.items()):
        s = season_sums(m, prefix, season).sum('lband')
        for col, (name, metric, kw) in enumerate([
            ('ACC', p_acc(s, f'{prefix}'), dict(vmin=0, vmax=1, cmap='coolwarm')),
            ('RMSE', p_rmse(s, f'{prefix}'), dict(vmin=0, cmap='viridis')),
        ]):
            ax = axes[row, col]
            pcm = ax.pcolormesh(metric['lon'], metric['lat'], metric, **kw)
            ax.set_ylabel(label)
            fig.colorbar(pcm, ax=ax, label=name)
    savefig(fig, path)

def plot_sections(m: xr.Dataset, prefix: str, history: int, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, sec in enumerate(['sec', 'mer']):
        q = f'{prefix}{sec}_'
        axis = 'lon' if sec == 'sec' else 'lat'
        for row, (label, metric) in enumerate([('ACC', p_acc(m, q)), ('RMSE', p_rmse(m, q))]):
            metric = to_leads(metric, history)
            ax = axes[row, col]
            pcm = ax.pcolormesh(metric[axis], metric['lead'], metric.transpose('lead', axis))
            ax.set_xlabel(axis)
            ax.set_ylabel('Lead')
            fig.colorbar(pcm, ax=ax, label=label)
    savefig(fig, path)

def plot_spectra(m: xr.Dataset, prefix: str, history: int, path: Path):
    # banded zonal power ratio forecast/observed over all leads — spectra are only used as ratios
    ratio = to_leads(m[f'{prefix}pff'] / m[f'{prefix}poo'], history).transpose('lead', 'band')
    fig = plt.figure(figsize=(8, 4))
    pcm = plt.pcolormesh(ratio['band'], ratio['lead'], np.log2(ratio), cmap='coolwarm', vmin=-2, vmax=2)
    plt.colorbar(pcm, label='log2 power ratio')
    plt.xlabel('Wavenumber band')
    plt.ylabel('Lead')
    savefig(fig, path)

### EVOLUTION AND CROSS-RUN

def plot_evolution(path: Path, dataset: str, task: str, prefix: str, history: int, out: Path):
    # skill across training: pcc vs lead per dispatch, and the full (step, lead) surface
    m = open_pieces(path, dataset, task)
    pcc = to_leads(field_mean(p_pcc, m, prefix), history)
    steps = pcc['dispatch'].values
    colors = plt.colormaps['viridis'](np.linspace(0.2, 0.85, len(steps)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for step, c in zip(steps, colors):
        axes[0].plot(pcc['lead'], pcc.sel(dispatch=step), color=c, label=f'{step}')
    axes[0].axhline(0.5, color='r', linestyle='dashed')
    axes[0].set_xlabel('Lead')
    axes[0].set_ylabel('PCC')
    axes[0].legend(title='Step', fontsize=7)
    surface = pcc.transpose('dispatch', 'lead')
    pcm = axes[1].pcolormesh(surface['lead'], surface['dispatch'], surface, cmap='coolwarm', vmin=0, vmax=1)
    fig.colorbar(pcm, ax=axes[1], label='PCC')
    axes[1].set_xlabel('Lead')
    axes[1].set_ylabel('Training step')
    savefig(fig, out)

def plot_comparison(paths: dict, dataset: str, task: str, prefix: str, history: int, out: Path, metric_fn=p_pcc):
    # metric vs lead, one line per run — paths = {label : eval.zarr path}
    fig = plt.figure(figsize=(8, 4))
    for label, path in paths.items():
        m = open_pieces(path, dataset, task)
        if 'dispatch' in m.dims:
            m = m.isel(dispatch=-1)
        curve = to_leads(field_mean(metric_fn, m, prefix), history)
        plt.plot(curve['lead'], curve, label=label)
    plt.axhline(0.5, color='r', linestyle='dashed')
    plt.xlabel('Lead')
    plt.legend()
    savefig(fig, out)

def plot_run(path: Path, dataset: str, task: str, out_dir: Path, prefix: str = 'temp_ocn_0a_', history: int = 12, step: int = None, E: int = None):
    # the default per-run figure battery from one (dataset, task) cell
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = open_pieces(path, dataset, task, step=step)
    if 'dispatch' in m.dims:
        m = m.isel(dispatch=-1)
    tag = f'{dataset}_{task}'
    plot_skill(m, prefix, history, out_dir / f'{tag}_skill.png')
    plot_info_noise(m, prefix, history, out_dir / f'{tag}_info_noise.png')
    if f'{prefix}rank' in m or f'{prefix}pit' in m:
        plot_histograms(m, prefix, history, out_dir / f'{tag}_hist.png')
    if f'{prefix}rel_n' in m:
        plot_calibration(m, prefix, history, out_dir / f'{tag}_calibration.png', E=E)
    if f'{prefix}sec_sf' in m:
        plot_sections(m, prefix, history, out_dir / f'{tag}_sections.png')
    if f'{prefix}mon_sf' in m:
        plot_monthly_init(m, prefix, history, out_dir / f'{tag}_monthly.png', E=E)
        plot_seasonal_maps(m, f'{prefix}', out_dir / f'{tag}_seasonal_maps.png')
    if f'{prefix}pff' in m:
        plot_spectra(m, prefix, history, out_dir / f'{tag}_spectra.png')
