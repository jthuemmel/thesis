import math
import shutil
import zarr
import numpy as np
import xarray as xr

from pathlib import Path
from scipy.special import erf
from scipy.ndimage import distance_transform_edt

from utils.config import exists, default, EvaluationConfig
from analysis.metrics import gaussian_crps, gaussian_ign

SPACE = ('lat', 'lon')

TIERS = ('light', 'standard', 'full')

### EVALUATION
# config-driven assembly of tiered sufficient statistics ("pieces") and their storage
# one eval.zarr per run: group per (dataset, task), append along a dispatch axis, static subgroup written once

class Evaluation:
    def __init__(self, config: EvaluationConfig = None):
        self.cfg = default(config, EvaluationConfig())

    # CONFIG ACCESS
    @property
    def dist_edges(self) -> tuple:
        return tuple(self.cfg.kwargs.get('dist_edges', (0.5, 1.5, 2.5))) + (np.inf,)

    @property
    def equator(self) -> dict:
        lo, hi = self.cfg.kwargs.get('section_band', (-5, 5))
        return {'lat' : slice(lo, hi)}

    def region(self, name: str) -> dict:
        return {ax : slice(*bounds) for ax, bounds in self.cfg.regions[name].items()}

    def rel_edges(self, std: float = 1.) -> np.ndarray:
        # σ bins scaled by the variable's reference std, so wind stress resolves as well as SST
        # edges must be identical across every chunk and dispatch of a group — always pass the same stds
        lo, hi = self.cfg.kwargs.get('rel_range', (1e-2, 1e1))
        return std * np.geomspace(lo, hi, self.cfg.rel_bins + 1)

    # HELPERS
    @staticmethod
    def discover(ds: xr.Dataset) -> list:
        return sorted(k[:-4] for k in ds.data_vars if k.endswith('_obs'))

    @staticmethod
    def box_mean(da: xr.DataArray, sel: dict) -> xr.DataArray:
        return da.sel(**sel).mean(SPACE, skipna=True)

    @staticmethod
    def valid_mask(ds: xr.Dataset, var: str, sea: xr.DataArray) -> xr.DataArray:
        # NaN obs are excluded once here so counts match every skipna sum exactly
        return ~ds[f'{var}_visible'].astype(bool) & sea & ds[f'{var}_obs'].notnull()

    @staticmethod
    def heads(ds: xr.Dataset, var: str):
        # f is the point forecast: ens mean or mu head
        if f'{var}_pred' in ds:
            pred = ds[f'{var}_pred']
            return pred.mean('ens', skipna=True), pred, None, None
        return ds[f'{var}_pred_mu'], None, ds[f'{var}_pred_mu'], ds[f'{var}_pred_sigma']

    @staticmethod
    def pairwise_sum(pred: xr.DataArray) -> xr.DataArray:
        E = pred.sizes['ens']
        def _pairwise(p):
            total = np.zeros(p.shape[:-1], dtype=np.float64)
            for i in range(E):
                total = total + np.sum(np.abs(p[..., i:i + 1] - p[..., i + 1:]), axis=-1)
            return total
        return xr.apply_ufunc(_pairwise, pred, input_core_dims=[['ens']])

    @staticmethod
    def rfft_lon(da: xr.DataArray) -> xr.DataArray:
        out = xr.apply_ufunc(np.fft.rfft, da, input_core_dims=[['lon']], output_core_dims=[['k']], kwargs={'axis' : -1})
        return out.assign_coords(k=np.arange(out.sizes['k']))

    @staticmethod
    def binned_counts(idx: xr.DataArray, valid: xr.DataArray, nbins: int) -> xr.DataArray:
        # one bincount pass over a combined (frame, bin) index instead of a loop over bins
        T, S = idx.sizes['time'], idx.sizes['step']
        iv = idx.transpose('time', 'step', *SPACE).values.reshape(T * S, -1)
        ok = valid.transpose('time', 'step', *SPACE).values.reshape(T * S, -1)
        iv = np.nan_to_num(iv, nan=0.).astype(np.int64)
        flat = (np.arange(T * S)[:, None] * nbins + iv.clip(0, nbins - 1))[ok]
        counts = np.bincount(flat, minlength=T * S * nbins).reshape(T, S, nbins)
        return xr.DataArray(counts.transpose(2, 0, 1), dims=('bin', 'time', 'step'),
                            coords={'bin' : np.arange(nbins), 'time' : idx['time'].values, 'step' : idx['step'].values})

    @staticmethod
    def band_edges(num_k: int, nbands: int) -> np.ndarray:
        return np.unique(np.geomspace(1, num_k - 1, nbands + 1).round())

    @staticmethod
    def bandsum(P: xr.DataArray, edges: np.ndarray) -> xr.DataArray:
        g = P.isel(k=slice(1, None)).sum('lat', skipna=True).groupby_bins('k', edges, include_lowest=True).sum()
        g = g.rename(k_bins='band').assign_coords(band=np.arange(len(edges) - 1))
        return g.sum('time') if 'time' in g.dims else g

    @staticmethod
    def bymonth(da: xr.DataArray, nleads: int) -> xr.DataArray:
        # pool pointwise sums into (init month, lead band) maps
        edges = np.linspace(-0.5, da.sizes['step'] - 0.5, nleads + 1)
        g = da.groupby('time.month').sum('time')
        g = g.groupby_bins('step', edges, include_lowest=True).sum().rename(step_bins='lband')
        return g.assign_coords(lband=np.arange(nleads)).reindex(month=np.arange(1, 13), fill_value=0.)

    @staticmethod
    def spatial_distance(vis: xr.DataArray) -> xr.DataArray:
        v = vis.transpose('time', 'step', *SPACE).values
        dist = np.full(v.shape, np.inf, dtype=np.float32)
        for i in range(v.shape[0]):
            for t in range(v.shape[1]):
                if v[i, t].any():
                    dist[i, t] = distance_transform_edt(~v[i, t])
        ref = vis.transpose('time', 'step', *SPACE)
        return ref.astype('float32').copy(data=dist)

    @staticmethod
    def nearest_fill(obs: xr.DataArray, visible: xr.DataArray, sea: xr.DataArray) -> xr.DataArray:
        # fill only from usable observations — NaN sources would desync the baseline sums from the obs counts
        vis = (visible.astype(bool) & sea & obs.notnull()).transpose('time', 'step', *SPACE).values
        vals = obs.transpose('time', 'step', *SPACE).values
        filled = np.full_like(vals, np.nan)
        ok = np.zeros(vals.shape[:2], dtype=bool)
        for i in range(vals.shape[0]):
            for t in range(vals.shape[1]):
                if vis[i, t].any():
                    _, ind = distance_transform_edt(~vis[i, t], return_indices=True)
                    filled[i, t] = vals[i, t][ind[0], ind[1]]
                    ok[i, t] = True
            good = np.flatnonzero(ok[i])
            for t in np.flatnonzero(~ok[i]):
                if good.size > 0:
                    filled[i, t] = filled[i, good[np.abs(good - t).argmin()]]
        return obs.transpose('time', 'step', *SPACE).copy(data=filled)

    @staticmethod
    def is_partial(ds: xr.Dataset, var: str, sea: xr.DataArray) -> bool:
        visfrac = ds[f'{var}_visible'].astype(bool).where(sea).mean(SPACE, skipna=True)
        return bool(((visfrac > 0) & (visfrac < 1)).any())

    # PIECES
    # sums over space (or time) that suffice to recover the metrics, accumulated in float64
    @staticmethod
    def obs_moments(obs: xr.DataArray, valid: xr.DataArray, prefix: str, dim: tuple[str] = SPACE) -> dict:
        om = obs.where(valid).astype('float64')
        return {
            f'{prefix}n' :   valid.sum(dim),
            f'{prefix}so' :  om.sum(dim, skipna=True),
            f'{prefix}soo' : (om**2).sum(dim, skipna=True),
        }

    @staticmethod
    def pred_moments(f: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str, dim: tuple[str] = SPACE) -> dict:
        fm, om = f.where(valid).astype('float64'), obs.where(valid).astype('float64')
        return {
            f'{prefix}sf' :  fm.sum(dim, skipna=True),
            f'{prefix}sff' : (fm**2).sum(dim, skipna=True),
            f'{prefix}sfo' : (fm * om).sum(dim, skipna=True),
            f'{prefix}sad' : np.abs(fm - om).sum(dim, skipna=True),
        }

    @staticmethod
    def clim_sums(obs: xr.DataArray, valid: xr.DataArray, prefix: str, std: float = 1.) -> dict:
        crps = gaussian_crps(xr.zeros_like(obs), xr.full_like(obs, std), obs)
        return {f'{prefix}scrps_clim' : crps.where(valid).astype('float64').sum(SPACE, skipna=True)}

    def ens_sums(self, pred: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        return {
            f'{prefix}svar' : pred.var('ens', ddof=1).where(valid).astype('float64').sum(SPACE, skipna=True),
            f'{prefix}sae' :  np.abs(pred - obs).sum('ens').where(valid).astype('float64').sum(SPACE, skipna=True),
            f'{prefix}sap' :  self.pairwise_sum(pred).where(valid).sum(SPACE, skipna=True),
        }

    @staticmethod
    def mve_sums(mu: xr.DataArray, sigma: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        z = (obs - mu) / sigma.clip(min=1e-6)
        return {
            f'{prefix}scrps' : gaussian_crps(mu, sigma, obs).where(valid).astype('float64').sum(SPACE, skipna=True),
            f'{prefix}sign' :  gaussian_ign(mu, sigma, obs).where(valid).astype('float64').sum(SPACE, skipna=True),
            f'{prefix}ssig2' : (sigma.clip(min=1e-6)**2).where(valid).astype('float64').sum(SPACE, skipna=True),
            f'{prefix}sz' :    z.where(valid).astype('float64').sum(SPACE, skipna=True),
            f'{prefix}szz' :   (z**2).where(valid).astype('float64').sum(SPACE, skipna=True),
        }

    def member_moments(self, pred: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        # per-member sums for the info/noise diagram, first few members only
        fm = pred.isel(ens=slice(self.cfg.num_members)).where(valid).astype('float64')
        om = obs.where(valid).astype('float64')
        return {
            f'{prefix}msf' :  fm.sum(SPACE, skipna=True),
            f'{prefix}msff' : (fm**2).sum(SPACE, skipna=True),
            f'{prefix}msfo' : (fm * om).sum(SPACE, skipna=True),
        }

    def rank_counts(self, pred: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        E = pred.sizes['ens']
        rank = (pred < obs).sum('ens')
        return {f'{prefix}rank' : self.binned_counts(rank, valid, E + 1)}

    def pit_counts(self, mu: xr.DataArray, sigma: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        nbins = self.cfg.pit_bins
        z = (obs - mu) / sigma.clip(min=1e-6)
        pit = 0.5 * (1 + xr.apply_ufunc(erf, z / math.sqrt(2)))
        idx = np.floor(nbins * pit).clip(0, nbins - 1)
        return {f'{prefix}pit' : self.binned_counts(idx, valid, nbins)}

    def reliability_sums(self, f: xr.DataArray, sigma: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray,
                         prefix: str, std: float = 1.) -> dict:
        # binned joint of (σ², err²) for the spread-reliability curve, pooled over time and space
        # squares keep the binned curve consistent with the SSR recovered from svar/ssig2
        edges = self.rel_edges(std)
        nbins = len(edges) - 1
        S = f.sizes['step']
        sv = sigma.clip(min=1e-6).where(valid).transpose('step', ...).values.reshape(S, -1)
        ev = np.abs(f - obs).where(valid).transpose('step', ...).values.reshape(S, -1)
        ok = np.isfinite(sv) & np.isfinite(ev)
        # one bincount pass over a combined (step, bin) index instead of a loop over bins
        idx = np.digitize(sv, edges).clip(1, nbins) - 1
        flat = (np.arange(S)[:, None] * nbins + idx)[ok]
        n = np.bincount(flat, minlength=S * nbins).reshape(S, nbins)
        ssig2 = np.bincount(flat, weights=(sv[ok].astype('float64'))**2, minlength=S * nbins).reshape(S, nbins)
        serr2 = np.bincount(flat, weights=(ev[ok].astype('float64'))**2, minlength=S * nbins).reshape(S, nbins)
        # own bin dim — sharing 'bin' with the rank/pit counts would outer-join the two on merge
        coords = {'sbin' : np.arange(nbins), 'step' : f['step'].values}
        wrap = lambda a: xr.DataArray(a.T, dims=('sbin', 'step'), coords=coords)
        return {f'{prefix}rel_n' : wrap(n), f'{prefix}rel_sig2' : wrap(ssig2), f'{prefix}rel_err2' : wrap(serr2)}

    def zonal_spectra(self, f: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str, members: xr.DataArray = None) -> dict:
        # banded zonal power, pooled over time — only ever used as ratios; poo lives in the static group
        F, O = self.rfft_lon(f.where(valid, 0.)), self.rfft_lon(obs.where(valid, 0.))
        edges = self.band_edges(F.sizes['k'], self.cfg.num_bands)
        pieces = {
            f'{prefix}pff' : self.bandsum((F * F.conj()).real, edges),
            f'{prefix}pfo' : self.bandsum((F * O.conj()).real, edges),
        }
        if exists(members):
            M = self.rfft_lon(members.where(valid, 0.))
            pieces[f'{prefix}pmm'] = self.bandsum((M * M.conj()).real.mean('ens'), edges)
        return pieces

    def zonal_obs_spectra(self, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        O = self.rfft_lon(obs.where(valid, 0.))
        return {f'{prefix}poo' : self.bandsum((O * O.conj()).real, self.band_edges(O.sizes['k'], self.cfg.num_bands))}

    def field_maps(self, f: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, spread2: xr.DataArray, prefix: str) -> dict:
        out = self.pred_moments(f, obs, valid, f'{prefix}map_', dim=('time',))
        out[f'{prefix}map_svar'] = spread2.where(valid).astype('float64').sum('time', skipna=True)
        return out

    def monthly_maps(self, f: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, spread2: xr.DataArray, prefix: str) -> dict:
        # init-month × lead-band conditioned maps — explicit pointwise sums, no space reduction
        # the float64 intermediates are full-field: call per chunk, never on an assembled dataset
        fm = f.where(valid).astype('float64').fillna(0.)
        om = obs.where(valid).astype('float64').fillna(0.)
        pieces = {
            f'{prefix}mon_sf' :   fm,
            f'{prefix}mon_sff' :  fm**2,
            f'{prefix}mon_sfo' :  fm * om,
            f'{prefix}mon_sad' :  np.abs(fm - om),
            f'{prefix}mon_svar' : spread2.where(valid).astype('float64').fillna(0.),
        }
        return {k : self.bymonth(v, self.cfg.num_leads) for k, v in pieces.items()}

    def monthly_obs_maps(self, obs: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        om = obs.where(valid).astype('float64').fillna(0.)
        pieces = {
            f'{prefix}mon_n' :   valid.astype('int64'),
            f'{prefix}mon_so' :  om,
            f'{prefix}mon_soo' : om**2,
        }
        return {k : self.bymonth(v, self.cfg.num_leads) for k, v in pieces.items()}

    def equatorial_section(self, f: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str, spread2: xr.DataArray = None) -> dict:
        band = lambda da: da.sel(**self.equator)
        out = self.pred_moments(band(f), band(obs), band(valid), f'{prefix}sec_', dim=('time', 'lat'))
        if exists(spread2):
            out[f'{prefix}sec_svar'] = band(spread2.where(valid)).astype('float64').sum(('time', 'lat'), skipna=True)
        return out

    def meridional_section(self, f: xr.DataArray, obs: xr.DataArray, valid: xr.DataArray, prefix: str, spread2: xr.DataArray = None) -> dict:
        out = self.pred_moments(f, obs, valid, f'{prefix}mer_', dim=('time', 'lon'))
        if exists(spread2):
            out[f'{prefix}mer_svar'] = spread2.where(valid).astype('float64').sum(('time', 'lon'), skipna=True)
        return out

    def band_moments(self, f: xr.DataArray, obs: xr.DataArray, vis: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        # pred metrics stratified by spatial distance to the nearest visible pixel; obs terms live in static
        dist = self.spatial_distance(vis)
        edges = self.dist_edges
        pieces = {}
        for b in range(len(edges) - 1):
            sel = valid & (dist > edges[b]) & (dist <= edges[b + 1])
            pieces.update(self.pred_moments(f, obs, sel, f'{prefix}d{b + 1}_'))
        return pieces

    def band_obs_moments(self, obs: xr.DataArray, vis: xr.DataArray, valid: xr.DataArray, prefix: str) -> dict:
        dist = self.spatial_distance(vis)
        edges = self.dist_edges
        pieces = {}
        for b in range(len(edges) - 1):
            sel = valid & (dist > edges[b]) & (dist <= edges[b + 1])
            pieces.update(self.obs_moments(obs, sel, f'{prefix}d{b + 1}_'))
        return pieces

    def index_obs(self, name: str, sel: dict, ds: xr.Dataset, var: str, sea: xr.DataArray) -> dict:
        return {
            f'{name}_obs' :     self.box_mean(ds[f'{var}_obs'].where(sea), sel),
            f'{name}_visfrac' : self.box_mean(ds[f'{var}_visible'].where(sea), sel),
        }

    def index_pred(self, name: str, sel: dict, ds: xr.Dataset, var: str, sea: xr.DataArray) -> dict:
        valid = self.valid_mask(ds, var, sea)
        if f'{var}_pred' in ds:
            return {f'{name}_member' : self.box_mean(ds[f'{var}_pred'].where(valid), sel)}
        return {
            f'{name}_mu' :    self.box_mean(ds[f'{var}_pred_mu'].where(valid), sel),
            f'{name}_sigma' : self.box_mean(ds[f'{var}_pred_sigma'].where(valid), sel),
        }

    @staticmethod
    def mask_geometry(ds: xr.Dataset, var: str, sea: xr.DataArray) -> dict:
        visfrac = ds[f'{var}_visible'].astype(bool).where(sea).mean(SPACE, skipna=True)
        def _dist(v):
            idx = np.flatnonzero(v)
            if idx.size == 0:
                return np.full(v.shape, np.nan, dtype=np.float32)
            return np.abs(np.arange(v.size)[:, None] - idx[None, :]).min(-1).astype(np.float32)
        tdist = xr.apply_ufunc(_dist, visfrac > 0, input_core_dims=[['step']], output_core_dims=[['step']], vectorize=True)
        return {f'{var}_visfrac' : visfrac, f'{var}_tdist' : tdist}

    def column_series(self, fn, name: str, ds: xr.Dataset, sea: xr.DataArray) -> dict:
        # depth-averaged index series over the wwv column
        subs = [v for v in self.cfg.column_variables if f'{v}_obs' in ds]
        if len(subs) == 0:
            return {}
        parts = [fn(name, self.region(name), ds, v, sea) for v in subs]
        return {key : sum(p[key] for p in parts) / len(subs) for key in parts[0]}

    def index_series(self, fn, ds: xr.Dataset, sea: xr.DataArray, tier: str) -> dict:
        pieces = {}
        if f'{self.cfg.index_variable}_obs' in ds:
            for name in self.cfg.index_regions:
                pieces.update(fn(name, self.region(name), ds, self.cfg.index_variable, sea))
        if tier != 'light':
            for name in self.cfg.column_regions:
                pieces.update(self.column_series(fn, name, ds, sea))
        return pieces

    # ASSEMBLY
    def variables(self, ds: xr.Dataset) -> list:
        # intersect with the dataset — a config-listed variable absent from an older library must not raise
        return [v for v in default(self.cfg.variables, self.discover(ds)) if f'{v}_obs' in ds]

    def compute(self, ds: xr.Dataset, tier: str = 'full', stds: dict = None, partial: bool = None) -> xr.Dataset:
        # model-dependent pieces for one eval pass; obs/mask-only counterparts live in static
        # pass partial explicitly for stochastic mask tasks — per-chunk auto-detection can differ between chunks
        variables = self.variables(ds)
        full = tuple(v for v in self.cfg.full_variables if v in variables) or (variables[0],)
        sea = ~ds['lsm'].astype(bool)
        pieces = {}
        for var in variables:
            obs = ds[f'{var}_obs']
            valid = self.valid_mask(ds, var, sea)
            f, pred, mu, sigma = self.heads(ds, var)
            spread2 = pred.var('ens', ddof=1) if exists(pred) else sigma.clip(min=1e-6)**2
            std = (stds or {}).get(var, 1.)
            # n is re-recorded per dispatch as a mask-drift canary against the static group
            pieces[f'{var}_n'] = valid.sum(SPACE)
            pieces.update(self.pred_moments(f, obs, valid, f'{var}_'))
            if exists(pred):
                pieces.update(self.ens_sums(pred, obs, valid, f'{var}_'))
            else:
                pieces.update(self.mve_sums(mu, sigma, obs, valid, f'{var}_'))
            if tier == 'light':
                continue
            pieces.update(self.equatorial_section(f, obs, valid, f'{var}_', spread2=spread2))
            pieces.update(self.meridional_section(f, obs, valid, f'{var}_', spread2=spread2))
            pieces.update(self.reliability_sums(f, np.sqrt(spread2), obs, valid, f'{var}_', std=std))
            if exists(pred):
                counts = self.rank_counts(pred, obs, valid, f'{var}_')
            else:
                counts = self.pit_counts(mu, sigma, obs, valid, f'{var}_')
            if tier == 'standard':
                counts = {k : v.sum('time') for k, v in counts.items()}
            pieces.update(counts)
            if tier == 'full' and var in full:
                pieces.update(self.field_maps(f, obs, valid, spread2, f'{var}_'))
                pieces.update(self.monthly_maps(f, obs, valid, spread2, f'{var}_'))
                pieces.update(self.zonal_spectra(f, obs, valid, f'{var}_', members=pred))
                if exists(pred):
                    pieces.update(self.member_moments(pred, obs, valid, f'{var}_'))
                if partial if exists(partial) else self.is_partial(ds, var, sea):
                    pieces.update(self.band_moments(f, obs, ds[f'{var}_visible'].astype(bool) & sea, valid, f'{var}_'))
        pieces.update(self.index_series(self.index_pred, ds, sea, tier))
        return xr.Dataset(pieces)

    def static(self, ds: xr.Dataset, stds: dict = None, partial: bool = None) -> xr.Dataset:
        # everything depending only on (obs, mask) — dispatch-invariant, written once per (dataset, task)
        # always the full piece set: a group's static must serve every later dispatch tier
        variables = self.variables(ds)
        full = tuple(v for v in self.cfg.full_variables if v in variables) or (variables[0],)
        sea = ~ds['lsm'].astype(bool)
        band = lambda da: da.sel(**self.equator)
        pieces = {}
        for var in variables:
            obs = ds[f'{var}_obs']
            valid = self.valid_mask(ds, var, sea)
            pieces.update(self.obs_moments(obs, valid, f'{var}_'))
            pieces.update(self.clim_sums(obs, valid, f'{var}_', std=(stds or {}).get(var, 1.)))
            pieces.update(self.obs_moments(band(obs), band(valid), f'{var}_sec_', dim=('time', 'lat')))
            pieces.update(self.obs_moments(obs, valid, f'{var}_mer_', dim=('time', 'lon')))
            if var in full:
                pieces.update(self.mask_geometry(ds, var, sea))
                pieces.update(self.obs_moments(obs, valid, f'{var}_map_', dim=('time',)))
                pieces.update(self.monthly_obs_maps(obs, valid, f'{var}_'))
                pieces.update(self.zonal_obs_spectra(obs, valid, f'{var}_'))
                base = self.nearest_fill(obs, ds[f'{var}_visible'], sea)
                pieces.update(self.pred_moments(base, obs, valid, f'{var}_nn_'))
                pieces.update(self.equatorial_section(base, obs, valid, f'{var}_nn_'))
                pieces.update(self.meridional_section(base, obs, valid, f'{var}_nn_'))
                if partial if exists(partial) else self.is_partial(ds, var, sea):
                    pieces.update(self.band_obs_moments(obs, ds[f'{var}_visible'].astype(bool) & sea, valid, f'{var}_'))
        pieces.update(self.index_series(self.index_obs, ds, sea, tier='full'))
        return xr.Dataset(pieces)

    @staticmethod
    def combine(chunks: list) -> xr.Dataset:
        # pieces carrying a time dim concatenate along it, pieces without one sum
        with_time = [c[[k for k in c.data_vars if 'time' in c[k].dims]] for c in chunks]
        without_time = [c[[k for k in c.data_vars if 'time' not in c[k].dims]] for c in chunks]
        combined = xr.concat(with_time, dim='time') if len(with_time[0].data_vars) > 0 else with_time[0]
        # union over keys — a piece present in only some chunks must not vanish silently
        keys = set().union(*(set(c.data_vars) for c in without_time))
        pooled = xr.Dataset({k : sum(c[k] for c in without_time if k in c) for k in keys})
        return xr.merge([combined, pooled])

    # STORAGE
    # the path is per-run runtime state and stays a call argument; only the root rank calls the write methods
    @staticmethod
    def encode(pieces: xr.Dataset) -> xr.Dataset:
        # accumulate float64, store nothing above float32
        out = {}
        for k, da in pieces.data_vars.items():
            if k.endswith(('rank', 'pit')):
                # counts per (time, step) frame are bounded by the sea points (~3.7k at 1°) — uint16 is safe
                out[k] = da.astype('uint16') if 'time' in da.dims else da.astype('int32')
            elif k.endswith('_n'):
                out[k] = da.astype('int32')
            elif da.dtype in (np.float64, np.int64):
                out[k] = da.astype('float32')
            else:
                out[k] = da
        return xr.Dataset(out, coords=pieces.coords, attrs=pieces.attrs)

    @staticmethod
    def has_group(path: str | Path, group: str) -> bool:
        return (Path(path) / group / 'zarr.json').exists() or (Path(path) / group / '.zgroup').exists()

    def check_masks(self, path: Path, pieces: xr.Dataset, group: str):
        # fail fast on mask drift — the canary n must match the static group before anything is appended
        # note: compared on the coordinate intersection, so a shortened time axis passes silently
        if not self.has_group(path, f'{group}/static'):
            return
        static = xr.open_zarr(path, group=f'{group}/static', consolidated=False)
        for k in pieces.data_vars:
            if k in static.data_vars and not bool((pieces[k].squeeze('dispatch') == static[k]).all()):
                raise ValueError(f'{k} differs from the static group — task masks drifted, refusing to write {group}')

    def write_static(self, path: str | Path, pieces: xr.Dataset, group: str):
        path = Path(path)
        if self.has_group(path, f'{group}/static'):
            return
        # write to a temporary group and rename — a crash mid-write must not look like a complete group
        tmp = path / group / 'static_tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        self.encode(pieces).to_zarr(path, group=f'{group}/static_tmp', mode='a', consolidated=False)
        shutil.rmtree(path / group / 'static', ignore_errors=True)
        tmp.rename(path / group / 'static')

    def write(self, path: str | Path, pieces: xr.Dataset, group: str, step: int, attrs: dict = None):
        path = Path(path)
        step = int(step)  # a stray numpy scalar fails JSON serialization inside zarr
        pieces = self.encode(pieces).expand_dims(dispatch=[step])
        self.check_masks(path, pieces, group)
        if self.has_group(path, f'{group}/dispatch'):
            # committed → skip, present-uncommitted → stop, absent → append: no silent path
            g = zarr.open_group(store=path, path=f'{group}/dispatch', mode='a')
            committed = list(g.attrs.get('committed', []))
            if step in committed:
                print(f'Evaluation: dispatch {step} already in {group}, skipping')
                return
            if step in g['dispatch'][:]:
                raise ValueError(f'dispatch {step} in {group} is present but uncommitted — repair before continuing')
            # tier and partial are properties of the group for the whole run — append would corrupt on mismatch
            existing = {k : g[k].shape[1:] for k in g.array_keys() if k != 'dispatch' and k not in pieces.coords}
            incoming = {k : pieces[k].shape[1:] for k in pieces.data_vars}
            if existing != incoming:
                raise ValueError(f'dispatch pieces for {group} differ from the existing group — tier or partial changed mid-run')
            # xarray's append replaces the group attrs with the incoming dataset's — carry the existing ones
            pieces = pieces.assign_attrs(dict(g.attrs))
            pieces.to_zarr(path, group=f'{group}/dispatch', append_dim='dispatch', consolidated=False)
            g.attrs['committed'] = committed + [step]
        else:
            encoding = {k : {'chunks' : (1, *pieces[k].shape[1:])} for k in pieces.data_vars}
            pieces.assign_attrs(default(attrs, {})).to_zarr(path, group=f'{group}/dispatch', mode='a', encoding=encoding, consolidated=False)
            g = zarr.open_group(store=path, path=f'{group}/dispatch', mode='a')
            g.attrs['committed'] = [step]
