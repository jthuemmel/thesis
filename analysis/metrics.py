import re
import math
import numpy as np
import xarray as xr

from pathlib import Path
from scipy.special import erf

from utils.config import exists, default

### STORE ACCESS

def open_pieces(path: str | Path, dataset: str, task: str, step: int = None) -> xr.Dataset:
    # merged (static + dispatch) view; step selects one dispatch, None keeps the dispatch dim
    group = f'{dataset}/{task}'
    static = xr.open_zarr(Path(path), group=f'{group}/static', consolidated=False)
    dispatched = xr.open_zarr(Path(path), group=f'{group}/dispatch', consolidated=False)
    # mask-drift guard: the dispatch group re-records n per variable
    for k in [k for k in dispatched.data_vars if k in static.data_vars]:
        if not bool((dispatched[k] == static[k]).all()):
            raise ValueError(f'{k} differs between static and dispatch groups — task masks drifted across dispatches')
        dispatched = dispatched.drop_vars(k)
    if exists(step):
        dispatched = dispatched.sel(dispatch=step)
    merged = xr.merge([static, dispatched], compat='no_conflicts', combine_attrs='no_conflicts')
    # pool across time in float64 at analysis time
    return merged.map(lambda da: da.astype('float64') if da.dtype in (np.float32, np.float16) else da, keep_attrs=True)

### SCORING RULES

def gaussian_crps(mu: xr.DataArray, sigma: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    sqrtPi, sqrtTwo = math.sqrt(math.pi), math.sqrt(2)
    sigma = sigma.clip(min=1e-6)
    z = (obs - mu) / sigma
    phi = np.exp(-z**2 / 2) / (sqrtTwo * sqrtPi)
    return sigma * (z * xr.apply_ufunc(erf, z / sqrtTwo) + 2 * phi - 1 / sqrtPi)

def gaussian_ign(mu: xr.DataArray, sigma: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    sigma = sigma.clip(min=1e-6)
    z = (obs - mu) / sigma
    return 0.5 * math.log(2 * math.pi) + np.log(sigma) + 0.5 * z**2

def expected_clim_crps(sigma_c, sigma_o):
    # closed form for gaussian clim forecast N(0, σ_c) verified against N(0, σ_o) draws
    return math.sqrt(2 / math.pi) * np.sqrt(sigma_c**2 + sigma_o**2) - sigma_c / math.sqrt(math.pi)

### RECOVERY
# metrics from merged (static + dispatch) pieces; caller reduces the surviving (time, step) dims
# obs_prefix points baselines (nn_, member ms*) at the shared obs terms, e.g. p_rmse(m, 'ssta_nn_', 'ssta_')

def moments(m: xr.Dataset, prefix: str = '', obs_prefix: str = None):
    q, o = prefix, default(obs_prefix, prefix)
    return m[f'{o}n'], m[f'{q}sf'], m[f'{o}so'], m[f'{q}sff'], m[f'{o}soo'], m[f'{q}sfo']

def centered(m: xr.Dataset, prefix: str = '', obs_prefix: str = None):
    n, sf, so, sff, soo, sfo = moments(m, prefix, obs_prefix)
    var_f = sff / n - (sf / n)**2
    var_t = soo / n - (so / n)**2
    cov = sfo / n - sf * so / n**2
    return var_f, var_t, cov

def p_bias(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    n, sf, so, *_ = moments(m, prefix, obs_prefix)
    return (sf - so) / n

def p_pcc(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    _, _, _, sff, soo, sfo = moments(m, prefix, obs_prefix)
    return sfo / (np.sqrt(sff) * np.sqrt(soo))

def p_rmse(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    n, _, _, sff, soo, sfo = moments(m, prefix, obs_prefix)
    return np.sqrt((sff - 2 * sfo + soo) / n)

def p_mae(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    return m[f'{prefix}sad'] / m[f'{default(obs_prefix, prefix)}n']

def p_acc(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    var_f, var_t, cov = centered(m, prefix, obs_prefix)
    return cov / (np.sqrt(var_f) * np.sqrt(var_t))

def p_fi(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    var_f, var_t, cov = centered(m, prefix, obs_prefix)
    return cov / var_t

def p_info(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    _, var_t, _ = centered(m, prefix, obs_prefix)
    return p_fi(m, prefix, obs_prefix) * np.sqrt(var_t)

def p_ie(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    _, var_t, _ = centered(m, prefix, obs_prefix)
    return np.abs(1 - p_fi(m, prefix, obs_prefix)) * np.sqrt(var_t)

def p_ne(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    var_f, var_t, cov = centered(m, prefix, obs_prefix)
    fi = cov / var_t
    return np.sqrt(var_f - 2 * fi * cov + fi**2 * var_t)

def p_stde(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    var_f, var_t, cov = centered(m, prefix, obs_prefix)
    acc = cov / (np.sqrt(var_f) * np.sqrt(var_t))
    return np.sqrt(var_t + var_f - 2 * np.sqrt(var_t) * np.sqrt(var_f) * acc)

def p_sdav(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    _, var_t, _ = centered(m, prefix, obs_prefix)
    return np.sqrt(var_t)

def p_activ(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    _, _, _, sff, soo, _ = moments(m, prefix, obs_prefix)
    return np.sqrt(sff / soo)

def p_spread(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    s2 = m[f'{prefix}svar'] if f'{prefix}svar' in m else m[f'{prefix}ssig2']
    return np.sqrt(s2 / m[f'{default(obs_prefix, prefix)}n'])

def p_crps(m: xr.Dataset, prefix: str = '', obs_prefix: str = None, E: int = None, fair: bool = True) -> xr.DataArray:
    n = m[f'{default(obs_prefix, prefix)}n']
    if f'{prefix}scrps' in m:
        return m[f'{prefix}scrps'] / n
    coef = -1 / (E * (E - 1)) if fair else -1 / E**2
    return (m[f'{prefix}sae'] / E + coef * m[f'{prefix}sap']) / n

def p_ign(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    return m[f'{prefix}sign'] / m[f'{default(obs_prefix, prefix)}n']

def p_ssr(m: xr.Dataset, prefix: str = '', obs_prefix: str = None, E: int = None) -> xr.DataArray:
    correction = math.sqrt((E + 1) / E) if exists(E) and f'{prefix}svar' in m else 1.
    return correction * p_spread(m, prefix, obs_prefix) / p_rmse(m, prefix, obs_prefix)

def p_crps_ss(m: xr.Dataset, prefix: str = '', obs_prefix: str = None, E: int = None, fair: bool = True) -> xr.DataArray:
    # skill score against the self-estimated climatology (σ_c = σ_o = sdav)
    sdav = p_sdav(m, prefix, obs_prefix)
    clim = expected_clim_crps(sdav, sdav)
    return 1 - p_crps(m, prefix, obs_prefix, E=E, fair=fair) / clim

def p_rmse_ss(m: xr.Dataset, prefix: str = '', obs_prefix: str = None) -> xr.DataArray:
    n, _, _, _, soo, _ = moments(m, prefix, obs_prefix)
    return 1 - p_rmse(m, prefix, obs_prefix) / np.sqrt(soo / n)

### HELPERS

def depth_stack(pieces: xr.Dataset, piece: str) -> xr.DataArray:
    pattern = re.compile(rf'temp_ocn_(\d+)a_{piece}$')
    matches = sorted((int(pattern.match(k).group(1)), k) for k in pieces.data_vars if pattern.match(k))
    return xr.concat([pieces[k] for _, k in matches], dim='depth').assign_coords(depth=[d for d, _ in matches])
