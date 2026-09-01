import math
import shutil
import zarr
import numpy as np
import xarray as xr

from pathlib import Path
from scipy.special import erf

from analysis.pieces import *
from analysis.metrics import *

# closure tests: metrics recovered from pieces must match direct computation on the same data
# reference formulas mirror the xr_* methods in experiment.py

SEED = 0
ATOL = 1e-10

EV = Evaluation(EvaluationConfig())

### SYNTHETIC DATA

def make_dataset(ens: bool = False, T: int = 24, S: int = 6, H: int = 8, W: int = 12, E: int = 4) -> xr.Dataset:
    rng = np.random.default_rng(SEED)
    time = xr.date_range('2000-01', periods=T, freq='MS')
    coords = {'time' : time, 'step' : np.arange(S), 'lat' : np.linspace(-6, 6, H), 'lon' : np.linspace(150, 250, W)}
    dims = ('time', 'step', 'lat', 'lon')
    shape = (T, S, H, W)

    obs = rng.normal(size=shape)
    obs[rng.random(shape) < 0.01] = np.nan # NaN obs at sea must drop out of valid everywhere
    visible = np.zeros(shape, dtype=bool)
    visible[:, :2] = True
    lsm = rng.random((H, W)) < 0.2

    data = {
        'temp_ocn_0a_obs' :     (dims, obs),
        'temp_ocn_0a_visible' : (dims, visible),
        'lsm' :                 (('lat', 'lon'), lsm),
    }
    if ens:
        pred = obs[..., None] + rng.normal(size=(*shape, E))
        data['temp_ocn_0a_pred'] = ((*dims, 'ens'), pred)
        coords['ens'] = np.arange(E)
    else:
        data['temp_ocn_0a_pred_mu'] = (dims, obs + rng.normal(size=shape))
        data['temp_ocn_0a_pred_sigma'] = (dims, np.abs(rng.normal(size=shape)) + 0.1)
    return xr.Dataset(data, coords=coords)

def masked_arrays(ds: xr.Dataset):
    sea = ~ds['lsm'].astype(bool)
    valid = EV.valid_mask(ds, 'temp_ocn_0a', sea)
    f, pred, mu, sigma = EV.heads(ds, 'temp_ocn_0a')
    obs = ds['temp_ocn_0a_obs'].where(valid)
    f = f.where(valid)
    return obs, f, pred, mu, sigma, valid, sea

### REFERENCE METRICS (as in experiment.py)

def ref_pcc(pred, obs, dim):
    num = (pred * obs).sum(dim, skipna=True)
    denom = np.sqrt((pred**2).sum(dim, skipna=True)) * np.sqrt((obs**2).sum(dim, skipna=True))
    return num / denom

def ref_rmse(pred, obs, dim):
    return np.sqrt(((pred - obs)**2).mean(dim, skipna=True))

def _debias(x, dim):
    return x - x.mean(dim, skipna=True)

def _winner(x, y, dim):
    return (x * y).mean(dim, skipna=True)

def ref_acc(pred, obs, dim):
    af, at = _debias(pred, dim), _debias(obs, dim)
    return _winner(af, at, dim) / (np.sqrt(_winner(af, af, dim)) * np.sqrt(_winner(at, at, dim)))

def ref_fi(pred, obs, dim):
    af, at = _debias(pred, dim), _debias(obs, dim)
    return _winner(af, at, dim) / _winner(at, at, dim)

def ref_info(pred, obs, dim):
    at = _debias(obs, dim)
    return ref_fi(pred, obs, dim) * np.sqrt(_winner(at, at, dim))

def ref_ie(pred, obs, dim):
    at = _debias(obs, dim)
    return np.abs(1 - ref_fi(pred, obs, dim)) * np.sqrt(_winner(at, at, dim))

def ref_ne(pred, obs, dim):
    af, at = _debias(pred, dim), _debias(obs, dim)
    var_f, var_t = _winner(af, af, dim), _winner(at, at, dim)
    cov = _winner(af, at, dim)
    fi = cov / var_t
    return np.sqrt(var_f - 2 * fi * cov + fi**2 * var_t)

def ref_stde(pred, obs, dim):
    af, at = _debias(pred, dim), _debias(obs, dim)
    var_f, var_t = _winner(af, af, dim), _winner(at, at, dim)
    acc = _winner(af, at, dim) / (np.sqrt(var_f) * np.sqrt(var_t))
    return np.sqrt(var_t + var_f - 2 * np.sqrt(var_t) * np.sqrt(var_f) * acc)

def ref_kernel_crps(pred, obs, fair):
    E = pred.sizes['ens']
    coef = -1 / (E * (E - 1)) if fair else -1 / E**2
    mae = np.abs(pred - obs).mean('ens', skipna=True)
    def _pairwise(p):
        total = np.zeros(p.shape[:-1], dtype=np.float64)
        for i in range(E):
            total = total + np.sum(np.abs(p[..., i:i + 1] - p[..., i + 1:]), axis=-1)
        return total
    ens_var = xr.apply_ufunc(_pairwise, pred, input_core_dims=[['ens']])
    return mae + coef * ens_var

### TESTS

def check(name, got, want, atol=ATOL):
    got, want = np.asarray(got), np.asarray(want)
    mask = ~np.isnan(want)
    assert np.allclose(got[mask], want[mask], atol=atol, equal_nan=True), \
        f'{name}: max abs diff {np.nanmax(np.abs(got - want))}'
    print(f'PASS {name}')

def test_pair_recovery(ens):
    ds = make_dataset(ens)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    pieces = xr.merge([EV.compute(ds, tier='light'), EV.static(ds)])
    q = 'temp_ocn_0a_'
    check(f'pcc[{ens=}]', p_pcc(pieces, q), ref_pcc(f, obs, SPACE))
    check(f'rmse[{ens=}]', p_rmse(pieces, q), ref_rmse(f, obs, SPACE))
    check(f'mae[{ens=}]', p_mae(pieces, q), np.abs(f - obs).mean(SPACE, skipna=True))
    check(f'acc[{ens=}]', p_acc(pieces, q), ref_acc(f, obs, SPACE))
    check(f'fi[{ens=}]', p_fi(pieces, q), ref_fi(f, obs, SPACE))
    check(f'info[{ens=}]', p_info(pieces, q), ref_info(f, obs, SPACE))
    check(f'ie[{ens=}]', p_ie(pieces, q), ref_ie(f, obs, SPACE))
    check(f'ne[{ens=}]', p_ne(pieces, q), ref_ne(f, obs, SPACE))
    check(f'stde[{ens=}]', p_stde(pieces, q), ref_stde(f, obs, SPACE))
    check(f'sdav[{ens=}]', p_sdav(pieces, q), np.sqrt(_winner(_debias(obs, SPACE), _debias(obs, SPACE), SPACE)))
    check(f'activ[{ens=}]', p_activ(pieces, q), np.sqrt((f**2).sum(SPACE, skipna=True) / (obs**2).sum(SPACE, skipna=True)))
    check(f'bias[{ens=}]', p_bias(pieces, q), (f - obs).mean(SPACE, skipna=True))
    check(f'rmse_ss[{ens=}]', p_rmse_ss(pieces, q), 1 - ref_rmse(f, obs, SPACE) / np.sqrt((obs**2).mean(SPACE, skipna=True)))

def test_prob_recovery_ens():
    ds = make_dataset(ens=True)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    E = pred.sizes['ens']
    pieces = xr.merge([EV.compute(ds, tier='light'), EV.static(ds)])
    q = 'temp_ocn_0a_'
    check('crps_fair', p_crps(pieces, q, E=E, fair=True),
          ref_kernel_crps(pred.where(valid), obs, fair=True).mean(SPACE, skipna=True))
    check('crps_unfair', p_crps(pieces, q, E=E, fair=False),
          ref_kernel_crps(pred.where(valid), obs, fair=False).mean(SPACE, skipna=True))
    check('spread', p_spread(pieces, q),
          np.sqrt(pred.where(valid).var('ens', ddof=1).mean(SPACE, skipna=True)))
    ssr = math.sqrt((E + 1) / E) * np.sqrt(pred.where(valid).var('ens', ddof=1).mean(SPACE, skipna=True)) / ref_rmse(f, obs, SPACE)
    check('ssr', p_ssr(pieces, q, E=E), ssr)

def test_prob_recovery_mve():
    ds = make_dataset(ens=False)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    mu, sigma = mu.where(valid), sigma.where(valid)
    pieces = xr.merge([EV.compute(ds, tier='light'), EV.static(ds)])
    q = 'temp_ocn_0a_'
    check('crps_mve', p_crps(pieces, q), gaussian_crps(mu, sigma, obs).mean(SPACE, skipna=True))
    check('ign_mve', p_ign(pieces, q), gaussian_ign(mu, sigma, obs).mean(SPACE, skipna=True))
    check('spread_mve', p_spread(pieces, q), np.sqrt((sigma.clip(min=1e-6)**2).mean(SPACE, skipna=True)))
    check('ssr_mve', p_ssr(pieces, q), np.sqrt((sigma.clip(min=1e-6)**2).mean(SPACE, skipna=True)) / ref_rmse(mu, obs, SPACE))

def test_histograms():
    ds = make_dataset(ens=True)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    pieces = EV.compute(ds, tier='full')
    rank = pieces['temp_ocn_0a_rank']
    check('rank_total', rank.sum('bin'), valid.sum(SPACE))
    ds = make_dataset(ens=False)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    pieces = EV.compute(ds, tier='full')
    pit = pieces['temp_ocn_0a_pit']
    check('pit_total', pit.sum('bin'), valid.sum(SPACE))
    rel = pieces['temp_ocn_0a_rel_n']
    check('rel_total', rel.sum('sbin'), valid.sum(SPACE + ('time',)))
    # RMS spread recovered from the reliability bins matches the pooled spread
    sig2 = pieces['temp_ocn_0a_rel_sig2'].sum(('sbin', 'step'))
    check('rel_sig2', np.sqrt(sig2 / rel.sum(('sbin', 'step'))),
          np.sqrt((sigma.where(valid).clip(min=1e-6)**2).mean(SPACE + ('time', 'step'), skipna=True)))

def test_monthly_maps():
    ds = make_dataset(ens=False)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    m = xr.merge([EV.compute(ds, tier='full'), EV.static(ds)])
    q = 'temp_ocn_0a_mon_'
    mon = xr.Dataset({k : m[k].sum('lband') for k in m.data_vars if k.startswith(q)})
    month = 3
    sel = ds['time.month'] == month
    fm, om = f.sel(time=sel), obs.sel(time=sel)
    dim = ('time', 'step')
    check('monthly_map_rmse', p_rmse(mon, q).sel(month=month), ref_rmse(fm, om, dim), atol=1e-7)
    check('monthly_map_acc', p_acc(mon, q).sel(month=month), ref_acc(fm, om, dim), atol=1e-7)

def test_combine():
    ds = make_dataset(ens=True)
    whole = EV.compute(ds, tier='full')
    halves = [EV.compute(ds.isel(time=s), tier='full') for s in (slice(0, 12), slice(12, None))]
    combined = EV.combine(halves)
    for k in whole.data_vars:
        check(f'combine[{k}]', combined[k].transpose(*whole[k].dims), whole[k], atol=1e-8)

def test_encoding():
    ds = make_dataset(ens=True)
    pieces = EV.encode(xr.merge([EV.compute(ds, tier='full'), EV.static(ds)]))
    assert pieces['temp_ocn_0a_n'].dtype == np.int32, 'counts should be int32'
    assert pieces['temp_ocn_0a_sff'].dtype == np.float32, 'moments should be float32'
    assert pieces['temp_ocn_0a_rank'].dtype == np.uint16, 'time-resolved counts should be uint16'
    assert pieces['nino4_member'].dtype == np.float32, 'member series stay f32 — fp16 quantization ties ranks'
    ds = make_dataset(ens=False)
    pieces = EV.encode(EV.compute(ds, tier='full'))
    assert pieces['temp_ocn_0a_sign'].dtype == np.float32, 'sign is a float sum, not a count'
    print('PASS encoding')

def test_baselines():
    # obs_prefix points the recovery at the shared obs terms of the nearest-neighbour baseline
    ds = make_dataset(ens=False)
    obs, f, pred, mu, sigma, valid, sea = masked_arrays(ds)
    m = xr.merge([EV.compute(ds, tier='full'), EV.static(ds)])
    base = EV.nearest_fill(ds['temp_ocn_0a_obs'], ds['temp_ocn_0a_visible'], sea).where(valid)
    check('nn_rmse', p_rmse(m, 'temp_ocn_0a_nn_', 'temp_ocn_0a_'), ref_rmse(base, obs, SPACE), atol=1e-7)
    check('nn_pcc', p_pcc(m, 'temp_ocn_0a_nn_', 'temp_ocn_0a_'), ref_pcc(base, obs, SPACE), atol=1e-7)

def test_writer():
    path = Path('analysis/_test_eval.zarr')
    shutil.rmtree(path, ignore_errors=True)
    ds = make_dataset(ens=True)
    EV.write_static(path, EV.static(ds), 'picontrol/frcst')
    EV.write_static(path, EV.static(ds), 'picontrol/frcst')  # no-op
    EV.write(path, EV.compute(ds, tier='light'), 'picontrol/frcst', 100, attrs={'job_name' : 'test'})
    EV.write(path, EV.compute(ds, tier='light'), 'picontrol/frcst', 100)  # duplicate step, must skip
    EV.write(path, EV.compute(ds, tier='light'), 'picontrol/frcst', 200)
    out = open_pieces(path, 'picontrol', 'frcst')
    assert list(out['dispatch'].values) == [100, 200], 'duplicate dispatch must be skipped'
    assert 'temp_ocn_0a_soo' in out and 'temp_ocn_0a_sff' in out, 'static and dispatch groups should merge'
    assert out.attrs.get('job_name') == 'test', 'group attrs must survive the append'
    check('writer_roundtrip', p_pcc(out.sel(dispatch=100), 'temp_ocn_0a_'),
          p_pcc(xr.merge([EV.compute(ds, tier='light'), EV.static(ds)]), 'temp_ocn_0a_'),
          atol=1e-5)
    # mask-drift canary: a dispatch with different masks must fail loudly at write time
    drifted = ds.copy()
    vis = drifted['temp_ocn_0a_visible'].copy()
    vis.loc[dict(step=2)] = True
    drifted['temp_ocn_0a_visible'] = vis
    EV.write_static(path, EV.static(ds), 'picontrol/drift')
    try:
        EV.write(path, EV.compute(drifted, tier='light'), 'picontrol/drift', 100)
        raise AssertionError('mask drift must raise at write time')
    except ValueError as err:
        assert 'drifted' in str(err)
    print('PASS mask_drift_guard')
    # a changed tier or partial mid-run must be rejected before the append corrupts the group
    try:
        EV.write(path, EV.compute(ds, tier='full'), 'picontrol/frcst', 300)
        raise AssertionError('tier change must raise at write time')
    except ValueError as err:
        assert 'tier or partial' in str(err)
    print('PASS tier_change_guard')
    # a crash between append and the committed update leaves the step present but uncommitted
    g = zarr.open_group(store=path, path='picontrol/frcst/dispatch', mode='a')
    g.attrs['committed'] = [100]
    try:
        EV.write(path, EV.compute(ds, tier='light'), 'picontrol/frcst', np.int64(200))
        raise AssertionError('present-uncommitted dispatch must raise')
    except ValueError as err:
        assert 'uncommitted' in str(err)
    print('PASS uncommitted_guard')
    shutil.rmtree(path, ignore_errors=True)

if __name__ == '__main__':
    test_pair_recovery(ens=True)
    test_pair_recovery(ens=False)
    test_prob_recovery_ens()
    test_prob_recovery_mve()
    test_histograms()
    test_monthly_maps()
    test_combine()
    test_encoding()
    test_baselines()
    test_writer()
    print('ALL PASS')
