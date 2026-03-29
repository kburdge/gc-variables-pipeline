#!/usr/bin/env python
"""
Combined Segment3+Segment4 coherent period search for Liller 1.
Uses the 1.75-day baseline between segments to get much better
period resolution than single-segment searches.

For each source, concatenates the best LC from both segments
(in MJD) and runs LS + BLS on the combined time series.

Usage:
    python combined_period_search.py [--source 441]
"""
import numpy as np
import h5py
import os
import re
import sys
import time
import json
from astropy.io import fits
from astropy.timeseries import LombScargle, BoxLeastSquares
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING_PATH = f'{BASE}/catalogs/master_source_mapping.json'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

MIN_PERIOD_MIN = 20.0
MAX_PERIOD_MIN = 720.0

# Load actual MJD time arrays for each seg/det
_MJD_CACHE = {}

def get_mjd_times(target, seg, det):
    """Get MJD times from groupdiff FITS headers."""
    key = (target, seg, det)
    if key not in _MJD_CACHE:
        path = f'{BASE}/refs/groupdiffs_{target}_{seg}_{det}.fits'
        if os.path.exists(path):
            with fits.open(path) as hdul:
                _MJD_CACHE[key] = hdul['DIFF_TIMES'].data['MID_BARY_MJD'].astype(np.float64)
        else:
            _MJD_CACHE[key] = None
    return _MJD_CACHE[key]


def get_best_lc_mjd(h5, target, mid, seg, ch, idx, sw_det):
    """Read best-stage LC and return times in MJD (not hours)."""
    stage = 'groupdiff'
    if f'best_stage/{target}' in h5:
        bs = h5[f'best_stage/{target}'][:]
        row = bs[bs['master_id'] == mid]
        if len(row) > 0 and f'{seg}_{ch}' in bs.dtype.names:
            stage = row[0][f'{seg}_{ch}'].decode()

    det = sw_det if ch == 'SW' else 'nrcblong'

    if stage in ('special', 'sat_corrected', 'slope_corrected', 'sat_slope'):
        group = 'special_reduction' if stage == 'special' else stage
        path = f'{group}/{target}/{mid}/{seg}_{ch}'
        if path in h5:
            tc = h5[path]['times'][:].astype(np.float64)
            fc = h5[path]['flux_norm'][:].astype(np.float64)
            if len(tc) >= 20:
                # These are in hours from start — convert to MJD
                mjd_times = get_mjd_times(target, seg, det)
                if mjd_times is not None:
                    # Map hours back to MJD using the time grid
                    t0_mjd = mjd_times[0]
                    tc_mjd = t0_mjd + tc / 24.0
                    so = np.argsort(tc_mjd)
                    return tc_mjd[so], fc[so]

    # Groupdiff fallback
    if det is None: return None, None
    path_lc = f'{target}/lightcurves/{seg}/ramp/{det}'
    if path_lc not in h5: return None, None
    flux = h5[path_lc][idx, :].astype(np.float64)
    mjd_times = get_mjd_times(target, seg, det)
    if mjd_times is None: return None, None
    valid = flux != 0
    if valid.sum() < 20: return None, None
    fv = flux[valid]; tv = mjd_times[valid]
    fn = fv / np.nanmedian(fv)
    so = np.argsort(tv)
    return tv[so], fn[so]


def run_ls(times_mjd, flux, min_period_min=20.0, max_period_min=720.0):
    """Run LS with frequency grid appropriate for the baseline."""
    if len(times_mjd) < 50:
        return MAX_PERIOD_MIN, 0.0, 0.0

    baseline_hr = (times_mjd.max() - times_mjd.min()) * 24.0
    times_hr = (times_mjd - times_mjd.min()) * 24.0

    min_freq = 1.0 / (max_period_min / 60.0)
    max_freq = 1.0 / (min_period_min / 60.0)

    # Oversample by 5x relative to 1/T
    df = 1.0 / (baseline_hr * 5)
    n_freq = int((max_freq - min_freq) / df) + 1
    n_freq = max(n_freq, 5000)
    freq = np.linspace(min_freq, max_freq, n_freq)

    ls = LombScargle(times_hr, flux)
    power = ls.power(freq)

    best_idx = np.argmax(power)
    best_period_min = 60.0 / freq[best_idx]
    best_power = float(power[best_idx])

    med = np.median(power)
    mad = np.median(np.abs(power - med))
    significance = (best_power - med) / mad if mad > 0 else 0.0

    return float(best_period_min), float(best_power), float(significance)


def run_bls(times_mjd, flux, min_period_min=20.0, max_period_min=720.0):
    """Run BLS on combined time series."""
    if len(times_mjd) < 50:
        return MAX_PERIOD_MIN, 0.0, 0.0

    baseline_hr = (times_mjd.max() - times_mjd.min()) * 24.0
    times_hr = (times_mjd - times_mjd.min()) * 24.0

    min_p_hr = min_period_min / 60.0
    max_p_hr = max_period_min / 60.0
    min_dur_hr = 1.0 / 60.0
    max_dur_hr = 0.95 * min_p_hr
    if max_dur_hr <= min_dur_hr:
        return MAX_PERIOD_MIN, 0.0, 0.0

    n_periods = max(2000, int(baseline_hr / min_p_hr * 5))
    periods = np.linspace(min_p_hr, max_p_hr, n_periods)
    durations = np.geomspace(min_dur_hr, max_dur_hr, 30)

    try:
        bls = BoxLeastSquares(times_hr, flux)
        result = bls.power(periods, durations)
        best_idx = np.argmax(result.power)
        best_period_min = float(result.period[best_idx]) * 60.0
        best_depth = float(result.depth[best_idx])
        mean_pow = np.mean(result.power)
        std_pow = np.std(result.power)
        significance = (float(result.power[best_idx]) - mean_pow) / std_pow if std_pow > 0 else 0.0
        return best_period_min, best_depth, significance
    except Exception:
        return MAX_PERIOD_MIN, 0.0, 0.0


def main():
    single_source = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--source='):
            single_source = int(arg.split('=')[1])
        elif arg == '--source' and i < len(sys.argv) - 1:
            single_source = int(sys.argv[i + 1])

    t0 = time.time()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    h5 = h5py.File(CATALOG, 'r')
    target = 'Liller1'
    srcs = h5[f'{target}/sources'][:]

    results = []

    for i, src in enumerate(srcs):
        mid = int(src['master_id'])
        if single_source is not None and mid != single_source:
            continue

        sw_det = None
        if mid < len(mapping):
            for dk, dv in mapping[mid].get('detections', {}).items():
                if 'ramp' in dk and '_LW' not in dk:
                    dm = re.search(r'(nrcb[1-4])', dv.get('filename', ''))
                    if dm: sw_det = dm.group(1); break

        for ch in ['SW', 'LW']:
            det = sw_det if ch == 'SW' else 'nrcblong'
            if det is None and ch == 'SW': continue

            # Get LCs from both segments in MJD
            t3, f3 = get_best_lc_mjd(h5, target, mid, 'Segment3', ch, i, sw_det)
            t4, f4 = get_best_lc_mjd(h5, target, mid, 'Segment4', ch, i, sw_det)

            if t3 is None and t4 is None: continue

            # Concatenate
            if t3 is not None and t4 is not None:
                t_comb = np.concatenate([t3, t4])
                f_comb = np.concatenate([f3, f4])
            elif t3 is not None:
                t_comb, f_comb = t3, f3
            else:
                t_comb, f_comb = t4, f4

            so = np.argsort(t_comb)
            t_comb, f_comb = t_comb[so], f_comb[so]

            if len(t_comb) < 50: continue

            ls_per, ls_pow, ls_sig = run_ls(t_comb, f_comb)
            bls_per, bls_dep, bls_sig = run_bls(t_comb, f_comb)

            results.append({
                'master_id': mid, 'ch': ch,
                'n_pts': len(t_comb),
                'baseline_days': float(t_comb.max() - t_comb.min()),
                'ls_period_min': ls_per, 'ls_significance': ls_sig,
                'bls_period_min': bls_per, 'bls_depth': bls_dep,
                'bls_significance': bls_sig,
            })

        if single_source is not None:
            for r in results:
                print(f'  {r["ch"]}: {r["n_pts"]}pt baseline={r["baseline_days"]:.2f}d '
                      f'LS P={r["ls_period_min"]:.2f}min sig={r["ls_significance"]:.0f}, '
                      f'BLS P={r["bls_period_min"]:.1f}min depth={r["bls_depth"]:.4f}')

        if single_source is None and (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(srcs)} ({len(results)} LCs, {time.time()-t0:.0f}s)',
                  flush=True)

    h5.close()

    if single_source is not None:
        return

    # Write to HDF5
    print(f'\n{len(results)} combined LCs searched in {(time.time()-t0)/60:.1f} min')
    print('Writing to catalog...')

    h5 = h5py.File(CATALOG, 'r+')
    dt = np.dtype([
        ('master_id', 'i4'), ('ch', 'S2'), ('n_pts', 'i4'),
        ('baseline_days', 'f4'),
        ('ls_period_min', 'f4'), ('ls_significance', 'f4'),
        ('bls_period_min', 'f4'), ('bls_depth', 'f4'), ('bls_significance', 'f4'),
    ])
    arr = np.zeros(len(results), dtype=dt)
    for j, r in enumerate(results):
        arr[j]['master_id'] = r['master_id']
        arr[j]['ch'] = r['ch']
        arr[j]['n_pts'] = r['n_pts']
        arr[j]['baseline_days'] = r['baseline_days']
        arr[j]['ls_period_min'] = r['ls_period_min']
        arr[j]['ls_significance'] = r['ls_significance']
        arr[j]['bls_period_min'] = r['bls_period_min']
        arr[j]['bls_depth'] = r['bls_depth']
        arr[j]['bls_significance'] = r['bls_significance']

    path = 'period_search_combined/Liller1'
    if path in h5: del h5[path]
    h5.require_group('period_search_combined')
    h5.create_dataset(path, data=arr)
    h5.close()
    print(f'  Written {len(results)} entries to {path}')

    # Summary
    ls_sigs = [r['ls_significance'] for r in results]
    n_sig = sum(1 for r in results if r['ls_significance'] > 50 and r['ls_period_min'] < 700)
    print(f'  {n_sig} with significant combined period')
    print(f'Done in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
