#!/usr/bin/env python
"""
Run period search (Lomb-Scargle + BLS) on the best-stage lightcurves
from master_variable_catalog.h5.

For each source/seg/ch, reads the best available lightcurve (using
best_stage table), runs LS and BLS, and stores results in the catalog.

Adds to HDF5:
  period_search/{target} — compound dataset with columns:
    master_id, seg_ch,
    ls_period_min, ls_power, ls_significance,
    bls_period_min, bls_depth, bls_significance

Usage:
    python run_period_search.py [--source 624] [--min-period 20]
"""
import numpy as np
import h5py
import os
import sys
import time
import json
from astropy.timeseries import LombScargle, BoxLeastSquares
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
MIN_PERIOD_MIN = 20.0  # minimum period in minutes
MAX_PERIOD_MIN = 720.0  # maximum period (half observation window)


def run_ls(times, flux, min_period_min=20.0, max_period_min=720.0, n_freq=5000):
    """Run Lomb-Scargle periodogram. Returns (best_period_min, power, significance)."""
    if len(times) < 50:
        return MAX_PERIOD_MIN, 0.0, 0.0

    # Convert period limits to frequency (cycles/hr)
    min_freq = 1.0 / (max_period_min / 60.0)  # cycles/hr
    max_freq = 1.0 / (min_period_min / 60.0)  # cycles/hr

    freq = np.linspace(min_freq, max_freq, n_freq)
    ls = LombScargle(times, flux)
    power = ls.power(freq)

    best_idx = np.argmax(power)
    best_freq = freq[best_idx]
    best_period_hr = 1.0 / best_freq
    best_period_min = best_period_hr * 60.0
    best_power = power[best_idx]

    # Significance: (peak - median) / MAD
    med = np.median(power)
    mad = np.median(np.abs(power - med))
    significance = (best_power - med) / mad if mad > 0 else 0.0

    return float(best_period_min), float(best_power), float(significance)


def run_bls(times, flux, min_period_min=20.0, max_period_min=720.0, n_periods=1000):
    """Run Box Least Squares. Returns (best_period_min, depth, significance)."""
    if len(times) < 50:
        return MAX_PERIOD_MIN, 0.0, 0.0

    min_period_hr = min_period_min / 60.0
    max_period_hr = max_period_min / 60.0

    # Duration range: 1 min to just under min_period
    # BLS requires max(duration) < min(period)
    min_dur_hr = 1.0 / 60.0
    max_dur_hr = 0.95 * min_period_hr  # just under min period
    if max_dur_hr <= min_dur_hr:
        return MAX_PERIOD_MIN, 0.0, 0.0
    durations = np.geomspace(min_dur_hr, max_dur_hr, 30)

    periods = np.linspace(min_period_hr, max_period_hr, n_periods)

    try:
        bls = BoxLeastSquares(times, flux)
        result = bls.power(periods, durations)

        best_idx = np.argmax(result.power)
        best_period_hr = result.period[best_idx]
        best_period_min = float(best_period_hr) * 60.0
        best_power = float(result.power[best_idx])
        best_depth = float(result.depth[best_idx])

        # Signal Detection Efficiency (SDE): standard BLS significance
        mean_pow = np.mean(result.power)
        std_pow = np.std(result.power)
        significance = (best_power - mean_pow) / std_pow if std_pow > 0 else 0.0

        return best_period_min, best_depth, float(significance)
    except Exception:
        return MAX_PERIOD_MIN, 0.0, 0.0


def get_best_lc(h5, target, mid, seg, ch, idx):
    """Read the best-stage lightcurve for a source/seg/ch."""
    # Check best_stage table
    stage = 'groupdiff'
    if f'best_stage/{target}' in h5:
        bs = h5[f'best_stage/{target}'][:]
        row = bs[bs['master_id'] == mid]
        if len(row) > 0:
            col = f'{seg}_{ch}'
            if col in bs.dtype.names:
                stage = row[0][col].decode()

    # Read from appropriate group
    if stage in ('special', 'sat_corrected', 'slope_corrected', 'sat_slope'):
        group = 'special_reduction' if stage == 'special' else stage
        path = f'{group}/{target}/{mid}/{seg}_{ch}'
        if path in h5:
            tc = h5[path]['times'][:].astype(np.float64)
            fc = h5[path]['flux_norm'][:].astype(np.float64)
            if len(tc) >= 20:
                so = np.argsort(tc)
                return tc[so], fc[so]

    # Fall back to groupdiff
    det_map = {'SW': ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'],
               'LW': ['nrcblong']}
    for det in det_map.get(ch, []):
        path = f'{target}/lightcurves/{seg}/ramp/{det}'
        if path not in h5:
            continue
        flux = h5[path][idx, :].astype(np.float64)
        times = h5[f'{target}/times/{seg}/ramp/{det}'][:].astype(np.float64)
        valid = flux != 0
        if valid.sum() < 20:
            continue
        fv = flux[valid]
        tv = times[valid]
        fn = fv / np.median(fv)
        tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
        so = np.argsort(tv_hr)
        # Check if this detector has real signal (not just noise)
        if np.median(fv) > 50:
            return tv_hr[so], fn[so]

    return None, None


def main():
    single_source = None
    min_period = MIN_PERIOD_MIN
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--source='):
            single_source = int(arg.split('=')[1])
        elif arg == '--source' and i < len(sys.argv) - 1:
            single_source = int(sys.argv[i + 1])
        elif arg.startswith('--min-period='):
            min_period = float(arg.split('=')[1])
        elif arg == '--min-period' and i < len(sys.argv) - 1:
            min_period = float(sys.argv[i + 1])

    t0 = time.time()
    h5 = h5py.File(CATALOG, 'r')

    # Collect all results
    results = []  # list of dicts

    for target in ['Liller1', 'Terzan5']:
        if target not in h5:
            continue
        srcs = h5[f'{target}/sources'][:]
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        for i, src in enumerate(srcs):
            mid = int(src['master_id'])
            if single_source is not None and mid != single_source:
                continue

            for seg in segments:
                for ch in ['SW', 'LW']:
                    t_lc, f_lc = get_best_lc(h5, target, mid, seg, ch, i)
                    if t_lc is None or len(t_lc) < 50:
                        continue

                    # Run LS
                    ls_per, ls_pow, ls_sig = run_ls(t_lc, f_lc,
                                                     min_period_min=min_period)
                    # Run BLS
                    bls_per, bls_depth, bls_sig = run_bls(t_lc, f_lc,
                                                           min_period_min=min_period)

                    results.append({
                        'target': target,
                        'master_id': mid,
                        'seg_ch': f'{seg}_{ch}',
                        'n_pts': len(t_lc),
                        'ls_period_min': ls_per,
                        'ls_power': ls_pow,
                        'ls_significance': ls_sig,
                        'bls_period_min': bls_per,
                        'bls_depth': bls_depth,
                        'bls_significance': bls_sig,
                    })

            if single_source is None and (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f'  {target}: {i+1}/{len(srcs)} sources '
                      f'({len(results)} LCs, {elapsed:.0f}s)', flush=True)

    h5.close()

    if single_source is not None:
        for r in results:
            print(f'  {r["seg_ch"]}: LS P={r["ls_period_min"]:.1f}min '
                  f'(sig={r["ls_significance"]:.0f}), '
                  f'BLS P={r["bls_period_min"]:.1f}min '
                  f'(sig={r["bls_significance"]:.0f}, depth={r["bls_depth"]:.4f})')
        return

    print(f'\n{len(results)} lightcurves searched in {(time.time()-t0)/60:.1f} min')

    # Write to HDF5
    print('Writing to catalog...')
    h5 = h5py.File(CATALOG, 'r+')

    # Remove old period_search if exists
    if 'period_search' in h5:
        del h5['period_search']

    for target in ['Liller1', 'Terzan5']:
        target_results = [r for r in results if r['target'] == target]
        if not target_results:
            continue

        dt = np.dtype([
            ('master_id', 'i4'),
            ('seg_ch', 'S20'),
            ('n_pts', 'i4'),
            ('ls_period_min', 'f4'),
            ('ls_power', 'f4'),
            ('ls_significance', 'f4'),
            ('bls_period_min', 'f4'),
            ('bls_depth', 'f4'),
            ('bls_significance', 'f4'),
        ])
        arr = np.zeros(len(target_results), dtype=dt)
        for j, r in enumerate(target_results):
            arr[j]['master_id'] = r['master_id']
            arr[j]['seg_ch'] = r['seg_ch']
            arr[j]['n_pts'] = r['n_pts']
            arr[j]['ls_period_min'] = r['ls_period_min']
            arr[j]['ls_power'] = r['ls_power']
            arr[j]['ls_significance'] = r['ls_significance']
            arr[j]['bls_period_min'] = r['bls_period_min']
            arr[j]['bls_depth'] = r['bls_depth']
            arr[j]['bls_significance'] = r['bls_significance']

        h5.create_dataset(f'period_search/{target}', data=arr)
        print(f'  {target}: {len(target_results)} entries')

    h5.attrs['period_search_min_period'] = min_period
    h5.attrs['period_search_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    h5.close()

    # Print summary stats
    ls_sigs = [r['ls_significance'] for r in results]
    bls_sigs = [r['bls_significance'] for r in results]
    ls_periodic = sum(1 for r in results if r['ls_period_min'] < MAX_PERIOD_MIN and r['ls_significance'] > 50)
    bls_periodic = sum(1 for r in results if r['bls_period_min'] < MAX_PERIOD_MIN and r['bls_significance'] > 50)
    print(f'\nLS: {ls_periodic} with significant period (sig>50)')
    print(f'BLS: {bls_periodic} with significant period (sig>50)')
    print(f'Done in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
