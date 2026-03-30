#!/usr/bin/env python
"""
Generate phase-folded lightcurve plots for all sources at their best periods.
Outputs to diagnostics/phase_folds/{Liller1,Terzan5}/

For Liller 1: uses combined period search (S3+S4 baseline).
For Terzan 5: uses single-segment period search.

Usage:
    python generate_phase_folds.py
"""
import numpy as np
import h5py
import os
import re
import json
import time
from astropy.io import fits
from KBB_Utils.LC_Tools import PhaseFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING_PATH = f'{BASE}/catalogs/master_source_mapping.json'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

with open(MAPPING_PATH) as _f:
    _MAPPING = json.load(_f)

_MJD_CACHE = {}  # (target, seg, det) -> first MJD


def get_sw_det(mid):
    if mid < len(_MAPPING):
        for dk, dv in _MAPPING[mid].get('detections', {}).items():
            if 'ramp' in dk and '_LW' not in dk:
                dm = re.search(r'(nrcb[1-4])', dv.get('filename', ''))
                if dm: return dm.group(1)
    return None


def get_best_lc(h5, target, mid, seg, ch, idx, sw_det):
    """Read best-stage lightcurve."""
    stage = 'groupdiff'
    if f'best_stage/{target}' in h5:
        bs = h5[f'best_stage/{target}'][:]
        row = bs[bs['master_id'] == mid]
        if len(row) > 0 and f'{seg}_{ch}' in bs.dtype.names:
            stage = row[0][f'{seg}_{ch}'].decode()

    if stage in ('special', 'sat_corrected', 'slope_corrected', 'sat_slope'):
        group = 'special_reduction' if stage == 'special' else stage
        path = f'{group}/{target}/{mid}/{seg}_{ch}'
        if path in h5:
            tc = h5[path]['times'][:].astype(np.float64)
            fc = h5[path]['flux_norm'][:].astype(np.float64)
            if len(tc) >= 20:
                so = np.argsort(tc)
                return tc[so], fc[so]

    det = sw_det if ch == 'SW' else 'nrcblong'
    if det is None: return None, None
    path = f'{target}/lightcurves/{seg}/ramp/{det}'
    if path not in h5: return None, None
    flux = h5[path][idx, :].astype(np.float64)
    times = h5[f'{target}/times/{seg}/ramp/{det}'][:].astype(np.float64)
    valid = flux != 0
    if valid.sum() < 20: return None, None
    fv = flux[valid]; tv = times[valid]
    fn = fv / np.nanmedian(fv)
    tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
    so = np.argsort(tv_hr)
    return tv_hr[so], fn[so]


def phase_bin(t, f, period_hr, n_bins=100):
    phase = PhaseFold(t, period_hr, t0=t[0])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_flux = np.full(n_bins, np.nan)
    bin_err = np.full(n_bins, np.nan)
    for bi in range(n_bins):
        mask = (phase >= bin_edges[bi]) & (phase < bin_edges[bi + 1])
        if mask.sum() >= 3:
            bin_flux[bi] = np.mean(f[mask])
            bin_err[bi] = np.std(f[mask])
    return bin_centers, bin_flux, bin_err


def main():
    t0 = time.time()
    h5 = h5py.File(CATALOG, 'r')

    for target in ['Liller1', 'Terzan5']:
        if target not in h5: continue
        cluster = 'Liller 1' if target == 'Liller1' else 'Terzan 5'
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        out_dir = f'{BASE}/diagnostics/phase_folds/{target}'
        os.makedirs(out_dir, exist_ok=True)

        # Get best periods
        # Use combined for Liller1 if available
        if target == 'Liller1' and 'period_search_combined/Liller1' in h5:
            ps = h5['period_search_combined/Liller1'][:]
            ps_key = 'ch'
        else:
            ps = h5[f'period_search/{target}'][:]
            ps_key = 'seg_ch'

        source_periods = {}  # mid -> (period_min, sig, ch_or_seg_ch)
        for row in ps:
            mid = int(row['master_id'])
            sig = float(row['ls_significance'])
            per = float(row['ls_period_min'])
            if per >= 700: continue
            if mid not in source_periods or sig > source_periods[mid][1]:
                source_periods[mid] = (per, sig, row[ps_key].decode())

        print(f'{target}: {len(source_periods)} sources with significant period')

        srcs = h5[f'{target}/sources'][:]
        n_plotted = 0

        for i, src in enumerate(srcs):
            mid = int(src['master_id'])
            if mid not in source_periods: continue
            period_min, sig, best_key = source_periods[mid]
            period_hr = period_min / 60.0
            snr = float(src['best_snr'])
            sw_det = get_sw_det(mid)

            # For Liller1: coherently combine S3+S4 into one SW and one LW panel
            # using MJD times so the phase fold is coherent across the gap.
            # For Terzan5: just SW and LW from Segment2.
            channels = ['SW', 'LW']
            fig, axes = plt.subplots(1, 2, figsize=(8, 3), squeeze=False)

            has_data = False
            for ci, ch in enumerate(channels):
                ax = axes[0, ci]
                det = sw_det if ch == 'SW' else 'nrcblong'

                # Collect LCs from all segments, in MJD for coherent folding
                all_t = []
                all_f = []
                total_pts = 0
                for seg in segments:
                    t_lc, f_lc = get_best_lc(h5, target, mid, seg, ch, i, sw_det)
                    if t_lc is None or len(t_lc) < 20:
                        continue
                    # Convert hours-from-start to MJD
                    if det is not None:
                        mjd_key = (target, seg, det)
                        if mjd_key not in _MJD_CACHE:
                            mjd_path = f'{BASE}/refs/groupdiffs_{target}_{seg}_{det}.fits'
                            if os.path.exists(mjd_path):
                                with fits.open(mjd_path) as hdul:
                                    _MJD_CACHE[mjd_key] = hdul['DIFF_TIMES'].data['MID_BARY_MJD'][0]
                            else:
                                _MJD_CACHE[mjd_key] = 0.0
                        t0_mjd = _MJD_CACHE[mjd_key]
                        t_mjd = t0_mjd + t_lc / 24.0
                    else:
                        t_mjd = t_lc  # fallback
                    all_t.append(t_mjd)
                    all_f.append(f_lc)
                    total_pts += len(f_lc)

                if not all_t:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            fontsize=8, color='gray', transform=ax.transAxes)
                    ax.set_title(f'{ch}', fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.set_xlabel('Phase', fontsize=8)
                    continue

                has_data = True
                t_comb = np.concatenate(all_t)
                f_comb = np.concatenate(all_f)
                so = np.argsort(t_comb)
                t_comb = t_comb[so]
                f_comb = f_comb[so]

                # Phase fold at 2x period to reveal primary/secondary structure
                period_days = (period_hr * 2.0) / 24.0
                bc, bf, be = phase_bin(t_comb, f_comb, period_days)
                valid = np.isfinite(bf)
                clr = '#d62728' if ch == 'SW' else '#1f77b4'
                for offset in [0.0, 1.0]:
                    ax.errorbar(bc[valid] + offset, bf[valid], yerr=be[valid],
                                fmt='o', ms=1.5, color=clr,
                                ecolor=clr, elinewidth=0.4, capsize=0, alpha=0.8)
                ax.set_xlim(-0.05, 2.05)
                n_segs = len(all_t)
                seg_label = f'{n_segs} seg' if n_segs > 1 else '1 seg'
                ax.set_title(f'{ch} ({total_pts}pt, {seg_label})', fontsize=9)
                ax.tick_params(labelsize=7)
                if ci == 0: ax.set_ylabel('Norm flux', fontsize=8)
                ax.set_xlabel('Phase', fontsize=8)

            if not has_data:
                plt.close(fig); continue

            fig.suptitle(f'obj{mid:04d} ({cluster}) SNR={snr:.1f} — '
                         f'folded at 2P={2*period_min:.1f}min (LS P={period_min:.1f}min, sig={sig:.0f})',
                         fontsize=10, y=1.02)
            fig.tight_layout()
            out = os.path.join(out_dir,
                               f'obj{mid:04d}_P{period_min:.0f}min_sig{sig:.0f}.png')
            fig.savefig(out, dpi=100, bbox_inches='tight')
            plt.close(fig)
            n_plotted += 1

            if n_plotted % 100 == 0:
                print(f'  {n_plotted} plotted ({time.time()-t0:.0f}s)', flush=True)

        print(f'  {n_plotted} phase fold plots in {out_dir}/')

    h5.close()
    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
