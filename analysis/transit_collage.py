#!/usr/bin/env python
"""
Generate a collage of the best final lightcurves for all vetted transit sources.
Cross-matches vetted transit filenames to master catalog by RA/Dec,
then plots the best-stage lightcurve from the master catalog.

Usage:
    python transit_collage.py
"""
import numpy as np
import os
import re
import json
import h5py
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
VETTED_DIR = f'{BASE}/diagnostics/vetted_transits_ramp_pngs'
OUT_DIR = f'{BASE}/diagnostics/transit_collage'
os.makedirs(OUT_DIR, exist_ok=True)

with open(f'{BASE}/catalogs/master_source_mapping.json') as f:
    MAPPING = json.load(f)


def get_sw_det(mid):
    if mid < len(MAPPING):
        for dk, dv in MAPPING[mid].get('detections', {}).items():
            if 'ramp' in dk and '_LW' not in dk:
                dm = re.search(r'(nrcb[1-4])', dv.get('filename', ''))
                if dm: return dm.group(1)
    return None


def get_best_lc(h5, target, mid, seg, ch, idx, sw_det):
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


def main():
    # Parse vetted transit filenames
    transit_sources = []
    for fn in sorted(os.listdir(VETTED_DIR)):
        if not fn.endswith('.png'): continue
        m = re.search(r'(nrcb\w+)_P(\d+)min_LS(\d+)_amp([\d.]+)_([\d.]+)_([-\d.]+)\.png$', fn)
        if m:
            transit_sources.append({
                'filename': fn,
                'det': m.group(1),
                'period': int(m.group(2)),
                'ls_sig': int(m.group(3)),
                'amp': float(m.group(4)),
                'ra': float(m.group(5)),
                'dec': float(m.group(6)),
            })

    print(f'{len(transit_sources)} vetted transit sources')

    # Cross-match to master catalog
    h5 = h5py.File(CATALOG, 'r')
    master_pos = {}
    for target in ['Liller1', 'Terzan5']:
        if target not in h5: continue
        srcs = h5[f'{target}/sources'][:]
        for i, s in enumerate(srcs):
            mid = int(s['master_id'])
            master_pos[mid] = {
                'target': target, 'ra': float(s['ra']), 'dec': float(s['dec']),
                'idx': i, 'snr': float(s['best_snr']),
            }

    matched = {}  # mid -> transit info (deduplicated)
    for ts in transit_sources:
        sky = SkyCoord(ra=ts['ra']*u.deg, dec=ts['dec']*u.deg)
        best_mid = None; best_sep = 999.
        for mid, info in master_pos.items():
            sep = sky.separation(SkyCoord(ra=info['ra']*u.deg, dec=info['dec']*u.deg)).arcsec
            if sep < best_sep:
                best_sep = sep; best_mid = mid
        if best_sep < 1.0 and best_mid not in matched:
            matched[best_mid] = ts

    unique_mids = sorted(matched.keys())
    print(f'{len(unique_mids)} unique sources matched to master catalog')

    # Generate collage — 4 columns (S3 SW, S3 LW, S4 SW, S4 LW or S2 SW, S2 LW)
    # Group by page of ~20 sources
    page_size = 20
    n_pages = (len(unique_mids) + page_size - 1) // page_size

    for page in range(n_pages):
        start = page * page_size
        end = min(start + page_size, len(unique_mids))
        page_mids = unique_mids[start:end]
        n_rows = len(page_mids)

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.2 * n_rows), squeeze=False)

        for row, mid in enumerate(page_mids):
            info = master_pos[mid]
            target = info['target']
            idx = info['idx']
            snr = info['snr']
            sw_det = get_sw_det(mid)
            segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']
            cluster = 'Liller 1' if target == 'Liller1' else 'Terzan 5'
            ts = matched[mid]

            for ci, ch in enumerate(['SW', 'LW']):
                ax = axes[row, ci]
                # Concatenate all segments
                all_t = []; all_f = []
                for seg in segments:
                    t_lc, f_lc = get_best_lc(h5, target, mid, seg, ch, idx, sw_det)
                    if t_lc is not None:
                        all_t.append(t_lc); all_f.append(f_lc)

                if all_t:
                    # For single segment, just use it; for multi, offset times
                    if len(all_t) == 1:
                        tc = all_t[0]; fc = all_f[0]
                    else:
                        # Offset S4 times to follow S3
                        tc = np.concatenate(all_t)
                        fc = np.concatenate(all_f)
                    so = np.argsort(tc); tc = tc[so]; fc = fc[so]

                    clr = '#d62728' if ch == 'SW' else '#1f77b4'
                    ax.scatter(tc, fc, s=0.5, c='black', alpha=1.0, rasterized=True)
                    bsz = max(1, len(fc) // 100); nb = len(fc) // bsz
                    if nb > 1:
                        nu = nb * bsz
                        ax.plot(tc[:nu].reshape(nb, bsz).mean(1),
                                fc[:nu].reshape(nb, bsz).mean(1), color=clr, lw=1.5)
                    ax.set_title(f'obj{mid:04d} ({cluster}) {ch} — {len(fc)}pt',
                                 fontsize=7, loc='left')
                else:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            fontsize=7, color='gray', transform=ax.transAxes)
                    ax.set_title(f'obj{mid:04d} {ch}', fontsize=7, loc='left')

                ax.tick_params(labelsize=6)
                if ci == 0: ax.set_ylabel('Flux', fontsize=7)
                if row == n_rows - 1: ax.set_xlabel('Time (hr)', fontsize=7)

        fig.suptitle(f'Vetted Transit Candidates — Final Lightcurves (page {page+1}/{n_pages})',
                     fontsize=12, y=1.005)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'transit_collage_page{page+1:02d}.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

    h5.close()
    print(f'\nDone: {len(unique_mids)} sources across {n_pages} pages in {OUT_DIR}/')


if __name__ == '__main__':
    main()
