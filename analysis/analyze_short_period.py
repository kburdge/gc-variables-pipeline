#!/usr/bin/env python
"""
Analyze user-identified short-period candidates from the 'short' folder.
Cross-matches with master catalog, looks up period search results,
and generates comparison plots (best corrected LC vs original ramp LC).

Usage:
    python analyze_short_period.py
"""
import numpy as np
import h5py
import os
import re
import json
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
SHORT_DIR = f'{BASE}/short'
OUT_DIR = f'{SHORT_DIR}/analysis'
os.makedirs(OUT_DIR, exist_ok=True)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

H = 2
ap = CircularAperture([(H, H)], r=1.5)
ap_mask = ap.to_mask(method='exact')[0].to_image((2*H+1, 2*H+1)).astype(np.float32)


def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    for b in range(nc):
        s, e = b*cs, (b+1)*cs; sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75]); r = q3 - q1
        m[s:e] = (sg >= q1 - iq*r) & (sg <= q3 + iq*r)
    return f[m], t[m]


def parse_master_thumbnails():
    """Parse master IDs from top-level short/ folder."""
    sources = []
    for fn in sorted(os.listdir(SHORT_DIR)):
        m = re.match(r'obj(\d{4})_snr([\d.]+)_([\d.]+)_([-\d.]+)\.png$', fn)
        if m:
            sources.append({
                'master_id': int(m.group(1)),
                'snr': float(m.group(2)),
                'ra': float(m.group(3)),
                'dec': float(m.group(4)),
            })
    return sources


def parse_ramp_folder():
    """Parse ramp diagnostic filenames for cross-matching."""
    ramp_dir = os.path.join(SHORT_DIR, 'ramp')
    if not os.path.isdir(ramp_dir):
        return []
    entries = []
    for fn in sorted(os.listdir(ramp_dir)):
        m = re.match(r'SNR([\d.]+)_src(\d+)_(nrcb\w+)_P(\d+)min_LS(\d+)_amp([\d.]+)_([\d.]+)_([-\d.]+)\.png$', fn)
        if m:
            entries.append({
                'snr': float(m.group(1)),
                'src_id': int(m.group(2)),
                'det': m.group(3),
                'period_min': int(m.group(4)),
                'ls_sig': int(m.group(5)),
                'amplitude': float(m.group(6)),
                'ra': float(m.group(7)),
                'dec': float(m.group(8)),
                'filename': fn,
            })
    return entries


def get_best_lc(h5, target, mid, seg, ch, idx, sw_det):
    """Read the best-stage lightcurve. sw_det from mapping."""
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
                return tc[so], fc[so], stage

    det = sw_det if ch == 'SW' else 'nrcblong'
    path = f'{target}/lightcurves/{seg}/ramp/{det}'
    if path not in h5: return None, None, None
    flux = h5[path][idx, :].astype(np.float64)
    times = h5[f'{target}/times/{seg}/ramp/{det}'][:].astype(np.float64)
    valid = flux != 0
    if valid.sum() < 20: return None, None, None
    fv = flux[valid]; tv = times[valid]
    fn = fv / np.nanmedian(fv)
    tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
    so = np.argsort(tv_hr)
    return tv_hr[so], fn[so], 'groupdiff'


def get_raw_ramp_lc(target, seg, det, ix, iy):
    """Extract raw ramp LC from groupdiff cube (no corrections)."""
    cube_path = f'{BASE}/refs/groupdiffs_{target}_{seg}_{det}.fits'
    h5_path = f'{BASE}/extraction/{target}/{seg}/{det}_ramp.h5'
    if not os.path.exists(cube_path) or not os.path.exists(h5_path):
        return None, None
    cube = fits.getdata(cube_path, memmap=True)
    with h5py.File(h5_path, 'r') as f:
        times = f['times'][:]
    nf, ny, nx = cube.shape
    if ix-H < 0 or ix+H+1 > nx or iy-H < 0 or iy+H+1 > ny:
        return None, None
    cutout = np.array(cube[:, iy-H:iy+H+1, ix-H:ix+H+1], dtype=np.float32)
    np.nan_to_num(cutout, nan=0., copy=False)
    flux = np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))
    valid = np.isfinite(flux) & (flux != 0)
    if valid.sum() < 50: return None, None
    fn = flux[valid] / np.median(flux[valid])
    so = np.argsort(times[valid])
    return times[valid][so], fn[so]


def main():
    master_sources = parse_master_thumbnails()
    ramp_entries = parse_ramp_folder()
    print(f'{len(master_sources)} master sources, {len(ramp_entries)} ramp entries')

    # Cross-match ramp entries to master IDs
    ramp_matched = {}  # master_id -> list of ramp entries
    h5 = h5py.File(CATALOG, 'r')

    # Build master source positions
    all_master = {}
    for target in ['Liller1', 'Terzan5']:
        if target not in h5: continue
        srcs = h5[f'{target}/sources'][:]
        for i, s in enumerate(srcs):
            mid = int(s['master_id'])
            all_master[mid] = {
                'target': target, 'ra': float(s['ra']),
                'dec': float(s['dec']), 'idx': i,
            }

    # Match ramp entries to master IDs by position
    for entry in ramp_entries:
        sky = SkyCoord(ra=entry['ra']*u.deg, dec=entry['dec']*u.deg)
        best_mid = None
        best_sep = 999.
        for mid, info in all_master.items():
            sep = sky.separation(SkyCoord(ra=info['ra']*u.deg, dec=info['dec']*u.deg)).arcsec
            if sep < best_sep:
                best_sep = sep
                best_mid = mid
        if best_sep < 1.0:
            if best_mid not in ramp_matched:
                ramp_matched[best_mid] = []
            ramp_matched[best_mid].append(entry)

    # Combine: all master IDs from the short folder
    all_mids = sorted(set(s['master_id'] for s in master_sources))
    print(f'\nUnique master IDs: {all_mids}')
    print(f'Ramp cross-matches: {sorted(ramp_matched.keys())}')

    # Add any ramp-only sources not in master thumbnails
    for mid in ramp_matched:
        if mid not in all_mids:
            all_mids.append(mid)
    all_mids = sorted(set(all_mids))

    # Look up period search results
    print(f'\n{"="*80}')
    print(f'PERIOD SEARCH RESULTS FOR USER-IDENTIFIED SHORT-PERIOD CANDIDATES')
    print(f'{"="*80}')

    for mid in all_mids:
        if mid not in all_master:
            print(f'\n  obj{mid:04d}: NOT IN MASTER CATALOG')
            continue
        info = all_master[mid]
        target = info['target']
        cluster = 'Liller 1' if target == 'Liller1' else 'Terzan 5'

        # Get period search entries
        if f'period_search/{target}' in h5:
            ps = h5[f'period_search/{target}'][:]
            ps_rows = ps[ps['master_id'] == mid]
        else:
            ps_rows = []

        # Get ramp pipeline periods
        ramp_info = ramp_matched.get(mid, [])

        print(f'\n  obj{mid:04d} ({cluster}, SNR={info.get("snr", all_master[mid].get("snr", "?"))}):')

        if len(ramp_info) > 0:
            print(f'    Ramp pipeline detections:')
            for r in ramp_info:
                print(f'      {r["det"]} P={r["period_min"]}min LS={r["ls_sig"]} amp={r["amplitude"]:.3f}')

        if len(ps_rows) > 0:
            print(f'    Period search (on corrected LCs):')
            for row in ps_rows:
                seg_ch = row['seg_ch'].decode()
                print(f'      {seg_ch}: LS P={row["ls_period_min"]:.1f}min sig={row["ls_significance"]:.0f}, '
                      f'BLS P={row["bls_period_min"]:.1f}min depth={row["bls_depth"]:.3f}')
        else:
            print(f'    No period search results')

    # Generate comparison plots
    print(f'\n{"="*80}')
    print(f'GENERATING COMPARISON PLOTS')
    print(f'{"="*80}')

    with open(f'{BASE}/catalogs/master_source_mapping.json') as f:
        mapping = json.load(f)

    for mid in all_mids:
        if mid not in all_master: continue
        info = all_master[mid]
        target = info['target']
        idx = info['idx']
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        # Get best LS period for this source
        best_period = None
        if f'period_search/{target}' in h5:
            ps = h5[f'period_search/{target}'][:]
            ps_rows = ps[ps['master_id'] == mid]
            if len(ps_rows) > 0:
                best_row = ps_rows[np.argmax(ps_rows['ls_significance'])]
                best_period = float(best_row['ls_period_min'])

        # Determine SW detector
        sw_det = None
        if mid < len(mapping):
            for dk, dv in mapping[mid].get('detections', {}).items():
                fn = dv.get('filename', '')
                if 'ramp' in dk and '_LW' not in dk:
                    dm = re.search(r'(nrcb[1-4])', fn)
                    if dm: sw_det = dm.group(1); break

        fig, axes = plt.subplots(len(segments) * 2, 2, figsize=(14, 4 * len(segments) * 2),
                                 squeeze=False)

        has_data = False
        row_idx = 0
        for seg in segments:
            for ch in ['SW', 'LW']:
                det = sw_det if ch == 'SW' else 'nrcblong'
                if det is None and ch == 'SW': continue

                # Get pixel position
                rp = f'{BASE}/refs/{target}_{seg}_{det}_ref.fits'
                if not os.path.exists(rp): continue
                wcs = WCS(fits.getheader(rp))
                sky = SkyCoord(ra=info['ra']*u.deg, dec=info['dec']*u.deg)
                px, py = wcs.world_to_pixel(sky)
                ix, iy = int(round(float(px))), int(round(float(py)))
                if ix < H or ix >= 2048-H or iy < H or iy >= 2048-H: continue

                ss = seg.replace('Segment', 'S')
                col = 0 if ch == 'SW' else 1
                clr = '#d62728' if ch == 'SW' else '#1f77b4'

                # Row 1: Raw ramp LC
                ax = axes[row_idx, col]
                t_raw, f_raw = get_raw_ramp_lc(target, seg, det, ix, iy)
                if t_raw is not None:
                    has_data = True
                    ax.scatter(t_raw, f_raw, s=0.3, c='grey', alpha=0.3, rasterized=True)
                    bsz = max(1, len(f_raw) // 100); nb = len(f_raw) // bsz
                    if nb > 1:
                        nu = nb * bsz
                        ax.plot(t_raw[:nu].reshape(nb, bsz).mean(1),
                                f_raw[:nu].reshape(nb, bsz).mean(1), 'grey', lw=2)
                    ax.set_title(f'{ss} {ch} — Raw ramp ({len(f_raw)}pt)', fontsize=9)
                    ax.set_ylabel('Norm flux', fontsize=8)

                # Row 2: Best corrected LC
                ax = axes[row_idx + 1, col]
                t_best, f_best, stage = get_best_lc(h5, target, mid, seg, ch, idx, sw_det)
                if t_best is not None:
                    has_data = True
                    ax.scatter(t_best, f_best, s=0.5, c=clr, alpha=0.3, rasterized=True)
                    bsz = max(1, len(f_best) // 100); nb = len(f_best) // bsz
                    if nb > 1:
                        nu = nb * bsz
                        ax.plot(t_best[:nu].reshape(nb, bsz).mean(1),
                                f_best[:nu].reshape(nb, bsz).mean(1), clr, lw=2)
                    stage_label = {'groupdiff': '', 'sat_corrected': '*',
                                   'slope_corrected': 'S', 'sat_slope': '*S',
                                   'special': 'spec'}.get(stage, '')
                    ax.set_title(f'{ss} {ch} — Best ({stage_label}) ({len(f_best)}pt)',
                                 fontsize=9)
                    ax.set_ylabel('Norm flux', fontsize=8)
                    ax.set_xlabel('Time (hr)', fontsize=8)

            row_idx += 2

        if not has_data:
            plt.close(fig); continue

        period_str = f'P={best_period:.1f}min' if best_period and best_period < 700 else 'no sig period'
        cluster = 'Liller 1' if target == 'Liller1' else 'Terzan 5'
        fig.suptitle(f'obj{mid:04d} ({cluster}) — {period_str}\n'
                     f'Top: raw ramp, Bottom: best corrected',
                     fontsize=12, y=1.01)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f'obj{mid:04d}_comparison.png')
        fig.savefig(out, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

    h5.close()
    print(f'\nDone. Plots in {OUT_DIR}/')


if __name__ == '__main__':
    main()
