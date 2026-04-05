#!/usr/bin/env python
"""
Forced photometry at arbitrary sky coordinates across all datasets.

For a given RA/Dec, extracts lightcurves from every available
segment/detector/mode combination and generates a diagnostic collage
showing raw groupdiff, IQR-clipped, sat-corrected, slope-corrected,
sat+slope, and ZF lightcurves.

Usage:
    python test_photometry.py 17:48:05.13 -24:46:38.7
    python test_photometry.py 263.35403 -33.38660
    python test_photometry.py 263.35403 -33.38660 --target Liller1
    python test_photometry.py 263.35403 -33.38660 --output /tmp/test.png
"""
import sys
import os
import re
import glob
import argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import median_filter
from photutils.aperture import CircularAperture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
REFS_DIR = f'{BASE}/refs'

H = 2
ap = CircularAperture([(H, H)], r=1.5)
ap_mask = ap.to_mask(method='exact')[0].to_image((2*H+1, 2*H+1)).astype(np.float32)
AP_AREA = float(np.sum(ap_mask))

# Background annulus radii per channel
# SW (0.031"/px): r_in=10, r_out=16 -> 0.31-0.50"
# LW (0.063"/px): r_in=5,  r_out=8  -> 0.32-0.50"
BKG_ANNULUS = {'SW': (10, 16), 'LW': (5, 8)}

TARGETS = {
    'Liller1': {'segments': ['Segment3', 'Segment4']},
    'Terzan5': {'segments': ['Segment2']},
}
SW_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
ALL_DETS = SW_DETS + ['nrcblong']

STAGE_COLORS = {
    'groupdiff_raw': '#888888',
    'groupdiff': '#444444',
    'sat_corrected': '#d62728',
    'slope_corrected': '#2ca02c',
    'sat_slope': '#1f77b4',
    'zf': '#9467bd',
}


def parse_coords(ra_str, dec_str):
    """Parse RA/Dec from either decimal degrees or sexagesimal."""
    try:
        ra = float(ra_str)
        dec = float(dec_str)
    except ValueError:
        sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
        ra, dec = sc.ra.deg, sc.dec.deg
    return ra, dec


def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    for b in range(nc):
        s, e = b*cs, (b+1)*cs; sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75]); r = q3 - q1
        m[s:e] = (sg >= q1 - iq*r) & (sg <= q3 + iq*r)
    return f[m], t[m]


def find_integrations(t):
    if len(t) < 3: return []
    dt = np.diff(t)
    med_dt = np.median(dt[dt > 0])
    breaks = np.where(dt > 2 * med_dt)[0] + 1
    boundaries = np.concatenate([[0], breaks, [len(t)]])
    return [(int(boundaries[i]), int(boundaries[i+1]))
            for i in range(len(boundaries) - 1)
            if boundaries[i+1] - boundaries[i] >= 3]


def integration_scatter(t, f):
    ints = find_integrations(t)
    if len(ints) < 3: return np.inf
    return np.mean([np.std(f[s:e]) for s, e in ints])


def apply_slope_correction(t, f):
    ints = find_integrations(t)
    if len(ints) < 10: return None
    slopes = []; med_fluxes = []
    for s, e in ints:
        x = np.arange(e - s, dtype=float)
        slopes.append(np.polyfit(x, f[s:e], 1)[0])
        med_fluxes.append(np.median(f[s:e]))
    slopes = np.array(slopes); med_fluxes = np.array(med_fluxes)
    try:
        poly = np.polyfit(med_fluxes, slopes, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    pred_slopes = np.polyval(poly, med_fluxes)
    f_corr = f.copy()
    for i, (s, e) in enumerate(ints):
        x = np.arange(e - s, dtype=float)
        f_corr[s:e] -= pred_slopes[i] * (x - x.mean())
    return f_corr / np.median(f_corr)


def get_wcs(target, seg, det):
    """Load LW-aligned WCS (preferred) or Gaia WCS."""
    for suffix in ['wcs_lw', 'wcs_gaia']:
        path = f'{ASTROM_DIR}/{target}_{seg}_{det}_{suffix}.fits'
        if os.path.exists(path):
            return WCS(fits.getheader(path))
    # Fallback to autocorr header
    ac_path = f'{REFS_DIR}/{target}_{seg}_{det}_autocorr.fits'
    if os.path.exists(ac_path):
        return WCS(fits.getheader(ac_path))
    return None


def measure_bkg_offset(cube, ix, iy, ch='SW'):
    """Measure static background offset from annulus.
    Annulus radii depend on channel (SW vs LW pixel scale).
    Returns scalar background flux scaled to the source aperture area.
    """
    r_in, r_out = BKG_ANNULUS[ch]
    nf, ny, nx = cube.shape
    y0 = max(0, iy - r_out)
    y1 = min(ny, iy + r_out + 1)
    x0 = max(0, ix - r_out)
    x1 = min(nx, ix + r_out + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r2 = (xx - ix)**2 + (yy - iy)**2
    ann_mask = (r2 >= r_in**2) & (r2 <= r_out**2)
    if ann_mask.sum() < 10:
        return 0.0
    ann_data = np.array(cube[:, y0:y1, x0:x1], dtype=np.float32)
    per_frame_med = np.array([np.nanmedian(ann_data[f][ann_mask]) for f in range(nf)])
    bkg_per_pixel = float(np.nanmedian(per_frame_med))
    return bkg_per_pixel * AP_AREA


def apply_bkg_rescale(fn, med_raw, bkg_offset):
    """Rescale normalized lightcurve to account for background.
    Given flux_norm = flux / med_raw, the true source-only normalized
    flux is: (flux_norm * med_raw - bkg) / (med_raw - bkg)
    """
    source_med = med_raw - bkg_offset
    if source_med <= 0 or bkg_offset <= 0:
        return fn
    return (fn * med_raw - bkg_offset) / source_med


def extract_groupdiff(target, seg, det, ix, iy):
    """Extract raw + clipped groupdiff lightcurve with background subtraction."""
    cube_path = f'{REFS_DIR}/groupdiffs_{target}_{seg}_{det}.fits'
    h5_path = f'{BASE}/extraction/{target}/{seg}/{det}_ramp.h5'
    if not os.path.exists(cube_path) or not os.path.exists(h5_path):
        return None
    cube = fits.getdata(cube_path, memmap=True)
    with h5py.File(h5_path, 'r') as f:
        times = f['times'][:]
    nf, ny, nx = cube.shape
    if ix-H < 0 or ix+H+1 > nx or iy-H < 0 or iy+H+1 > ny:
        return None
    cutout = np.array(cube[:, iy-H:iy+H+1, ix-H:ix+H+1], dtype=np.float32)
    np.nan_to_num(cutout, nan=0., copy=False)
    flux = np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))
    valid = np.isfinite(flux) & (flux != 0)
    if valid.sum() < 50:
        return None
    med_raw = float(np.median(flux[valid]))
    # Measure background
    ch = 'LW' if det == 'nrcblong' else 'SW'
    bkg_offset = measure_bkg_offset(cube, ix, iy, ch=ch)
    fn_raw = flux[valid] / med_raw
    t_raw = times[valid]
    so = np.argsort(t_raw)
    fn_raw, t_raw = fn_raw[so], t_raw[so]
    # IQR clip
    fn_c, t_c = clip_iqr(fn_raw, t_raw)
    fn_c, t_c = clip_iqr(fn_c, t_c)
    # Rescale amplitude for background
    fn_raw = apply_bkg_rescale(fn_raw, med_raw, bkg_offset)
    fn_c = apply_bkg_rescale(fn_c, med_raw, bkg_offset)
    return {
        'raw': (t_raw, fn_raw),
        'clipped': (t_c, fn_c),
        'median_adu': med_raw,
        'bkg_offset': float(bkg_offset),
        'med_raw': med_raw,
    }


def extract_zf(target, seg, det, ix, iy):
    """Extract ZF lightcurve."""
    cube_path = f'{REFS_DIR}/zeroframes_{target}_{seg}_{det}.fits'
    h5_path = f'{BASE}/extraction/{target}/{seg}/{det}_zf.h5'
    if not os.path.exists(cube_path):
        return None
    # Try to get times
    times = None
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            if 'times' in f:
                times = f['times'][:]
    cube = fits.getdata(cube_path, memmap=True)
    nf, ny, nx = cube.shape
    if ix-H < 0 or ix+H+1 > nx or iy-H < 0 or iy+H+1 > ny:
        return None
    cutout = np.array(cube[:, iy-H:iy+H+1, ix-H:ix+H+1], dtype=np.float32)
    np.nan_to_num(cutout, nan=0., copy=False)
    flux = np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))
    valid = np.isfinite(flux) & (flux != 0)
    if valid.sum() < 5:
        return None
    fv = flux[valid]
    if times is not None and len(times) == nf:
        tv = times[valid]
        tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
    else:
        tv_hr = np.arange(valid.sum(), dtype=float)
    so = np.argsort(tv_hr)
    fv, tv_hr = fv[so], tv_hr[so]
    # IQR clip (chunk_size=4 for ZF)
    fv_c, tv_c = clip_iqr(fv, tv_hr, cs=4)
    fv_c, tv_c = clip_iqr(fv_c, tv_c, cs=4)
    fn = fv_c / np.median(fv_c)
    return {'clipped': (tv_c, fn), 'median_adu': float(np.median(fv))}


def sat_correct_source(target, seg, det, ix, iy, groupdiff_times):
    """Saturation correction for a single source from uncal files."""
    RATIO_THRESH = 0.05
    TEMPORAL_WINDOW = 7
    CLIP_SIGMA = 3.0

    uncals = sorted(glob.glob(f'/data/JWST/{target}/{seg}/*{det}*uncal.fits'))
    if not uncals:
        return None
    # Get barycentric time base
    fb = None
    for uf in uncals:
        try:
            it = fits.getdata(uf, 'INT_TIMES')
            fb = it['int_start_BJD_TDB'].min(); break
        except:
            continue
    if fb is None:
        return None

    # Collect pixels in aperture
    pix_list = []
    for dy in range(-H, H+1):
        for dx in range(-H, H+1):
            w = ap_mask[dy+H, dx+H]
            if w < 0.01: continue
            py, px = iy+dy, ix+dx
            if 0 <= py < 2048 and 0 <= px < 2048:
                pix_list.append((py, px, w))

    # Read all ramps
    all_ramps = {(py, px): [] for py, px, w in pix_list}
    t0s = []
    for uf in uncals:
        d = fits.getdata(uf, 'SCI')
        ni, ng = d.shape[0], d.shape[1]
        try:
            it = fits.getdata(uf, 'INT_TIMES')
            ts = (it['int_mid_BJD_TDB'] - fb) * 24.0
        except:
            continue
        for i_int in range(ni):
            t0s.append(ts[i_int] if i_int < len(ts) else ts[-1])
            for py, px, w in pix_list:
                all_ramps[(py, px)].append(d[i_int, :, py, px].astype(np.float64))

    if not t0s:
        return None
    n_ints = len(t0s)
    time_arr = np.array(t0s)

    # Build ratio models per pixel
    max_gd = 0
    pix_contribs = []
    for py, px, w in pix_list:
        ramps = all_ramps[(py, px)]
        if len(ramps) != n_ints:
            continue
        # Group differences
        gds = []
        for r in ramps:
            gd = np.diff(r)
            gds.append(gd)
            if len(gd) > max_gd:
                max_gd = len(gd)
        if max_gd == 0:
            continue

        # Ratio model: for each group g, fit ratio g_i/g_0 vs g_0
        ratios_by_g = {}
        for i_int, gd in enumerate(gds):
            g0 = gd[0] if len(gd) > 0 else 0
            if g0 <= 0: continue
            for g in range(1, min(len(gd), max_gd)):
                ratios_by_g.setdefault(g, ([], []))
                ratios_by_g[g][0].append(g0)
                ratios_by_g[g][1].append(gd[g] / g0)

        rm_d = {}
        for g, (x, y) in ratios_by_g.items():
            x, y = np.array(x), np.array(y)
            if len(x) < 20: continue
            # Clip outliers
            med_y = np.median(y)
            mad_y = np.median(np.abs(y - med_y)) * 1.4826
            if mad_y > 0:
                good = np.abs(y - med_y) < 3 * mad_y
                x, y = x[good], y[good]
            if len(x) < 10: continue
            if np.std(y) / (np.abs(np.mean(y)) + 1e-10) < RATIO_THRESH:
                continue
            try:
                rm_d[g] = np.polyfit(x, y, 2)
            except:
                continue

        mg = max(rm_d.keys()) if rm_d else 0
        n_slots = n_ints * max_gd
        pix_vals = np.full((n_ints, max_gd), np.nan)
        for i_int, gd in enumerate(gds):
            g0 = gd[0] if len(gd) > 0 else 0
            if g0 <= 0: continue
            pix_vals[i_int, 0] = g0
            for g in range(1, min(len(gd), max_gd)):
                if g <= mg and g in rm_d:
                    pr = np.polyval(rm_d[g], g0)
                    if pr > 0.03:
                        pix_vals[i_int, g] = gd[g] / pr
        valid = np.isfinite(pix_vals) & (pix_vals > 0)
        if valid.sum() < 20: continue
        pmed = np.nanmedian(pix_vals[valid])
        if pmed <= 0: continue
        pix_contribs.append((pix_vals / pmed, w))

    if not pix_contribs:
        return None

    # Weighted combination
    n_slots = n_ints * max_gd
    ws = np.zeros(n_slots); wa = np.zeros(n_slots)
    for idx in range(n_slots):
        ii = idx // max_gd; g = idx % max_gd
        vals = []; wts = []
        for (cn, w) in pix_contribs:
            if ii < cn.shape[0] and g < cn.shape[1] and np.isfinite(cn[ii, g]) and cn[ii, g] > 0:
                vals.append(cn[ii, g]); wts.append(w)
        if len(vals) >= 3:
            vals = np.array(vals); wts = np.array(wts)
            med = np.median(vals); mad = np.median(np.abs(vals - med)) * 1.4826
            good = np.abs(vals - med) < CLIP_SIGMA * mad if mad > 0 else np.ones(len(vals), bool)
            ws[idx] = np.sum(vals[good] * wts[good]); wa[idx] = np.sum(wts[good])
        elif vals:
            ws[idx] = sum(v*wt for v, wt in zip(vals, wts)); wa[idx] = sum(wts)

    vc = wa > 0
    if vc.sum() < 50:
        return None
    cf = ws[vc] / wa[vc]
    cfn = cf / np.median(cf)
    tc = np.repeat(time_arr, max_gd)[vc]
    so = np.argsort(tc)
    cfn_c, tc_c = clip_iqr(cfn[so], tc[so])
    cfn_c, tc_c = clip_iqr(cfn_c, tc_c)
    if len(cfn_c) < 20:
        return None
    return (tc_c, cfn_c)


def bin_lc(t, f, bs=9):
    n = len(t) // bs
    if n < 1: return t, f
    tb = [np.mean(t[i*bs:(i+1)*bs]) for i in range(n)]
    fb = [np.median(f[i*bs:(i+1)*bs]) for i in range(n)]
    return np.array(tb), np.array(fb)


def main():
    parser = argparse.ArgumentParser(description='Forced photometry at sky coordinates')
    parser.add_argument('ra', help='RA (decimal degrees or HH:MM:SS.ss)')
    parser.add_argument('dec', help='Dec (decimal degrees or +/-DD:MM:SS.s)')
    parser.add_argument('--target', default=None, help='Force target (Liller1 or Terzan5)')
    parser.add_argument('--output', default=None, help='Output PNG path')
    parser.add_argument('--no-sat', action='store_true', help='Skip saturation correction (faster)')
    args = parser.parse_args()

    ra, dec = parse_coords(args.ra, args.dec)
    print(f'Coordinates: RA={ra:.6f}, Dec={dec:.6f}')

    # Determine target from coordinates if not specified
    if args.target:
        target = args.target
    else:
        sc = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        liller1_cen = SkyCoord(ra=263.337*u.deg, dec=-33.398*u.deg)
        terzan5_cen = SkyCoord(ra=267.020*u.deg, dec=-24.779*u.deg)
        if sc.separation(liller1_cen).arcmin < 3:
            target = 'Liller1'
        elif sc.separation(terzan5_cen).arcmin < 3:
            target = 'Terzan5'
        else:
            print('Could not determine target from coordinates. Use --target.')
            return
    print(f'Target: {target}')

    segments = TARGETS[target]['segments']

    # Find pixel positions on each detector/segment
    positions = {}  # (seg, det) -> (ix, iy)
    for seg in segments:
        for det in ALL_DETS:
            wcs = get_wcs(target, seg, det)
            if wcs is None:
                continue
            px, py = wcs.world_to_pixel_values(ra, dec)
            ix, iy = int(round(float(px))), int(round(float(py)))
            if 0 <= ix < 2048 and 0 <= iy < 2048:
                positions[(seg, det)] = (ix, iy)
                ch = 'LW' if det == 'nrcblong' else 'SW'
                print(f'  {seg} {det} ({ch}): pixel ({ix}, {iy})')

    if not positions:
        print('No valid pixel positions found.')
        return

    # Find which SW detector the source lands on
    sw_det = None
    for seg in segments:
        for det in SW_DETS:
            if (seg, det) in positions:
                ix, iy = positions[(seg, det)]
                ac_path = f'{REFS_DIR}/{target}_{seg}_{det}_autocorr.fits'
                if os.path.exists(ac_path):
                    ac = fits.getdata(ac_path)
                    if H < ix < 2048-H and H < iy < 2048-H:
                        val = ac[iy, ix]
                        if sw_det is None or val > positions.get('_best_ac', -1):
                            sw_det = det
                            positions['_best_ac'] = val
    if sw_det:
        print(f'  Best SW detector: {sw_det}')

    # Extract lightcurves
    results = {}  # (seg, ch) -> {stage: (t, f), ...}

    for seg in segments:
        for det in ALL_DETS:
            if (seg, det) not in positions:
                continue
            ch = 'LW' if det == 'nrcblong' else 'SW'
            # Skip non-matching SW detectors
            if ch == 'SW' and sw_det and det != sw_det:
                continue

            ix, iy = positions[(seg, det)]
            key = (seg, ch)
            if key not in results:
                results[key] = {}

            # Groupdiff (ramp)
            print(f'  Extracting {seg} {det} groupdiff...', end='', flush=True)
            gd = extract_groupdiff(target, seg, det, ix, iy)
            if gd:
                results[key]['groupdiff_raw'] = gd['raw']
                results[key]['groupdiff'] = gd['clipped']
                results[key]['median_adu'] = gd['median_adu']
                print(f' median={gd["median_adu"]:.0f} ADU, n={len(gd["clipped"][0])}')

                # Slope correction
                t_c, f_c = gd['clipped']
                f_sl = apply_slope_correction(t_c, f_c)
                if f_sl is not None:
                    f_sl_c, t_sl_c = clip_iqr(f_sl, t_c)
                    f_sl_c, t_sl_c = clip_iqr(f_sl_c, t_sl_c)
                    results[key]['slope_corrected'] = (t_sl_c, f_sl_c)

                # Saturation correction
                if not args.no_sat:
                    print(f'  Extracting {seg} {det} sat correction...', end='', flush=True)
                    sat = sat_correct_source(target, seg, det, ix, iy, gd['clipped'][0])
                    if sat:
                        results[key]['sat_corrected'] = sat
                        print(f' n={len(sat[0])}')
                        # Sat + slope
                        t_s, f_s = sat
                        f_ss = apply_slope_correction(t_s, f_s)
                        if f_ss is not None:
                            f_ss_c, t_ss_c = clip_iqr(f_ss, t_s)
                            f_ss_c, t_ss_c = clip_iqr(f_ss_c, t_ss_c)
                            results[key]['sat_slope'] = (t_ss_c, f_ss_c)
                    else:
                        print(' no correction needed')
            else:
                print(' no data')

            # ZF
            print(f'  Extracting {seg} {det} ZF...', end='', flush=True)
            zf = extract_zf(target, seg, det, ix, iy)
            if zf:
                results[key]['zf'] = zf['clipped']
                print(f' n={len(zf["clipped"][0])}, median={zf["median_adu"]:.0f} ADU')
            else:
                print(' no data')

    if not results:
        print('No data extracted.')
        return

    # Build collage
    combos = [(seg, ch) for seg in segments for ch in ['SW', 'LW']]
    # Filter to combos that have data
    combos = [c for c in combos if c in results]
    n_cols = len(combos)

    stages_order = ['groupdiff_raw', 'groupdiff', 'sat_corrected',
                    'slope_corrected', 'sat_slope', 'zf']
    stage_labels = {
        'groupdiff_raw': 'Groupdiff (raw)',
        'groupdiff': 'Groupdiff (IQR clipped)',
        'sat_corrected': 'Sat corrected',
        'slope_corrected': 'Slope corrected',
        'sat_slope': 'Sat + Slope',
        'zf': 'Zero-frame',
    }

    # +1 row for autocorr cutouts at the top
    n_rows = len(stages_order) + 1
    fig, axes = plt.subplots(n_rows, max(n_cols, 1),
                             figsize=(5 * max(n_cols, 1), 3 * n_rows),
                             squeeze=False)

    # Row 0: autocorr cutouts with aperture overlay
    CUTOUT_HALF = 12
    for ci, (seg, ch) in enumerate(combos):
        ax = axes[0, ci]
        seg_short = seg.replace('Segment', 'S')
        det_for_cutout = 'nrcblong' if ch == 'LW' else sw_det
        if det_for_cutout and (seg, det_for_cutout) in positions:
            ix, iy = positions[(seg, det_for_cutout)]
            ac_path = f'{REFS_DIR}/{target}_{seg}_{det_for_cutout}_autocorr.fits'
            if os.path.exists(ac_path):
                ac_data = fits.getdata(ac_path)
                y0 = max(0, iy - CUTOUT_HALF)
                y1 = min(ac_data.shape[0], iy + CUTOUT_HALF + 1)
                x0 = max(0, ix - CUTOUT_HALF)
                x1 = min(ac_data.shape[1], ix + CUTOUT_HALF + 1)
                cutout = ac_data[y0:y1, x0:x1]
                vmin, vmax = np.nanpercentile(cutout, [5, 99])
                ax.imshow(cutout, origin='lower', cmap='magma',
                          vmin=vmin, vmax=vmax,
                          extent=[x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5])
                # Aperture circle (r=1.5px) at the WCS position
                wcs_here = get_wcs(target, seg, det_for_cutout)
                if wcs_here:
                    px_f, py_f = wcs_here.world_to_pixel_values(ra, dec)
                    px_f, py_f = float(px_f), float(py_f)
                else:
                    px_f, py_f = float(ix), float(iy)
                circle = plt.Circle((px_f, py_f), 1.5, fill=False, ec='cyan',
                                    lw=1.5, ls='-')
                ax.add_patch(circle)
                ax.plot(px_f, py_f, '+', color='cyan', ms=8, mew=1)
                ax.set_title(f'{seg_short} {ch} autocorr ({det_for_cutout})\n'
                             f'px=({px_f:.1f}, {py_f:.1f})', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'no autocorr', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_title(f'{seg_short} {ch} autocorr', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'off detector', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(f'{seg_short} {ch} autocorr', fontsize=8)
        ax.tick_params(labelsize=7)

    # Lightcurve rows
    for ci, (seg, ch) in enumerate(combos):
        seg_short = seg.replace('Segment', 'S')
        data = results[(seg, ch)]
        med_adu = data.get('median_adu', 0)

        for ri, stage in enumerate(stages_order):
            ax = axes[ri + 1, ci]  # +1 for cutout row
            if stage in data:
                t, f = data[stage]
                color = STAGE_COLORS.get(stage, '#333333')
                ax.scatter(t, f, s=0.3, c=color, alpha=0.3, rasterized=True)
                # Binned overlay
                if len(t) > 18:
                    bs = 4 if stage == 'zf' else 9
                    tb, fb = bin_lc(t, f, bs=bs)
                    ax.plot(tb, fb, '-', color=color, lw=0.8, alpha=0.7)
                sc = integration_scatter(t, f) if stage != 'zf' else np.std(f)
                ax.set_title(f'{seg_short} {ch} -- {stage_labels[stage]}\n'
                             f'n={len(f)}, scatter={sc:.6f}, std={np.std(f):.4f}',
                             fontsize=8)
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
                ax.set_title(f'{seg_short} {ch} -- {stage_labels[stage]}', fontsize=8)
            ax.tick_params(labelsize=7)
            if ri == len(stages_order) - 1:
                ax.set_xlabel('Time (hr)', fontsize=8)

    coord_str = SkyCoord(ra=ra*u.deg, dec=dec*u.deg).to_string('hmsdms', precision=2)
    fig.suptitle(f'{target}  RA={ra:.5f}  Dec={dec:.5f}  ({coord_str})',
                 fontsize=11, y=1.01)
    fig.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = f'{BASE}/diagnostics/test_phot_{ra:.5f}_{dec:.5f}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
