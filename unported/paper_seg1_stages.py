#!/usr/bin/env python
"""
Generate paper figures showing Segment 1 dithered extraction processing stages.

For each source, shows 6 panels:
  1. Raw groupdiff aperture photometry (all exposures, color-coded)
  2. After saturation correction (per-group median ratio)
  3. After per-exposure IQR clip
  4. After integration slope correction (raw ADU)
  5. After normalization + slope-aware stitching
  6. Segment 2 corrected lightcurve (reference)

Usage:
    python paper_seg1_stages.py --sources 6 14 115
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
SEG1_DIR = '/data/JWST/Terzan5/Segment1'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'

H = 2
ap = CircularAperture([(H, H)], r=1.5)
ap_mask = ap.to_mask(method='exact')[0].to_image((2*H+1, 2*H+1)).astype(np.float32)
AP_AREA = float(np.sum(ap_mask))
TGROUP_HR = 21.47354 / 3600.0
BKG_ANNULUS = {'SW': (10, 16), 'LW': (5, 8)}


def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    last_chunk = nc - 1 if n % cs != 0 and nc > 0 else nc
    for b in range(nc):
        s = b * cs
        e = (b+1)*cs if b < last_chunk else n
        sg = f[s:e]
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


def correct_integration_slopes_raw(t, f):
    ints = find_integrations(t)
    if len(ints) < 3:
        return f.copy()
    slopes = []; med_fluxes = []
    for s, e in ints:
        x = np.arange(e - s, dtype=float)
        slopes.append(np.polyfit(x, f[s:e], 1)[0])
        med_fluxes.append(np.median(f[s:e]))
    slopes = np.array(slopes); med_fluxes = np.array(med_fluxes)
    try:
        poly = np.polyfit(med_fluxes, slopes, 2)
    except:
        return f.copy()
    pred_slopes = np.polyval(poly, med_fluxes)
    f_corr = f.copy()
    for i, (s, e) in enumerate(ints):
        x = np.arange(e - s, dtype=float)
        f_corr[s:e] -= pred_slopes[i] * (x - x.mean())
    return f_corr


def slope_stitch(blocks):
    valid_blocks = [b for b in blocks if b is not None]
    if not valid_blocks:
        return None
    offsets = np.zeros(len(blocks))
    models = []
    for b in blocks:
        if b is None:
            models.append(None)
            continue
        if len(b[0]) < 10:
            models.append(None)
            continue
        p = np.polyfit(b[0], b[1], 1)
        models.append({
            'slope': p[0],
            'val_end': np.polyval(p, b[0][-1]),
            'val_start': np.polyval(p, b[0][0]),
            't_end': b[0][-1],
            't_start': b[0][0],
        })
    for i in range(1, len(blocks)):
        if blocks[i] is None or models[i] is None:
            continue
        prev = None
        for j in range(i-1, -1, -1):
            if blocks[j] is not None and models[j] is not None:
                prev = j
                break
        if prev is None:
            continue
        pm, cm = models[prev], models[i]
        dt_gap = cm['t_start'] - pm['t_end']
        avg_slope = (pm['slope'] + cm['slope']) / 2.0
        expected_start = pm['val_end'] + offsets[prev] + avg_slope * dt_gap
        offsets[i] = expected_start - cm['val_start']

    all_t, all_f = [], []
    for i, b in enumerate(blocks):
        if b is None:
            continue
        all_t.append(b[0])
        all_f.append(b[1] + offsets[i])
    t_cat = np.concatenate(all_t)
    f_cat = np.concatenate(all_f)
    so = np.argsort(t_cat)
    t_cat, f_cat = t_cat[so], f_cat[so]
    global_med = np.median(f_cat)
    if global_med <= 0:
        return None
    return t_cat, f_cat / global_med, global_med


def apply_bkg_rescale(fn, med_raw, bkg_offset):
    source_med = med_raw - bkg_offset
    if source_med <= 0 or bkg_offset <= 0:
        return fn
    return (fn * med_raw - bkg_offset) / source_med


def extract_source_stages(mid, ra, dec, det):
    """Extract one source from Segment 1, returning intermediate stages.

    Returns dict of stage_name -> list of (t_hr, flux) per exposure.
    Stages: 'raw', 'sat_corr', 'iqr_clip', 'slope_corr', 'normalized', 'stitched'
    """
    ch = 'LW' if det == 'nrcblong' else 'SW'
    r_in, r_out = BKG_ANNULUS[ch]

    uncals = sorted(glob.glob(f'{SEG1_DIR}/*{det}_uncal.fits'))
    calints_files = sorted(glob.glob(f'{SEG1_DIR}/*{det}_calints.fits'))
    if len(uncals) != len(calints_files) or len(uncals) == 0:
        return None

    n_exp = len(uncals)
    stages = {
        'raw': [None] * n_exp,
        'sat_corr': [None] * n_exp,
    }
    bkg_vals = []

    print(f'  Extracting obj{mid} from {det} ({n_exp} exposures)...')

    for ei, (uncal_path, calints_path) in enumerate(zip(uncals, calints_files)):
        wcs = WCS(fits.getheader(calints_path, 'SCI'), naxis=2)
        try:
            int_times = fits.getdata(calints_path, 'INT_TIMES')
            times_bjd = int_times['int_mid_BJD_TDB']
        except:
            continue

        px_f, py_f = wcs.world_to_pixel_values(ra, dec)
        ix, iy = int(round(float(px_f))), int(round(float(py_f)))

        if ix < H or ix >= 2048-H or iy < H or iy >= 2048-H:
            continue

        uncal_data = fits.getdata(uncal_path, 'SCI')
        n_ints, n_groups, ny, nx = uncal_data.shape
        n_gd = n_groups - 1

        flux_raw = np.zeros(n_ints * n_gd)
        group_idx = np.zeros(n_ints * n_gd, dtype=int)
        time_arr = np.zeros(n_ints * n_gd)

        pos = np.array([[px_f, py_f]])
        ap_phot = CircularAperture(pos, r=1.5)
        bkg_ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)

        for i_int in range(n_ints):
            ramp = uncal_data[i_int].astype(np.float64)
            for g in range(n_gd):
                gd_frame = (ramp[g+1] - ramp[g]).astype(np.float32)
                idx = i_int * n_gd + g
                phot = aperture_photometry(gd_frame, ap_phot)
                flux_raw[idx] = float(phot['aperture_sum'][0])
                group_idx[idx] = g
                t_off = (g - n_gd / 2.0) * TGROUP_HR
                time_arr[idx] = float(times_bjd[i_int]) + t_off / 24.0

            # Background from first group-diff
            gd0 = (ramp[1] - ramp[0]).astype(np.float32)
            bkg_phot = aperture_photometry(gd0, bkg_ann)
            bkg_vals.append(float(bkg_phot['aperture_sum'][0]) / bkg_ann.area)

        valid = np.isfinite(flux_raw) & (flux_raw > 0)
        if valid.sum() < 10:
            continue

        stages['raw'][ei] = (time_arr[valid].copy(), flux_raw[valid].copy())

        # Saturation correction
        g0_med = np.median(flux_raw[group_idx == 0])
        if g0_med <= 0:
            continue
        fl_corr = flux_raw.copy()
        for g in range(n_gd):
            gmask = group_idx == g
            g_med = np.median(flux_raw[gmask & valid])
            ratio = g_med / g0_med if g0_med > 0 else 1.0
            if ratio > 0.05:
                fl_corr[gmask] = flux_raw[gmask] / ratio
            else:
                fl_corr[gmask] = np.nan

        valid_corr = np.isfinite(fl_corr) & (fl_corr > 0)
        if valid_corr.sum() > 5:
            stages['sat_corr'][ei] = (time_arr[valid_corr].copy(), fl_corr[valid_corr].copy())

        if (ei + 1) % 5 == 0:
            print(f'    exposure {ei+1}/{n_exp}')

    # Compute bkg
    bkg_off = float(np.median(bkg_vals)) * AP_AREA if bkg_vals else 0.0

    # Convert times to hours from earliest
    all_t = []
    for b in stages['sat_corr']:
        if b is not None:
            all_t.extend(b[0].tolist())
    if not all_t:
        return None
    t0 = min(all_t)

    # Convert raw blocks to hours
    raw_hr = []
    for b in stages['raw']:
        if b is None:
            raw_hr.append(None)
        else:
            raw_hr.append(((b[0] - t0) * 24.0, b[1]))
    stages['raw_hr'] = raw_hr

    # Convert sat_corr to hours
    hr_blocks = []
    for b in stages['sat_corr']:
        if b is None:
            hr_blocks.append(None)
        else:
            hr_blocks.append(((b[0] - t0) * 24.0, b[1]))
    stages['sat_corr_hr'] = hr_blocks

    # IQR clip per exposure
    iqr_blocks = []
    for b in hr_blocks:
        if b is None:
            iqr_blocks.append(None)
            continue
        t_b, f_b = b
        f_bc, t_bc = clip_iqr(f_b, t_b)
        f_bc, t_bc = clip_iqr(f_bc, t_bc)
        iqr_blocks.append((t_bc, f_bc))
    stages['iqr_clip'] = iqr_blocks

    # Integration slope correction in raw ADU
    slope_blocks = []
    for b in iqr_blocks:
        if b is None:
            slope_blocks.append(None)
            continue
        t_b, f_b = b
        f_corr = correct_integration_slopes_raw(t_b, f_b)
        slope_blocks.append((t_b, f_corr))
    stages['slope_corr'] = slope_blocks

    # Normalize per block
    norm_blocks = []
    for b in slope_blocks:
        if b is None:
            norm_blocks.append(None)
            continue
        med_b = np.median(b[1])
        if med_b <= 0:
            norm_blocks.append(None)
            continue
        norm_blocks.append((b[0], b[1] / med_b))
    stages['normalized'] = norm_blocks

    # 4-IQR exposure rejection
    block_raw_meds = []
    for b in hr_blocks:
        if b is None:
            block_raw_meds.append(None)
        else:
            block_raw_meds.append(np.median(b[1]) if len(b[1]) > 0 else None)
    valid_meds = [m for m in block_raw_meds if m is not None and m > 0]
    rejected_exposures = set()
    if len(valid_meds) >= 3:
        q1, q3 = np.percentile(valid_meds, [25, 75])
        iqr_val = q3 - q1
        lo, hi = q1 - 4 * iqr_val, q3 + 4 * iqr_val
        for bi in range(len(norm_blocks)):
            if block_raw_meds[bi] is not None and (block_raw_meds[bi] < lo or block_raw_meds[bi] > hi):
                norm_blocks[bi] = None
                rejected_exposures.add(bi)

    stages['rejected'] = rejected_exposures

    # Slope-aware stitch
    result = slope_stitch(norm_blocks)
    if result is None:
        return None
    t_hr, fn, med_raw = result

    # Final IQR clip + bkg rescale
    fn_c, t_c = clip_iqr(fn, t_hr)
    fn_c, t_c = clip_iqr(fn_c, t_c)
    fn_c = apply_bkg_rescale(fn_c, med_raw, bkg_off)

    stages['stitched'] = (t_c, fn_c)
    stages['med_raw'] = med_raw
    stages['bkg_off'] = bkg_off

    return stages


def get_seg2_lc(h5, mid, ch='SW'):
    """Get the best-stage Segment 2 lightcurve."""
    bs = h5['best_stage/Terzan5'][:]
    row = bs[bs['master_id'] == mid]
    if len(row) == 0:
        return None, None
    col = f'Segment2_{ch}'
    if col not in bs.dtype.names:
        return None, None
    stage = row[0][col].decode()
    if not stage:
        return None, None
    path = f'{stage}/Terzan5/{mid}/Segment2_{ch}'
    if path not in h5:
        path = f'groupdiff/Terzan5/{mid}/Segment2_{ch}'
        if path not in h5:
            return None, None
    ds = h5[path]
    return ds['times'][:], ds['flux_norm'][:]


def add_binned_median(ax, t_all, f_all, n_bins=60, color='k', ms=3):
    """Overlay binned median on a scatter panel."""
    if len(t_all) < 20:
        return
    t_edges = np.linspace(t_all.min(), t_all.max(), n_bins + 1)
    for i in range(n_bins):
        m = (t_all >= t_edges[i]) & (t_all < t_edges[i + 1])
        if m.sum() > 5:
            ax.plot((t_edges[i] + t_edges[i + 1]) / 2, np.median(f_all[m]),
                    'o', color=color, ms=ms, zorder=5)


def plot_exposure_blocks(ax, blocks, exp_colors, rejected, s=0.4, alpha=1.0):
    """Plot per-exposure scatter with color coding. Returns concatenated (t, f)."""
    all_t, all_f = [], []
    for ei, b in enumerate(blocks):
        if b is None:
            continue
        c = exp_colors[ei]
        a = alpha if ei not in rejected else 0.1
        ax.scatter(b[0], b[1], s=s, c=[c], alpha=a, rasterized=True)
        if ei not in rejected:
            all_t.append(b[0])
            all_f.append(b[1])
    if all_t:
        return np.concatenate(all_t), np.concatenate(all_f)
    return np.array([]), np.array([])


def make_figure(mid, det, stages, seg2_t, seg2_f, out_path):
    """Create 6-panel processing stages figure."""
    ch = 'LW' if det == 'nrcblong' else 'SW'

    fig = plt.figure(figsize=(14, 16))
    # Use gridspec: panels a-d share x-axis, e-f are independent
    gs = fig.add_gridspec(6, 1, hspace=0.35, top=0.96, bottom=0.04, left=0.08, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    # Color palette for exposures
    n_exp = len(stages['raw_hr'])
    cmap = plt.cm.turbo
    exp_colors = [cmap(i / max(n_exp - 1, 1)) for i in range(n_exp)]

    rejected = stages.get('rejected', set())

    # Determine shared x range for panels a-d
    all_times = []
    for stage_key in ['raw_hr', 'sat_corr_hr', 'iqr_clip', 'slope_corr']:
        for b in stages[stage_key]:
            if b is not None:
                all_times.extend(b[0].tolist())
    if all_times:
        t_lo, t_hi = min(all_times), max(all_times)
        t_pad = (t_hi - t_lo) * 0.02
        shared_xlim = (t_lo - t_pad, t_hi + t_pad)
    else:
        shared_xlim = None

    # --- Panel (a): Raw groupdiff ---
    ax = axes[0]
    t_cat, f_cat = plot_exposure_blocks(ax, stages['raw_hr'], exp_colors, rejected)
    ax.set_ylabel('Flux (ADU)', fontsize=10)
    ax.set_title('(a) Raw group-difference aperture photometry', fontsize=11, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)
    if shared_xlim:
        ax.set_xlim(shared_xlim)
    ax.set_xticklabels([])

    # --- Panel (b): After saturation correction ---
    ax = axes[1]
    t_cat, f_cat = plot_exposure_blocks(ax, stages['sat_corr_hr'], exp_colors, rejected)
    ax.set_ylabel('Flux (ADU)', fontsize=10)
    ax.set_title('(b) After saturation correction (per-group median ratio)', fontsize=11, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)
    if shared_xlim:
        ax.set_xlim(shared_xlim)
    ax.set_xticklabels([])

    # --- Panel (c): After IQR clip ---
    ax = axes[2]
    t_cat, f_cat = plot_exposure_blocks(ax, stages['iqr_clip'], exp_colors, rejected)
    ax.set_ylabel('Flux (ADU)', fontsize=10)
    ax.set_title('(c) After per-exposure IQR clip', fontsize=11, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)
    if shared_xlim:
        ax.set_xlim(shared_xlim)
    ax.set_xticklabels([])

    # --- Panel (d): After integration slope correction ---
    ax = axes[3]
    t_cat, f_cat = plot_exposure_blocks(ax, stages['slope_corr'], exp_colors, rejected)
    ax.set_ylabel('Flux (ADU)', fontsize=10)
    ax.set_title('(d) After integration slope correction', fontsize=11, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)
    if shared_xlim:
        ax.set_xlim(shared_xlim)
    ax.set_xlabel('Time (hr from start of Segment 1)', fontsize=10)

    # --- Panel (e): Normalized + stitched ---
    ax = axes[4]
    t_s, f_s = stages['stitched']
    ax.scatter(t_s, f_s, s=0.5, c='k', alpha=1.0, rasterized=True)
    ax.set_ylabel('Normalized Flux', fontsize=10)
    ax.set_xlabel('Time (hr from start of Segment 1)', fontsize=10)
    n_rej = len(rejected)
    rej_text = f' ({n_rej} exposures rejected)' if n_rej > 0 else ''
    ax.set_title(f'(e) Normalized + slope-aware stitch{rej_text}',
                 fontsize=11, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)

    # --- Panel (f): Seg2 reference ---
    ax = axes[5]
    if seg2_t is not None and seg2_f is not None:
        ax.scatter(seg2_t, seg2_f, s=0.5, c='k', alpha=1.0, rasterized=True)
        ax.set_ylabel('Normalized Flux', fontsize=10)
        ax.set_xlabel('Time (hr from start of Segment 2)', fontsize=10)
        ax.set_title('(f) Segment 2 corrected lightcurve (reference)',
                     fontsize=11, fontweight='bold', loc='left')
    else:
        ax.text(0.5, 0.5, 'No Segment 2 data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_title('(f) Segment 2 (no data)', fontsize=11, fontweight='bold', loc='left')
    ax.tick_params(labelsize=8)

    fig.suptitle(f'Terzan 5 \u2014 Object {mid} ({ch}): Dithered Segment 1 Processing Pipeline',
                 fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=int, nargs='+', default=[6, 14, 115])
    args = parser.parse_args()

    h5 = h5py.File(CATALOG, 'r')
    srcs = h5['Terzan5/sources'][:]

    # Use mapping file for RA/Dec (same source add_segment1_terzan5.py uses)
    import json
    with open(f'{BASE}/catalogs/master_source_mapping.json') as f:
        mapping = json.load(f)
    mapping_by_mid = {m['master_id']: m for m in mapping}

    out_dir = f'{BASE}/diagnostics'
    os.makedirs(out_dir, exist_ok=True)

    for mid in args.sources:
        row = srcs[srcs['master_id'] == mid]
        if len(row) == 0:
            print(f'Source {mid} not found in catalog')
            continue
        r = row[0]
        det = r['detector'].decode()
        # Use mapping RA/Dec to match what add_segment1_terzan5.py extracts
        m = mapping_by_mid.get(mid)
        if m is None:
            print(f'Source {mid} not in mapping')
            continue
        ra, dec = m['ra'], m['dec']
        print(f'\n=== Object {mid}: det={det}, ra={ra:.6f}, dec={dec:.6f} ===')

        # Extract all stages from Seg1
        stages = extract_source_stages(mid, ra, dec, det)
        if stages is None:
            print(f'  No Segment 1 data for obj{mid}')
            continue

        # Get Seg2 reference
        ch = 'LW' if det == 'nrcblong' else 'SW'
        seg2_t, seg2_f = get_seg2_lc(h5, mid, ch)

        # Make figure
        out_path = f'{out_dir}/paper_seg1_stages_obj{mid:04d}.png'
        make_figure(mid, det, stages, seg2_t, seg2_f, out_path)

    h5.close()
    print('\nDone.')


if __name__ == '__main__':
    main()
