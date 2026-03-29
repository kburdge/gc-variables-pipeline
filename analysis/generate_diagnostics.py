#!/usr/bin/env python
"""Generate diagnostic PNGs with SW + LW cross-channel forced photometry.

Each source gets a multi-panel diagnostic: the detection lightcurve on row 1,
then forced photometry from all available ramp cubes (same/other segment,
same/cross channel) on subsequent rows.

Panel layout (Liller1 ramp example, 4 rows):
  Row 1: Detection LC (Seg3 SW ramp)
  Row 2: Other-segment same-channel forced (Seg4 SW ramp)
  Row 3: Same-segment cross-channel forced (Seg3 LW ramp)
  Row 4: Other-segment cross-channel forced (Seg4 LW ramp)

Terzan5 (single segment) gets 2 rows: detection + cross-channel.
ZF sources use ramp cubes for all comparison panels.

Usage:
    python generate_diagnostics.py --target Liller1 --mode ramp
    python generate_diagnostics.py --target Liller1 --mode ramp --channel lw
    python generate_diagnostics.py --target Terzan5 --mode zf
"""
import os
import sys
import argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as timer

BASE_DIR = '/data/Globulars_Pipeline'
CUTOUT_RADIUS = 10
AP_RADIUS = 1.5
DEDUP_RADIUS = 0.3  # arcsec

# IQR clipping params (match pipeline.yaml)
IQR_CHUNK_RAMP = 18
IQR_CHUNK_ZF = 4
IQR_FACTOR = 2.0

SW_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW_DETS = ['nrcblong']

TARGETS = {
    'Liller1': {'segments': ['Segment3', 'Segment4']},
    'Terzan5': {'segments': ['Segment2']},
}


# ============================================================================
# Utility functions
# ============================================================================

def extract_cutout(ref_img, px, py, radius=CUTOUT_RADIUS):
    """Extract (2*radius+1) x (2*radius+1) cutout, NaN-padded at edges."""
    ix, iy = int(np.round(px)), int(np.round(py))
    ny, nx = ref_img.shape
    y0, y1 = max(0, iy - radius), min(ny, iy + radius + 1)
    x0, x1 = max(0, ix - radius), min(nx, ix + radius + 1)
    size = 2 * radius + 1
    cutout = np.full((size, size), np.nan, dtype=np.float32)
    cy0 = radius - (iy - y0)
    cx0 = radius - (ix - x0)
    cutout[cy0:cy0 + (y1 - y0), cx0:cx0 + (x1 - x0)] = ref_img[y0:y1, x0:x1]
    return cutout


def bin_lightcurve(times, flux, pts_per_bin=9):
    """Bin by grouping every pts_per_bin consecutive points."""
    n = len(times)
    n_bins = max(1, n // pts_per_bin)
    n_use = n_bins * pts_per_bin
    t_reshape = times[:n_use].reshape(n_bins, pts_per_bin)
    f_reshape = flux[:n_use].reshape(n_bins, pts_per_bin)
    return np.mean(t_reshape, axis=1), np.nanmean(f_reshape, axis=1)


def clip_outliers_iqr(flux, times, chunk_size=18, iqr_factor=IQR_FACTOR):
    """IQR clipping in chunks — matches pipeline clip_outliers_iqr exactly."""
    n = len(flux)
    n_chunks = n // chunk_size
    if n_chunks < 1:
        return flux, times
    mask = np.ones(n, dtype=bool)
    for b in range(n_chunks):
        s, e = b * chunk_size, (b + 1) * chunk_size
        seg = flux[s:e]
        q1, q3 = np.percentile(seg, [25, 75])
        iqr = q3 - q1
        good = (seg >= q1 - iqr_factor * iqr) & (seg <= q3 + iqr_factor * iqr)
        mask[s:e] = good
    return flux[mask], times[mask]


def dedup_sources(sources):
    """Spatial dedup via KD-tree: keep highest SNR within DEDUP_RADIUS."""
    if len(sources) == 0:
        return []
    sources.sort(key=lambda x: -x['snr'])
    coords = np.array([[s['ra'], s['dec']] for s in sources])
    mean_dec = np.mean(coords[:, 1])
    cos_dec = np.cos(np.radians(mean_dec))
    xy = np.column_stack([coords[:, 0] * cos_dec * 3600, coords[:, 1] * 3600])
    tree = cKDTree(xy)
    kept = np.ones(len(sources), dtype=bool)
    for i in range(len(sources)):
        if not kept[i]:
            continue
        for j in tree.query_ball_point(xy[i], DEDUP_RADIUS):
            if j > i and kept[j]:
                kept[j] = False
    return [s for s, k in zip(sources, kept) if k]


def forced_photometry(cube, px, py, ap_radius=AP_RADIUS):
    """Extract aperture photometry lightcurve from cube at pixel position."""
    ix, iy = int(np.round(px)), int(np.round(py))
    r = int(np.ceil(ap_radius)) + 1
    ny, nx = cube.shape[1], cube.shape[2]
    if ix < r or ix >= nx - r or iy < r or iy >= ny - r:
        return np.full(cube.shape[0], np.nan, dtype=np.float32)
    size = 2 * r + 1
    yy, xx = np.mgrid[:size, :size]
    mask = ((xx - r) ** 2 + (yy - r) ** 2) <= ap_radius ** 2
    cutout = cube[:, iy - r:iy + r + 1, ix - r:ix + r + 1]
    return np.nansum(cutout[:, mask], axis=1).astype(np.float32)


# ============================================================================
# Data loading
# ============================================================================

def load_segment_sources(base_dir, target, segment, mode, detectors):
    """Load sources from extraction HDF5s for one segment, dedup across detectors."""
    all_src = []
    for det in detectors:
        h5_path = f'{base_dir}/extraction/{target}/{segment}/{det}_{mode}.h5'
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, 'r') as f:
            srcs = f['sources'][:]
        for i in range(len(srcs)):
            s = srcs[i]
            all_src.append({
                'det': det, 'idx': i,
                'ra': float(s['ra']), 'dec': float(s['dec']),
                'snr': float(s['det_snr']),
                'period': float(s['best_period_min']),
                'ls_sig': float(s['ls_significance']),
                'amp': float(s['amplitude']),
            })
    print(f'  {segment} {mode}: {len(all_src)} raw sources')
    deduped = dedup_sources(all_src)
    print(f'  {segment} {mode}: {len(deduped)} after dedup')
    return deduped


def load_extraction_data(base_dir, target, segment, mode, detectors):
    """Open extraction HDF5s, return {det: {sources, flux_clipped, times, handle}}."""
    data = {}
    for det in detectors:
        h5_path = f'{base_dir}/extraction/{target}/{segment}/{det}_{mode}.h5'
        if not os.path.exists(h5_path):
            continue
        f = h5py.File(h5_path, 'r')
        data[det] = {
            'sources': f['sources'][:],
            'flux_clipped': f['flux_clipped'],
            'times': f['times'][:],
            'handle': f,
        }
    return data


def load_autocorr_refs(base_dir, target, segment, mode, detectors):
    """Load autocorrelation reference images for one segment."""
    refs = {}
    for det in detectors:
        if mode == 'zf':
            ac_path = f'{base_dir}/refs/{target}_{segment}_{det}_zf_autocorr.fits'
        else:
            ac_path = f'{base_dir}/refs/{target}_{segment}_{det}_autocorr.fits'
        if os.path.exists(ac_path):
            refs[det] = fits.getdata(ac_path)
    return refs


def load_cubes(base_dir, target, segment, mode, detectors):
    """Load group-diff or ZF cubes (memmap) for forced photometry."""
    cubes = {}
    times = {}
    for det in detectors:
        if mode == 'ramp':
            cube_path = f'{base_dir}/refs/groupdiffs_{target}_{segment}_{det}.fits'
        else:
            cube_path = f'{base_dir}/refs/zeroframes_{target}_{segment}_{det}.fits'
        if not os.path.exists(cube_path):
            continue
        hdul = fits.open(cube_path, memmap=True)
        cubes[det] = hdul[0].data
        h5_path = f'{base_dir}/extraction/{target}/{segment}/{det}_{mode}.h5'
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                times[det] = f['times'][:]
        else:
            n_frames = cubes[det].shape[0]
            times[det] = np.linspace(0, 7.0, n_frames)
    return cubes, times


def load_wcs_refs(base_dir, target, segment, detectors):
    """Load WCS from ramp reference images (always use ramp WCS, even for ZF)."""
    wcs_dict = {}
    for det in detectors:
        ref_path = f'{base_dir}/refs/{target}_{segment}_{det}_ref.fits'
        if os.path.exists(ref_path):
            wcs_dict[det] = WCS(fits.getheader(ref_path))
    return wcs_dict


# ============================================================================
# Comparison panel logic
# ============================================================================

def build_comparison_panels(target, segment, mode, is_lw, segments):
    """Return list of (seg, comp_mode, detectors, label) for comparison panels."""
    same_dets = LW_DETS if is_lw else SW_DETS
    cross_dets = SW_DETS if is_lw else LW_DETS
    same_ch = 'LW' if is_lw else 'SW'
    cross_ch = 'SW' if is_lw else 'LW'

    panels = []

    if mode == 'ramp':
        if len(segments) > 1:
            other = [s for s in segments if s != segment][0]
            panels.append((other, 'ramp', same_dets,
                           f'{other} {same_ch} ramp'))
            panels.append((segment, 'ramp', cross_dets,
                           f'{segment} {cross_ch} ramp'))
            panels.append((other, 'ramp', cross_dets,
                           f'{other} {cross_ch} ramp'))
        else:
            # Single segment (Terzan5): just cross-channel
            panels.append((segment, 'ramp', cross_dets,
                           f'{segment} {cross_ch} ramp'))
    elif mode == 'zf':
        # ZF: all comparisons use ramp cubes
        panels.append((segment, 'ramp', same_dets,
                       f'{segment} {same_ch} ramp'))
        panels.append((segment, 'ramp', cross_dets,
                       f'{segment} {cross_ch} ramp'))
        if len(segments) > 1:
            other = [s for s in segments if s != segment][0]
            panels.append((other, 'ramp', cross_dets,
                           f'{other} {cross_ch} ramp'))

    return panels


def find_detector_for_position(ra, dec, wcs_dict, cubes_dict):
    """Find which detector a sky position falls on, return (det, px, py) or None."""
    for det, wcs_obj in wcs_dict.items():
        if det not in cubes_dict:
            continue
        px, py = wcs_obj.world_to_pixel_values(ra, dec)
        px, py = float(px), float(py)
        if 0 <= px < 2048 and 0 <= py < 2048:
            return det, px, py
    return None, None, None


# ============================================================================
# Plotting
# ============================================================================

def plot_lc_row(ax_lc, ax_cut, times, flux, src_px, src_py, seg_snr,
                title_label, autocorr_ref, det_name):
    """Plot one row: lightcurve panel + autocorrelation cutout."""
    good = np.isfinite(flux)
    if np.any(good):
        t_good = times[good]
        f_good = flux[good]
        ax_lc.plot(t_good, f_good, '.', ms=1.5, color='black', rasterized=True)
        bc, bm = bin_lightcurve(t_good, f_good, pts_per_bin=9)
        gb = np.isfinite(bm)
        if np.any(gb):
            ax_lc.plot(bc[gb], bm[gb], '-', color='red', lw=1.2)
    else:
        ax_lc.text(0.5, 0.5, 'no valid points', ha='center', va='center',
                   transform=ax_lc.transAxes, fontsize=8, color='gray')
    ax_lc.set_xlabel('Time (hr)', fontsize=8)
    ax_lc.set_ylabel('Flux (DN)', fontsize=8)
    ax_lc.set_title(title_label, fontsize=9)
    ax_lc.tick_params(labelsize=7)

    if autocorr_ref is not None:
        cutout = extract_cutout(autocorr_ref, src_px, src_py)
        r = CUTOUT_RADIUS
        ax_cut.imshow(cutout, origin='lower', cmap='RdBu_r',
                      vmin=-0.5, vmax=0.5,
                      extent=[-r-0.5, r+0.5, -r-0.5, r+0.5])
        circle = plt.Circle((0, 0), AP_RADIUS, fill=False, color='lime', lw=1.5)
        ax_cut.add_patch(circle)
        center_val = cutout[r, r]
        ax_cut.set_title(f'ac={center_val:.2f} S/N={seg_snr:.0f}', fontsize=8)
    else:
        ax_cut.text(0.5, 0.5, 'no ref', ha='center', va='center',
                    transform=ax_cut.transAxes, fontsize=8, color='gray')
    ax_cut.set_xticks([])
    ax_cut.set_yticks([])


def blank_row(ax_lc, ax_cut, label):
    """Fill a row with 'out of FOV' text."""
    for ax in (ax_lc, ax_cut):
        ax.text(0.5, 0.5, f'{label}: out of FOV', ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])


def generate_plot(gid, src, primary_data, comparisons, primary_ac,
                  primary_seg, mode, out_dir):
    """Generate multi-panel diagnostic PNG."""
    det = src['det']
    ra, dec = src['ra'], src['dec']
    idx = src['idx']
    period = src['period']
    ls_sig = src['ls_sig']
    amp = src['amp']

    n_rows = 1 + len(comparisons)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 3.5 * n_rows),
                             squeeze=False,
                             gridspec_kw={'width_ratios': [3, 1]})

    max_snr = src['snr']

    # Row 0: Primary detection LC from extraction HDF5
    pd = primary_data.get(det)
    if pd is not None:
        times = pd['times']
        flux = pd['flux_clipped'][idx, :]
        src_px = pd['sources']['px'][idx]
        src_py = pd['sources']['py'][idx]
        seg_snr = float(pd['sources']['det_snr'][idx])
        max_snr = max(max_snr, seg_snr)
        plot_lc_row(axes[0, 0], axes[0, 1], times, flux,
                    src_px, src_py, seg_snr,
                    f'{primary_seg} {mode}', primary_ac.get(det), det)
    else:
        for ax in axes[0]:
            ax.text(0.5, 0.5, f'{primary_seg}: no data', ha='center',
                    va='center', transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    # Rows 1..N: Comparison forced photometry panels
    for row_idx, comp in enumerate(comparisons, start=1):
        # Find which detector in this comparison covers the source
        found_det, proj_px, proj_py = find_detector_for_position(
            ra, dec, comp['wcs'], comp['cubes'])

        if found_det is None:
            blank_row(axes[row_idx, 0], axes[row_idx, 1], comp['label'])
            continue

        # Extract forced photometry from the cube
        flux_raw = forced_photometry(comp['cubes'][found_det], proj_px, proj_py)
        t_raw = comp['times'][found_det]

        # IQR clip
        chunk = IQR_CHUNK_RAMP if comp['comp_mode'] == 'ramp' else IQR_CHUNK_ZF
        good = np.isfinite(flux_raw)
        if np.sum(good) > chunk:
            flux_c, t_c = clip_outliers_iqr(flux_raw[good], t_raw[good], chunk)
        else:
            flux_c, t_c = flux_raw, t_raw

        ac_ref = comp['autocorr'].get(found_det)
        plot_lc_row(axes[row_idx, 0], axes[row_idx, 1], t_c, flux_c,
                    proj_px, proj_py, 0,
                    f'{comp["label"]} (forced)', ac_ref, found_det)

    fig.suptitle(
        f'src{gid:05d}  {det}  RA={ra:.5f} Dec={dec:.5f}  '
        f'P={period:.1f}min  LS={ls_sig:.1f}  amp={amp:.3f}',
        fontsize=10, y=1.02)
    fig.tight_layout()

    basename = (f'SNR{max_snr:07.1f}_src{gid:05d}_{det}_P{period:.0f}min_'
                f'LS{ls_sig:.0f}_amp{amp:.3f}_{ra:.5f}_{dec:.5f}.png')
    out_path = os.path.join(out_dir, basename)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ============================================================================
# Main processing
# ============================================================================

def process_segment(base_dir, target, segment, mode, detectors):
    """Process one segment: load sources, build comparisons, generate PNGs."""
    segments = TARGETS[target]['segments']
    is_lw = (detectors == ['nrcblong'])
    suffix = '_LW' if is_lw else ''

    # Output directory
    if len(segments) > 1:
        diag_dir = f'{base_dir}/diagnostics/{target}_{mode}_{segment}{suffix}'
    else:
        diag_dir = f'{base_dir}/diagnostics/{target}_{mode}{suffix}'
    fake_dir = f'{diag_dir}/FAKE'
    real_dir = f'{diag_dir}/REAL'
    os.makedirs(fake_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    # Load and dedup sources
    print(f'\n=== {target} {segment} {mode} {"LW" if is_lw else "SW"} ===')
    sources = load_segment_sources(base_dir, target, segment, mode, detectors)
    if not sources:
        print('  No sources, skipping')
        return

    # Primary extraction data
    print(f'Loading primary data ({segment} {mode})...')
    primary_data = load_extraction_data(base_dir, target, segment, mode, detectors)
    primary_ac = load_autocorr_refs(base_dir, target, segment, mode, detectors)
    print(f'  {len(primary_data)} detectors, {len(primary_ac)} autocorr refs')

    # Build and load comparison panels
    panel_specs = build_comparison_panels(target, segment, mode, is_lw, segments)
    comparisons = []
    for seg, comp_mode, dets, label in panel_specs:
        print(f'Loading comparison: {label} ({len(dets)} det(s))...')
        cubes, times = load_cubes(base_dir, target, seg, comp_mode, dets)
        ac = load_autocorr_refs(base_dir, target, seg, comp_mode, dets)
        wcs_d = load_wcs_refs(base_dir, target, seg, dets)
        comparisons.append({
            'cubes': cubes, 'times': times, 'autocorr': ac, 'wcs': wcs_d,
            'label': label, 'comp_mode': comp_mode,
        })
        print(f'  {len(cubes)} cubes, {len(ac)} autocorr, {len(wcs_d)} WCS')

    # Generate PNGs
    t0 = timer.time()
    n_done = 0
    n_err = 0
    n_total = len(sources)

    for i, src in enumerate(sources):
        try:
            generate_plot(i, src, primary_data, comparisons, primary_ac,
                          segment, mode, fake_dir)
            n_done += 1
        except Exception as e:
            if n_err < 5:
                print(f'  Error src {i}: {e}')
            n_err += 1

        if (n_done + n_err) % 500 == 0:
            elapsed = timer.time() - t0
            rate = (n_done + n_err) / elapsed if elapsed > 0 else 0
            print(f'  {n_done + n_err}/{n_total}: {n_done} plotted, '
                  f'{n_err} errors, {rate:.0f}/s')

    # Close HDF5 handles
    for d in primary_data.values():
        d['handle'].close()

    elapsed = timer.time() - t0
    print(f'\nDone: {n_done} PNGs in {fake_dir}/ ({elapsed:.1f}s)')
    if n_err:
        print(f'  Errors: {n_err}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-panel diagnostic PNGs with SW+LW forced photometry')
    parser.add_argument('--target', required=True, choices=list(TARGETS.keys()))
    parser.add_argument('--mode', required=True, choices=['ramp', 'zf'])
    parser.add_argument('--segment', default=None, help='Single segment (default: all)')
    parser.add_argument('--channel', default='sw', choices=['sw', 'lw'],
                        help='Channel: sw (nrcb1-4) or lw (nrcblong)')
    parser.add_argument('--det', default=None,
                        help='Single detector override (e.g. nrcblong)')
    parser.add_argument('--base-dir', default=BASE_DIR)
    args = parser.parse_args()

    # Determine detectors from channel/det args
    if args.det:
        detectors = [args.det]
    elif args.channel == 'lw':
        detectors = ['nrcblong']
    else:
        detectors = SW_DETS

    segments = TARGETS[args.target]['segments']
    if args.segment:
        if args.segment not in segments:
            print(f"ERROR: {args.segment} not valid for {args.target}")
            sys.exit(1)
        segments = [args.segment]

    for seg in segments:
        process_segment(args.base_dir, args.target, seg, args.mode, detectors)


if __name__ == '__main__':
    main()
