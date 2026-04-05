#!/usr/bin/env python
"""
Rebuild master catalog from REAL-sorted diagnostic folders.

Ground truth: PIXEL POSITIONS from extraction H5 files.
WCS is used ONLY for:
  - Cross-matching SW↔LW during dedup (LW-aligned WCS)
  - Cross-detector lightcurve extraction (calints WCS, the native frame)
  - Final catalog RA/Dec (LW-aligned WCS)

Steps:
  1. Parse REAL filenames → match to extraction H5 → pixel positions
  2. Build master mapping with 0.2" dedup (LW-aligned WCS for sky projection)
  3. Centroid refinement (Gaussian on autocorr, reject shifts > 0.2")
  4. Extract groupdiff lightcurves at pixel positions + IQR clip
  5. Compute final RA/Dec from refined pixels via LW-aligned WCS
  6. Then run build_corrected_catalog.py separately for sat/slope corrections

Usage:
    python rebuild_master_catalog_v2.py
"""
import os
import sys
import re
import json
import time
import numpy as np
import h5py
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from photutils.aperture import CircularAperture
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
REFS = f'{BASE}/refs'
ASTROM_DIR = f'{BASE}/astrometry'
EXTRACT_DIR = f'{BASE}/extraction'
DIAG_DIR = f'{BASE}/diagnostics'
OUTPUT = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING_PATH = f'{BASE}/catalogs/master_source_mapping.json'

AP_RADIUS = 1.5
H = 2  # cutout half-size
FIT_R = 4  # Gaussian fit radius
DEDUP_ARCSEC = 0.2

ALL_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']
SW_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
TARGETS = {'Liller1': ['Segment3', 'Segment4'], 'Terzan5': ['Segment2']}

FOLDER_MAP = {
    'Liller1_ramp_Segment3': ('Liller1', 'Segment3', 'ramp', 'sw'),
    'Liller1_ramp_Segment3_LW': ('Liller1', 'Segment3', 'ramp', 'lw'),
    'Liller1_ramp_Segment4': ('Liller1', 'Segment4', 'ramp', 'sw'),
    'Liller1_ramp_Segment4_LW': ('Liller1', 'Segment4', 'ramp', 'lw'),
    'Liller1_zf_Segment3': ('Liller1', 'Segment3', 'zf', 'sw'),
    'Liller1_zf_Segment3_LW': ('Liller1', 'Segment3', 'zf', 'lw'),
    'Liller1_zf_Segment4': ('Liller1', 'Segment4', 'zf', 'sw'),
    'Liller1_zf_Segment4_LW': ('Liller1', 'Segment4', 'zf', 'lw'),
    'Terzan5_ramp': ('Terzan5', 'Segment2', 'ramp', 'sw'),
    'Terzan5_ramp_LW': ('Terzan5', 'Segment2', 'ramp', 'lw'),
    'Terzan5_zf': ('Terzan5', 'Segment2', 'zf', 'sw'),
    'Terzan5_zf_LW': ('Terzan5', 'Segment2', 'zf', 'lw'),
}

ap = CircularAperture([(H, H)], r=AP_RADIUS)
ap_mask = ap.to_mask(method='exact')[0].to_image((2 * H + 1, 2 * H + 1)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def parse_filename(fname):
    m = re.match(
        r'SNR(\d+\.\d+)_src(\d+)_(nrc\w+)_P(\d+)min_LS(\d+)_amp([\d.]+)_'
        r'([\d.]+)_([-\d.]+)\.png$', fname)
    if not m:
        return None
    return {
        'snr': float(m.group(1)), 'src_id': int(m.group(2)),
        'det': m.group(3), 'period_min': float(m.group(4)),
        'ls_sig': float(m.group(5)), 'amplitude': float(m.group(6)),
        'ra': float(m.group(7)), 'dec': float(m.group(8)),
    }


def gauss2d(coords, amp, x0, y0, sigma, bg):
    x, y = coords
    return (amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + bg).ravel()


def refine_centroid(img, ix, iy, R=FIT_R):
    ny, nx = img.shape
    if ix - R < 0 or ix + R + 1 > nx or iy - R < 0 or iy + R + 1 > ny:
        return float(ix), float(iy), np.nan, np.nan, False
    cutout = img[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(float)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    bg_est = np.median(np.concatenate([cutout[0, :], cutout[-1, :], cutout[:, 0], cutout[:, -1]]))
    pk_y, pk_x = np.unravel_index(np.argmax(cutout), cutout.shape)
    amp_est = cutout[pk_y, pk_x] - bg_est
    if amp_est <= 0:
        return float(ix), float(iy), np.nan, np.nan, False
    try:
        popt, pcov = curve_fit(gauss2d, (xx, yy), cutout.ravel(),
                               p0=[amp_est, pk_x - R, pk_y - R, 1.0, bg_est],
                               bounds=([0, -R, -R, 0.3, -np.inf], [np.inf, R, R, 5.0, np.inf]),
                               maxfev=2000)
        perr = np.sqrt(np.diag(pcov))
        if abs(popt[1]) > R or abs(popt[2]) > R or perr[1] > 2 or perr[2] > 2:
            return float(ix), float(iy), np.nan, np.nan, False
        return ix + popt[1], iy + popt[2], perr[1], perr[2], True
    except Exception:
        return float(ix), float(iy), np.nan, np.nan, False


def clip_iqr(data, times, chunk_size=18, iqr_factor=2.0):
    n = len(data)
    nc = n // chunk_size
    if nc < 1:
        return data, times
    mask = np.ones(n, dtype=bool)
    for b in range(nc):
        s, e = b * chunk_size, (b + 1) * chunk_size
        seg = data[s:e]
        q1, q3 = np.percentile(seg, [25, 75])
        iqr = q3 - q1
        mask[s:e] = (seg >= q1 - iqr_factor * iqr) & (seg <= q3 + iqr_factor * iqr)
    return data[mask], times[mask]


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Collect REAL sources with pixel positions from extraction H5
# ═══════════════════════════════════════════════════════════════════════

def collect_real_sources():
    """Parse all REAL folders, match to extraction H5, return list with pixel positions."""
    print('Step 1: Collecting REAL sources with pixel positions...')

    # Load extraction H5 data
    ext_data = {}
    for target, segs in TARGETS.items():
        for seg in segs:
            for det in ALL_DETS:
                path = f'{EXTRACT_DIR}/{target}/{seg}/{det}_ramp.h5'
                if os.path.exists(path):
                    with h5py.File(path, 'r') as f:
                        ext_data[(target, seg, det)] = f['sources'][:]
                path_zf = f'{EXTRACT_DIR}/{target}/{seg}/{det}_zf.h5'
                if os.path.exists(path_zf):
                    with h5py.File(path_zf, 'r') as f:
                        ext_data[(target, seg, det, 'zf')] = f['sources'][:]

    all_sources = []
    for folder_name, (target, seg, mode, channel) in FOLDER_MAP.items():
        real_dir = os.path.join(DIAG_DIR, folder_name, 'REAL')
        if not os.path.isdir(real_dir):
            continue

        dets_for_channel = ['nrcblong'] if channel == 'lw' else SW_DETS
        n_matched = 0

        for fname in os.listdir(real_dir):
            if not fname.endswith('.png'):
                continue
            info = parse_filename(fname)
            if info is None:
                continue

            det = info['det']
            fn_ra = info['ra']
            fn_dec = info['dec']
            fn_ls = info['ls_sig']

            # Find in extraction H5 by RA/Dec + LS match
            ext_key = (target, seg, det) if mode == 'ramp' else (target, seg, det, 'zf')
            if ext_key not in ext_data:
                # Try ramp extraction for ZF sources (they share the same pixel grid)
                ext_key = (target, seg, det)
                if ext_key not in ext_data:
                    continue

            ext = ext_data[ext_key]
            dist = np.sqrt((ext['ra'] - fn_ra)**2 + (ext['dec'] - fn_dec)**2)
            candidates = np.where(dist < 0.001)[0]
            if len(candidates) == 0:
                candidates = [np.argmin(dist)]

            best = candidates[0]
            for c in candidates:
                ls_field = 'ls_significance' if 'ls_significance' in ext.dtype.names else 'ls_sig'
                if abs(ext[c][ls_field] - fn_ls) < 1:
                    best = c
                    break

            px = int(ext[best]['px'])
            py = int(ext[best]['py'])

            all_sources.append({
                'folder': folder_name,
                'target': target,
                'segment': seg,
                'mode': mode,
                'channel': channel,
                'det': det,
                'px': px,
                'py': py,
                'ra': fn_ra,
                'dec': fn_dec,
                'snr': info['snr'],
                'ls_sig': fn_ls,
                'filename': fname,
            })
            n_matched += 1

        print(f'  {folder_name}/REAL: {n_matched} matched')

    print(f'  Total: {len(all_sources)} sources with pixel positions')
    return all_sources


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Build master mapping with dedup
# ═══════════════════════════════════════════════════════════════════════

def build_mapping(all_sources):
    """Dedup using LW-aligned WCS for sky projection, 0.2" radius."""
    print(f'\nStep 2: Building mapping (dedup at {DEDUP_ARCSEC}")')

    # Load LW-aligned WCS for sky projection during dedup
    lw_wcs = {}
    for target, segs in TARGETS.items():
        for seg in segs:
            for det in ALL_DETS:
                path = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_lw.fits'
                if os.path.exists(path):
                    lw_wcs[(target, seg, det)] = WCS(fits.getheader(path))
                else:
                    # Fallback to calints WCS
                    cal_files = sorted([f for f in glob.glob(
                        f'/data/JWST/{target}/{seg}/*{det}*cal.fits') if 'uncal' not in f])
                    if not cal_files:
                        cal_files = sorted(glob.glob(
                            f'/data/JWST/{target}/{seg}/calints/*{det}*calints.fits'))
                    if cal_files:
                        lw_wcs[(target, seg, det)] = WCS(fits.getheader(cal_files[0], 'SCI'))

    # Convert each source's pixel to sky using LW-aligned WCS
    for src in all_sources:
        key = (src['target'], src['segment'], src['det'])
        if key in lw_wcs:
            sky = lw_wcs[key].pixel_to_world(float(src['px']), float(src['py']))
            src['sky_ra'] = float(sky.ra.deg)
            src['sky_dec'] = float(sky.dec.deg)
        else:
            src['sky_ra'] = src['ra']
            src['sky_dec'] = src['dec']

    # Sort by SNR descending
    all_sources.sort(key=lambda s: -s['snr'])

    # Dedup
    master = []
    master_sc = None

    for src in all_sources:
        matched_idx = None
        if master_sc is not None and len(master_sc) > 0:
            src_sc = SkyCoord(ra=src['sky_ra'] * u.deg, dec=src['sky_dec'] * u.deg)
            same_target = np.array([m['target'] == src['target'] for m in master])
            if same_target.any():
                seps = src_sc.separation(master_sc).arcsec
                seps[~same_target] = 999
                best = np.argmin(seps)
                if seps[best] < DEDUP_ARCSEC:
                    matched_idx = best

        if matched_idx is not None:
            m = master[matched_idx]
            if src['folder'] not in m['detections']:
                m['detections'][src['folder']] = {
                    'snr': src['snr'], 'filename': src['filename'],
                    'det': src['det'], 'segment': src['segment'],
                    'px': src['px'], 'py': src['py'],
                }
                m['n_folders'] = len(m['detections'])
                if src['snr'] > m['best_snr']:
                    m['best_snr'] = src['snr']
        else:
            master.append({
                'target': src['target'],
                'sky_ra': src['sky_ra'], 'sky_dec': src['sky_dec'],
                'best_snr': src['snr'], 'n_folders': 1,
                'detections': {
                    src['folder']: {
                        'snr': src['snr'], 'filename': src['filename'],
                        'det': src['det'], 'segment': src['segment'],
                        'px': src['px'], 'py': src['py'],
                    }
                },
            })
            master_sc = SkyCoord(
                ra=[m['sky_ra'] for m in master] * u.deg,
                dec=[m['sky_dec'] for m in master] * u.deg)

    # Assign IDs
    master.sort(key=lambda m: (-{'Liller1': 0, 'Terzan5': 1}[m['target']], -m['best_snr']))
    for i, m in enumerate(master):
        m['master_id'] = i

    # Save mapping
    # Convert to JSON-serializable (include pixel positions)
    mapping_out = []
    for m in master:
        entry = {
            'master_id': m['master_id'],
            'target': m['target'],
            'ra': m['sky_ra'],
            'dec': m['sky_dec'],
            'best_snr': m['best_snr'],
            'n_folders': m['n_folders'],
            'detections': m['detections'],
        }
        mapping_out.append(entry)

    with open(MAPPING_PATH, 'w') as f:
        json.dump(mapping_out, f, indent=2)

    by_target = {}
    for m in master:
        by_target.setdefault(m['target'], []).append(m)
    for t, sources in by_target.items():
        print(f'  {t}: {len(sources)} sources')
    print(f'  Total: {len(master)} master entries')

    return master, by_target


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Build HDF5 with centroids + lightcurves
# ═══════════════════════════════════════════════════════════════════════

def build_catalog(by_target):
    """Build master HDF5: centroids, groupdiff extraction, IQR clip."""
    print('\nStep 3: Building master catalog...')

    # Load autocorr images for centroid refinement (both ramp and ZF)
    autocorr = {}
    zf_autocorr = {}
    for target, segs in TARGETS.items():
        for seg in segs:
            for det in ALL_DETS:
                path = f'{REFS}/{target}_{seg}_{det}_autocorr.fits'
                if os.path.exists(path):
                    autocorr[(target, seg, det)] = fits.getdata(path)
                zf_path = f'{REFS}/{target}_{seg}_{det}_zf_autocorr.fits'
                if os.path.exists(zf_path):
                    zf_autocorr[(target, seg, det)] = fits.getdata(zf_path)

    # Load LW-aligned WCS (for cross-detector projection, dedup, and final RA/Dec)
    lw_wcs = {}
    for target, segs in TARGETS.items():
        for seg in segs:
            for det in ALL_DETS:
                path = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_lw.fits'
                if os.path.exists(path):
                    lw_wcs[(target, seg, det)] = WCS(fits.getheader(path))
                else:
                    # Fallback: Gaia-corrected LW WCS
                    path2 = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_gaia.fits'
                    if os.path.exists(path2):
                        lw_wcs[(target, seg, det)] = WCS(fits.getheader(path2))

    first_target = True
    for target, src_list in by_target.items():
        segs = TARGETS[target]
        n_t = len(src_list)
        print(f'\n  === {target}: {n_t} sources ===')

        dt = np.dtype([
            ('master_id', 'i4'), ('ra', 'f8'), ('dec', 'f8'),
            ('best_snr', 'f4'), ('n_detections', 'i4'),
            ('detector', 'S8'),
            ('det_px', 'f4'), ('det_py', 'f4'),
            ('refined_px', 'f4'), ('refined_py', 'f4'),
            ('refined_px_err', 'f4'), ('refined_py_err', 'f4'),
        ])
        src_arr = np.zeros(n_t, dtype=dt)

        # Per-source: primary detection pixel + detector + segment
        primary_info = []  # (det, seg, px, py) for each source

        # ── Phase A: Find pixels + refine centroids ──
        n_refined = 0
        n_failed = 0
        for i, m in enumerate(src_list):
            src_arr[i]['master_id'] = m['master_id']
            src_arr[i]['best_snr'] = m['best_snr']
            src_arr[i]['n_detections'] = m['n_folders']

            # Find primary detection for centroiding.
            # For bright sources (ZF SNR>10), prefer ZF centroid since
            # ramp pixels may be saturated/masked, offsetting the centroid.
            # Priority: SW ZF (SNR>10) > LW ZF (SNR>10) > SW ramp > SW ZF > LW
            primary = None

            # Check for bright ZF detection (SNR>10)
            sw_zf = None
            lw_zf = None
            for dk, dv in m['detections'].items():
                if '_LW' not in dk and 'zf' in dk and dv.get('snr', 0) > 10:
                    if sw_zf is None or dv['snr'] > sw_zf['snr']:
                        sw_zf = dv
                elif '_LW' in dk and 'zf' in dk and dv.get('snr', 0) > 10:
                    if lw_zf is None or dv['snr'] > lw_zf['snr']:
                        lw_zf = dv

            primary_is_zf = False
            if sw_zf is not None:
                primary = sw_zf
                primary_is_zf = True
            elif lw_zf is not None:
                primary = lw_zf
                primary_is_zf = True
            else:
                # Fall back to ramp detections
                for dk, dv in m['detections'].items():
                    if '_LW' not in dk and 'ramp' in dk:
                        primary = dv
                        break
                if primary is None:
                    for dk, dv in m['detections'].items():
                        if '_LW' not in dk and 'zf' in dk:
                            primary = dv
                            break
                if primary is None:
                    # LW-only: use LW detection
                    for dk, dv in m['detections'].items():
                        if '_LW' in dk:
                            primary = dv
                            break

            if primary is None:
                primary_info.append(None)
                src_arr[i]['ra'] = m['sky_ra']
                src_arr[i]['dec'] = m['sky_dec']
                n_failed += 1
                continue

            det = primary['det']
            seg = primary['segment']
            px = primary['px']
            py = primary['py']

            src_arr[i]['detector'] = det
            src_arr[i]['det_px'] = px
            src_arr[i]['det_py'] = py
            primary_info.append((det, seg, px, py))

            # Centroid refinement: use ZF autocorr for ZF-selected sources,
            # ramp autocorr otherwise. Prevents bright ramp neighbors from
            # pulling the centroid away from the actual ZF source.
            ac_key = (target, seg, det)
            if primary_is_zf and ac_key in zf_autocorr:
                ac_img = zf_autocorr[ac_key]
            elif ac_key in autocorr:
                ac_img = autocorr[ac_key]
            else:
                ac_img = None
            if ac_img is not None:
                fit_x, fit_y, err_x, err_y, success = refine_centroid(ac_img, px, py)
                # Reject if shift > ~6.5 px (0.2" at 31 mas/px)
                shift_px = np.sqrt((fit_x - px)**2 + (fit_y - py)**2)
                pix_scale = 0.063 if det == 'nrcblong' else 0.031
                shift_arcsec = shift_px * pix_scale
                if success and shift_arcsec < 0.2:
                    src_arr[i]['refined_px'] = fit_x
                    src_arr[i]['refined_py'] = fit_y
                    src_arr[i]['refined_px_err'] = err_x
                    src_arr[i]['refined_py_err'] = err_y
                    n_refined += 1
                else:
                    src_arr[i]['refined_px'] = px
                    src_arr[i]['refined_py'] = py
                    n_failed += 1
            else:
                src_arr[i]['refined_px'] = px
                src_arr[i]['refined_py'] = py
                n_failed += 1

            # Final RA/Dec from refined pixel via LW-aligned WCS
            wcs_key = (target, seg, det)
            use_px = float(src_arr[i]['refined_px'])
            use_py = float(src_arr[i]['refined_py'])
            if wcs_key in lw_wcs:
                sky = lw_wcs[wcs_key].pixel_to_world(use_px, use_py)
                src_arr[i]['ra'] = float(sky.ra.deg)
                src_arr[i]['dec'] = float(sky.dec.deg)
            else:
                src_arr[i]['ra'] = m['sky_ra']
                src_arr[i]['dec'] = m['sky_dec']

        print(f'  Centroids: {n_refined} refined, {n_failed} fallback')

        # ── Phase B: Extract groupdiff + ZF lightcurves ──
        print(f'  Extracting lightcurves...')
        lc_data = {}
        time_data = {}

        for seg in segs:
            for mode in ['ramp', 'zf']:
                cube_prefix = 'groupdiffs' if mode == 'ramp' else 'zeroframes'
                chunk_size = 18 if mode == 'ramp' else 4
                for det in ALL_DETS:
                    cube_path = f'{REFS}/{cube_prefix}_{target}_{seg}_{det}.fits'
                    if not os.path.exists(cube_path):
                        continue

                    cube = fits.getdata(cube_path, memmap=True)
                    nf, ny, nx = cube.shape
                    with fits.open(cube_path) as hdul:
                        times_mjd = np.array(hdul['DIFF_TIMES'].data['MID_BARY_MJD'],
                                             dtype=np.float64)

                    flux = np.zeros((n_t, nf), dtype=np.float32)
                    n_ext = 0

                    for j in range(n_t):
                        if primary_info[j] is None:
                            continue

                        p_det, p_seg, p_px, p_py = primary_info[j]
                        use_px = float(src_arr[j]['refined_px'])
                        use_py = float(src_arr[j]['refined_py'])

                        if p_det == det and p_seg == seg:
                            # Same detector + same segment: use pixel directly
                            ix = int(round(use_px))
                            iy = int(round(use_py))
                        elif p_det == det:
                            # Same detector, different segment: project through WCS
                            src_wcs = lw_wcs.get((target, p_seg, p_det))
                            det_wcs = lw_wcs.get((target, seg, det))
                            if src_wcs is None or det_wcs is None:
                                ix = int(round(use_px))
                                iy = int(round(use_py))
                            else:
                                sky = src_wcs.pixel_to_world(use_px, use_py)
                                px_d, py_d = det_wcs.world_to_pixel_values(
                                    sky.ra.deg, sky.dec.deg)
                                ix = int(round(float(px_d)))
                                iy = int(round(float(py_d)))
                        else:
                            # Different detector: project via LW-aligned WCS
                            src_wcs = lw_wcs.get((target, p_seg, p_det))
                            det_wcs = lw_wcs.get((target, seg, det))
                            if src_wcs is None or det_wcs is None:
                                continue
                            sky = src_wcs.pixel_to_world(use_px, use_py)
                            px_d, py_d = det_wcs.world_to_pixel_values(
                                sky.ra.deg, sky.dec.deg)
                            ix = int(round(float(px_d)))
                            iy = int(round(float(py_d)))

                        if ix - H < 0 or ix + H + 1 > nx or iy - H < 0 or iy + H + 1 > ny:
                            continue

                        cutout = np.array(cube[:, iy - H:iy + H + 1, ix - H:ix + H + 1],
                                          dtype=np.float32)
                        np.nan_to_num(cutout, nan=0.0, copy=False)
                        raw_flux = np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))

                        # IQR clip and store
                        valid = (raw_flux != 0) & np.isfinite(raw_flux)
                        if valid.sum() < 20:
                            continue
                        # Set clipped points to 0 in the fixed-size array
                        t_hr = (times_mjd - times_mjd[0]) * 24.0
                        f_v = raw_flux[valid]
                        t_v = t_hr[valid]
                        f_c, t_c = clip_iqr(f_v, t_v, chunk_size, 2.0)
                        f_c, t_c = clip_iqr(f_c, t_c, chunk_size, 2.0)
                        # Keep clipped version in full array (set rejected to 0)
                        keep_times = set(np.round(t_c, 8))
                        for k in range(nf):
                            if round(t_hr[k], 8) in keep_times:
                                flux[j, k] = raw_flux[k]
                            # else stays 0

                        n_ext += 1

                    lc_data[(seg, mode, det)] = flux
                    time_data[(seg, mode, det)] = times_mjd
                    print(f'    {seg}/{mode}/{det}: {n_ext}/{n_t}')

        # ── Phase C: Write HDF5 ──
        print(f'  Writing HDF5...')
        h5_mode = 'w' if first_target else 'r+'
        first_target = False

        with h5py.File(OUTPUT, h5_mode) as f:
            if f'{target}/sources' in f:
                del f[f'{target}/sources']
            f.create_dataset(f'{target}/sources', data=src_arr)
            for (seg, mode, det), flux in lc_data.items():
                f.create_dataset(f'{target}/lightcurves/{seg}/{mode}/{det}', data=flux)
            for (seg, mode, det), t in time_data.items():
                f.create_dataset(f'{target}/times/{seg}/{mode}/{det}', data=t)

        print(f'  {target} done')

    # Restore special_reduction from backup
    print('\nRestoring special_reduction from backup...')
    backup_path = f'{BASE}_backup/catalogs/master_variable_catalog.h5'
    if os.path.exists(backup_path):
        with h5py.File(backup_path, 'r') as bk, h5py.File(OUTPUT, 'r+') as f:
            if 'special_reduction' in bk:
                if 'special_reduction' in f:
                    del f['special_reduction']
                bk.copy('special_reduction', f)
                print('  Restored')


def load_existing_mapping():
    """Load mapping from JSON and convert to the format build_catalog expects."""
    print('Loading existing mapping...')
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)

    by_target = {}
    for m in mapping:
        target = m['target']
        # Find primary detection (highest SNR SW, fallback to ZF/LW)
        primary = None
        for dk, dv in m['detections'].items():
            if '_LW' not in dk and 'ramp' in dk and dv.get('px', 0) != 0:
                if primary is None or dv['snr'] > primary['snr']:
                    primary = dv
        if primary is None:
            for dk, dv in m['detections'].items():
                if '_LW' not in dk and 'zf' in dk and dv.get('px', 0) != 0:
                    if primary is None or dv['snr'] > primary['snr']:
                        primary = dv
        if primary is None:
            for dk, dv in m['detections'].items():
                if '_LW' in dk and dv.get('px', 0) != 0:
                    if primary is None or dv['snr'] > primary['snr']:
                        primary = dv

        entry = {
            'master_id': m['master_id'],
            'target': target,
            'sky_ra': m['ra'],
            'sky_dec': m['dec'],
            'best_snr': m['best_snr'],
            'n_folders': m['n_folders'],
            'detections': m['detections'],
        }
        by_target.setdefault(target, []).append(entry)

    for t, sources in by_target.items():
        print(f'  {t}: {len(sources)} sources')
    return by_target


def main():
    t0 = time.time()

    use_existing = '--use-existing-mapping' in sys.argv

    if use_existing:
        by_target = load_existing_mapping()
    else:
        all_sources = collect_real_sources()
        master, by_target = build_mapping(all_sources)

    build_catalog(by_target)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed / 60:.1f} min')
    print(f'Output: {OUTPUT}')
    print(f'Mapping: {MAPPING_PATH}')
    print(f'\nNext steps:')
    print(f'  1. Run build_corrected_catalog.py for sat/slope corrections')
    print(f'  2. Remap special_reduction IDs if needed')
    print(f'  3. Restart catalog server')


if __name__ == '__main__':
    main()
