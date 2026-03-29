#!/usr/bin/env python
"""
Rebuild master catalog with refined Gaussian centroids and fresh lightcurves.

For each source:
  1. Look up detector from mapping JSON
  2. Load autocorr image for that detector
  3. Fit 2D Gaussian to get sub-pixel centroid + uncertainty
  4. Re-extract raw lightcurves from groupdiff + ZF cubes at refined pixel
  5. Store everything in a fresh master catalog HDF5

Then run build_corrected_catalog.py separately for sat/slope corrections.

Usage:
    python rebuild_master_catalog.py
"""
import os
import sys
import re
import json
import time
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from photutils.aperture import CircularAperture
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
BACKUP = f'{BASE}_backup/catalogs/master_variable_catalog.h5'
OUTPUT = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING_PATH = f'{BASE}/catalogs/master_source_mapping.json'
REFS_DIR = f'{BASE}/refs'
ASTROM_DIR = f'{BASE}/astrometry'

AP_RADIUS = 1.5
CUTOUT_HALF = 2
FIT_RADIUS = 4

TARGETS = {
    'Liller1': {'segments': ['Segment3', 'Segment4']},
    'Terzan5': {'segments': ['Segment2']},
}
SW_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
ALL_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']


def build_aperture_mask():
    ap = CircularAperture([(CUTOUT_HALF, CUTOUT_HALF)], r=AP_RADIUS)
    mask = ap.to_mask(method='exact')[0]
    return mask.to_image((2 * CUTOUT_HALF + 1, 2 * CUTOUT_HALF + 1)).astype(np.float32)


def gauss2d(coords, amp, x0, y0, sigma, bg):
    x, y = coords
    return (amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + bg).ravel()


def refine_centroid(img, ix, iy, R=FIT_RADIUS):
    ny, nx = img.shape
    if ix - R < 0 or ix + R + 1 > nx or iy - R < 0 or iy + R + 1 > ny:
        return float(ix), float(iy), np.nan, np.nan, False

    cutout = img[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(float)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]

    bg_est = np.median(np.concatenate([cutout[0, :], cutout[-1, :],
                                        cutout[:, 0], cutout[:, -1]]))
    pk_y, pk_x = np.unravel_index(np.argmax(cutout), cutout.shape)
    amp_est = cutout[pk_y, pk_x] - bg_est
    if amp_est <= 0:
        return float(ix), float(iy), np.nan, np.nan, False

    try:
        popt, pcov = curve_fit(
            gauss2d, (xx, yy), cutout.ravel(),
            p0=[amp_est, pk_x - R, pk_y - R, 1.0, bg_est],
            bounds=([0, -R, -R, 0.3, -np.inf], [np.inf, R, R, 5.0, np.inf]),
            maxfev=2000)
        perr = np.sqrt(np.diag(pcov))
        fit_x = ix + popt[1]
        fit_y = iy + popt[2]
        if abs(popt[1]) > R or abs(popt[2]) > R or perr[1] > 2 or perr[2] > 2:
            return float(ix), float(iy), np.nan, np.nan, False
        return fit_x, fit_y, perr[1], perr[2], True
    except Exception:
        return float(ix), float(iy), np.nan, np.nan, False


def extract_lc(cube, ix, iy, ap_mask, H=CUTOUT_HALF):
    nf, ny, nx = cube.shape
    if ix - H < 0 or ix + H + 1 > nx or iy - H < 0 or iy + H + 1 > ny:
        return np.full(nf, np.nan, dtype=np.float32)
    cutout = np.array(cube[:, iy - H:iy + H + 1, ix - H:ix + H + 1], dtype=np.float32)
    np.nan_to_num(cutout, nan=0.0, copy=False)
    return np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))


def main():
    t0_global = time.time()
    ap_mask = build_aperture_mask()

    # Load mapping
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    print(f'Loaded mapping: {len(mapping)} sources')

    # Load backup catalog structure
    print(f'Loading backup catalog...')
    with h5py.File(BACKUP, 'r') as f:
        backup_data = {}
        for target in TARGETS:
            if target not in f:
                continue
            backup_data[target] = {
                'sources': f[f'{target}/sources'][:],
            }
            # Load times (needed for output)
            backup_data[target]['times'] = {}
            for seg in TARGETS[target]['segments']:
                for mode in ['ramp', 'zf']:
                    for det in ALL_DETS:
                        path = f'{target}/times/{seg}/{mode}/{det}'
                        if path in f:
                            backup_data[target]['times'][(seg, mode, det)] = f[path][:]

    # ================================================================
    # Phase 1: Refine centroids for all sources
    # ================================================================
    print(f'\n{"="*60}')
    print(f'Phase 1: Refining centroids')
    print(f'{"="*60}')

    # Cache autocorr images
    autocorr_cache = {}

    all_refined = {}  # mid -> {det, det_px, det_py, refined_px, refined_py, err_x, err_y}

    for target in TARGETS:
        if target not in backup_data:
            continue
        src = backup_data[target]['sources']
        segments = TARGETS[target]['segments']
        # Use first segment for autocorr detection reference
        seg = segments[0]

        n_refined = 0
        n_failed = 0

        for i in range(len(src)):
            mid = int(src[i]['master_id'])
            if mid >= len(mapping):
                continue

            # Find SW detector from mapping
            m = mapping[mid]
            det = None
            fn = None
            for dk, dv in m.get('detections', {}).items():
                if target in dk and '_LW' not in dk:
                    fn = dv.get('filename', '')
                    match = re.search(r'(nrcb[1-4])', fn)
                    if match:
                        det = match.group(1)
                    break

            if det is None:
                n_failed += 1
                continue

            # Find original detection pixel from extraction H5
            fn_match = re.match(
                r'SNR[\d.]+_src\d+_(nrcb[1-4])_P\d+min_LS(\d+)_amp[\d.]+_([\d.]+)_([-\d.]+)\.png$', fn)
            if not fn_match:
                n_failed += 1
                continue

            fn_ls = int(fn_match.group(2))
            fn_ra = float(fn_match.group(3))
            fn_dec = float(fn_match.group(4))

            # Find in extraction H5
            ext_path = f'{BASE}/extraction/{target}/{seg}/{det}_ramp.h5'
            if not os.path.exists(ext_path):
                # Try other segments
                for s in segments[1:]:
                    ext_path = f'{BASE}/extraction/{target}/{s}/{det}_ramp.h5'
                    if os.path.exists(ext_path):
                        seg = s
                        break

            if not os.path.exists(ext_path):
                n_failed += 1
                continue

            with h5py.File(ext_path, 'r') as ef:
                ext = ef['sources'][:]
            dist = np.sqrt((ext['ra'] - fn_ra)**2 + (ext['dec'] - fn_dec)**2)
            candidates = np.where(dist < 0.001)[0]
            if len(candidates) == 0:
                candidates = [np.argmin(dist)]
            best = candidates[0]
            for c in candidates:
                if abs(ext[c]['ls_significance'] - fn_ls) < 1:
                    best = c
                    break

            orig_px = int(ext[best]['px'])
            orig_py = int(ext[best]['py'])

            # Load autocorr image
            ac_key = (target, seg, det)
            if ac_key not in autocorr_cache:
                ac_path = f'{REFS_DIR}/{target}_{seg}_{det}_autocorr.fits'
                if os.path.exists(ac_path):
                    autocorr_cache[ac_key] = fits.getdata(ac_path)
                else:
                    autocorr_cache[ac_key] = None

            ac_img = autocorr_cache[ac_key]
            if ac_img is None:
                all_refined[mid] = {
                    'target': target, 'det': det, 'seg': seg,
                    'det_px': orig_px, 'det_py': orig_py,
                    'refined_px': float(orig_px), 'refined_py': float(orig_py),
                    'err_x': np.nan, 'err_y': np.nan, 'success': False}
                n_failed += 1
                continue

            # Gaussian refinement
            fit_x, fit_y, err_x, err_y, success = refine_centroid(ac_img, orig_px, orig_py)
            all_refined[mid] = {
                'target': target, 'det': det, 'seg': seg,
                'det_px': orig_px, 'det_py': orig_py,
                'refined_px': fit_x, 'refined_py': fit_y,
                'err_x': err_x, 'err_y': err_y, 'success': success}

            if success:
                n_refined += 1
            else:
                n_failed += 1

        print(f'  {target}: {n_refined} refined, {n_failed} failed/skipped')

    # ================================================================
    # Phase 2: Build new master catalog with refined positions + fresh LCs
    # ================================================================
    print(f'\n{"="*60}')
    print(f'Phase 2: Extracting fresh lightcurves at refined positions')
    print(f'{"="*60}')

    with h5py.File(OUTPUT, 'w') as out:
        for target in TARGETS:
            if target not in backup_data:
                continue

            segments = TARGETS[target]['segments']
            old_src = backup_data[target]['sources']
            n = len(old_src)

            # New source dtype with centroid columns
            new_dt = np.dtype([
                ('master_id', 'i4'), ('ra', 'f8'), ('dec', 'f8'),
                ('best_snr', 'f4'), ('n_detections', 'i4'),
                ('detector', 'S8'),
                ('det_px', 'f4'), ('det_py', 'f4'),
                ('refined_px', 'f4'), ('refined_py', 'f4'),
                ('refined_px_err', 'f4'), ('refined_py_err', 'f4'),
            ])
            new_src = np.zeros(n, dtype=new_dt)
            for field in ['master_id', 'best_snr', 'n_detections']:
                new_src[field] = old_src[field]

            # Fill positions
            for i in range(n):
                mid = int(old_src[i]['master_id'])
                if mid in all_refined:
                    r = all_refined[mid]
                    new_src[i]['detector'] = r['det']
                    new_src[i]['det_px'] = r['det_px']
                    new_src[i]['det_py'] = r['det_py']
                    new_src[i]['refined_px'] = r['refined_px']
                    new_src[i]['refined_py'] = r['refined_py']
                    new_src[i]['refined_px_err'] = r['err_x']
                    new_src[i]['refined_py_err'] = r['err_y']

                    # Compute RA/Dec from refined pixel
                    det = r['det']
                    seg = r['seg']
                    # For Terzan5: use LW-aligned WCS if available
                    lw_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_lw.fits'
                    ref_path = f'{REFS_DIR}/{target}_{seg}_{det}_ref.fits'
                    if target == 'Terzan5' and os.path.exists(lw_path):
                        wcs = WCS(fits.getheader(lw_path))
                    elif os.path.exists(ref_path):
                        wcs = WCS(fits.getheader(ref_path))
                    else:
                        new_src[i]['ra'] = old_src[i]['ra']
                        new_src[i]['dec'] = old_src[i]['dec']
                        continue

                    sky = wcs.pixel_to_world(r['refined_px'], r['refined_py'])
                    new_src[i]['ra'] = float(sky.ra.deg)
                    new_src[i]['dec'] = float(sky.dec.deg)
                else:
                    # Fallback: keep backup position
                    new_src[i]['ra'] = old_src[i]['ra']
                    new_src[i]['dec'] = old_src[i]['dec']

            # Safety check: if position moved >0.5" from backup, revert
            # (likely a bad extraction match or failed centroid)
            if mid in all_refined:
                dra = (new_src[i]['ra'] - old_src[i]['ra']) * np.cos(np.radians(new_src[i]['dec'])) * 3600
                ddec = (new_src[i]['dec'] - old_src[i]['dec']) * 3600
                if np.sqrt(dra**2 + ddec**2) > 0.5:
                    new_src[i]['ra'] = old_src[i]['ra']
                    new_src[i]['dec'] = old_src[i]['dec']

            out.create_dataset(f'{target}/sources', data=new_src)

            # Write times
            for (seg, mode, det), t in backup_data[target]['times'].items():
                out.create_dataset(f'{target}/times/{seg}/{mode}/{det}', data=t)

            # Extract lightcurves at refined positions
            for seg in segments:
                for mode in ['ramp', 'zf']:
                    cube_prefix = 'groupdiffs' if mode == 'ramp' else 'zeroframes'
                    for det in ALL_DETS:
                        cube_path = f'{REFS_DIR}/{cube_prefix}_{target}_{seg}_{det}.fits'
                        if not os.path.exists(cube_path):
                            continue

                        t1 = time.time()
                        with fits.open(cube_path, memmap=True) as hdul:
                            cube = hdul[0].data
                            nf = cube.shape[0]

                        flux = np.full((n, nf), np.nan, dtype=np.float32)

                        for i in range(n):
                            mid = int(new_src[i]['master_id'])
                            if mid not in all_refined:
                                continue
                            r = all_refined[mid]

                            # For the detection detector, use refined position
                            if r['det'] == det:
                                ix = int(round(r['refined_px']))
                                iy = int(round(r['refined_py']))
                            else:
                                # For other detectors, project RA/Dec to pixel
                                ref_path = f'{REFS_DIR}/{target}_{seg}_{det}_ref.fits'
                                if not os.path.exists(ref_path):
                                    continue
                                w = WCS(fits.getheader(ref_path))
                                px, py = w.world_to_pixel_values(
                                    float(new_src[i]['ra']), float(new_src[i]['dec']))
                                ix = int(round(float(px)))
                                iy = int(round(float(py)))

                            H = CUTOUT_HALF
                            if (ix - H < 0 or ix + H + 1 > cube.shape[2] or
                                    iy - H < 0 or iy + H + 1 > cube.shape[1]):
                                continue
                            cutout = np.array(
                                cube[:, iy - H:iy + H + 1, ix - H:ix + H + 1],
                                dtype=np.float32)
                            np.nan_to_num(cutout, nan=0.0, copy=False)
                            flux[i] = np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))

                        out.create_dataset(
                            f'{target}/lightcurves/{seg}/{mode}/{det}',
                            data=flux, chunks=(1, nf),
                            compression='gzip', compression_opts=4)

                        dt = time.time() - t1
                        n_valid = np.sum(np.any(np.isfinite(flux) & (flux != 0), axis=1))
                        print(f'  {target}/{seg}/{mode}/{det}: {n_valid}/{n} sources, '
                              f'{nf} frames, {dt:.1f}s')

            print(f'  {target}: done ({n} sources)')

    # Preserve special_reduction from backup (these are manually curated)
    print(f'\nPreserving special_reduction from backup...')
    with h5py.File(BACKUP, 'r') as bk, h5py.File(OUTPUT, 'r+') as out:
        if 'special_reduction' in bk:
            bk.copy('special_reduction', out)
            n_special = 0
            def count_special(name, obj):
                nonlocal n_special
                if isinstance(obj, h5py.Group) and 'flux_norm' in obj:
                    n_special += 1
            bk['special_reduction'].visititems(count_special)
            print(f'  Copied {n_special} special reduction lightcurves')
        else:
            print(f'  No special_reduction in backup')

    # Update mapping JSON
    print(f'\nUpdating mapping JSON...')
    for mid, r in all_refined.items():
        if mid < len(mapping):
            mapping[mid]['ra'] = float(new_src[new_src['master_id'] == mid]['ra'][0]) \
                if mid in [int(s['master_id']) for s in new_src] else mapping[mid].get('ra', 0)
            mapping[mid]['dec'] = float(new_src[new_src['master_id'] == mid]['dec'][0]) \
                if mid in [int(s['master_id']) for s in new_src] else mapping[mid].get('dec', 0)

    with open(MAPPING_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)

    elapsed = time.time() - t0_global
    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'Output: {OUTPUT}')
    print(f'Next: run build_corrected_catalog.py to add sat/slope corrections')


if __name__ == '__main__':
    main()
