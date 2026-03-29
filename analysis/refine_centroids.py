#!/usr/bin/env python
"""
Refine source centroids using 2D Gaussian fits on autocorr images.

For each source in the master catalog:
  1. Look up detector and original integer detection pixel from extraction H5
  2. Extract small cutout from autocorr image around that pixel
  3. Fit 2D Gaussian to get sub-pixel centroid + uncertainty
  4. Convert refined pixel position to RA/Dec using LW-aligned WCS
  5. Store original + refined positions in the master catalog

New columns added to master catalog:
  - det_px, det_py: original integer detection pixels
  - refined_px, refined_py: Gaussian-fit sub-pixel positions
  - refined_px_err, refined_py_err: fit uncertainties (pixels)
  - detector: detection detector name

Usage:
    python refine_centroids.py
"""
import os
import re
import json
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING = f'{BASE}/catalogs/master_source_mapping.json'

FIT_RADIUS = 4  # fit within ±4 pixels of peak


def gauss2d(coords, amp, x0, y0, sigma, bg):
    """2D circular Gaussian."""
    x, y = coords
    return (amp * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + bg).ravel()


def refine_centroid(img, ix, iy, R=FIT_RADIUS):
    """Fit 2D Gaussian around (ix, iy), return (x, y, x_err, y_err, sigma, success)."""
    ny, nx = img.shape
    if ix - R < 0 or ix + R + 1 > nx or iy - R < 0 or iy + R + 1 > ny:
        return float(ix), float(iy), np.nan, np.nan, np.nan, False

    cutout = img[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(float)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]

    # Initial guess
    peak_val = cutout[R, R]
    bg_est = np.median(np.concatenate([cutout[0, :], cutout[-1, :],
                                        cutout[:, 0], cutout[:, -1]]))
    amp_est = peak_val - bg_est
    if amp_est <= 0:
        return float(ix), float(iy), np.nan, np.nan, np.nan, False

    # Find the local max within the cutout to better initialize
    pk_y, pk_x = np.unravel_index(np.argmax(cutout), cutout.shape)
    x0_init = pk_x - R
    y0_init = pk_y - R

    try:
        popt, pcov = curve_fit(
            gauss2d, (xx, yy), cutout.ravel(),
            p0=[amp_est, x0_init, y0_init, 1.0, bg_est],
            bounds=([0, -R, -R, 0.3, -np.inf],
                    [np.inf, R, R, 5.0, np.inf]),
            maxfev=2000)
        perr = np.sqrt(np.diag(pcov))

        fit_x = ix + popt[1]
        fit_y = iy + popt[2]
        x_err = perr[1]
        y_err = perr[2]
        sigma = popt[3]

        # Sanity: reject if centroid moved too far or errors are huge
        if abs(popt[1]) > R or abs(popt[2]) > R or x_err > 2 or y_err > 2:
            return float(ix), float(iy), np.nan, np.nan, np.nan, False

        return fit_x, fit_y, x_err, y_err, sigma, True
    except Exception:
        return float(ix), float(iy), np.nan, np.nan, np.nan, False


def main():
    # Load mapping to get detector + filename for each source
    with open(MAPPING) as f:
        mapping = json.load(f)

    # Load extraction data per detector
    ext_data = {}
    for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']:
        h5_path = f'{BASE}/extraction/Terzan5/Segment2/{det}_ramp.h5'
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                ext_data[det] = f['sources'][:]

    # Load autocorr images and LW-aligned WCS
    autocorr = {}
    lw_wcs = {}
    for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']:
        ac_path = f'{BASE}/refs/Terzan5_Segment2_{det}_autocorr.fits'
        if os.path.exists(ac_path):
            autocorr[det] = fits.getdata(ac_path)
        wcs_path = f'{ASTROM_DIR}/Terzan5_Segment2_{det}_wcs_lw.fits'
        if os.path.exists(wcs_path):
            lw_wcs[det] = WCS(fits.getheader(wcs_path))

    # Load current master catalog
    with h5py.File(CATALOG, 'r') as f:
        old_src = f['Terzan5/sources'][:]
        # Also preserve lightcurves and times
        lc_data = {}
        times_data = {}
        for key in f['Terzan5/lightcurves/Segment2/ramp'].keys():
            lc_data[('ramp', key)] = f[f'Terzan5/lightcurves/Segment2/ramp/{key}'][:]
        for key in f['Terzan5/lightcurves/Segment2/zf'].keys():
            lc_data[('zf', key)] = f[f'Terzan5/lightcurves/Segment2/zf/{key}'][:]
        for key in f['Terzan5/times/Segment2/ramp'].keys():
            times_data[('ramp', key)] = f[f'Terzan5/times/Segment2/ramp/{key}'][:]
        for key in f['Terzan5/times/Segment2/zf'].keys():
            times_data[('zf', key)] = f[f'Terzan5/times/Segment2/zf/{key}'][:]
        # Preserve other targets by collecting all datasets recursively
        other_datasets = {}
        for tgt in f.keys():
            if tgt != 'Terzan5':
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        other_datasets[f'{tgt}/{name}'] = obj[:]
                f[tgt].visititems(collect)

    n = len(old_src)
    print(f'Processing {n} Terzan5 sources')

    # Build new dtype with additional columns
    new_fields = [
        ('master_id', 'i4'),
        ('ra', 'f8'), ('dec', 'f8'),
        ('best_snr', 'f4'), ('n_detections', 'i4'),
        ('detector', 'S8'),
        ('det_px', 'f4'), ('det_py', 'f4'),
        ('refined_px', 'f4'), ('refined_py', 'f4'),
        ('refined_px_err', 'f4'), ('refined_py_err', 'f4'),
    ]
    new_dt = np.dtype(new_fields)
    new_src = np.zeros(n, dtype=new_dt)

    # Copy existing fields
    for field in old_src.dtype.names:
        new_src[field] = old_src[field]

    n_refined = 0
    n_failed = 0
    offsets = []

    for i in range(n):
        mid = int(old_src[i]['master_id'])
        if mid >= len(mapping):
            continue

        m = mapping[mid]
        # Find SW detection
        det = None
        fn = None
        for dk, dv in m.get('detections', {}).items():
            if 'Terzan5' in dk and '_LW' not in dk:
                fn = dv.get('filename', '')
                match = re.search(r'(nrcb[1-4])', fn)
                if match:
                    det = match.group(1)
                break

        if det is None or det not in ext_data or det not in autocorr:
            continue

        new_src[i]['detector'] = det

        # Find original extraction pixel by matching filename RA/Dec + LS
        fn_match = re.match(
            r'SNR[\d.]+_src\d+_(nrcb[1-4])_P\d+min_LS(\d+)_amp[\d.]+_([\d.]+)_([-\d.]+)\.png$', fn)
        if not fn_match:
            continue
        fn_ls = int(fn_match.group(2))
        fn_ra = float(fn_match.group(3))
        fn_dec = float(fn_match.group(4))

        ext = ext_data[det]
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
        new_src[i]['det_px'] = orig_px
        new_src[i]['det_py'] = orig_py

        # Refine centroid
        fit_x, fit_y, x_err, y_err, sigma, success = refine_centroid(
            autocorr[det], orig_px, orig_py)

        new_src[i]['refined_px'] = fit_x
        new_src[i]['refined_py'] = fit_y
        new_src[i]['refined_px_err'] = x_err if success else np.nan
        new_src[i]['refined_py_err'] = y_err if success else np.nan

        # Update RA/Dec from refined position
        if success and det in lw_wcs:
            sky = lw_wcs[det].pixel_to_world(fit_x, fit_y)
            new_src[i]['ra'] = float(sky.ra.deg)
            new_src[i]['dec'] = float(sky.dec.deg)
            n_refined += 1
            offsets.append(np.sqrt((fit_x - orig_px)**2 + (fit_y - orig_py)**2))
        else:
            # Keep the integer-pixel position RA/Dec
            if det in lw_wcs:
                sky = lw_wcs[det].pixel_to_world(float(orig_px), float(orig_py))
                new_src[i]['ra'] = float(sky.ra.deg)
                new_src[i]['dec'] = float(sky.dec.deg)
            n_failed += 1

    offsets = np.array(offsets)
    print(f'\nRefined: {n_refined}, Failed: {n_failed}')
    print(f'Centroid shift: median={np.median(offsets):.3f} px, '
          f'mean={np.mean(offsets):.3f} px, max={np.max(offsets):.3f} px')

    # Write updated catalog
    with h5py.File(CATALOG, 'w') as f:
        f.create_dataset('Terzan5/sources', data=new_src)
        for (mode, det_key), lc in lc_data.items():
            f.create_dataset(f'Terzan5/lightcurves/Segment2/{mode}/{det_key}', data=lc)
        for (mode, det_key), t in times_data.items():
            f.create_dataset(f'Terzan5/times/Segment2/{mode}/{det_key}', data=t)
        # Restore other targets
        for ds_path, ds_data in other_datasets.items():
            f.create_dataset(ds_path, data=ds_data)

    print(f'\nWrote {CATALOG}')

    # Also update mapping JSON
    with open(MAPPING) as f:
        mapping = json.load(f)
    for i in range(n):
        mid = int(new_src[i]['master_id'])
        if mid < len(mapping):
            mapping[mid]['ra'] = float(new_src[i]['ra'])
            mapping[mid]['dec'] = float(new_src[i]['dec'])
    with open(MAPPING, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f'Updated {MAPPING}')

    # Verify
    print(f'\nVerification:')
    for sid in [1021, 364, 280]:
        s = new_src[new_src['master_id'] == sid]
        if len(s) == 0:
            continue
        s = s[0]
        print(f'  Source {sid} ({s["detector"].decode()}): '
              f'det=({s["det_px"]:.0f},{s["det_py"]:.0f}), '
              f'refined=({s["refined_px"]:.3f},{s["refined_py"]:.3f}), '
              f'err=({s["refined_px_err"]:.4f},{s["refined_py_err"]:.4f})')


if __name__ == '__main__':
    main()
