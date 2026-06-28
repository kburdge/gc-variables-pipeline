"""Aperture photometry on the group-diff cube and IQR outlier clipping.

Ported from do_aperture_photometry / clip_outliers_iqr in jwst_utils.py.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

try:
    from photutils.aperture import CircularAperture, aperture_photometry
except ImportError:  # older photutils
    from photutils.aperture import CircularAperture
    from photutils.photometry import aperture_photometry


def load_cube(cube_path):
    """Memory-map a group-diff cube; return (cube, times_mjd)."""
    hdul = fits.open(cube_path, memmap=True)
    cube = hdul[0].data  # (nframes, ny, nx), memory-mapped
    times = hdul["DIFF_TIMES"].data["MID_BARY_MJD"].astype(np.float64)
    return cube, times


def aperture_lightcurves(cube, positions, ap_radius=1.5, high_flux_thresh=50000):
    """Aperture photometry at fixed positions on every frame.

    Returns flux array of shape (n_frames, n_sources). NaNs are zeroed per frame
    so the effective aperture is identical across time.
    """
    n_frames = cube.shape[0]
    aperture = CircularAperture(positions, r=ap_radius)
    hi_mask = cube[0] > high_flux_thresh
    flux = np.zeros((n_frames, len(positions)), dtype=float)
    for i in range(n_frames):
        frame = np.nan_to_num(np.asarray(cube[i]), nan=0.0)
        phot = aperture_photometry(frame, aperture, mask=hi_mask)
        flux[i] = phot["aperture_sum"]
    return flux


def clip_outliers_iqr(data, times, chunk_size=18, iqr_factor=2.0):
    """Chunked IQR outlier clip. Returns (clipped_data, clipped_times).

    Removes impulsive outliers (cosmic rays) locally within chunks while
    preserving smooth astrophysical variability across chunks.
    """
    data = np.asarray(data, dtype=float)
    times = np.asarray(times, dtype=float)
    n = len(data)
    n_chunks = n // chunk_size
    if n_chunks < 1:
        return data, times
    mask = np.ones(n, dtype=bool)
    for b in range(n_chunks):
        start = b * chunk_size
        end = n if b == n_chunks - 1 else (b + 1) * chunk_size  # merge final partial chunk
        seg = data[start:end]
        finite = seg[np.isfinite(seg)]
        if finite.size < 4:
            continue
        q1, q3 = np.percentile(finite, [25, 75])
        iqr = q3 - q1
        good = (seg >= q1 - iqr_factor * iqr) & (seg <= q3 + iqr_factor * iqr)
        mask[start:end] = good
    return data[mask], times[mask]
