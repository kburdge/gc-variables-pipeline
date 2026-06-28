"""Detection: lag-1 autocorrelation reference image + PSF-matched filtering.

Variable sources are found on a *temporal autocorrelation* image rather than the
raw frames: for each pixel, the lag-1 autocorrelation of the calints time series
is ~0 for white (shot) noise but >>0 for genuinely variable sources, making
detection independent of source brightness. Sources are then located by a
PSF-matched filter (FFT convolution with the real NIRCam PSF) on that image.

Ported from create_autocorr_reference / fast_psf_detect / load_psf_kernel in the
original jwst_utils.py.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def find_calints(data_root, target, segment, detector):
    """Locate calints files for a detector (checks both calints/ layouts)."""
    candidates = []
    for sub in ("calints", os.path.join("detector1_output", "calints")):
        candidates += glob.glob(
            os.path.join(str(data_root), target, segment, sub, f"*_{detector}_calints.fits")
        )
    return sorted(set(candidates))


def create_autocorr_reference(calints_files, output_path=None):
    """Lag-1 temporal autocorrelation image from a list of calints files.

    autocorr[pix] = sum_t r[t] r[t+1] / sum_t r[t]^2,   r[t] = f[t] - mean_t(f).
    """
    cubes = []
    for fn in sorted(calints_files):
        with fits.open(fn, memmap=True) as hdul:
            sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
            cubes.append(sci.data.astype(np.float32))
    big = np.concatenate(cubes, axis=0)

    mean = np.nanmean(big, axis=0)
    resid = big - mean[None, :, :]
    cross = np.nansum(resid[:-1] * resid[1:], axis=0)
    var = np.nansum(resid**2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ac = np.where(var > 0, cross / var, 0.0)
    ac = np.clip(ac, -1.0, 1.0)
    ac[~np.isfinite(ac)] = 0.0

    if output_path:
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=ac.astype(np.float32)).writeto(output_path, overwrite=True)
    return ac


def load_psf_kernel(psf_path, size=21):
    """Load a detector-sampled, unit-sum PSF kernel from a WebbPSF FITS file."""
    with fits.open(psf_path) as hd:
        psf = hd["DET_SAMP"].data.astype(np.float64)
    cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
    half = size // 2
    kern = psf[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return kern / kern.sum()


def fast_psf_detect(image, psf_kernel, threshold_sigma=3.0, min_separation=1):
    """Single-pass PSF matched-filter detection. Returns (positions Nx2 (x,y), snr)."""
    from scipy.ndimage import maximum_filter
    from scipy.signal import fftconvolve

    img_clean = image.copy()
    p995 = np.nanpercentile(img_clean, 99.5)
    img_clean[(img_clean > p995) | (img_clean < -p995)] = np.nan
    _, median_bg, std_bg = sigma_clipped_stats(img_clean, sigma=3.0)
    residual = np.nan_to_num(image.astype(np.float64) - median_bg, nan=0.0, posinf=0.0, neginf=0.0)

    psf_norm = np.sqrt(np.sum(psf_kernel**2))
    filtered = fftconvolve(residual, psf_kernel, mode="same")
    snr_map = filtered / (std_bg * psf_norm + 1e-10)

    data_max = maximum_filter(snr_map, size=min_separation * 2 + 1)
    is_peak = (snr_map == data_max) & (snr_map > threshold_sigma)
    py, px = np.where(is_peak)
    if len(px) == 0:
        return np.empty((0, 2)), np.array([])
    snr = snr_map[py, px]
    order = np.argsort(snr)[::-1]
    positions = np.column_stack([px[order].astype(float), py[order].astype(float)])
    return positions, snr[order]
