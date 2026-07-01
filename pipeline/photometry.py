"""Aperture photometry on the group-diff cube and IQR outlier clipping.

Ported from the production extraction path — analyze_source_worker() in
ramp_pipeline.py — NOT from jwst_utils.do_aperture_photometry (which the
production pipeline imports but never calls). The production behavior this
reproduces:

* whole-pixel circular aperture: pixels with center distance <= r from the
  detected (integer) source position — 9 pixels for r = 1.5;
* pixels that are NaN in ANY frame are excluded from the aperture entirely,
  so the effective aperture is identical across all time steps;
* no background annulus (config default), no high-flux masking.

The catalog-build stage (pipeline/catalog.py) intentionally uses a different
aperture (photutils exact fractional overlap on 5x5 cutouts), matching its own
production counterpart rebuild_master_catalog_v2.py. Do not unify them.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits


def load_cube(cube_path):
    """Memory-map a group-diff cube; return (cube, times_mjd)."""
    hdul = fits.open(cube_path, memmap=True)
    cube = hdul[0].data  # (nframes, ny, nx), memory-mapped
    times = hdul["DIFF_TIMES"].data["MID_BARY_MJD"].astype(np.float64)
    return cube, times


def extraction_lightcurve(cube, cx, cy, ap_radius=1.5):
    """Production-faithful extraction photometry for one source.

    Returns flux array of shape (nframes,), or None if the aperture is empty
    (off-detector or fully NaN-excluded).
    """
    nframes, ny, nx = cube.shape
    max_radius = int(np.ceil(ap_radius)) + 1
    x0 = max(0, int(cx - max_radius))
    x1 = min(nx, int(cx + max_radius + 1))
    y0 = max(0, int(cy - max_radius))
    y1 = min(ny, int(cy + max_radius + 1))
    if x1 <= x0 or y1 <= y0:
        return None

    local_x = np.arange(x0, x1) - cx
    local_y = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(local_x, local_y)
    dist = np.sqrt(xx ** 2 + yy ** 2)
    ap_mask = dist <= ap_radius
    if not ap_mask.any():
        return None

    cutout = np.asarray(cube[:, y0:y1, x0:x1]).astype(np.float64)
    # Exclude pixels that are NaN in ANY frame so the effective aperture
    # is constant across all frames.
    any_nan = np.any(np.isnan(cutout), axis=0)
    good_ap = ap_mask & ~any_nan
    if not good_ap.any():
        return None

    cutout_clean = np.nan_to_num(cutout, nan=0.0)
    return np.sum(cutout_clean[:, good_ap], axis=1)


def aperture_lightcurves(cube, positions, ap_radius=1.5):
    """Extraction photometry at fixed positions on every frame.

    Returns flux array of shape (n_frames, n_sources). Sources whose aperture
    is empty get an all-NaN column.
    """
    n_frames = cube.shape[0]
    flux = np.full((n_frames, len(positions)), np.nan, dtype=float)
    for s, (cx, cy) in enumerate(positions):
        lc = extraction_lightcurve(cube, cx, cy, ap_radius=ap_radius)
        if lc is not None:
            flux[:, s] = lc
    return flux


def test_saturation(flux, sat_thresh=10.0, group_size=9):
    """Detect saturating sources: first vs last group-diff within each ramp.

    In a saturating ramp the late group differences collapse toward zero, so
    the mean of last-in-ramp samples departs from the first-in-ramp mean by
    many sigma. Ported from jwst_utils.test_saturation.
    """
    flux = np.asarray(flux)
    first = flux[0::group_size]
    last = flux[group_size - 1::group_size]
    if len(first) == 0 or len(last) == 0:
        return False
    mu1, sigma1 = first.mean(), first.std()
    mu_last = last.mean()
    return abs(mu_last - mu1) > sat_thresh * sigma1


def sat_first_groups_lightcurve(flux_raw, times, n_groups=2, bin_size=9):
    """Saturated-source extraction: average the first N group-diffs per ramp.

    Early groups carry the least accumulated charge and stay below saturation,
    so instead of IQR clipping (which would shred a saturated lightcurve) the
    production pipeline averaged the first ``n_groups`` samples of each ramp.
    Returns (y, t) with one point per ramp. Ported from analyze_source_worker.
    """
    flux_raw = np.asarray(flux_raw, dtype=float)
    times = np.asarray(times, dtype=float)
    n_raw = len(flux_raw)
    n_ramps = n_raw // bin_size
    y = np.empty(n_ramps)
    t = np.empty(n_ramps)
    for r in range(n_ramps):
        start = r * bin_size
        end = min(start + n_groups, n_raw)
        y[r] = flux_raw[start:end].mean()
        t[r] = times[start:end].mean()
    return y, t


def clip_outliers_iqr(data, times, chunk_size=18, iqr_factor=2.0,
                      merge_final_chunk=True):
    """Chunked IQR outlier clip. Returns (clipped_data, clipped_times).

    Removes impulsive outliers (cosmic rays) locally within chunks while
    preserving smooth astrophysical variability across chunks.

    merge_final_chunk controls the trailing ``n % chunk_size`` points:
    * True  — fold them into the last full chunk (behavior of the corrections
      stage, build_corrected_catalog.clip_iqr, and the paper's description);
    * False — leave them unclipped (behavior of the production extraction and
      catalog-population code, jwst_utils.clip_outliers_iqr /
      rebuild_master_catalog_v2.clip_iqr, which built the published
      lightcurves — use this to reproduce the shipped catalog exactly).
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
        if merge_final_chunk and b == n_chunks - 1:
            end = n
        else:
            end = (b + 1) * chunk_size
        seg = data[start:end]
        finite = seg[np.isfinite(seg)]
        if finite.size < 4:
            continue
        q1, q3 = np.percentile(finite, [25, 75])
        iqr = q3 - q1
        good = (seg >= q1 - iqr_factor * iqr) & (seg <= q3 + iqr_factor * iqr)
        mask[start:end] = good
    return data[mask], times[mask]
