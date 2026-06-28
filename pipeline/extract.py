"""Stage 3 orchestration: detect variables and extract their light curves.

For one target/segment/detector:
  1. build (or reuse) the lag-1 autocorrelation reference image from calints,
  2. detect sources on it by PSF-matched filtering,
  3. aperture-photometer every source on the group-diff cube,
  4. IQR-clip each light curve and run the period search,
  5. write everything to a per-detector extraction HDF5.

This is the demo's path "all the way to a light curve". The heavyweight
saturation/slope corrections and the catalog build are downstream stages.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import h5py

from .detect import find_calints, create_autocorr_reference, load_psf_kernel, fast_psf_detect
from .photometry import load_cube, aperture_lightcurves, clip_outliers_iqr
from .periods import search


def _psf_path(cfg, detector):
    return cfg["paths"]["psf_f356w"] if detector == cfg.get("detector_lw") else cfg["paths"]["psf_f200w"]


def extract_detector(cfg, target, segment, detector, overwrite=False, max_sources=None):
    """Run detection + extraction for one detector; write an extraction HDF5.

    max_sources: if set, keep only the N highest-detection-SNR sources. Use this
    for the demo / quick runs; leave None for the full source list.
    """
    refs_dir = cfg["paths"]["refs_dir"]
    extr_dir = os.path.join(cfg["paths"]["extraction_dir"], target, segment)
    Path(extr_dir).mkdir(parents=True, exist_ok=True)
    out_h5 = os.path.join(extr_dir, f"{detector}_ramp.h5")
    if os.path.exists(out_h5) and not overwrite:
        print(f"[extract] exists, skipping: {out_h5}")
        return out_h5

    # 1. autocorrelation reference (reuse if present)
    ac_path = os.path.join(refs_dir, f"{target}_{segment}_{detector}_autocorr.fits")
    if os.path.exists(ac_path) and not overwrite:
        from astropy.io import fits
        ac = fits.getdata(ac_path).astype(np.float64)
    else:
        calints = find_calints(cfg["paths"]["data_root"], target, segment, detector)
        if not calints:
            raise FileNotFoundError(
                f"No calints for {target}/{segment}/{detector}; run stage 1 (calibrate)."
            )
        print(f"[extract] autocorr from {len(calints)} calints files")
        ac = create_autocorr_reference(calints, ac_path)

    # 2. detection
    sigma = cfg["detection"]["ramp_sigma"]
    kern = load_psf_kernel(_psf_path(cfg, detector), size=cfg["detection"].get("psf_kernel_size", 21))
    positions, snr = fast_psf_detect(ac, kern, threshold_sigma=sigma,
                                     min_separation=cfg["detection"].get("min_separation", 1))
    print(f"[extract] detected {len(positions)} sources at >{sigma}sigma")
    if len(positions) == 0:
        raise RuntimeError("No sources detected — check PSF path / threshold.")
    if max_sources and len(positions) > max_sources:
        positions, snr = positions[:max_sources], snr[:max_sources]  # already SNR-sorted
        print(f"[extract] capping to top {max_sources} sources by detection SNR")

    # 3. photometry on the group-diff cube
    cube_path = os.path.join(refs_dir, f"groupdiffs_{target}_{segment}_{detector}.fits")
    if not os.path.exists(cube_path):
        raise FileNotFoundError(f"Missing cube {cube_path}; run stage 2 (build_cubes).")
    cube, times_mjd = load_cube(cube_path)
    flux = aperture_lightcurves(cube, positions, ap_radius=cfg["photometry"]["aperture_radius"])
    times_hr = (times_mjd - times_mjd[0]) * 24.0

    # 4. per-source IQR clip + period search (store clipped LCs as fixed-length w/ NaN)
    chunk = cfg["clipping"]["ramp"]["chunk_size"]
    iqrf = cfg["clipping"]["ramp"]["iqr_factor"]
    n_src, n_frm = len(positions), flux.shape[0]
    flux_clipped = np.full((n_src, n_frm), np.nan, dtype=np.float32)
    best_period = np.full(n_src, np.nan, dtype=np.float32)
    ls_sig = np.full(n_src, np.nan, dtype=np.float32)
    bls_sig = np.full(n_src, np.nan, dtype=np.float32)

    for s in range(n_src):
        lc = flux[:, s]
        cl, ct = clip_outliers_iqr(lc, times_hr, chunk_size=chunk, iqr_factor=iqrf)
        # map kept points back onto the fixed grid by value-matching times
        keep = np.isin(times_hr, ct)
        flux_clipped[s, keep] = lc[keep].astype(np.float32)
        if cl.size >= cfg["clipping"]["ramp"].get("min_points", 500) and np.nanmedian(cl) != 0:
            res = search(ct, cl / np.nanmedian(cl), cfg)
            best_period[s] = res["best_period_min"]
            ls_sig[s] = res["ls_significance"]
            bls_sig[s] = res["bls_significance"]

    # 5. write HDF5
    with h5py.File(out_h5, "w") as f:
        f.attrs.update(dict(target=target, segment=segment, detector=detector,
                            mode="ramp", n_sources=n_src, n_frames=n_frm,
                            detection_sigma=sigma, aperture_radius=cfg["photometry"]["aperture_radius"]))
        f.create_dataset("times_hr", data=times_hr.astype(np.float64))
        f.create_dataset("times_mjd", data=times_mjd.astype(np.float64))
        f.create_dataset("px", data=positions[:, 0].astype(np.float32))
        f.create_dataset("py", data=positions[:, 1].astype(np.float32))
        f.create_dataset("det_snr", data=snr.astype(np.float32))
        f.create_dataset("best_period_min", data=best_period)
        f.create_dataset("ls_significance", data=ls_sig)
        f.create_dataset("bls_significance", data=bls_sig)
        f.create_dataset("flux", data=flux.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("flux_clipped", data=flux_clipped, compression="gzip", compression_opts=4)
    print(f"[extract] wrote {out_h5} ({n_src} sources x {n_frm} frames)")
    return out_h5
