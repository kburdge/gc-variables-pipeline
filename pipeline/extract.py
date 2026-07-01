"""Stage 3 orchestration: detect variables and extract their light curves.

For one target/segment/detector in a given mode:

ramp mode:
  1. build (or reuse) the lag-1 autocorrelation reference image from calints,
  2. detect sources on it by PSF-matched filtering (3 sigma),
  3. aperture-photometer every source on the group-diff cube,
  4. IQR-clip each light curve (saturated sources instead average the first
     two group-diffs per ramp) and run the period search,
  5. write everything to {det}_ramp.h5.

zf mode:
  same flow on the 96-frame zeroframe cube: lag-1 autocorrelation of the ZF
  cube, detection at 5 sigma, chunk-4 IQR clip (no saturation branch — zero
  frames do not saturate), ZF period grid, written to {det}_zf.h5.

This is the demo's path "all the way to a light curve". The heavyweight
saturation/slope corrections and the catalog build are downstream stages.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS

from .detect import (find_calints, create_autocorr_reference,
                     create_zf_autocorr_reference, load_psf_kernel, fast_psf_detect)
from .groupdiff import create_zeroframe_cube
from .photometry import (load_cube, aperture_lightcurves, clip_outliers_iqr,
                         test_saturation, sat_first_groups_lightcurve)
from .periods import search

SOURCE_DTYPE = np.dtype([
    ("source_id", "i4"), ("px", "f4"), ("py", "f4"),
    ("ra", "f8"), ("dec", "f8"), ("det_snr", "f4"),
    ("best_period_min", "f4"), ("ls_significance", "f4"),
    ("bls_significance", "f4"), ("n_points", "i4"), ("is_saturated", "?"),
])


def _psf_path(cfg, detector):
    return cfg["paths"]["psf_f356w"] if detector == cfg.get("detector_lw") else cfg["paths"]["psf_f200w"]


def _sky_from_ref(ref_path, positions):
    """RA/Dec for (x, y) positions via the reference image's WCS, if present."""
    ra = np.full(len(positions), np.nan)
    dec = np.full(len(positions), np.nan)
    try:
        hdr = fits.getheader(ref_path)
        if "CRVAL1" in hdr:
            w = WCS(hdr)
            sky = w.pixel_to_world(positions[:, 0], positions[:, 1])
            ra, dec = np.asarray(sky.ra.deg), np.asarray(sky.dec.deg)
    except Exception:
        pass
    return ra, dec


def extract_detector(cfg, target, segment, detector, mode="ramp",
                     overwrite=False, max_sources=None):
    """Run detection + extraction for one detector; write an extraction HDF5.

    mode: "ramp" (group-diff cube, 972 frames) or "zf" (zeroframe cube, 96).
    max_sources: if set, keep only the N highest-detection-SNR sources. Use this
    for the demo / quick runs; leave None for the full source list.
    """
    if mode not in ("ramp", "zf"):
        raise ValueError(f"mode must be 'ramp' or 'zf', got {mode!r}")
    refs_dir = cfg["paths"]["refs_dir"]
    extr_dir = os.path.join(cfg["paths"]["extraction_dir"], target, segment)
    Path(extr_dir).mkdir(parents=True, exist_ok=True)
    out_h5 = os.path.join(extr_dir, f"{detector}_{mode}.h5")
    if os.path.exists(out_h5) and not overwrite:
        print(f"[extract] exists, skipping: {out_h5}")
        return out_h5

    kern = load_psf_kernel(_psf_path(cfg, detector),
                           size=cfg["detection"].get("psf_kernel_size", 21))

    if mode == "ramp":
        # autocorrelation reference from calints (reuse if present)
        ac_path = os.path.join(refs_dir, f"{target}_{segment}_{detector}_autocorr.fits")
        if os.path.exists(ac_path) and not overwrite:
            ac = fits.getdata(ac_path).astype(np.float64)
        else:
            calints = find_calints(cfg["paths"]["data_root"], target, segment, detector)
            if not calints:
                raise FileNotFoundError(
                    f"No calints for {target}/{segment}/{detector}; run stage 1 (calibrate)."
                )
            print(f"[extract] autocorr from {len(calints)} calints files")
            ac = create_autocorr_reference(calints, ac_path)
        cube_path = os.path.join(refs_dir, f"groupdiffs_{target}_{segment}_{detector}.fits")
        if not os.path.exists(cube_path):
            raise FileNotFoundError(f"Missing cube {cube_path}; run stage 2 (build_cubes).")
        sigma = cfg["detection"]["ramp_sigma"]
        clip_cfg = cfg["clipping"]["ramp"]
    else:
        # zeroframe cube (built here if stage 2 didn't) + ZF autocorr reference
        cube_path = create_zeroframe_cube(cfg, target, segment, detector)
        ac_path = os.path.join(refs_dir, f"{target}_{segment}_{detector}_zf_autocorr.fits")
        if os.path.exists(ac_path) and not overwrite:
            ac = fits.getdata(ac_path).astype(np.float64)
        else:
            with fits.open(cube_path, memmap=True) as hd:
                zf_cube = hd[0].data.astype(np.float32)
                wcs_hdr = hd[0].header
            print(f"[extract] ZF autocorr from {zf_cube.shape[0]}-frame cube")
            ac = create_zf_autocorr_reference(zf_cube, ac_path, wcs_header=wcs_hdr)
            del zf_cube
        sigma = cfg["detection"]["zf_sigma"]
        clip_cfg = cfg["clipping"]["zf"]

    # detection
    positions, snr = fast_psf_detect(ac, kern, threshold_sigma=sigma,
                                     min_separation=cfg["detection"].get("min_separation", 1))
    print(f"[extract] detected {len(positions)} sources at >{sigma}sigma ({mode})")
    if len(positions) == 0:
        raise RuntimeError("No sources detected — check PSF path / threshold.")
    if max_sources and len(positions) > max_sources:
        positions, snr = positions[:max_sources], snr[:max_sources]  # already SNR-sorted
        print(f"[extract] capping to top {max_sources} sources by detection SNR")
    ra, dec = _sky_from_ref(ac_path, positions)

    # photometry on the cube
    cube, times_mjd = load_cube(cube_path)
    flux = aperture_lightcurves(cube, positions, ap_radius=cfg["photometry"]["aperture_radius"])
    times_hr = (times_mjd - times_mjd[0]) * 24.0

    # per-source clipping (or saturated-source rebinning) + period search
    chunk = clip_cfg["chunk_size"]
    iqrf = clip_cfg.get("iqr_factor", 2.0)
    min_points = clip_cfg.get("min_points", 500 if mode == "ramp" else 30)
    sat_cfg = cfg.get("saturation", {})
    sat_thresh = sat_cfg.get("sat_thresh", 10.0)
    sat_first = sat_cfg.get("first_groups", 2)
    bin_size = cfg.get("binning", {}).get("ramp", 9)
    sat_min_points = sat_cfg.get("min_points", 50)

    n_src, n_frm = len(positions), flux.shape[0]
    flux_clipped = np.full((n_src, n_frm), np.nan, dtype=np.float32)
    src_arr = np.zeros(n_src, dtype=SOURCE_DTYPE)

    for s in range(n_src):
        lc = flux[:, s]
        if not np.isfinite(lc).any():
            continue
        is_sat = False
        if mode == "ramp":
            is_sat = test_saturation(lc, sat_thresh, group_size=bin_size)
        if is_sat:
            # early groups stay below saturation: one averaged point per ramp
            cl, ct = sat_first_groups_lightcurve(lc, times_hr, n_groups=sat_first,
                                                 bin_size=bin_size)
            src_min_points = sat_min_points
        else:
            cl, ct = clip_outliers_iqr(lc, times_hr, chunk_size=chunk, iqr_factor=iqrf)
            src_min_points = min_points
        # map kept points back onto the fixed frame grid (nearest time)
        idx = np.searchsorted(times_hr, ct)
        idx = np.clip(idx, 0, n_frm - 1)
        left = np.clip(idx - 1, 0, n_frm - 1)
        use_left = np.abs(times_hr[left] - ct) < np.abs(times_hr[idx] - ct)
        idx[use_left] = left[use_left]
        flux_clipped[s, idx] = cl.astype(np.float32)

        src_arr[s]["n_points"] = len(cl)
        src_arr[s]["is_saturated"] = is_sat
        med = np.nanmedian(cl)
        if len(cl) >= src_min_points and med != 0:
            res = search(ct, cl / med, cfg, mode=mode)
            src_arr[s]["best_period_min"] = res["best_period_min"]
            src_arr[s]["ls_significance"] = res["ls_significance"]
            src_arr[s]["bls_significance"] = res["bls_significance"]
        else:
            src_arr[s]["best_period_min"] = np.nan
            src_arr[s]["ls_significance"] = np.nan
            src_arr[s]["bls_significance"] = np.nan

    src_arr["source_id"] = np.arange(n_src)
    src_arr["px"] = positions[:, 0]
    src_arr["py"] = positions[:, 1]
    src_arr["ra"] = ra
    src_arr["dec"] = dec
    src_arr["det_snr"] = snr

    # write HDF5
    with h5py.File(out_h5, "w") as f:
        f.attrs.update(dict(target=target, segment=segment, detector=detector,
                            mode=mode, n_sources=n_src, n_frames=n_frm,
                            detection_sigma=sigma,
                            aperture_radius=cfg["photometry"]["aperture_radius"]))
        f.create_dataset("times_hr", data=times_hr.astype(np.float64))
        f.create_dataset("times_mjd", data=times_mjd.astype(np.float64))
        f.create_dataset("sources", data=src_arr)
        # flat aliases kept for convenience/back-compat with early port scripts
        f.create_dataset("px", data=positions[:, 0].astype(np.float32))
        f.create_dataset("py", data=positions[:, 1].astype(np.float32))
        f.create_dataset("det_snr", data=snr.astype(np.float32))
        f.create_dataset("best_period_min", data=src_arr["best_period_min"])
        f.create_dataset("ls_significance", data=src_arr["ls_significance"])
        f.create_dataset("bls_significance", data=src_arr["bls_significance"])
        f.create_dataset("flux", data=flux.T.astype(np.float32),
                         compression="gzip", compression_opts=4)
        f.create_dataset("flux_clipped", data=flux_clipped,
                         compression="gzip", compression_opts=4)
    print(f"[extract] wrote {out_h5} ({n_src} sources x {n_frm} frames)")
    return out_h5
