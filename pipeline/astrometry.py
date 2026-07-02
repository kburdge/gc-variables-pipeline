"""LW (nrcblong) astrometric calibration to Gaia DR3.

Faithful, config-driven port of the production calibrate_lw_astrometry.py
(the "DO NOT DELETE" recipe of March 2026, preserved verbatim in unported/).
The LW-aligned WCS produced here is the absolute astrometric reference for
all downstream SW alignment and catalog positions.

Recipe:
  1. DAOStarFinder on the uncal zeroframe median image over a grid of FWHM
     values (4..30 px), threshold 10 sigma, brightest 500 per FWHM.
  2. For each Gaia DR3 source (proper motions propagated to the observation
     epoch), try every FWHM catalog and keep the match with the smallest
     |roundness1| ("best-roundness" selection -- finds the FWHM whose
     Gaussian model best fits each source, which varies with saturation).
  3. Keep matches with |roundness1| < 0.1.
  4. 3x IQR clip on the RA/Dec offsets.
  5. Apply the median shift as a rigid CRVAL correction, preserving the
     JWST calints SIP distortion model.

Inputs (all shipped data products or stage outputs):
  {astrometry_dir}/{target}_{seg}_nrcblong_uncal_zf_median.fits
  {astrometry_dir}/gaia_{target}.vot          (Gaia DR3 cone-search cache)
  a calints SCI header for the initial WCS (data_root)

Determinism: DAOStarFinder and the matching are deterministic given these
inputs; rerunning reproduces the published solution exactly.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
import astropy.units as u

GAIA_EPOCH_JYEAR = 2016.0
FWHM_GRID = (4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
DETECT_THRESHOLD_SIGMA = 10
DETECT_BRIGHTEST = 500
EDGE = 10
LW_PIXEL_MAS = 63.0

# Per-target matching parameters of the published reduction.
TARGET_PARAMS = {
    "Terzan5": {"gaia_mag_limit": 17.5, "match_radius_arcsec": 0.5, "roundness_cut": 0.1},
    "Liller1": {"gaia_mag_limit": 18.0, "match_radius_arcsec": 0.3, "roundness_cut": 0.1},
}


def detect_all_fwhm(zf_image, std):
    """DAOStarFinder at each FWHM in the grid; returns {fwhm: catalog dict}."""
    from photutils.detection import DAOStarFinder

    median_val = np.median(zf_image[np.isfinite(zf_image)])
    det_cats = {}
    for fwhm in FWHM_GRID:
        finder = DAOStarFinder(fwhm=fwhm, threshold=DETECT_THRESHOLD_SIGMA * std,
                               brightest=DETECT_BRIGHTEST)
        sources = finder(zf_image - median_val)
        if sources is None:
            continue
        xp = np.array(sources["xcentroid"])
        yp = np.array(sources["ycentroid"])
        mask = (xp > EDGE) & (xp < 2048 - EDGE) & (yp > EDGE) & (yp < 2048 - EDGE)
        det_cats[fwhm] = {
            "x": xp[mask], "y": yp[mask],
            "r1": np.array(sources["roundness1"])[mask],
            "sharp": np.array(sources["sharpness"])[mask],
            "peak": np.array(sources["peak"])[mask],
        }
    return det_cats


def propagate_gaia(vot_path, obs_epoch_mjd):
    """Load the cached Gaia DR3 table; propagate proper motions to the epoch."""
    gaia = Table.read(vot_path, format="votable")
    dt_yr = (Time(obs_epoch_mjd, format="mjd") - Time(GAIA_EPOCH_JYEAR, format="jyear")).to(u.yr).value

    ra = np.array(gaia["ra"], dtype=np.float64)
    dec = np.array(gaia["dec"], dtype=np.float64)
    pmra = np.array(gaia["pmra"], dtype=np.float64)
    pmdec = np.array(gaia["pmdec"], dtype=np.float64)
    ra_err = np.array(gaia["ra_error"], dtype=np.float64)
    dec_err = np.array(gaia["dec_error"], dtype=np.float64)
    pmra_err = np.array(gaia["pmra_error"], dtype=np.float64)
    pmdec_err = np.array(gaia["pmdec_error"], dtype=np.float64)

    has_pm = np.isfinite(pmra) & np.isfinite(pmdec)
    cos_dec = np.cos(np.radians(dec))
    ra_prop, dec_prop = ra.copy(), dec.copy()
    ra_prop[has_pm] += (pmra[has_pm] * dt_yr / 3600000.0) / cos_dec[has_pm]
    dec_prop[has_pm] += pmdec[has_pm] * dt_yr / 3600000.0

    pmra_err[~np.isfinite(pmra_err)] = 0
    pmdec_err[~np.isfinite(pmdec_err)] = 0
    return {
        "ra_prop": ra_prop, "dec_prop": dec_prop,
        "gmag": np.array(gaia["phot_g_mean_mag"], dtype=np.float64),
        "ra_err": np.sqrt(ra_err ** 2 + (pmra_err * dt_yr) ** 2),
        "dec_err": np.sqrt(dec_err ** 2 + (pmdec_err * dt_yr) ** 2),
        "dt_yr": dt_yr,
    }


def match_best_roundness(det_cats, wcs_use, gaia_info, params, std):
    """Best-|roundness1| match across all FWHM catalogs, per Gaia source."""
    glim = params["gaia_mag_limit"]
    match_rad = params["match_radius_arcsec"]
    r1_cut = params["roundness_cut"]

    bright = gaia_info["gmag"] < glim
    gpx, gpy = wcs_use.world_to_pixel_values(
        gaia_info["ra_prop"][bright], gaia_info["dec_prop"][bright])
    on_det = (gpx > EDGE) & (gpx < 2048 - EDGE) & (gpy > EDGE) & (gpy < 2048 - EDGE)

    gaia_sky = SkyCoord(ra=gaia_info["ra_prop"][bright][on_det] * u.deg,
                        dec=gaia_info["dec_prop"][bright][on_det] * u.deg)
    gmags_on = gaia_info["gmag"][bright][on_det]
    ra_err_on = gaia_info["ra_err"][bright][on_det]
    dec_err_on = gaia_info["dec_err"][bright][on_det]

    # pre-compute detection sky coords per FWHM (identical math to original)
    det_sky = {}
    for fwhm, cat in det_cats.items():
        sky = wcs_use.pixel_to_world(cat["x"], cat["y"])
        det_sky[fwhm] = SkyCoord(ra=sky.ra, dec=sky.dec)

    rows = []
    for gi in range(len(gaia_sky)):
        best_r1 = 999.0
        best = None
        for fwhm in FWHM_GRID:
            if fwhm not in det_cats:
                continue
            cat = det_cats[fwhm]
            sep = gaia_sky[gi].separation(det_sky[fwhm]).arcsec
            idx = int(np.argmin(sep))
            if sep[idx] < match_rad and abs(cat["r1"][idx]) < best_r1:
                best_r1 = abs(cat["r1"][idx])
                sc = det_sky[fwhm][idx]
                cos_d = np.cos(np.radians(float(sc.dec.deg)))
                snr = cat["peak"][idx] / std
                centroid_err_mas = (fwhm / (2 * max(snr, 3))) * LW_PIXEL_MAS
                best = {
                    "gaia_ra": float(gaia_sky[gi].ra.deg),
                    "gaia_dec": float(gaia_sky[gi].dec.deg),
                    "gaia_gmag": float(gmags_on[gi]),
                    "gaia_ra_err_mas": float(ra_err_on[gi]),
                    "gaia_dec_err_mas": float(dec_err_on[gi]),
                    "jwst_x": float(cat["x"][idx]),
                    "jwst_y": float(cat["y"][idx]),
                    "dra_mas": (float(sc.ra.deg) - float(gaia_sky[gi].ra.deg)) * cos_d * 3600e3,
                    "ddec_mas": (float(sc.dec.deg) - float(gaia_sky[gi].dec.deg)) * 3600e3,
                    "sep_mas": float(sep[idx] * 1000),
                    "best_fwhm": float(fwhm),
                    "roundness1": float(cat["r1"][idx]),
                    "sharpness": float(cat["sharp"][idx]),
                    "peak": float(cat["peak"][idx]),
                    "centroid_err_mas": float(centroid_err_mas),
                    "era_mas": float(np.sqrt(ra_err_on[gi] ** 2 + centroid_err_mas ** 2)),
                    "edec_mas": float(np.sqrt(dec_err_on[gi] ** 2 + centroid_err_mas ** 2)),
                }
        if best is not None and abs(best["roundness1"]) < r1_cut:
            rows.append(best)
    return rows


def iqr_clip(rows, factor=3.0):
    """factor x IQR clip on dra_mas and ddec_mas."""
    dra = np.array([r["dra_mas"] for r in rows])
    ddec = np.array([r["ddec_mas"] for r in rows])
    q1r, q3r = np.percentile(dra, [25, 75]); iqr_r = q3r - q1r
    q1d, q3d = np.percentile(ddec, [25, 75]); iqr_d = q3d - q1d
    keep = ((dra >= q1r - factor * iqr_r) & (dra <= q3r + factor * iqr_r) &
            (ddec >= q1d - factor * iqr_d) & (ddec <= q3d + factor * iqr_d))
    return [r for r, k in zip(rows, keep) if k]


def compute_shift(rows):
    """Median shift + uncertainty statistics from clipped matches."""
    dra = np.array([r["dra_mas"] for r in rows])
    ddec = np.array([r["ddec_mas"] for r in rows])
    n = len(rows)
    shift_ra, shift_dec = np.median(dra), np.median(ddec)
    std_ra, std_dec = np.std(dra), np.std(ddec)
    unc_ra, unc_dec = std_ra / np.sqrt(n), std_dec / np.sqrt(n)
    resid = np.hypot(dra - shift_ra, ddec - shift_dec)
    return {
        "n": n, "shift_ra": shift_ra, "shift_dec": shift_dec,
        "std_ra": std_ra, "std_dec": std_dec,
        "unc_ra": unc_ra, "unc_dec": unc_dec,
        "unc_tot": float(np.hypot(unc_ra, unc_dec)),
        "median_resid": float(np.median(resid)),
        "p90_resid": float(np.percentile(resid, 90)),
    }


def _find_calints_header(data_root, target, seg, det="nrcblong"):
    pats = [
        os.path.join(data_root, target, seg, f"*{det}*cal.fits"),
        os.path.join(data_root, target, seg, "calints", f"*{det}*calints.fits"),
        os.path.join(data_root, target, seg, "detector1_output", "calints", f"*{det}*calints.fits"),
    ]
    for p in pats:
        files = sorted(f for f in glob.glob(p) if "uncal" not in f)
        if files:
            return fits.getheader(files[0], "SCI")
    raise FileNotFoundError(f"no calints for {target}/{seg}/{det} under {data_root}")


def calibrate_lw_gaia(cfg, target, seg, obs_epoch_mjd, out_dir=None, write_wcs=True):
    """Full LW->Gaia calibration for one target/segment.

    Returns the summary dict (pre-correction fit + post-correction residuals).
    Writes {target}_{seg}_nrcblong_wcs_gaia.fits + the match table to out_dir
    (defaults to cfg paths astrometry_dir).
    """
    det = "nrcblong"
    astrom = cfg["paths"]["astrometry_dir"]
    out_dir = out_dir or astrom
    os.makedirs(out_dir, exist_ok=True)
    params = TARGET_PARAMS[target]

    zf = fits.getdata(os.path.join(astrom, f"{target}_{seg}_{det}_uncal_zf_median.fits"))
    hdr = _find_calints_header(cfg["paths"]["data_root"], target, seg, det)
    wcs_cal = WCS(hdr)
    _, _, std = sigma_clipped_stats(zf[np.isfinite(zf)], sigma=3.0)

    det_cats = detect_all_fwhm(zf, std)
    gaia_info = propagate_gaia(os.path.join(astrom, f"gaia_{target}.vot"), obs_epoch_mjd)

    rows_pre = match_best_roundness(det_cats, wcs_cal, gaia_info, params, std)
    rows_pre_c = iqr_clip(rows_pre)
    stats_pre = compute_shift(rows_pre_c)

    hdr_new = hdr.copy()
    cos_d = np.cos(np.radians(hdr_new["CRVAL2"]))
    hdr_new["CRVAL1"] -= stats_pre["shift_ra"] / 1000 / 3600 / cos_d
    hdr_new["CRVAL2"] -= stats_pre["shift_dec"] / 1000 / 3600
    hdr_new["GAIADRA"] = float(stats_pre["shift_ra"] / 1000)
    hdr_new["GAIADDEC"] = float(stats_pre["shift_dec"] / 1000)
    hdr_new["GAIANMAT"] = stats_pre["n"]
    wcs_corr = WCS(hdr_new)

    rows_post = match_best_roundness(det_cats, wcs_corr, gaia_info, params, std)
    rows_post_c = iqr_clip(rows_post)
    stats_post = compute_shift(rows_post_c)
    hdr_new["GAIARESI"] = float(stats_post["median_resid"])

    if write_wcs:
        out_wcs = os.path.join(out_dir, f"{target}_{seg}_{det}_wcs_gaia.fits")
        fits.writeto(out_wcs, zf.astype(np.float32), hdr_new, overwrite=True)
        Table(rows_pre_c).write(
            os.path.join(out_dir, f"{target}_{seg}_{det}_gaia_match_final.fits"),
            overwrite=True)

    return {
        "target": target, "seg": seg,
        "pre": stats_pre, "post": stats_post,
        "n_raw": len(rows_pre),
    }
