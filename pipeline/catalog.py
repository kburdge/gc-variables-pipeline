"""Downstream catalog construction from shipped vetting labels.

Stage 5 of the pipeline (paper §3.7). Takes the human REAL/FAKE vetting labels
(exported by scripts/export_vetting_labels.py, shipped on Zenodo) and:

  build_mapping()      deduplicate REAL detections across detectors/segments/
                       modes/channels at 0.2" (SNR-ordered) -> unique objects
                       with a master_id (the 1,315-object source list).
  write_source_table() write the per-target `sources` compound table into
                       master_variable_catalog.h5.

The lightcurve population (centroid refinement + forced photometry on the
group-diff cubes + RA/Dec from the LW-aligned WCS) is the next stage; the
faithful helpers it needs (refine_centroid, clip_iqr) are ported here.

Ported from build_master_mapping.py / rebuild_master_catalog_v2.py.
"""
from __future__ import annotations

import csv

import numpy as np
import h5py
from astropy.coordinates import SkyCoord
import astropy.units as u

# compound dtype of the published catalog's {target}/sources table
SOURCE_DTYPE = np.dtype([
    ("master_id", "i4"), ("ra", "f8"), ("dec", "f8"),
    ("best_snr", "f4"), ("n_detections", "i4"),
    ("detector", "S8"),
    ("det_px", "f4"), ("det_py", "f4"),
    ("refined_px", "f4"), ("refined_py", "f4"),
    ("refined_px_err", "f4"), ("refined_py_err", "f4"),
])


def load_labels(csv_path):
    """Read vetting_labels.csv into a list of per-detection dicts."""
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            def _f(key):
                v = r.get(key, "")
                return float(v) if v not in ("", "nan") else float("nan")
            rows.append({
                "target": r["target"], "segment": r["segment"], "mode": r["mode"],
                "channel": r["channel"], "detector": r["detector"],
                "folder": r["folder"], "src_id": int(r["src_id"]),
                "ra": float(r["ra"]), "dec": float(r["dec"]), "snr": float(r["snr"]),
                "px": _f("px"), "py": _f("py"), "ls_sig": _f("ls_sig"),
            })
    return rows


def build_mapping(labels, dedup_arcsec=0.2):
    """Deduplicate REAL detections into unique objects.

    SNR-ordered single-link merge: the brightest detection seeds each object;
    later detections within ``dedup_arcsec`` (and same target) attach to it,
    otherwise they seed a new object. Mirrors build_master_mapping.py exactly.

    Returns a list of master entries sorted by (target, -best_snr) with a
    contiguous master_id assigned per the published convention (Liller1 first).
    """
    srcs = sorted(labels, key=lambda s: -s["snr"])
    master = []
    master_sc = None  # SkyCoord cache, rebuilt when a new entry is added

    for s in srcs:
        matched = None
        if master_sc is not None and len(master_sc) > 0:
            same = np.array([m["target"] == s["target"] for m in master])
            if same.any():
                sc = SkyCoord(ra=s["ra"] * u.deg, dec=s["dec"] * u.deg)
                seps = sc.separation(master_sc).arcsec
                seps[~same] = 999.0
                j = int(np.argmin(seps))
                if seps[j] < dedup_arcsec:
                    matched = j
        if matched is not None:
            m = master[matched]
            if s["folder"] not in m["detections"]:
                m["detections"][s["folder"]] = s  # full per-detection label (px,py,seg,mode,channel,...)
                if s["snr"] > m["best_snr"]:
                    m["best_snr"], m["ra"], m["dec"] = s["snr"], s["ra"], s["dec"]
                    m["detector"] = s["detector"]
        else:
            master.append({
                "target": s["target"], "ra": s["ra"], "dec": s["dec"],
                "best_snr": s["snr"], "detector": s["detector"],
                "detections": {s["folder"]: s},
            })
            master_sc = SkyCoord(ra=[m["ra"] for m in master] * u.deg,
                                 dec=[m["dec"] for m in master] * u.deg)

    # sort Liller1 first then by descending SNR, assign contiguous master_id
    order = {"Liller1": 0, "Terzan5": 1}
    master.sort(key=lambda m: (order.get(m["target"], 2), -m["best_snr"]))
    for i, m in enumerate(master):
        m["master_id"] = i
        m["n_detections"] = len(m["detections"])
    return master


def write_source_table(master, out_h5, targets=("Liller1", "Terzan5")):
    """Write per-target `sources` compound tables into the catalog HDF5.

    Pixel/refined fields are left as sentinels (NaN/empty) here; they are filled
    by the lightcurve-population stage. RA/Dec/SNR/n_detections come from the
    vetting labels and are final.
    """
    counts = {}
    with h5py.File(out_h5, "a") as f:
        for target in targets:
            rows = [m for m in master if m["target"] == target]
            arr = np.zeros(len(rows), dtype=SOURCE_DTYPE)
            for i, m in enumerate(rows):
                arr[i]["master_id"] = m["master_id"]
                arr[i]["ra"] = m["ra"]
                arr[i]["dec"] = m["dec"]
                arr[i]["best_snr"] = m["best_snr"]
                arr[i]["n_detections"] = m["n_detections"]
                arr[i]["detector"] = m["detector"].encode()
                for fld in ("det_px", "det_py", "refined_px", "refined_py",
                            "refined_px_err", "refined_py_err"):
                    arr[i][fld] = np.nan
            if f"{target}/sources" in f:
                del f[f"{target}/sources"]
            f.create_dataset(f"{target}/sources", data=arr)
            counts[target] = len(rows)
    return counts


# --------------------------------------------------------------------------
# Helpers ported for the lightcurve-population stage (centroiding + clipping)
# --------------------------------------------------------------------------

def _gauss2d(coords, amp, x0, y0, sigma, bg):
    x, y = coords
    return (amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) + bg).ravel()


def refine_centroid(img, ix, iy, R=4):
    """2D Gaussian centroid refinement on a cutout; reject shifts > R or big errors.

    Returns (x, y, x_err, y_err, ok). Ported from rebuild_master_catalog_v2.py.
    """
    from scipy.optimize import curve_fit

    ny, nx = img.shape
    if ix - R < 0 or ix + R + 1 > nx or iy - R < 0 or iy + R + 1 > ny:
        return float(ix), float(iy), np.nan, np.nan, False
    cutout = img[iy - R:iy + R + 1, ix - R:ix + R + 1].astype(float)
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    bg = np.median(np.concatenate([cutout[0], cutout[-1], cutout[:, 0], cutout[:, -1]]))
    py, px = np.unravel_index(np.argmax(cutout), cutout.shape)
    amp = cutout[py, px] - bg
    if amp <= 0:
        return float(ix), float(iy), np.nan, np.nan, False
    try:
        popt, pcov = curve_fit(_gauss2d, (xx, yy), cutout.ravel(),
                               p0=[amp, px - R, py - R, 1.0, bg],
                               bounds=([0, -R, -R, 0.3, -np.inf], [np.inf, R, R, 5.0, np.inf]),
                               maxfev=2000)
        perr = np.sqrt(np.diag(pcov))
        if abs(popt[1]) > R or abs(popt[2]) > R or perr[1] > 2 or perr[2] > 2:
            return float(ix), float(iy), np.nan, np.nan, False
        return ix + popt[1], iy + popt[2], perr[1], perr[2], True
    except Exception:
        return float(ix), float(iy), np.nan, np.nan, False


# --------------------------------------------------------------------------
# Lightcurve population (stage 5b): centroid refine + forced photometry
# --------------------------------------------------------------------------
_H = 2  # cutout half-size (5x5)


def _aperture_mask(ap_radius=1.5):
    """Exact circular-aperture weight mask on the (2H+1) cutout grid."""
    from photutils.aperture import CircularAperture
    ap = CircularAperture([(_H, _H)], r=ap_radius)
    return ap.to_mask(method="exact")[0].to_image((2 * _H + 1, 2 * _H + 1)).astype(np.float32)


def _load_wcs(astrom_dir, target, segments, detectors):
    """LW-aligned WCS per (target,seg,det): prefer *_wcs_lw.fits, else *_wcs_gaia.fits."""
    import os
    from astropy.io import fits
    from astropy.wcs import WCS
    out = {}
    for seg in segments:
        for det in detectors:
            for suffix in ("wcs_lw", "wcs_gaia"):
                p = os.path.join(astrom_dir, f"{target}_{seg}_{det}_{suffix}.fits")
                if os.path.exists(p):
                    out[(target, seg, det)] = WCS(fits.getheader(p))
                    break
    return out


def _load_autocorr(refs_dir, target, segments, detectors):
    """Ramp + ZF autocorr reference images per (target,seg,det)."""
    import os
    from astropy.io import fits
    ac, zf = {}, {}
    for seg in segments:
        for det in detectors:
            p = os.path.join(refs_dir, f"{target}_{seg}_{det}_autocorr.fits")
            if os.path.exists(p):
                ac[(target, seg, det)] = fits.getdata(p)
            pz = os.path.join(refs_dir, f"{target}_{seg}_{det}_zf_autocorr.fits")
            if os.path.exists(pz):
                zf[(target, seg, det)] = fits.getdata(pz)
    return ac, zf


def _pick_primary(detections):
    """Choose the detection to centroid on.

    Priority (matches rebuild_master_catalog_v2): bright SW zeroframe (snr>10),
    then bright LW zeroframe, then SW ramp, then SW zf, then LW. Returns
    (detection_dict, is_zf).
    """
    sw_zf = lw_zf = sw_ramp = sw_zf_any = lw_any = None
    for d in detections.values():
        sw = d["channel"] == "sw"
        zf = d["mode"] == "zf"
        if sw and zf and d["snr"] > 10 and (sw_zf is None or d["snr"] > sw_zf["snr"]):
            sw_zf = d
        elif (not sw) and zf and d["snr"] > 10 and (lw_zf is None or d["snr"] > lw_zf["snr"]):
            lw_zf = d
        if sw and d["mode"] == "ramp" and sw_ramp is None:
            sw_ramp = d
        if sw and zf and sw_zf_any is None:
            sw_zf_any = d
        if (not sw) and lw_any is None:
            lw_any = d
    if sw_zf is not None:
        return sw_zf, True
    if lw_zf is not None:
        return lw_zf, True
    for cand in (sw_ramp, sw_zf_any, lw_any):
        if cand is not None:
            return cand, (cand["mode"] == "zf")
    return None, False


def populate_lightcurves(cfg, master, out_h5, targets=("Liller1", "Terzan5")):
    """Fill `sources` pixel fields + groupdiff/ZF lightcurves for each object.

    Phase A: pick primary detection, centroid-refine on the autocorr image,
             set RA/Dec from the refined pixel via the LW-aligned WCS.
    Phase B: forced aperture photometry on every available cube (same-detector
             direct; cross-detector/segment via the LW-aligned WCS), double IQR clip.
    Phase C: write per-target sources + lightcurves/{seg}/{mode}/{det} + times.

    Ported from rebuild_master_catalog_v2.py. Requires the group-diff/ZF cubes
    (refs_dir), autocorr references (refs_dir) and LW-aligned WCS (astrometry dir).
    """
    import os
    from astropy.io import fits
    from .photometry import clip_outliers_iqr

    refs = cfg["paths"]["refs_dir"]
    astrom = cfg["paths"].get("astrometry_dir", os.path.join(os.path.dirname(refs), "astrometry"))
    ap_mask = _aperture_mask(cfg["photometry"]["aperture_radius"])
    all_dets = list(cfg["detectors_sw"]) + [cfg["detector_lw"]]
    first = True

    for target in targets:
        rows = [m for m in master if m["target"] == target]
        if not rows:
            continue
        segs = cfg["targets"][target]["segments"]
        wcs = _load_wcs(astrom, target, segs, all_dets)
        ac, zf_ac = _load_autocorr(refs, target, segs, all_dets)
        n = len(rows)
        src = np.zeros(n, dtype=SOURCE_DTYPE)
        primary = []  # (det, seg, px, py) per source

        # ── Phase A: primary detection + centroid refinement + RA/Dec ──
        for i, m in enumerate(rows):
            src[i]["master_id"] = m["master_id"]
            src[i]["ra"], src[i]["dec"] = m["ra"], m["dec"]
            src[i]["best_snr"] = m["best_snr"]
            src[i]["n_detections"] = m["n_detections"]
            p, is_zf = _pick_primary(m["detections"])
            if p is None or not np.isfinite(p.get("px", np.nan)):
                primary.append(None)
                continue
            det, seg = p["detector"], p["segment"]
            px, py = int(round(p["px"])), int(round(p["py"]))
            src[i]["detector"] = det.encode()
            src[i]["det_px"], src[i]["det_py"] = px, py
            ac_img = zf_ac.get((target, seg, det)) if is_zf else ac.get((target, seg, det))
            if ac_img is None:
                ac_img = ac.get((target, seg, det))
            rpx, rpy = float(px), float(py)
            if ac_img is not None:
                fx, fy, ex, ey, ok = refine_centroid(ac_img, px, py)
                scale = 0.063 if det == cfg["detector_lw"] else 0.031
                if ok and np.hypot(fx - px, fy - py) * scale < 0.2:
                    rpx, rpy, src[i]["refined_px_err"], src[i]["refined_py_err"] = fx, fy, ex, ey
            src[i]["refined_px"], src[i]["refined_py"] = rpx, rpy
            primary.append((det, seg, rpx, rpy))
            w = wcs.get((target, seg, det))
            if w is not None:
                sky = w.pixel_to_world(rpx, rpy)
                src[i]["ra"], src[i]["dec"] = float(sky.ra.deg), float(sky.dec.deg)

        # ── Phase B: forced photometry on every available cube ──
        lc_data, time_data = {}, {}
        for seg in segs:
            for mode in ("ramp", "zf"):
                prefix = "groupdiffs" if mode == "ramp" else "zeroframes"
                chunk = 18 if mode == "ramp" else 4
                for det in all_dets:
                    cube_path = os.path.join(refs, f"{prefix}_{target}_{seg}_{det}.fits")
                    if not os.path.exists(cube_path):
                        continue
                    cube = fits.getdata(cube_path, memmap=True)
                    nf, ny, nx = cube.shape
                    with fits.open(cube_path) as hd:
                        tmjd = np.array(hd["DIFF_TIMES"].data["MID_BARY_MJD"], dtype=np.float64)
                    t_hr = (tmjd - tmjd[0]) * 24.0
                    flux = np.zeros((n, nf), dtype=np.float32)
                    n_ext = 0
                    for j in range(n):
                        if primary[j] is None:
                            continue
                        p_det, p_seg, upx, upy = primary[j]
                        if p_det == det and p_seg == seg:
                            ix, iy = int(round(upx)), int(round(upy))
                        else:  # project via LW-aligned WCS
                            sw_, dw_ = wcs.get((target, p_seg, p_det)), wcs.get((target, seg, det))
                            if sw_ is None or dw_ is None:
                                if p_det == det:
                                    ix, iy = int(round(upx)), int(round(upy))
                                else:
                                    continue
                            else:
                                sky = sw_.pixel_to_world(upx, upy)
                                xd, yd = dw_.world_to_pixel_values(sky.ra.deg, sky.dec.deg)
                                ix, iy = int(round(float(xd))), int(round(float(yd)))
                        if ix - _H < 0 or ix + _H + 1 > nx or iy - _H < 0 or iy + _H + 1 > ny:
                            continue
                        cut = np.array(cube[:, iy - _H:iy + _H + 1, ix - _H:ix + _H + 1], dtype=np.float32)
                        np.nan_to_num(cut, nan=0.0, copy=False)
                        raw = np.sum(cut * ap_mask[None, :, :], axis=(1, 2))
                        valid = (raw != 0) & np.isfinite(raw)
                        if valid.sum() < 20:
                            continue
                        fc, tc = clip_outliers_iqr(raw[valid], t_hr[valid], chunk, 2.0)
                        fc, tc = clip_outliers_iqr(fc, tc, chunk, 2.0)
                        keep = set(np.round(tc, 8))
                        for k in range(nf):
                            if round(t_hr[k], 8) in keep:
                                flux[j, k] = raw[k]
                        n_ext += 1
                    lc_data[(seg, mode, det)] = flux
                    time_data[(seg, mode, det)] = tmjd
                    print(f"    {target}/{seg}/{mode}/{det}: extracted {n_ext}/{n}")

        # ── Phase C: write ──
        with h5py.File(out_h5, "w" if first else "r+") as f:
            if f"{target}/sources" in f:
                del f[f"{target}/sources"]
            f.create_dataset(f"{target}/sources", data=src)
            for (seg, mode, det), flux in lc_data.items():
                key = f"{target}/lightcurves/{seg}/{mode}/{det}"
                if key in f:
                    del f[key]
                f.create_dataset(key, data=flux, compression="gzip", compression_opts=4)
            for (seg, mode, det), t in time_data.items():
                key = f"{target}/times/{seg}/{mode}/{det}"
                if key in f:
                    del f[key]
                f.create_dataset(key, data=t)
        first = False
        print(f"  {target}: wrote {n} sources + {len(lc_data)} lightcurve sets")
