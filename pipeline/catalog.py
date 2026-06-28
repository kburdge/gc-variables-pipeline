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
            rows.append({
                "target": r["target"], "segment": r["segment"], "mode": r["mode"],
                "channel": r["channel"], "detector": r["detector"],
                "folder": r["folder"], "src_id": int(r["src_id"]),
                "ra": float(r["ra"]), "dec": float(r["dec"]), "snr": float(r["snr"]),
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
                m["detections"][s["folder"]] = {"snr": s["snr"], "detector": s["detector"]}
                if s["snr"] > m["best_snr"]:
                    m["best_snr"], m["ra"], m["dec"] = s["snr"], s["ra"], s["dec"]
                    m["detector"] = s["detector"]
        else:
            master.append({
                "target": s["target"], "ra": s["ra"], "dec": s["dec"],
                "best_snr": s["snr"], "detector": s["detector"],
                "detections": {s["folder"]: {"snr": s["snr"], "detector": s["detector"]}},
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
