"""Lightcurve correction pipeline: saturation, slope, and best-stage selection.

For each source / segment / channel the pipeline builds up to four light-curve
variants and keeps the one with the lowest mean per-integration scatter,
provided it improves on the baseline by at least the configured gate (10%):

  groupdiff       baseline IQR-clipped aperture photometry (from the cubes)
  sat_corrected   per-pixel quadratic ratio-model correction from the *uncal*
                  ramps (g_i/g_0 vs g_0), which removes saturation "banding"
  slope_corrected flux-dependent within-integration slope removal
  sat_slope       slope correction applied on top of sat_corrected

Each kept stage is stored under {stage}/{target}/{mid}/{seg}_{ch}, and a
best_stage/{target} table records the winner per seg/ch (special_reduction,
if present, always wins). Faithfully ported from build_corrected_catalog.py.

Saturation correction reads the raw uncal ramps on purpose: the JWST linearity
correction introduces artifacts near saturation.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import h5py
from astropy.io import fits
from scipy.ndimage import median_filter

TGROUP_HR = 21.47354 / 3600.0
H = 2                  # cutout half-size (5x5)
RATIO_THRESH = 0.05    # max scatter about the ratio fit to accept a group
TEMPORAL_WINDOW = 7    # running-median window for cosmic-ray masking of g0
CLIP_SIGMA = 3.0
BKG_ANNULUS = {"SW": (10, 16), "LW": (5, 8)}  # (r_in, r_out) px by channel
STAGE_NAMES = ("groupdiff", "sat_corrected", "slope_corrected", "sat_slope")


def _aperture_mask(ap_radius=1.5):
    """Exact circular-aperture weight mask on the (2H+1) cutout grid."""
    from photutils.aperture import CircularAperture
    ap = CircularAperture([(H, H)], r=ap_radius)
    return ap.to_mask(method="exact")[0].to_image((2 * H + 1, 2 * H + 1)).astype(np.float32)


def clip_iqr(f, t, cs=18, iq=2.0):
    """Chunked IQR clip; the final partial chunk is merged into the last full one."""
    n = len(f)
    nc = n // cs
    if nc < 1:
        return f, t
    m = np.ones(n, dtype=bool)
    last_chunk = nc - 1 if n % cs != 0 and nc > 0 else nc
    for b in range(nc):
        s = b * cs
        e = (b + 1) * cs if b < last_chunk else n
        sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75])
        r = q3 - q1
        m[s:e] = (sg >= q1 - iq * r) & (sg <= q3 + iq * r)
    return f[m], t[m]


def find_integrations(t):
    if len(t) < 3:
        return []
    dt = np.diff(t)
    med_dt = np.median(dt[dt > 0])
    breaks = np.where(dt > 2 * med_dt)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(t)]])
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] >= 3]


def integration_scatter(t, f):
    ints = find_integrations(t)
    if len(ints) < 3:
        return np.inf
    return float(np.mean([np.std(f[s:e]) for s, e in ints]))


def apply_slope_correction(t, f):
    """Remove a flux-dependent within-integration linear slope. Returns normalized LC or None."""
    ints = find_integrations(t)
    if len(ints) < 10:
        return None
    slopes, med_fluxes = [], []
    for s, e in ints:
        x = np.arange(e - s, dtype=float)
        slopes.append(np.polyfit(x, f[s:e], 1)[0])
        med_fluxes.append(np.median(f[s:e]))
    slopes, med_fluxes = np.array(slopes), np.array(med_fluxes)
    try:
        poly = np.polyfit(med_fluxes, slopes, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    pred = np.polyval(poly, med_fluxes)
    fc = f.copy()
    for i, (s, e) in enumerate(ints):
        x = np.arange(e - s, dtype=float)
        fc[s:e] -= pred[i] * (x - x.mean())
    return fc / np.median(fc)


def measure_bkg_offset(cube, ix, iy, ch, ap_area):
    r_in, r_out = BKG_ANNULUS[ch]
    nf, ny, nx = cube.shape
    y0, y1 = max(0, iy - r_out), min(ny, iy + r_out + 1)
    x0, x1 = max(0, ix - r_out), min(nx, ix + r_out + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r2 = (xx - ix) ** 2 + (yy - iy) ** 2
    ann = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)
    if ann.sum() < 10:
        return 0.0
    data = np.array(cube[:, y0:y1, x0:x1], dtype=np.float32)
    per_frame = np.array([np.nanmedian(data[f][ann]) for f in range(nf)])
    return float(np.nanmedian(per_frame)) * ap_area


def apply_bkg_rescale(fn, med_raw, bkg_offset):
    """Amplify fractional variability to remove dilution by static background light."""
    source_med = med_raw - bkg_offset
    if source_med <= 0 or bkg_offset <= 0:
        return fn
    return (fn * med_raw - bkg_offset) / source_med


def extract_original(refs_dir, extraction_dir, target, seg, det, ix, iy, ap_mask, ap_area):
    cube_path = os.path.join(refs_dir, f"groupdiffs_{target}_{seg}_{det}.fits")
    h5_path = os.path.join(extraction_dir, target, seg, f"{det}_ramp.h5")
    if not os.path.exists(cube_path) or not os.path.exists(h5_path):
        return None
    cube = fits.getdata(cube_path, memmap=True)
    with h5py.File(h5_path, "r") as f:
        times = f["times_hr"][:] if "times_hr" in f else f["times"][:]
    nf, ny, nx = cube.shape
    if ix - H < 0 or ix + H + 1 > nx or iy - H < 0 or iy + H + 1 > ny:
        return None
    cut = np.array(cube[:, iy - H:iy + H + 1, ix - H:ix + H + 1], dtype=np.float32)
    np.nan_to_num(cut, nan=0.0, copy=False)
    flux = np.sum(cut * ap_mask[None, :, :], axis=(1, 2))
    valid = np.isfinite(flux) & (flux != 0)
    if valid.sum() < 50:
        return None
    ch = "LW" if det == "nrcblong" else "SW"
    bkg = measure_bkg_offset(cube, ix, iy, ch, ap_area)
    med_raw = float(np.median(flux[valid]))
    fn = flux[valid] / med_raw
    so = np.argsort(times[valid])
    fc, tc = clip_iqr(fn[so], times[valid][so])
    fc, tc = clip_iqr(fc, tc)
    fc = apply_bkg_rescale(fc, med_raw, bkg)
    return tc, fc, med_raw, bkg


def sat_correct_det(data_root, target, seg, det, sources_on_det, ap_mask):
    """Per-pixel quadratic ratio-model saturation correction from uncal ramps.

    Returns {mid: (times_hr, flux_norm_clipped)} for sources it could correct.
    """
    uncals = sorted(glob.glob(os.path.join(data_root, target, seg, f"*{det}*uncal.fits")))
    if not uncals:
        return {}
    fb = None
    for uf in uncals:
        try:
            fb = fits.getdata(uf, "INT_TIMES")["int_start_BJD_TDB"].min()
            break
        except Exception:
            continue
    if fb is None:
        return {}

    allpx, spx = set(), {}
    for s in sources_on_det:
        pl = []
        for dy in range(-H, H + 1):
            for dx in range(-H, H + 1):
                w = ap_mask[dy + H, dx + H]
                if w < 0.01:
                    continue
                py, px = s["iy"] + dy, s["ix"] + dx
                if 0 <= py < 2048 and 0 <= px < 2048:
                    allpx.add((py, px))
                    pl.append((py, px, w))
        spx[s["mid"]] = pl

    all_ramps = {k: [] for k in allpx}
    t0s = []
    for uf in uncals:
        d = fits.getdata(uf, "SCI")
        ni = d.shape[0]
        try:
            tsb = fits.getdata(uf, "INT_TIMES")["int_start_BJD_TDB"]
        except Exception:
            continue
        for i in range(ni):
            t0s.append((float(tsb[i]) - fb) * 24.0)
            for (py, px) in allpx:
                all_ramps[(py, px)].append(d[i, :, py, px].astype(np.float64))
    nint = len(t0s)
    if nint == 0:
        return {}
    ng = all_ramps[next(iter(allpx))][0].shape[0]
    max_gd = ng - 1

    pix_info = {}
    for (py, px) in allpx:
        ramps = all_ramps[(py, px)]
        all_gds = [r[1:] - r[:-1] for r in ramps]
        g0v = np.array([g[0] for g in all_gds])
        g0_clean = g0v.copy()
        pos = g0_clean > 0
        if pos.sum() > TEMPORAL_WINDOW:
            g0_filled = np.where(pos, g0_clean, np.median(g0_clean[pos]))
            g0_rm = median_filter(g0_filled, size=TEMPORAL_WINDOW)
            resid = np.abs(g0_clean - g0_rm)
            gmad = np.median(resid[pos]) * 1.4826
            if gmad > 0:
                g0_clean[resid > CLIP_SIGMA * gmad] = np.nan
        max_g, rm = 0, {}
        for g in range(1, max_gd):
            giv = np.array([gd[g] for gd in all_gds])
            vl = np.isfinite(g0_clean) & (g0_clean > 0) & (giv > 0)
            if vl.sum() < 10:
                break
            r = giv[vl] / g0_clean[vl]
            mr = np.median(r)
            if mr < 0.03:
                break
            poly = np.polyfit(g0_clean[vl], r, 2)
            if np.std(r - np.polyval(poly, g0_clean[vl])) / mr > RATIO_THRESH:
                break
            rm[g] = poly
            max_g = g
        pix_info[(py, px)] = {"max_g": max_g, "rm": rm, "g0v": g0_clean, "all_gds": all_gds}

    n_slots = nint * max_gd
    time_grid = np.array([t0s[ii] + (g + 1.25) * TGROUP_HR  # mid-exposure of diff g
                          for ii in range(nint) for g in range(max_gd)])

    results = {}
    for s in sources_on_det:
        mid = s["mid"]
        contribs = []
        for (py, px, ap_w) in spx[mid]:
            info = pix_info.get((py, px))
            if info is None:
                continue
            mg, rm_d, g0v, all_gds_px = info["max_g"], info["rm"], info["g0v"], info["all_gds"]
            pix_vals = np.full((nint, max_gd), np.nan)
            for ii in range(nint):
                g0 = g0v[ii]
                if not np.isfinite(g0) or g0 <= 0:
                    continue
                gds = all_gds_px[ii]
                for g in range(max_gd):
                    if g == 0:
                        pix_vals[ii, g] = g0
                    elif g <= mg and g in rm_d:
                        pr = np.polyval(rm_d[g], g0)
                        if pr > 0.03:
                            pix_vals[ii, g] = gds[g] / pr
            valid = np.isfinite(pix_vals) & (pix_vals > 0)
            if valid.sum() < 20:
                continue
            pmed = np.nanmedian(pix_vals[valid])
            if pmed <= 0:
                continue
            contribs.append((pix_vals / pmed, ap_w))
        if not contribs:
            continue
        ws, wa = np.zeros(n_slots), np.zeros(n_slots)
        for idx in range(n_slots):
            ii, g = idx // max_gd, idx % max_gd
            vals, wts = [], []
            for (cn, w) in contribs:
                if np.isfinite(cn[ii, g]) and cn[ii, g] > 0:
                    vals.append(cn[ii, g])
                    wts.append(w)
            if len(vals) >= 3:
                vals, wts = np.array(vals), np.array(wts)
                med = np.median(vals)
                mad = np.median(np.abs(vals - med)) * 1.4826
                good = np.abs(vals - med) < CLIP_SIGMA * mad if mad > 0 else np.ones(len(vals), bool)
                ws[idx], wa[idx] = np.sum(vals[good] * wts[good]), np.sum(wts[good])
            elif vals:
                ws[idx] = sum(v * w for v, w in zip(vals, wts))
                wa[idx] = sum(wts)
        vc = wa > 0
        if vc.sum() < 50:
            continue
        cf = ws[vc] / wa[vc]
        cfn, tc = cf / np.median(cf), time_grid[vc]
        so = np.argsort(tc)
        cfn_c, tc_c = clip_iqr(cfn[so], tc[so])
        cfn_c, tc_c = clip_iqr(cfn_c, tc_c)
        if len(cfn_c) >= 20:
            results[mid] = (tc_c, cfn_c)
    return results


def _sw_det_for(master_entry):
    """The SW detector for a source: its SW ramp detection, else the primary detector."""
    for d in master_entry["detections"].values():
        if d["channel"] == "sw" and d["mode"] == "ramp":
            return d["detector"]
    return master_entry.get("detector")


def apply_corrections(cfg, master, catalog_path, targets=("Liller1", "Terzan5")):
    """Phase 1-3 of the correction pipeline; writes stages + best_stage into the catalog HDF5."""
    from astropy.wcs import WCS

    refs = cfg["paths"]["refs_dir"]
    data_root = cfg["paths"]["data_root"]
    extr = cfg["paths"]["extraction_dir"]
    astrom = cfg["paths"].get("astrometry_dir", os.path.join(os.path.dirname(refs), "astrometry"))
    ap_mask = _aperture_mask(cfg["photometry"]["aperture_radius"])
    ap_area = float(np.sum(ap_mask))
    gate = 1.0 - cfg["correction"]["improvement_gate"]      # e.g. 0.9 for a 10% gate
    sw_dets = list(cfg["detectors_sw"])
    lw_det = cfg["detector_lw"]
    all_dets = sw_dets + [lw_det]

    # sw_det per master_id
    sw_det_map = {m["master_id"]: _sw_det_for(m) for m in master}

    # ra/dec per source from the (populated) catalog
    src_meta = {}
    with h5py.File(catalog_path, "r") as h5:
        for target in targets:
            if f"{target}/sources" not in h5:
                continue
            for s in h5[f"{target}/sources"][:]:
                src_meta[int(s["master_id"])] = (target, float(s["ra"]), float(s["dec"]))

    # LW-aligned WCS cache
    wcs = {}
    for target in targets:
        segs = cfg["targets"][target]["segments"]
        for seg in segs:
            for det in all_dets:
                for suf in ("wcs_lw", "wcs_gaia"):
                    p = os.path.join(astrom, f"{target}_{seg}_{det}_{suf}.fits")
                    if os.path.exists(p):
                        wcs[(target, seg, det)] = WCS(fits.getheader(p))
                        break

    originals, bkg_info, sat = {}, {}, {}
    # ── Phase 1: extract originals + sat corrections per detector/segment ──
    for target in targets:
        mids = [m for m, (t, _, _) in src_meta.items() if t == target]
        segs = cfg["targets"][target]["segments"]
        for seg in segs:
            for det in all_dets:
                ch = "LW" if det == lw_det else "SW"
                dw = wcs.get((target, seg, det))
                if dw is None:
                    continue
                on_det = []
                for mid in mids:
                    if ch == "SW" and sw_det_map.get(mid) and sw_det_map[mid] != det:
                        continue
                    _, ra, dec = src_meta[mid]
                    xf, yf = dw.world_to_pixel_values(ra, dec)
                    ix, iy = int(round(float(xf))), int(round(float(yf)))
                    if ix < H or ix >= 2048 - H or iy < H or iy >= 2048 - H:
                        continue
                    on_det.append({"mid": mid, "ix": ix, "iy": iy})
                    orig = extract_original(refs, extr, target, seg, det, ix, iy, ap_mask, ap_area)
                    if orig is not None:
                        t_o, f_o, med_raw, bkg = orig
                        originals[(target, mid, seg, ch)] = (t_o, f_o)
                        bkg_info[(target, mid, seg, ch)] = (med_raw, bkg)
                if not on_det:
                    continue
                t1 = time.time()
                for mid_c, lc in sat_correct_det(data_root, target, seg, det, on_det, ap_mask).items():
                    sat[(target, mid_c, seg, ch)] = lc
                print(f"  {target}/{seg}/{det}({ch}): {len(on_det)} src "
                      f"({time.time() - t1:.0f}s)", flush=True)

    # ── Phase 2: build the four stages, gate, pick best ──
    results = {}
    stats = {s: 0 for s in STAGE_NAMES}
    for mid, (target, _, _) in src_meta.items():
        for seg in cfg["targets"][target]["segments"]:
            for ch in ("SW", "LW"):
                key = (target, mid, seg, ch)
                if key not in originals:
                    continue
                t_o, f_o = originals[key]
                sc_o = integration_scatter(t_o, f_o)
                stages = {"groupdiff": (t_o, f_o, sc_o)}
                med_raw, bkg = bkg_info.get(key, (1.0, 0.0))
                if key in sat:
                    t_s, f_s = sat[key]
                    f_s = apply_bkg_rescale(f_s, med_raw, bkg)
                    sc = integration_scatter(t_s, f_s)
                    if sc < gate * sc_o:
                        stages["sat_corrected"] = (t_s, f_s, sc)
                f_sl = apply_slope_correction(t_o, f_o)
                if f_sl is not None:
                    fc, tc = clip_iqr(f_sl, t_o)
                    fc, tc = clip_iqr(fc, tc)
                    fc = apply_bkg_rescale(fc, med_raw, bkg)
                    sc = integration_scatter(tc, fc)
                    if sc < gate * sc_o:
                        stages["slope_corrected"] = (tc, fc, sc)
                if key in sat:
                    t_s, f_s_raw = sat[key]
                    f_ss = apply_slope_correction(t_s, f_s_raw)
                    if f_ss is not None:
                        fc, tc = clip_iqr(f_ss, t_s)
                        fc, tc = clip_iqr(fc, tc)
                        fc = apply_bkg_rescale(fc, med_raw, bkg)
                        sc = integration_scatter(tc, fc)
                        if sc < gate * sc_o:
                            stages["sat_slope"] = (tc, fc, sc)
                cands = {k: v for k, v in stages.items() if k != "groupdiff"}
                best = min(cands, key=lambda k: cands[k][2]) if cands else "groupdiff"
                stages["best"] = best
                results[key] = stages
                stats[best] += 1
    print("Best-stage distribution:", {k: stats[k] for k in STAGE_NAMES})

    # ── Phase 3: write stages + best_stage table ──
    with h5py.File(catalog_path, "r+") as h5:
        for g in STAGE_NAMES:
            if g in h5:
                del h5[g]
        for (target, mid, seg, ch), stages in results.items():
            for sname in STAGE_NAMES:
                if sname in stages:
                    t_lc, f_lc, sc = stages[sname]
                    grp = h5.require_group(f"{sname}/{target}/{mid}")
                    sg = grp.create_group(f"{seg}_{ch}")
                    sg.create_dataset("times", data=t_lc.astype(np.float64))
                    sg.create_dataset("flux_norm", data=f_lc.astype(np.float64))
                    sg.attrs["scatter"] = float(sc)
        for target in targets:
            t_mids = sorted(m for m, (t, _, _) in src_meta.items() if t == target)
            if not t_mids:
                continue
            segs = cfg["targets"][target]["segments"]
            fields = [("master_id", "i4")] + [(f"{seg}_{ch}", "S20") for seg in segs for ch in ("SW", "LW")]
            arr = np.zeros(len(t_mids), dtype=np.dtype(fields))
            for i, mid in enumerate(t_mids):
                arr[i]["master_id"] = mid
                for seg in segs:
                    for ch in ("SW", "LW"):
                        sp = f"special_reduction/{target}/{mid}/{seg}_{ch}"
                        key = (target, mid, seg, ch)
                        if sp in h5:
                            arr[i][f"{seg}_{ch}"] = "special"
                        elif key in results:
                            arr[i][f"{seg}_{ch}"] = results[key]["best"]
                        else:
                            arr[i][f"{seg}_{ch}"] = "groupdiff"
            if f"best_stage/{target}" in h5:
                del h5[f"best_stage/{target}"]
            h5.require_group("best_stage")
            h5.create_dataset(f"best_stage/{target}", data=arr)
    print("Wrote stages + best_stage table.")
