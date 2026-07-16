"""Stage 2: group-differenced cube construction.

Differences consecutive non-destructive readout groups within each integration
ramp to synthesize a high-cadence image cube: for 10 groups/integration this
yields 9 group-differences per integration at the ~21.47 s group time, i.e.
972 frames per ~7 hr visit. Statically bad pixels (PIXELDQ != 0) are masked to
NaN. Per-frame timestamps are barycentric (TDB) mid-exposures (temporal
midpoint of each sample's collection interval, not flux-weighted),
read from the ramp file's GROUP table.

Note on GROUPDQ: the mask below keeps the original's ``dq > 10`` threshold,
which no GROUPDQ value in this program's data exceeds (values are 0-5:
DO_NOT_USE/SATURATED/JUMP combinations). Saturated and jump-flagged groups are
therefore deliberately left in the cube — they are handled downstream by IQR
clipping and the saturation-correction stage. Kept as-is for bit-compatibility
with the published cubes (verified identical to production output).

Ported from create_groupdiff_cube() in the original ramp_pipeline.py, made
config-driven (no hardcoded paths). The cubes are large (~16 GB per detector);
they are written once and memory-mapped by downstream stages.
"""
from __future__ import annotations

import glob
import re
import os
from pathlib import Path

import numpy as np
from astropy.io import fits


def get_ramp_files(data_root, target, segment, detector):
    """Return sorted calibrated ramp files for one target/segment/detector."""
    pattern = os.path.join(
        str(data_root), target, segment, "detector1_output", f"*_{detector}_ramp.fits"
    )
    return sorted(glob.glob(pattern))


def _make_3d_header(sci_header, nframes, ny, nx):
    """Build a 3D primary header from a 2D SCI header (drops higher WCS axes)."""
    hdr = sci_header.copy()
    for key in list(hdr.keys()):
        # strip per-extension array-shape keys; we set our own below
        if key in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "NAXIS4"):
            del hdr[key]
    hdr["NAXIS"] = 3
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny
    hdr["NAXIS3"] = nframes
    return hdr


_WCS_CARD = re.compile(
    r"^(WCSAXES|CRPIX[12]|CRVAL[12]|CDELT[12]|CUNIT[12]|CTYPE[12]|"
    r"CD[12]_[12]|PC[12]_[12]|LONPOLE|LATPOLE|RADESYS|EQUINOX|"
    r"A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d+_\d+|B_\d+_\d+|"
    r"AP_\d+_\d+|BP_\d+_\d+|A_DMAX|B_DMAX|WCSNAME)$"
)


def _inject_calibrated_wcs(hdr, cfg, target, segment, detector):
    """Replace the raw pipeline WCS with the calibrated Gaia-tied solution.

    Solutions live in paths.astrometry_dir: SW detectors (nrcb1-4) use
    *_wcs_lw.fits (SW registered onto the Gaia-anchored LW frame; validates to
    ~3-9 mas against Gaia DR3), the LW detector (nrcblong) uses *_wcs_gaia.fits
    (direct LW->Gaia fit, ~3 mas). Run/obtain the astrometry solutions BEFORE
    building cubes so every downstream product carries the good WCS from the
    start; if no solution is found the raw (uncalibrated, ~50-200 mas) pointing
    WCS is kept and a warning is printed.
    """
    tag = "wcs_gaia" if detector == "nrcblong" else "wcs_lw"
    sol_path = os.path.join(
        str(cfg["paths"].get("astrometry_dir", "")),
        f"{target}_{segment}_{detector}_{tag}.fits",
    )
    if not os.path.exists(sol_path):
        print(f"[groupdiff] WARNING: no astrometry solution {sol_path}; "
              "cube keeps the RAW pointing WCS (~0.05-0.2 arcsec off Gaia). "
              "Run the astrometry stage first for calibrated coordinates.")
        return hdr
    for key in list(hdr.keys()):
        if key and _WCS_CARD.match(key):
            del hdr[key]
    sol = fits.getheader(sol_path)
    for card in sol.cards:
        if card.keyword and _WCS_CARD.match(card.keyword):
            hdr.append(card)
    hdr["WCSORIG"] = (tag, "Gaia-tied WCS injected at cube creation")
    print(f"[groupdiff] injected calibrated WCS ({tag}) from {os.path.basename(sol_path)}")
    return hdr


def create_groupdiff_cube(cfg, target, segment, detector, overwrite=False):
    """Create (or reuse) the group-differenced cube for one detector.

    Returns the output FITS path. The file has a primary HDU with the
    (nframes, ny, nx) cube and a ``DIFF_TIMES`` bin-table of barycentric MJD
    midpoints.
    """
    refs_dir = cfg["paths"]["refs_dir"]
    Path(refs_dir).mkdir(parents=True, exist_ok=True)
    outname = os.path.join(refs_dir, f"groupdiffs_{target}_{segment}_{detector}.fits")
    if os.path.exists(outname) and not overwrite:
        print(f"[groupdiff] exists, skipping: {outname}")
        return outname

    ramp_files = get_ramp_files(cfg["paths"]["data_root"], target, segment, detector)
    if not ramp_files:
        raise FileNotFoundError(
            f"No ramp files for {target}/{segment}/{detector} under "
            f"{cfg['paths']['data_root']}. Run stage 1 (calibrate) first."
        )
    print(f"[groupdiff] {target}/{segment}/{detector}: {len(ramp_files)} ramp files")

    diff_frames, diff_times = [], []
    sci_header = None

    for fn in ramp_files:
        # No memmap here: PIXELDQ is a scaled (BZERO) HDU, which astropy refuses
        # to memory-map. Everything below is copied into memory regardless;
        # memmap belongs on cube *reads*, not on ramp reads.
        with fits.open(fn) as hdul:
            sci = hdul["SCI"].data.astype(np.float32)
            groupdq = hdul["GROUPDQ"].data
            pixeldq = hdul["PIXELDQ"].data
            grp_tab = hdul["GROUP"].data
            if sci_header is None:
                sci_header = hdul["SCI"].header.copy()

            n_int, n_grp, ny, nx = sci.shape
            static_bad = pixeldq != 0

            for i_int in range(n_int):
                ramp = sci[i_int]
                dq = groupdq[i_int]
                diffs = ramp[1:] - ramp[:-1]
                # mask static-bad pixels and any diff touching a flagged group
                mask = static_bad[None, ...] | (dq[:-1] > 10) | (dq[1:] > 10)
                diffs[mask] = np.nan

                sel = (
                    (grp_tab["integration_number"] == (i_int + 1))
                    & (grp_tab["group_number"] >= 1)
                    & (grp_tab["group_number"] <= n_grp)
                )
                this_int = grp_tab[sel]
                order_g = np.argsort(this_int["group_number"])
                bary_end = this_int["bary_end_time"][order_g]
                for k in range(n_grp - 1):
                    # mid-exposure: group k is the average of two
                    # frames (equivalent sample at its mean frame time), so the
                    # diff's mid-time is the group-END midpoint minus t_frame/2.
                    tf_half = 0.25 * (bary_end[1] - bary_end[0])
                    diff_times.append(0.5 * (bary_end[k] + bary_end[k + 1]) - tf_half)
                diff_frames.extend(diffs)

    cube = np.stack(diff_frames, axis=0)
    times = np.array(diff_times, dtype=np.float64)

    hdr = _make_3d_header(sci_header, *cube.shape)
    hdr = _inject_calibrated_wcs(hdr, cfg, target, segment, detector)
    hdr["TARGET"] = target
    hdr["SEGMENT"] = segment
    hdr["DETECTOR"] = detector
    hdr["CONTENT"] = "group-differenced cube"

    primary = fits.PrimaryHDU(data=cube, header=hdr)
    col = fits.Column(name="MID_BARY_MJD", format="D", unit="d", array=times)
    time_hdu = fits.BinTableHDU.from_columns([col], name="DIFF_TIMES")
    fits.HDUList([primary, time_hdu]).writeto(outname, overwrite=True)
    print(f"[groupdiff] wrote {outname} ({cube.shape[0]} frames, {cube.shape[1]}x{cube.shape[2]})")
    return outname


def create_zeroframe_cube(cfg, target, segment, detector, overwrite=False):
    """Create (or reuse) the zeroframe cube for one detector.

    Stacks the ZEROFRAME planes of every exposure, dropping the first
    integration's zero frame per exposure (it is systematically offset), so a
    12-exposure x 9-integration visit yields 96 frames. Timestamps are the
    integration midpoints from the INT_TIMES table.

    Ported from build_zeroframe_cube + the cube save in
    process_detector_zeroframes (ramp_pipeline.py). Inherited quirk kept for
    The time column (MID_BARY_MJD) holds the zeroframe mid-exposure
    (reset + tf/2) on the same barycentric timebase as the group-diff cubes.
    NOTE: cubes/catalogs built before 2026-07-02 held UTC integration
    midpoints here instead (~5-7 min early relative to the ramp timebase).
    """
    refs_dir = cfg["paths"]["refs_dir"]
    Path(refs_dir).mkdir(parents=True, exist_ok=True)
    outname = os.path.join(refs_dir, f"zeroframes_{target}_{segment}_{detector}.fits")
    if os.path.exists(outname) and not overwrite:
        print(f"[zeroframe] exists, skipping: {outname}")
        return outname

    ramp_files = get_ramp_files(cfg["paths"]["data_root"], target, segment, detector)
    if not ramp_files:
        raise FileNotFoundError(
            f"No ramp files for {target}/{segment}/{detector} under "
            f"{cfg['paths']['data_root']}. Run stage 1 (calibrate) first."
        )
    print(f"[zeroframe] {target}/{segment}/{detector}: {len(ramp_files)} ramp files")

    zf_list, time_list = [], []
    sci_header = None
    for fn in ramp_files:
        with fits.open(fn) as hdul:
            if "ZEROFRAME" not in hdul:
                print(f"[zeroframe] warning: no ZEROFRAME in {fn}")
                continue
            zf = hdul["ZEROFRAME"].data.astype(np.float32)
            # Zeroframe mid-exposure (~tf/2 = 5.4 s after reset), barycentric,
            # consistent with the group-diff timebase: bary_end(g1) = reset+2tf
            # so zf_mid = bary_end(g1) - 1.5 tf. (The pre-2026-07 catalogs used
            # INT_TIMES int_mid_MJD_UTC here: UTC integration midpoints, ~5-7
            # min off the ramp timebase.)
            gt = hdul["GROUP"].data
            tmid = []
            for intno in np.unique(gt["integration_number"]):
                gi = gt[gt["integration_number"] == intno]
                gi = gi[np.argsort(gi["group_number"])]
                e1, e2 = float(gi["bary_end_time"][0]), float(gi["bary_end_time"][1])
                tmid.append(e1 - 0.75 * (e2 - e1))
            tmid = np.asarray(tmid)
            if sci_header is None and "SCI" in hdul:
                sci_header = hdul["SCI"].header.copy()
        for i in range(1, zf.shape[0]):   # drop first zero frame per exposure
            zf_list.append(zf[i])
            time_list.append(tmid[i])
    if not zf_list:
        raise RuntimeError(f"No zeroframes found in ramps for {target}/{segment}/{detector}")

    cube = np.ascontiguousarray(np.stack(zf_list, axis=0), dtype=np.float32)
    times = np.array(time_list, dtype=np.float64)

    if sci_header is not None:
        hdr = _make_3d_header(sci_header, *cube.shape)
    else:
        hdr = fits.Header()
    hdr = _inject_calibrated_wcs(hdr, cfg, target, segment, detector)
    hdr["TARGET"] = target
    hdr["SEGMENT"] = segment
    hdr["DETECTOR"] = detector
    hdr["CUBETYPE"] = "ZEROFRAME"

    primary = fits.PrimaryHDU(data=cube, header=hdr)
    col = fits.Column(name="MID_BARY_MJD", format="D", unit="d", array=times)
    time_hdu = fits.BinTableHDU.from_columns([col], name="DIFF_TIMES")
    fits.HDUList([primary, time_hdu]).writeto(outname, overwrite=True)
    print(f"[zeroframe] wrote {outname} ({cube.shape[0]} frames)")
    return outname
