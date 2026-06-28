"""Stage 2: group-differenced cube construction.

Differences consecutive non-destructive readout groups within each integration
ramp to synthesize a high-cadence image cube: for 10 groups/integration this
yields 9 group-differences per integration at the ~21.47 s group time, i.e.
972 frames per ~7 hr visit. Bad pixels (PIXELDQ) and jump/saturation-flagged
samples (GROUPDQ) are masked to NaN. Per-frame timestamps are the barycentric
midpoints of each group pair, read from the ramp file's GROUP table.

Ported from create_groupdiff_cube() in the original ramp_pipeline.py, made
config-driven (no hardcoded paths). The cubes are large (~16 GB per detector);
they are written once and memory-mapped by downstream stages.
"""
from __future__ import annotations

import glob
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
        with fits.open(fn, memmap=True) as hdul:
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
                    diff_times.append(0.5 * (bary_end[k] + bary_end[k + 1]))
                diff_frames.extend(diffs)

    cube = np.stack(diff_frames, axis=0)
    times = np.array(diff_times, dtype=np.float64)

    hdr = _make_3d_header(sci_header, *cube.shape)
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
