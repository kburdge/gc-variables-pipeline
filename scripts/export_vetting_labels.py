#!/usr/bin/env python
"""Author tool: export the human REAL/FAKE vetting labels to a shipped data product.

The published catalog's source list comes from visual classification (paper
§3.8): diagnostic PNGs were sorted into REAL/ folders, with the source metadata
encoded in each filename. This script reads those REAL folders and writes a
single portable CSV — `vetting_labels.csv` — which is uploaded to Zenodo and
consumed by `scripts/04_build_catalog.py` so the catalog reproduces
deterministically WITHOUT anyone having to re-vet by eye.

Run once, on the machine that holds the diagnostics tree. Most users never run
this — they download the resulting CSV from Zenodo.

Usage:
    python scripts/export_vetting_labels.py --config config/pipeline.yaml \
        --out vetting_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402

# diagnostic folder name -> (target, segment, mode, channel)
FOLDER_MAP = {
    "Liller1_ramp_Segment3": ("Liller1", "Segment3", "ramp", "sw"),
    "Liller1_ramp_Segment3_LW": ("Liller1", "Segment3", "ramp", "lw"),
    "Liller1_ramp_Segment4": ("Liller1", "Segment4", "ramp", "sw"),
    "Liller1_ramp_Segment4_LW": ("Liller1", "Segment4", "ramp", "lw"),
    "Liller1_zf_Segment3": ("Liller1", "Segment3", "zf", "sw"),
    "Liller1_zf_Segment3_LW": ("Liller1", "Segment3", "zf", "lw"),
    "Liller1_zf_Segment4": ("Liller1", "Segment4", "zf", "sw"),
    "Liller1_zf_Segment4_LW": ("Liller1", "Segment4", "zf", "lw"),
    "Terzan5_ramp": ("Terzan5", "Segment2", "ramp", "sw"),
    "Terzan5_ramp_LW": ("Terzan5", "Segment2", "ramp", "lw"),
    "Terzan5_zf": ("Terzan5", "Segment2", "zf", "sw"),
    "Terzan5_zf_LW": ("Terzan5", "Segment2", "zf", "lw"),
}

# SNR{snr}_src{id}_{det}_P{period}min_LS{ls}_amp{amp}_{ra}_{dec}.png
FNAME_RE = re.compile(
    r"SNR(\d+\.\d+)_src(\d+)_(nrc\w+)_P(\d+)min_LS(\d+)_amp([\d.]+)_([\d.]+)_([-\d.]+)\.png$"
)

COLUMNS = ["target", "segment", "mode", "channel", "detector", "src_id",
           "ra", "dec", "px", "py", "snr", "period_min", "ls_sig", "amplitude", "folder", "filename"]


def _load_extraction_sources(extraction_dir):
    """Load each extraction HDF5's source table, keyed by (target,seg,det,mode)."""
    import glob
    import h5py
    ext = {}
    for path in glob.glob(os.path.join(extraction_dir, "*", "*", "*_ramp.h5")) + \
                glob.glob(os.path.join(extraction_dir, "*", "*", "*_zf.h5")):
        parts = path.split(os.sep)
        target, seg = parts[-3], parts[-2]
        det, mode = os.path.basename(path).replace(".h5", "").rsplit("_", 1)
        try:
            with h5py.File(path, "r") as f:
                ext[(target, seg, det, mode)] = f["sources"][:]
        except Exception:
            pass
    return ext


def _match_pixel(ext_sources, ra, dec, ls_sig):
    """Match a label to its extraction source by nearest ra/dec (+LS tiebreak); return (px,py)."""
    import numpy as np
    if ext_sources is None or len(ext_sources) == 0:
        return np.nan, np.nan
    dist = np.hypot(ext_sources["ra"] - ra, ext_sources["dec"] - dec)
    cand = np.where(dist < 0.001)[0]
    if len(cand) == 0:
        cand = [int(np.argmin(dist))]
    best = cand[0]
    lsf = "ls_significance" if "ls_significance" in ext_sources.dtype.names else "ls_sig"
    for c in cand:
        if abs(float(ext_sources[c][lsf]) - ls_sig) < 1:
            best = c
            break
    return float(ext_sources[best]["px"]), float(ext_sources[best]["py"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--out", default="vetting_labels.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    diag_dir = cfg["paths"]["diagnostics_dir"]
    ext = _load_extraction_sources(cfg["paths"]["extraction_dir"])
    print(f"Loaded {len(ext)} extraction source tables for pixel matching")

    rows = []
    for folder, (target, seg, mode, channel) in FOLDER_MAP.items():
        real_dir = os.path.join(diag_dir, folder, "REAL")
        if not os.path.isdir(real_dir):
            continue
        n = 0
        for fname in os.listdir(real_dir):
            if not fname.endswith(".png"):
                continue
            m = FNAME_RE.match(fname)
            if not m:
                continue
            det = m.group(3)
            ra, dec, ls = float(m.group(7)), float(m.group(8)), float(m.group(5))
            tbl = ext.get((target, seg, det, mode))
            if tbl is None:  # zf sources share the ramp pixel grid
                tbl = ext.get((target, seg, det, "ramp"))
            px, py = _match_pixel(tbl, ra, dec, ls)
            rows.append({
                "target": target, "segment": seg, "mode": mode, "channel": channel,
                "detector": det, "src_id": int(m.group(2)),
                "ra": ra, "dec": dec, "px": px, "py": py,
                "snr": float(m.group(1)), "period_min": float(m.group(4)),
                "ls_sig": ls, "amplitude": float(m.group(6)),
                "folder": folder, "filename": fname,
            })
            n += 1
        print(f"  {folder}/REAL: {n} labels")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out}: {len(rows)} REAL detection labels "
          f"across {len({r['target'] for r in rows})} targets.")


if __name__ == "__main__":
    main()
