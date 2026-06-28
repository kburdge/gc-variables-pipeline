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
           "ra", "dec", "snr", "period_min", "ls_sig", "amplitude", "folder", "filename"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--out", default="vetting_labels.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    diag_dir = cfg["paths"]["diagnostics_dir"]

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
            rows.append({
                "target": target, "segment": seg, "mode": mode, "channel": channel,
                "detector": m.group(3), "src_id": int(m.group(2)),
                "ra": float(m.group(7)), "dec": float(m.group(8)),
                "snr": float(m.group(1)), "period_min": float(m.group(4)),
                "ls_sig": float(m.group(5)), "amplitude": float(m.group(6)),
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
