#!/usr/bin/env python
"""Author tool: export the manual deduplication decisions to dedup_groups.csv.

The published 1,315-object catalog is defined by the author's MANUAL
deduplication (1-arcsec spatial grouping + visual inspection, including 83
groups split into multiple distinct sources). Those decisions live in the
private diagnostics/cross_match folder tree and its derived
master_source_mapping.json. This tool flattens them into a portable CSV — one
row per (object, diagnostics-folder detection) — released on Zenodo so
scripts/04_build_catalog.py --dedup can rebuild the catalog exactly, with the
published master_ids.

Usage
-----
    python scripts/export_dedup_groups.py \
        --mapping /path/to/master_source_mapping.json \
        --out dedup_groups.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.catalog import _mode_channel_from_folder  # noqa: E402

FIELDS = ["master_id", "target", "obj_group", "split", "obj_ra", "obj_dec",
          "obj_best_snr", "folder", "filename", "detector", "segment",
          "mode", "channel", "snr", "px", "py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True,
                    help="path to master_source_mapping.json (the canonical manual mapping)")
    ap.add_argument("--out", default="dedup_groups.csv")
    args = ap.parse_args()

    with open(args.mapping) as fh:
        entries = json.load(fh)

    n_rows = 0
    counts = {}
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for e in entries:
            counts[e["target"]] = counts.get(e["target"], 0) + 1
            for folder, d in e["detections"].items():
                mode, channel = _mode_channel_from_folder(folder)
                w.writerow({
                    "master_id": e["master_id"], "target": e["target"],
                    "obj_group": e.get("obj_folder", ""),
                    "split": e.get("sub_folder") or "",
                    "obj_ra": f"{e['ra']:.6f}", "obj_dec": f"{e['dec']:.6f}",
                    "obj_best_snr": e["best_snr"],
                    "folder": folder, "filename": d.get("filename", ""),
                    "detector": d["det"], "segment": d["segment"],
                    "mode": mode, "channel": channel,
                    "snr": d["snr"], "px": d["px"], "py": d["py"],
                })
                n_rows += 1

    print(f"wrote {args.out}: {len(entries)} objects "
          + " ".join(f"{t}={n}" for t, n in sorted(counts.items()))
          + f", {n_rows} detection rows")


if __name__ == "__main__":
    main()
