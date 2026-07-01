#!/usr/bin/env python
"""Stage 5: build the master catalog source list from shipped vetting labels.

Deduplicates the human-vetted REAL detections into unique objects (paper §3.7)
and writes the per-target `sources` tables into master_variable_catalog.h5.

The vetting labels are produced once by scripts/export_vetting_labels.py and
shipped on Zenodo; download them rather than re-vetting by eye.

Example
-------
    python scripts/04_build_catalog.py --config config/pipeline.yaml \
        --labels vetting_labels.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402
from pipeline.catalog import resolve_mapping, write_source_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--labels", default="vetting_labels.csv", help="shipped vetting-label CSV")
    ap.add_argument("--dedup", help="shipped manual dedup decisions "
                                    "(dedup_groups.csv or master_source_mapping.json); "
                                    "required to reproduce the published catalog exactly")
    ap.add_argument("--out", help="output catalog HDF5 (default: <catalogs_dir>/master_variable_catalog.h5)")
    ap.add_argument("--with-lightcurves", action="store_true",
                    help="also populate centroids + lightcurves (needs cubes, autocorr, WCS)")
    ap.add_argument("--target", action="append", help="restrict to target(s); repeatable")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or os.path.join(cfg["paths"]["catalogs_dir"], "master_variable_catalog.h5")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    master = resolve_mapping(cfg, dedup_path=args.dedup, labels_path=args.labels)
    targets = tuple(args.target) if args.target else ("Liller1", "Terzan5")

    if args.with_lightcurves:
        from pipeline.catalog import populate_lightcurves
        populate_lightcurves(cfg, master, out, targets=targets)
    else:
        counts = write_source_table(master, out, targets=targets)
        total = sum(counts.values())
        print(f"\nUnique objects: {total}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"Wrote source tables to {out}")
        print("(run with --with-lightcurves to also populate centroids + lightcurves)")


if __name__ == "__main__":
    main()
