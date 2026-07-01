#!/usr/bin/env python
"""Stage 5b: lightcurve correction pipeline (saturation + slope + best-stage).

Adds sat_corrected / slope_corrected / sat_slope lightcurves and a best_stage
table to an existing master_variable_catalog.h5 (built by 04_build_catalog
--with-lightcurves). Saturation correction reads the raw uncal ramps.

Example
-------
    python scripts/05_corrections.py --config config/pipeline.yaml \
        --labels vetting_labels.csv --catalog data/catalogs/master_variable_catalog.h5 \
        --target Terzan5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402
from pipeline.catalog import resolve_mapping  # noqa: E402
from pipeline.corrections import apply_corrections  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--labels", default="vetting_labels.csv")
    ap.add_argument("--dedup", help="shipped manual dedup decisions "
                                    "(dedup_groups.csv or master_source_mapping.json)")
    ap.add_argument("--catalog", help="catalog HDF5 (default: <catalogs_dir>/master_variable_catalog.h5)")
    ap.add_argument("--target", action="append", help="restrict to target(s); repeatable")
    args = ap.parse_args()

    cfg = load_config(args.config)
    catalog = args.catalog or os.path.join(cfg["paths"]["catalogs_dir"], "master_variable_catalog.h5")
    # 04 and 05 must resolve the mapping identically or master_ids diverge
    master = resolve_mapping(cfg, dedup_path=args.dedup, labels_path=args.labels)
    targets = tuple(args.target) if args.target else ("Liller1", "Terzan5")
    apply_corrections(cfg, master, catalog, targets=targets)


if __name__ == "__main__":
    main()
