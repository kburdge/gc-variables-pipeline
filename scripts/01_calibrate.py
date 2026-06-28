#!/usr/bin/env python
"""Stage 1: calibrate raw uncal -> calibrated ramps + calints (pinned CRDS).

Runs calwebb_detector1 (saving the calibrated ramps) then Image2Pipeline for
WCS. The CRDS context is pinned from the config; this script refuses to run if
it is left as the placeholder.

Example
-------
    python scripts/01_calibrate.py --config config/pipeline.yaml \
        --target Terzan5 --segment Segment2
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config, apply_crds_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--detector", help="(informational; detector1 runs on all uncal in the dir)")
    ap.add_argument("--max-cores", default="1")
    args = ap.parse_args()

    cfg = load_config(args.config)
    context = apply_crds_env(cfg)   # pin CRDS BEFORE importing jwst
    print(f"[calibrate] pinned CRDS_CONTEXT={context}")

    from pipeline.detector1 import calibrate

    uncal_dir = os.path.join(cfg["paths"]["data_root"], args.target, args.segment)
    out_dir = os.path.join(uncal_dir, "detector1_output")
    calibrate(uncal_dir, out_dir, max_cores=args.max_cores)


if __name__ == "__main__":
    main()
