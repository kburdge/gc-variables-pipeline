#!/usr/bin/env python
"""Stage 2: build group-differenced cubes from calibrated ramp files.

Example
-------
    # one detector (demo)
    python scripts/02_build_cubes.py --config config/pipeline.yaml \
        --target Terzan5 --segment Segment2 --detector nrcb4

    # all SW detectors for a target/segment
    python scripts/02_build_cubes.py --config config/pipeline.yaml \
        --target Liller1 --segment Segment3
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402
from pipeline.groupdiff import create_groupdiff_cube  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--detector", help="one detector; default = all SW + LW from config")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.detector:
        detectors = [args.detector]
    else:
        detectors = list(cfg["detectors_sw"]) + [cfg["detector_lw"]]

    for det in detectors:
        create_groupdiff_cube(cfg, args.target, args.segment, det, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
