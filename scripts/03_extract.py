#!/usr/bin/env python
"""Stage 3: detect variables and extract light curves for a detector.

Example
-------
    python scripts/03_extract.py --config config/pipeline.yaml \
        --target Terzan5 --segment Segment2 --detector nrcb4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402
from pipeline.extract import extract_detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--detector", help="one detector; default = all SW + LW from config")
    ap.add_argument("--mode", choices=["ramp", "zf", "both"], default="ramp",
                    help="extraction mode: group-diff cube (ramp), zeroframes (zf), or both")
    ap.add_argument("--max-sources", type=int, help="keep only top-N detections by SNR (demo/quick runs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    detectors = [args.detector] if args.detector else list(cfg["detectors_sw"]) + [cfg["detector_lw"]]
    modes = ["ramp", "zf"] if args.mode == "both" else [args.mode]
    for det in detectors:
        for mode in modes:
            extract_detector(cfg, args.target, args.segment, det, mode=mode,
                             overwrite=args.overwrite, max_sources=args.max_sources)


if __name__ == "__main__":
    main()
