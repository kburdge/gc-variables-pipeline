#!/usr/bin/env python
"""Stage 6: tie the LW (nrcblong) astrometry to Gaia DR3.

Runs the published best-roundness alignment recipe (pipeline/astrometry.py)
for every configured target/segment, printing the per-axis and total
uncertainties on the mean shift and the post-correction residuals. Requires
the uncal ZF median images and the cached Gaia DR3 tables (gaia_{target}.vot)
in paths.astrometry_dir, and calints for the initial WCS.

Example
-------
    python scripts/06_astrometry_lw_gaia.py --config config/pipeline.yaml \
        --out-dir validation_astrometry
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402
from pipeline.astrometry import calibrate_lw_gaia  # noqa: E402

# Observation epochs (MJD) of the undithered visits.
OBS_EPOCHS = {
    ("Terzan5", "Segment2"): 60786.49,
    ("Liller1", "Segment3"): 60787.5,
    ("Liller1", "Segment4"): 60788.97,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--target", action="append", help="restrict to target(s); repeatable")
    ap.add_argument("--out-dir", help="write outputs here instead of paths.astrometry_dir "
                                      "(use for validation runs against shipped products)")
    ap.add_argument("--no-write", action="store_true", help="fit only; write nothing")
    args = ap.parse_args()

    cfg = load_config(args.config)
    targets = args.target or ["Terzan5", "Liller1"]

    results = []
    for (target, seg), mjd in OBS_EPOCHS.items():
        if target not in targets:
            continue
        print(f"=== {target}/{seg} (epoch MJD {mjd}) ===")
        r = calibrate_lw_gaia(cfg, target, seg, mjd,
                              out_dir=args.out_dir, write_wcs=not args.no_write)
        p, q = r["pre"], r["post"]
        print(f"  matches: {r['n_raw']} raw -> {p['n']} after 3xIQR clip")
        print(f"  shift: ({p['shift_ra']:+.1f}, {p['shift_dec']:+.1f}) mas")
        print(f"  scatter: ({p['std_ra']:.1f}, {p['std_dec']:.1f}) mas")
        print(f"  sigma_Gaia: RA {p['unc_ra']:.2f}  Dec {p['unc_dec']:.2f}  "
              f"total {p['unc_tot']:.2f} mas")
        print(f"  post-correction median residual: {q['median_resid']:.1f} mas "
              f"(90th pct {q['p90_resid']:.1f})")
        results.append(r)

    print("\nSUMMARY (for the paper's astrometry tables)")
    for r in results:
        p, q = r["pre"], r["post"]
        print(f"  {r['target']:8s} {r['seg']}: N={p['n']:3d}  "
              f"shift=({p['shift_ra']:+6.1f},{p['shift_dec']:+6.1f}) mas  "
              f"sigma_Gaia={p['unc_tot']:.2f} mas  med_resid={q['median_resid']:.1f} mas")


if __name__ == "__main__":
    main()
