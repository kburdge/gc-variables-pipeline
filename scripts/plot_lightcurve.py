#!/usr/bin/env python
"""Plot a light curve from an extraction HDF5 (raw + phase-folded).

With --demo-source, picks the most significant variable (highest LS
significance) — a quick visual confirmation that the demo ran end-to-end.

Example
-------
    python scripts/plot_lightcurve.py --config config/pipeline.yaml \
        --target Terzan5 --segment Segment2 --detector nrcb4 --demo-source
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--segment", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--source", type=int, help="source index to plot")
    ap.add_argument("--demo-source", action="store_true", help="auto-pick highest-significance variable")
    ap.add_argument("--out", help="output PNG path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    h5path = os.path.join(cfg["paths"]["extraction_dir"], args.target, args.segment, f"{args.detector}_ramp.h5")
    with h5py.File(h5path, "r") as f:
        t = f["times_hr"][:]
        flux = f["flux_clipped"][:]
        ls_sig = f["ls_significance"][:]
        periods = f["best_period_min"][:]
        px, py = f["px"][:], f["py"][:]

    if args.source is not None:
        s = args.source
    else:  # demo: most significant variable
        s = int(np.nanargmax(np.where(np.isfinite(ls_sig), ls_sig, -np.inf)))
    lc = flux[s]
    good = np.isfinite(lc)
    tt, yy = t[good], lc[good]
    yy = yy / np.nanmedian(yy)
    P = periods[s]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(tt, yy, ".", ms=2, color="k")
    axes[0].set(xlabel="Time (hr)", ylabel="Normalized flux",
                title=f"{args.target} {args.segment} {args.detector} src#{s} "
                      f"(px={px[s]:.0f},py={py[s]:.0f}, LS={ls_sig[s]:.0f})")
    if np.isfinite(P) and P > 0:
        phase = ((tt * 60.0) % P) / P
        axes[1].plot(phase, yy, ".", ms=2, color="C0")
        axes[1].plot(phase + 1, yy, ".", ms=2, color="C0")
        axes[1].set(xlabel="Phase", ylabel="Normalized flux", title=f"Folded at P = {P:.1f} min")
    else:
        axes[1].text(0.5, 0.5, "no significant period", ha="center", va="center")
    fig.tight_layout()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo",
                                   f"lightcurve_{args.target}_{args.segment}_{args.detector}_src{s}.png")
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {os.path.abspath(out)}  (source {s}, LS sig {ls_sig[s]:.1f}, P {P:.1f} min)")


if __name__ == "__main__":
    main()
