#!/usr/bin/env python
"""Stage 0: download raw JWST data for program GO-5381 from MAST.

Queries MAST for the NIRCam exposures and downloads the requested product type
(``uncal`` by default) into the configured data tree, organized as
``<data_root>/<target>/<segment>/``.

Examples
--------
    # list what would be downloaded (no download)
    python scripts/00_download_mast.py --config config/pipeline.yaml --list

    # download the demo slice (one detector, one segment)
    python scripts/00_download_mast.py --config config/pipeline.yaml \
        --target Terzan5 --segment Segment2 --detector nrcb4

    # download everything (TB-scale)
    python scripts/00_download_mast.py --config config/pipeline.yaml --all
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import load_config  # noqa: E402

# Map our segment labels to the JWST observation numbers within program 5381.
# (obs 001 = Terzan5 Seg1, 002 = Terzan5 Seg2, 003 = Liller1 Seg3, 004 = Liller1 Seg4)
SEGMENT_OBS = {
    ("Terzan5", "Segment1"): "001",
    ("Terzan5", "Segment2"): "002",
    ("Liller1", "Segment3"): "003",
    ("Liller1", "Segment4"): "004",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--program", default="5381", help="JWST proposal ID")
    ap.add_argument("--target", help="Liller1 or Terzan5 (default: all in config)")
    ap.add_argument("--segment", help="e.g. Segment2 (default: all for the target)")
    ap.add_argument("--detector", help="restrict to one detector, e.g. nrcb4")
    ap.add_argument("--product", default="uncal", help="product subgroup (uncal, rate, cal, ...)")
    ap.add_argument("--all", action="store_true", help="download all targets/segments in config")
    ap.add_argument("--list", action="store_true", help="list matching products, do not download")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from astroquery.mast import Observations

    if not args.target and not args.all and not args.list:
        ap.error("downloading every target/segment is TB-scale; pass --all to "
                 "confirm, or select a --target (optionally --segment/--detector); "
                 "use --list to preview")

    targets = [args.target] if args.target else list(cfg["targets"].keys())
    jobs = []
    for tgt in targets:
        segs = [args.segment] if args.segment else cfg["targets"][tgt]["segments"]
        for seg in segs:
            obs_id = SEGMENT_OBS.get((tgt, seg))
            if obs_id is None:
                print(f"!! unknown target/segment {tgt}/{seg}; skipping")
                continue
            jobs.append((tgt, seg, obs_id))

    for tgt, seg, obs_id in jobs:
        outdir = os.path.join(cfg["paths"]["data_root"], tgt, seg)
        os.makedirs(outdir, exist_ok=True)
        # obs_id like jw05381<obs>* selects the visit; instrument filters to NIRCam.
        # MAST obs_id looks like 'jw05381-o002_t001_nircam_clear-f200w'
        obs = Observations.query_criteria(
            obs_collection="JWST",
            proposal_id=args.program,
            obs_id=f"jw{int(args.program):05d}-o{obs_id}_*",
        )
        if len(obs) == 0:
            print(f"!! no MAST observations for {tgt}/{seg} (obs {obs_id})")
            continue
        products = Observations.get_product_list(obs)
        filt = Observations.filter_products(
            products,
            productSubGroupDescription=args.product.upper(),
            dataproduct_type="image",
        )
        if args.detector:
            mask = [args.detector in str(fn) for fn in filt["productFilename"]]
            filt = filt[mask]
        print(f"== {tgt}/{seg}: {len(filt)} '{args.product}' products -> {outdir}")
        if args.list:
            for fn in filt["productFilename"][:20]:
                print("   ", fn)
            if len(filt) > 20:
                print(f"    ... (+{len(filt) - 20} more)")
            continue
        if len(filt):
            Observations.download_products(filt, download_dir=outdir, flat=True)

    if args.list:
        print("\n(--list only; nothing downloaded)")


if __name__ == "__main__":
    main()
