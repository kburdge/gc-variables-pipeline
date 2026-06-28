"""Stage 1: JWST calibration (calwebb_detector1 + Image2).

Runs ``Detector1Pipeline`` with ``save_calibrated_ramp=True`` — the calibrated
ramp files are the essential product for this pipeline (MAST does not archive
them by default), because they preserve the individual group reads with the
non-linearity correction applied, which is what group-differencing needs.
``Image2Pipeline`` is then run on the rate-ints to attach the WCS + SIP
distortion used for astrometry.

CRDS context MUST be pinned (see pipeline.config.apply_crds_env) for the
calibration to be reproducible. This module imports ``jwst`` lazily so the
caller can set the CRDS environment first.

Ported from the original process_detector1.py / process_image2.py used for the
published reduction.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path


def run_detector1(uncal_files, output_dir, max_cores: str = "1"):
    """Run Detector1Pipeline on a list of uncal files, saving calibrated ramps.

    Produces ``*_rateints.fits`` and ``*_ramp.fits`` in ``output_dir``.
    """
    from jwst.pipeline import Detector1Pipeline  # lazy import (after CRDS pin)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for infile in sorted(uncal_files):
        print(f"[detector1] {os.path.basename(infile)} (CRDS_CONTEXT={os.environ.get('CRDS_CONTEXT')})")
        Detector1Pipeline.call(
            infile,
            save_results=True,
            save_calibrated_ramp=True,   # <-- the key flag: keep the ramps
            output_dir=str(output_dir),
            steps={
                "jump": {"maximum_cores": max_cores},
                "ramp_fit": {"maximum_cores": max_cores},
            },
        )
    return sorted(glob.glob(os.path.join(str(output_dir), "*_rateints.fits")))


def run_image2(rateints_files, output_dir):
    """Run Image2Pipeline on rate-ints files, producing ``*_calints.fits`` with WCS."""
    from jwst.pipeline import Image2Pipeline  # lazy import (after CRDS pin)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipe = Image2Pipeline()
    for f in sorted(rateints_files):
        print(f"[image2] {os.path.basename(f)}")
        pipe.call(f, output_dir=str(output_dir), save_results=True)
    return sorted(glob.glob(os.path.join(str(output_dir), "*_calints.fits")))


def calibrate(uncal_dir, output_dir, max_cores: str = "1"):
    """Full stage 1: detector1 (-> ramp + rateints) then image2 (-> calints)."""
    uncal = sorted(glob.glob(os.path.join(str(uncal_dir), "*_uncal.fits")))
    if not uncal:
        raise FileNotFoundError(f"No *_uncal.fits in {uncal_dir}")
    rateints = run_detector1(uncal, output_dir, max_cores=max_cores)
    calints = run_image2(rateints, os.path.join(str(output_dir), "calints"))
    print(f"[calibrate] {len(uncal)} uncal -> {len(rateints)} rateints, {len(calints)} calints")
    return rateints, calints
