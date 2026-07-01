#!/bin/bash
# ===========================================================================
# Full astrometry pipeline
# ===========================================================================
# *** DO NOT DELETE THIS FILE ***
#
# Runs the complete astrometric calibration chain:
#   1. LW (nrcblong) -> Gaia DR3 alignment
#   2. SW (nrcb1-4) -> LW cross-match alignment
#
# Prerequisites:
#   - Uncal ZF medians in astrometry/ (built by build_uncal_zf_median.py)
#   - Gaia catalogs in astrometry/ (queried by align_to_gaia.py)
#   - Calints files in /data/JWST/{target}/{segment}/
#
# Usage:
#   bash run_astrometry_pipeline.sh
# ===========================================================================

set -e
PYTHON=/home/kburdge/miniconda3/bin/python
CODE=/data/Globulars_Pipeline/code/analysis

echo "============================================"
echo "Step 1: LW -> Gaia alignment"
echo "============================================"
$PYTHON $CODE/calibrate_lw_astrometry.py --target Terzan5
$PYTHON $CODE/calibrate_lw_astrometry.py --target Liller1

echo ""
echo "============================================"
echo "Step 2: SW -> LW alignment"
echo "============================================"
$PYTHON $CODE/calibrate_sw_astrometry_lw.py --target Terzan5
$PYTHON $CODE/calibrate_sw_astrometry_lw.py --target Liller1

echo ""
echo "============================================"
echo "Done. Outputs in astrometry/"
echo "============================================"
