#!/usr/bin/env bash
#
# Demo: run the pipeline end-to-end on a single detector / single segment.
# Downloads a small slice from MAST, calibrates it (pinned CRDS), builds the
# group-differenced cube, detects variables, extracts light curves, and plots
# the most significant one.
#
# Runs from raw MAST data; expect the calibration step to dominate the runtime.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/config/pipeline.yaml"
PY="${PYTHON:-python}"
[[ -f "$CONFIG" ]] || { echo "Copy config/pipeline.example.yaml -> config/pipeline.yaml first."; exit 1; }

DEMO_TARGET=Terzan5
DEMO_SEGMENT=Segment2
DEMO_DETECTOR=nrcb4

echo ">> [0/4] download demo slice from MAST ($DEMO_TARGET $DEMO_SEGMENT $DEMO_DETECTOR)"
$PY "$ROOT/scripts/00_download_mast.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR" --product uncal

echo ">> [1/4] calibrate (calwebb_detector1 + Image2, pinned CRDS)"
$PY "$ROOT/scripts/01_calibrate.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR"

echo ">> [2/4] build group-differenced cube"
$PY "$ROOT/scripts/02_build_cubes.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR"

echo ">> [3/4] detect variables + extract light curves (top 800 by SNR, for speed)"
$PY "$ROOT/scripts/03_extract.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR" --max-sources 800

echo ">> [4/4] plot the most significant variable"
$PY "$ROOT/scripts/plot_lightcurve.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR" --demo-source

echo ">> done — see demo/lightcurve_*.png for a recovered variable."
