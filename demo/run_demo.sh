#!/usr/bin/env bash
#
# Demo: run the front-end on a single detector / single segment.
# Downloads a small slice from MAST, calibrates it (pinned CRDS), and builds the
# group-differenced cube. Detection/extraction/plotting are the next stages
# being ported (see ROADMAP.md) and will be appended here.
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

echo ">> [0/2] download demo slice from MAST ($DEMO_TARGET $DEMO_SEGMENT $DEMO_DETECTOR)"
$PY "$ROOT/scripts/00_download_mast.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR" --product uncal

echo ">> [1/2] calibrate (calwebb_detector1 + Image2, pinned CRDS)"
$PY "$ROOT/scripts/01_calibrate.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR"

echo ">> [2/2] build group-differenced cube"
$PY "$ROOT/scripts/02_build_cubes.py" --config "$CONFIG" \
    --target "$DEMO_TARGET" --segment "$DEMO_SEGMENT" --detector "$DEMO_DETECTOR"

echo ">> front-end complete. Cube written under the configured refs_dir."
echo ">> (detection + extraction + light-curve plot: coming next — see ROADMAP.md)"
