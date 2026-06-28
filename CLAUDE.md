# CLAUDE.md — gc-variables-pipeline

Orientation for working in this repository with Claude Code (or for any reader
who wants the mental model fast). This is the public reproduction package for
Burdge et al. (2026), the JWST/NIRCam variable-star census of Liller 1 and
Terzan 5.

## What this pipeline is

It converts raw JWST/NIRCam `uncal` exposures into a catalog of variable stars,
by **group-differencing** the non-destructive detector reads within each
integration ramp to synthesize ~21 s cadence light curves. The headline methods
are (1) a **lag-1 autocorrelation** detection image that is independent of source
brightness, and (2) a multi-stage **saturation/slope correction** that recovers
photometry from sources whose cores saturate.

## The pipeline in stages

| Stage | Module / script | Input | Output |
|-------|-----------------|-------|--------|
| 0. Download | `scripts/00_download_mast.py` | MAST (program GO-5381) | `uncal` exposures |
| 1. Calibrate | `scripts/01_calibrate.py` → `pipeline/detector1.py` | `uncal` + **pinned CRDS** | calibrated `ramp` + `calints` |
| 2. Cubes | `pipeline/groupdiff.py` | `ramp` | group-diff cubes (16 GB/detector) |
| 3. Detect+extract | `scripts/03_extract.py` → `pipeline/{detect,photometry,periods,extract}.py` | cubes + calints | extraction HDF5 (+ `plot_lightcurve.py`) |
| 4. **Vet (manual)** | human, or shipped labels | diagnostic PNGs | REAL/FAKE labels |
| 5. Build catalog | `scripts/04_build_catalog.py` → `pipeline/catalog.py` (TODO) | labels + extraction | `master_variable_catalog.h5` |
| 6. View | `pipeline/server.py` (TODO) | catalog + mosaics | web viewer (Aladin Lite) |

Stage 4 is the one human-in-the-loop step. For reproducibility we **ship the
vetting labels** (Zenodo) so stage 5 is deterministic; you do not need to re-vet.

### Stage 3 detail (implemented + validated)
`pipeline/extract.py` orchestrates: build lag-1 autocorrelation reference from
calints (`detect.create_autocorr_reference`) → PSF-matched detection
(`detect.fast_psf_detect`) → r=1.5px aperture photometry on the group-diff cube
(`photometry`) → chunked IQR clip → Lomb-Scargle + BLS period search
(`periods`) → per-detector extraction HDF5. `--max-sources N` keeps the top-N
detections by SNR for fast/demo runs (the 3σ autocorr detection is permissive —
~16k raw peaks per detector — and is vetted down by eye in the full pipeline).
Validated end-to-end on Terzan5/Segment2/nrcb4: recovers the 3–7 hr binary
period population (e.g. a clean 3.77 hr variable). `plot_lightcurve.py` renders
raw + phase-folded light curves.

**Period window:** the search runs 20 min – 12 hr (`freq_max_cph: 3.0`). The
20-min floor is deliberate — shorter periods alias against the ~3.2 min
integration timescale and the saturation banding pattern (paper §3.6).

## Conventions that matter (learned the hard way)

- **Pixel positions are ground truth, not WCS.** WCS is used only to (a)
  cross-match SW↔LW detections and (b) compute final catalog RA/Dec. Never use
  WCS to determine an extraction pixel — astrometry was revised several times.
- **CRDS context must be pinned.** JWST calibration is only reproducible with a
  fixed `CRDS_CONTEXT`. It is set in `environment.yml` and the config; do not
  let it float to "latest."
- **Saturation correction reads `uncal`, not the calibrated ramp** — the JWST
  linearity correction introduces artifacts near saturation.
- **Group-diff cubes are huge (16 GB each).** Always memory-map; never load whole.
- **All paths come from `config/pipeline.yaml`.** There are no hardcoded
  machine paths — if you find one, it's a bug to fix, not a pattern to follow.
- **Don't pipe long-running stage output through `head`** — SIGPIPE kills the
  writer. Use `PYTHONUNBUFFERED=1` for live logs.

## Where to start a task

- Changing detection/photometry/period-search params → `config/pipeline.yaml`
  (read by `pipeline/ramp_pipeline.py`); no code edit needed.
- Reproducing the catalog → `docs/full_reproduction.md`, or the demo in `demo/`.
- Understanding data flow → `docs/flowchart.md`.
- The science / method rationale → the paper's Analysis section and Appendix.

## Environment

- Python 3.11+, conda env from `environment.yml` (key: `jwst`, `crds`, `astropy`,
  `photutils`, `astroquery`, `h5py`, `scipy`, `matplotlib`).
- A GPU is optional (only the every-pixel transit search uses CuPy).
