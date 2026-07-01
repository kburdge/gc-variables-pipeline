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
| 2. Cubes | `scripts/02_build_cubes.py` → `pipeline/groupdiff.py` | `ramp` | group-diff cubes (16 GB/detector) + zeroframe cubes |
| 3. Detect+extract | `scripts/03_extract.py` (`--mode ramp/zf/both`) → `pipeline/{detect,photometry,periods,extract}.py` | cubes + calints | extraction HDF5s (+ `plot_lightcurve.py`) |
| 4. **Vet (manual)** | human, or shipped labels (`export_vetting_labels.py`) | diagnostic PNGs | `vetting_labels.csv` |
| 5. Build catalog | `scripts/04_build_catalog.py --dedup` + `scripts/05_corrections.py` → `pipeline/{catalog,corrections}.py` | dedup groupings + labels | `master_variable_catalog.h5` |
| 6. View | web viewer (port pending) | catalog + mosaics | Aladin Lite viewer |

Stage 4 is the one human-in-the-loop step (two products: the REAL/FAKE vetting
labels AND the manual deduplication decisions). For reproducibility we **ship
both** (Zenodo: `vetting_labels.csv` + `dedup_groups.csv`) so stage 5 is
deterministic — you do not need to re-vet, and the rebuilt catalog carries the
exact published `master_id`s.

### Stage 3 detail (implemented + validated)
`pipeline/extract.py` orchestrates: build lag-1 autocorrelation reference from
calints (`detect.create_autocorr_reference`) → PSF-matched detection
(`detect.fast_psf_detect`) → r=1.5px aperture photometry on the group-diff cube
(`photometry`) → chunked IQR clip → Lomb-Scargle + BLS period search
(`periods`) → per-detector extraction HDF5. Photometry is the production
algorithm exactly: whole-pixel aperture, pixels NaN in any frame excluded (so
the aperture is constant in time); saturating sources skip the IQR clip and
average the first two group-diffs per ramp instead. `--mode zf` runs the same
flow on the 96-frame zeroframe cube (5σ detection, chunk-4 clip). `--max-sources
N` keeps the top-N detections by SNR for fast/demo runs (the 3σ autocorr
detection is permissive — ~16k raw peaks per detector — and is vetted down by
eye in the full pipeline). Validated bit-for-bit on Terzan5/Segment2/nrcb4:
raw fluxes identical to the production extraction HDF5s (both modes), and ZF
detection reproduces the published 1,649 sources exactly. `plot_lightcurve.py`
renders raw + phase-folded light curves.

**Period window:** the search runs 20 min – 12 hr (`freq_max_cph: 3.0`). The
20-min floor is deliberate — shorter periods alias against the ~3.2 min
integration timescale and the saturation banding pattern (paper §3.6).

### Stage 5 detail (catalog construction, implemented + validated)
`export_vetting_labels.py` (author tool) scans the REAL/ diagnostic folders and
writes `vetting_labels.csv` — the shipped, deterministic record of the human
classification (one row per REAL detection). `export_dedup_groups.py` (author
tool) flattens the manual deduplication decisions into `dedup_groups.csv`.
`catalog.resolve_mapping` (shared by stages 04/05) then loads the shipped
groupings via `load_manual_mapping` — preserving the published `master_id`s —
or falls back to the automatic 0.2" `build_mapping` with a warning (the
fallback yields a few percent more objects and different IDs). Validated:
1,315 objects (915 Liller 1 + 400 Terzan 5) round-trip identically from both
the JSON and CSV forms.

Lightcurve population (`populate_lightcurves`, `04_build_catalog --with-lightcurves`)
picks each object's primary detection, refines its centroid (2D Gaussian on the
autocorr image), does forced aperture photometry on every available cube
(same-detector direct; cross-detector/segment via the LW-aligned WCS),
double-IQR-clips, sets RA/Dec from the refined pixel, and fills the `sources`
pixel fields + `lightcurves/{seg}/{mode}/{det}`. The shipped labels carry
ground-truth px,py so this is decoupled from re-extraction. Validated on
Terzan 5: matched lightcurves reproduce the published catalog at corr = 1.000.
The sat/slope corrections live in `pipeline/corrections.py`
(`scripts/05_corrections.py`), validated on Terzan 5 (best-stage 116/116,
sat_corrected median corr = 1.000). Still unported: the PSF-wing special
reduction, Segment-1 dithered stitching, and the web viewer (see ROADMAP).

**Count note:** the published catalog is defined by the MANUAL dedup (1,315 =
915 + 400) shipped as `dedup_groups.csv`. The automatic fallback on today's
REAL folders yields 1,362 — more vetted sources exist than were published, and
automatic matching also fails to merge some PSF-wing detections by hand-merged
bright stars. Always rebuild with `--dedup` to reproduce the paper.

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
  (read by `pipeline/extract.py` and friends); no code edit needed.
- Reproducing the catalog → `docs/full_reproduction.md`, or the demo in `demo/`.
- Understanding data flow → `docs/flowchart.md`.
- The science / method rationale → the paper's Analysis section and Appendix.

## Environment

- Python 3.11+, conda env from `environment.yml` (key: `jwst`, `crds`, `astropy`,
  `photutils`, `astroquery`, `h5py`, `scipy`, `matplotlib`).
- A GPU is optional (only the every-pixel transit search uses CuPy).
