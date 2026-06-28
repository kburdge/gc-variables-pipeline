# Roadmap — porting the working pipeline into a portable, reproducible package

The science code already exists and produced the published catalog. This repo
makes it **portable** (config-driven, no machine-specific paths), **runnable by
others** (demo subset + documented full path), and **reproducible** (pinned CRDS,
shipped vetting labels, Zenodo data products). Tracking the porting work here.

## Status legend
✅ done · 🚧 in progress · ⬜ not started

### Scaffolding
- ✅ Repo structure, README, CLAUDE.md, environment.yml, config template
- ✅ Flowchart + full-reproduction docs (draft)
- ✅ Paper reproducibility appendix + flowchart figure (draft)

### Front-end (raw → cubes) — the new, reproducibility-critical part
- ⬜ `scripts/00_download_mast.py` — astroquery MAST query for GO-5381 uncal
- ⬜ `pipeline/detector1.py` — calwebb_detector1 wrapper, `save_calibrated_ramp=True`, **pinned CRDS context**
- ⬜ `pipeline/groupdiff.py` — group-diff cube construction from ramp files
- ⬜ Confirm/record the exact CRDS pmap used for the published reduction

### Core (port from /data/Globulars_Pipeline/code/core)
- ⬜ `pipeline/ramp_pipeline.py` — de-hardcode paths, read all params from config
- ⬜ `pipeline/jwst_utils.py` — detection / photometry / WCS utilities
- ⬜ Replace absolute `/data/...` and `/home/kburdge/...` paths with config values

### Downstream catalog build (port from code/analysis)
- ⬜ `pipeline/catalog.py` — mapping → rebuild → corrections → best_stage
- ⬜ Read **shipped vetting labels** instead of scanning local REAL/FAKE folders
- ⬜ `scripts/03_build_catalog.py` orchestrator (cf. build_catalog.sh in the source tree)
- ⬜ Determinism check: rebuilt catalog == published catalog (within tolerance)

### Demo subset
- ⬜ Pick the lightest representative slice (one detector, one segment) incl. a
      showcase variable (e.g. an eclipsing binary or the redback)
- ⬜ `demo/run_demo.sh` — download → calibrate → cube → detect → extract → plot
- ⬜ Verify it runs on a workstation in minutes and reproduces a known light curve

### Data products (Zenodo)
- ⬜ Final `master_variable_catalog.h5`
- ⬜ Vetting labels (REAL source list / `master_source_mapping.json`)
- ⬜ Demo-subset cubes (so the demo can skip the heavy calibration if desired)
- ⬜ Mint DOI; fill into CITATION.cff, README, and the paper appendix

### Packaging / hygiene
- ⬜ LICENSE (confirm: MIT vs BSD-3) and CITATION.cff DOI
- ⬜ `tests/` — at least a smoke test on the demo subset
- ⬜ CI (GitHub Actions) running the smoke test
- ⬜ Strip dead/superseded scripts; do not port v1/duplicate code

## Known gaps in the source pipeline to fix during the port
- `run_pipeline.sh` Phase 3 (`build_catalog_v3.py` → `*_viewer.h5`) is dead — don't port.
- `PIPELINE.md` references a `variability_filter_v3.py` that doesn't exist — vetting is manual.
- Pervasive hardcoded paths (`/data/Globulars_Pipeline`, `/data/JWST`, `/home/kburdge/...`).
