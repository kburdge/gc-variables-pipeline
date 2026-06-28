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
- ✅ `scripts/00_download_mast.py` — astroquery MAST query for GO-5381 uncal (validated against live MAST)
- ✅ `pipeline/detector1.py` — calwebb_detector1 wrapper, `save_calibrated_ramp=True`, **pinned CRDS context**
- ✅ `pipeline/groupdiff.py` — group-diff cube construction from ramp files (ported, config-driven)
- ✅ CRDS context recorded: **jwst_1322.pmap** (jwst cal 1.17.1), from the calibrated FITS headers
- ⬜ End-to-end test of the demo front-end on real data (download → calibrate → cube)

### Core (port from /data/Globulars_Pipeline/code/core)
- ✅ `pipeline/config.py` — YAML config loader, env expansion, CRDS-pin guard
- ✅ `pipeline/detect.py` — autocorr reference + PSF-matched detection (ported)
- ✅ `pipeline/photometry.py` — aperture photometry + chunked IQR clip (ported)
- ✅ `pipeline/periods.py` — Lomb-Scargle + BLS, (peak-median)/MAD significance (ported)
- ✅ `pipeline/extract.py` + `scripts/03_extract.py` — detect→photometer→clip→period→HDF5
- ✅ `scripts/plot_lightcurve.py` — raw + phase-folded light-curve plot
- ✅ **Demo validated end-to-end on real data** (Terzan5/Seg2/nrcb4): recovers the
      3–7 hr binary population; clean 3.77 hr variable. Period window set to the
      paper's 20 min–12 hr (was inheriting 1 min, which aliased on the integration timescale).
- ⬜ Optional: multiprocess the per-source period search for full (un-capped) runs

### Downstream catalog build (port from code/analysis)
- ✅ `scripts/export_vetting_labels.py` — REAL folders → shipped `vetting_labels.csv`
- ✅ `pipeline/catalog.build_mapping` — 0.2" SNR-ordered dedup → unique objects (faithful: matches original exactly)
- ✅ `pipeline/catalog.write_source_table` + `scripts/04_build_catalog.py` — per-target `sources` tables
- ✅ Read **shipped vetting labels** instead of scanning local REAL/FAKE folders
- ✅ `refine_centroid` ported (for the LC stage)
- ⬜ Lightcurve population: pixel positions from extraction → centroid refine → forced photometry on cubes → RA/Dec via LW WCS → fill `sources` pixel fields + `lightcurves/{seg}/{mode}/{det}`
- ⬜ Saturation + slope corrections + `best_stage` table (port build_corrected_catalog.py; needs uncal)
- ⬜ Determinism check: rebuilt catalog == published catalog (note: current folders → 1,362 vs published 1,315; folders grew ~47 since last build)

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
