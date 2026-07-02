# Roadmap — porting the working pipeline into a portable, reproducible package

The science code already exists and produced the published catalog. This repo
makes it **portable** (config-driven, no machine-specific paths), **runnable by
others** (demo subset + documented full path), and **reproducible** (pinned CRDS,
shipped vetting labels + dedup decisions, Zenodo data products). Tracking the
porting work here.

## Status legend
✅ done · 🚧 in progress · ⬜ not started

### Scaffolding
- ✅ Repo structure, README, CLAUDE.md, environment.yml, config template
- ✅ Flowchart + full-reproduction docs
- ✅ Paper reproducibility appendix + flowchart figure (draft)
- ✅ environment.yml pins `jwst==1.17.1` + sets CRDS env on activation

### Front-end (raw → cubes) — the new, reproducibility-critical part
- ✅ `scripts/00_download_mast.py` — astroquery MAST query for GO-5381 uncal (validated against live MAST)
- ✅ `pipeline/detector1.py` — calwebb_detector1 wrapper, `save_calibrated_ramp=True`, **pinned CRDS context**; `--detector` filter
- ✅ `pipeline/groupdiff.py` — group-diff cube construction (ported, config-driven).
      **Bit-identical to the production cubes** (972 timestamps to 0.0 s; frames max|diff| = 0.0)
- ✅ Zeroframe cube construction (`create_zeroframe_cube`; built by `02_build_cubes` by default)
- ✅ CRDS context recorded: **jwst_1322.pmap** (jwst cal 1.17.1), from the calibrated FITS headers
- ⬜ End-to-end test of the demo front-end from a fresh MAST download (cube-onward validated on real data)

### Core (port from the production core)
- ✅ `pipeline/config.py` — YAML config loader, env expansion, CRDS-pin guard
- ✅ `pipeline/detect.py` — autocorr reference (calints + ZF) + PSF-matched detection; WCS in ref headers
- ✅ `pipeline/photometry.py` — production extraction photometry (whole-pixel mask,
      any-NaN pixels excluded → time-constant aperture), saturation test +
      first-groups averaging, chunked IQR clip (both partial-chunk variants)
- ✅ `pipeline/periods.py` — the published search: LS 20 min–12 hr (peak−median)/MAD;
      BLS 30 geomspace durations, SDE significance; 720-min sentinel
- ✅ `pipeline/extract.py` + `scripts/03_extract.py` — ramp **and zf** modes;
      compound `sources` table compatible with the label exporter
- ✅ `scripts/plot_lightcurve.py` — raw + phase-folded light-curve plot
- ✅ **Stage 3 validated bit-for-bit** (Terzan5/Seg2/nrcb4): raw extraction fluxes
      identical to the production HDF5s (max|diff| = 0.0, 300/300 sources, both modes);
      ZF detection reproduces the published 1,649 sources at 5σ exactly
- ⬜ Optional: multiprocess the per-source period search for full (un-capped) runs

### Downstream catalog build
- ✅ `scripts/export_vetting_labels.py` — REAL folders → shipped `vetting_labels.csv`
- ✅ **Manual dedup shipped + ingested**: `scripts/export_dedup_groups.py` flattens
      `master_source_mapping.json` → `dedup_groups.csv`; `catalog.load_manual_mapping`
      ingests either format preserving the published master_ids;
      `catalog.resolve_mapping` shared by 04/05. **Validated: 1,315 objects
      (915 Liller 1 + 400 Terzan 5) round-trip identically; Rapid Burster = master_id 587**
- ✅ `pipeline/catalog.build_mapping` — automatic 0.2" fallback (Terzan5-first IDs; warns it won't reproduce the published catalog)
- ✅ `pipeline/catalog.write_source_table` + `scripts/04_build_catalog.py --dedup`
- ✅ `refine_centroid` ported (for the LC stage)
- ✅ Lightcurve population (`populate_lightcurves`, `04_build_catalog --with-lightcurves`):
      primary-detection pick → centroid refine → forced photometry (same/cross-detector WCS) →
      double IQR clip (production partial-chunk semantics) → RA/Dec.
      **Validated on Terzan 5: 11/12 sky-matched lightcurves reproduce the published catalog at corr = 1.000**
- ✅ Saturation + slope corrections + `best_stage` (`pipeline/corrections.py`, `scripts/05_corrections.py`):
      **validated on Terzan 5: best-stage 116/116 agreement; sat_corrected median corr = 1.000**
- ⬜ Port `add_special_reduction` (PSF-wing annulus extraction, e.g. Rapid Burster);
      key by RA/Dec → master_id, not a hardcoded ID
- ⬜ Segment 1 dithered stitching (port add_segment1_terzan5.py: per-group ratio sat corr →
      slope → per-block slope-aware stitching → exposure rejection → bkg rescale)
- ⬜ Full Liller 1 run (heavier: 4 segments) + web-viewer port (incl. `build_real_catalog`)
- ⬜ Determinism check: rebuild with `--dedup dedup_groups.csv` == published catalog
      (source list + master_ids now guaranteed; verify lightcurves/corrections end-to-end)

### Demo subset
- ✅ Slice picked: Terzan5 / Segment2 / nrcb4 (has the MSP companion + clean binaries)
- ✅ `demo/run_demo.sh` — download → calibrate → cube → detect → extract → plot
- ⬜ Verify wall-clock on a fresh machine and reword the "minutes" claim if calibration dominates

### Data products (Zenodo)
- ⬜ Final `master_variable_catalog.h5`
- ⬜ Label products: `vetting_labels.csv` + `dedup_groups.csv` (export tools done)
- ⬜ WCS products (`*_wcs_lw.fits` / `*_wcs_gaia.fits`) + Gaia DR3 caches (`gaia_*.vot`) + uncal ZF medians — stage-5 and astrometry inputs
- ✅ LW→Gaia astrometry PORTED (`pipeline/astrometry.py`, `scripts/06_astrometry_lw_gaia.py`);
      **validated bit-exact** vs the shipped WCS (0.000 mas CRVAL, N=97/46/45)
- ⬜ SW→LW transfer port (`calibrate_sw_astrometry_lw.py`) — last unported stage
- ⬜ Demo-subset cubes (so the demo can skip the heavy calibration if desired)
- ⬜ Mint DOI; fill into CITATION.cff, README, and the paper appendix
- ⬜ PSF models (F200W/F356W WebbPSF in-flight OPD) — small; consider committing to the repo instead

### Packaging / hygiene
- ✅ LICENSE: MIT (remove stale "confirm" comment in CITATION.cff at release)
- ✅ Pruned the April raw-code dump: `core/` + `preprocessing/` deleted (superseded
      by `pipeline/`); `analysis/` → `unported/` with an archival-only README
- ⬜ `tests/` — at least a smoke test on the demo subset
- ⬜ CI (GitHub Actions) running the smoke test
- ⬜ Fill CITATION.cff `journal`/DOI at release

## Known gaps in the source pipeline (documented, not ported)
- `run_pipeline.sh` Phase 3 (`build_catalog_v3.py` → `*_viewer.h5`) is dead — don't port.
- `PIPELINE.md` references a `variability_filter_v3.py` that doesn't exist — vetting is manual.
- `clean_catalog_lcs.py` (standalone IQR clip) is superseded by the double clip inside
  the LC population — do not port.
- Production `ramp_pipeline.py` has a latent quirk: with `--autocorr-detect`, a rerun
  today would take the ZF detection threshold from `autocorr_sigma` (3σ) rather than
  `zf_sigma`; the published run used 5σ (empirically verified — the port uses
  `detection.zf_sigma` explicitly).
