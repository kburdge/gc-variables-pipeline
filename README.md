# JWST Globular Cluster Variable Star Pipeline

Production pipeline for detecting and characterizing variable stars in globular clusters using JWST NIRCam time-series photometry. Primary targets: **Liller 1** and **Terzan 5**, observed in F200W (SW) and F356W (LW).

## Pipeline Overview

```
Observations (uncal/ramp/calints)
        |
        v
[ramp_pipeline.py] -- group-diff cubes, autocorr detection, aperture photometry, period search
        |
        v
[generate_diagnostics.py] -- multi-panel PNGs for user sorting (REAL/FAKE)
        |
        v
[build_real_catalog.py] -- HDF5 catalog from user-sorted sources
        |
        v
[rebuild_master_catalog.py] -- master catalog with refined centroids
        |
        v
[build_corrected_catalog.py] -- saturation correction, slope correction, best-stage selection
        |
        v
[run_period_search.py] -- LS + BLS on corrected lightcurves
        |
        v
[catalog_server.py] -- web viewer (Aladin Lite + lightcurve display)
```

## Astrometry

Two-stage absolute astrometry pipeline:

1. **LW to Gaia DR3** (`calibrate_lw_astrometry.py`): Aligns nrcblong to Gaia using adaptive FWHM detection with best-roundness selection. Achieves 1-2 mas uncertainty on the absolute frame.

2. **SW to LW** (`calibrate_sw_astrometry_lw.py`): Cross-matches SW detectors to the Gaia-corrected LW frame. 300-600 reference sources per detector, 2.5-6 mas median residuals.

3. **Centroid refinement** (`refine_centroids.py`): 2D Gaussian fitting on autocorrelation images for sub-pixel precision (~0.1 px = 3 mas).

Run the full astrometry pipeline:
```bash
bash analysis/run_astrometry_pipeline.sh
```

## Directory Structure

```
core/
  ramp_pipeline.py        # Main pipeline: cube creation, detection, photometry, period search
  jwst_utils.py           # Shared utilities: DAOStarFinder, aperture photometry, WCS
  source_tracker.py       # Per-source debug tracing

analysis/
  # Astrometry
  calibrate_lw_astrometry.py      # LW -> Gaia DR3 (production, DO NOT DELETE)
  calibrate_sw_astrometry_lw.py   # SW -> LW cross-match (production, DO NOT DELETE)
  build_lw_match_table.py         # FITS match table for TOPCAT inspection
  refine_centroids.py             # Gaussian centroid refinement
  run_astrometry_pipeline.sh      # Full astrometry runner

  # Catalog building
  rebuild_master_catalog.py       # Master catalog with refined positions
  build_corrected_catalog.py      # Sat/slope corrections + best-stage
  build_real_catalog.py           # Catalog from user-sorted diagnostics
  build_catalog_v3.py             # Catalog from extraction HDF5 (v3)

  # Diagnostics & viewer
  generate_diagnostics.py         # Multi-panel diagnostic PNGs
  catalog_server.py               # Web viewer (port 8085)
  static/index.html               # Viewer frontend (Aladin Lite)

  # Period search
  run_period_search.py            # LS + BLS on best-stage lightcurves
  combined_period_search.py       # Combined S3+S4 coherent period search

  # Special
  add_special_reduction.py        # PSF-wing extractions (e.g., rapid burster)
  sw_lw_crossmatch.py             # SW-LW diagnostic cross-match

framework/
  pipeline.py             # Pipeline orchestration class
  config.py               # YAML config management
```

## Key Data Products

- `catalogs/master_variable_catalog.h5` -- Master catalog with positions, lightcurves, corrections
- `catalogs/master_source_mapping.json` -- Source ID to detection filename mapping
- `astrometry/*_wcs_gaia.fits` -- Gaia-corrected LW WCS
- `astrometry/*_wcs_lw.fits` -- LW-aligned SW WCS
- `refs/*_autocorr.fits` -- Autocorrelation reference images
- `extraction/{target}/{seg}/{det}_{mode}.h5` -- Pipeline extraction HDF5

## Environment

- Python 3.12 (miniconda3)
- Key packages: astropy, photutils, numpy, scipy, matplotlib, h5py
- 125 GB RAM, JWST data at `/data/JWST/`
