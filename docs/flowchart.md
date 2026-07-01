# Pipeline data flow

```
                         JWST GO-5381 (NIRCam Module B, BRIGHT2)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                       │
  ┌───────────────────┐                                            │
  │ 0. DOWNLOAD        │  scripts/00_download_mast.py               │
  │   MAST → uncal     │  (astroquery, program 5381)               │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────┐                                            │
  │ 1. CALIBRATE       │  scripts/01_calibrate.py                   │
  │   calwebb_detector1│  → pipeline/detector1.py                   │
  │   + Image2          │  save_calibrated_ramp=True                 │
  │                    │  *** pinned CRDS_CONTEXT ***               │
  └─────────┬─────────┘                                            │
            ▼            ramp.fits (+ calints.fits w/ WCS+SIP)       │
  ┌───────────────────┐                                            │
  │ 2. BUILD CUBES     │  scripts/02_build_cubes.py                 │
  │   group-diffs      │  → pipeline/groupdiff.py                   │
  │   + zeroframes     │  → 972 frames @ 21.47 s, 16 GB/detector    │
  │                    │  → 96-frame zeroframe cube                 │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────┐    ┌──────────────────────────┐            │
  │ 3. DETECT+EXTRACT  │    │  lag-1 autocorrelation     │           │
  │   scripts/03_      │◀───│  reference image           │           │
  │   extract.py       │    │  (brightness-independent)  │           │
  │   • PSF-matched    │    └──────────────────────────┘            │
  │     detection      │                                            │
  │   • r=1.5px phot   │  → extraction/{target}/{seg}/{det}_{ramp,zf}.h5
  │   • IQR clip       │    (--mode ramp / zf / both)                │
  │   • LS + BLS       │                                            │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────┐                                            │
  │ 4. VET (MANUAL)    │  human visual inspection                   │
  │   REAL vs FAKE     │  ── OR ── load shipped vetting labels      │
  │                    │           (Zenodo; deterministic)          │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────────────────────────────────────────────┐   │
  │ 5. BUILD CATALOG   scripts/04_build_catalog.py +            │   │
  │                    scripts/05_corrections.py                │   │
  │   a. manual dedup groupings (--dedup dedup_groups.csv;      │   │
  │      automatic 0.2" dedup is the non-reproducing fallback)  │   │
  │   b. centroid refine (2D Gaussian on autocorr)             │   │
  │   c. extract groupdiff + ZF LCs at refined pixels          │   │
  │   d. saturation correction (per-pixel ratio model, uncal)  │   │
  │   e. slope correction                                       │◀──┘  astrometry:
  │   f. best-stage selection (10% scatter gate)               │      LW→Gaia DR3,
  │   → master_variable_catalog.h5                             │      SW→LW (WCS only
  └─────────┬─────────────────────────────────────────────────┘      for RA/Dec + xmatch)
            ▼
  ┌───────────────────┐
  │ 6. VIEW / PUBLISH  │  web viewer (Aladin Lite; port pending) ; Zenodo DOI
  └───────────────────┘
```

**Golden rule:** pixel positions are ground truth through stages 2–5. WCS is used
*only* to cross-match SW↔LW detections and to compute the final catalog RA/Dec.

See the paper's reproducibility appendix for the same flow with section
cross-references, and `docs/full_reproduction.md` for the command-by-command
TB-scale procedure.
