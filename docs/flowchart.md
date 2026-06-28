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
  │ 1. CALIBRATE       │  pipeline/detector1.py                     │
  │   calwebb_detector1│  save_calibrated_ramp=True                 │
  │   + Image2          │  *** pinned CRDS_CONTEXT ***               │
  └─────────┬─────────┘                                            │
            ▼            ramp.fits (+ calints.fits w/ WCS+SIP)       │
  ┌───────────────────┐                                            │
  │ 2. GROUP-DIFF CUBE │  pipeline/groupdiff.py                     │
  │   consecutive group│  → 972 frames @ 21.47 s, 16 GB/detector    │
  │   differences      │                                            │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────┐    ┌──────────────────────────┐            │
  │ 3. DETECT+EXTRACT  │    │  lag-1 autocorrelation     │           │
  │   pipeline/ramp_   │◀───│  reference image           │           │
  │   pipeline.py      │    │  (brightness-independent)  │           │
  │   • PSF-matched    │    └──────────────────────────┘            │
  │     detection      │                                            │
  │   • r=1.5px phot   │  → extraction/{target}/{seg}/{det}_{ramp,zf}.h5
  │   • IQR clip       │  → diagnostic PNGs                          │
  │   • LS + BLS       │                                            │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────┐                                            │
  │ 4. VET (MANUAL)    │  human visual inspection of PNGs           │
  │   REAL vs FAKE     │  ── OR ── load shipped vetting labels      │
  │                    │           (Zenodo; deterministic)          │
  └─────────┬─────────┘                                            │
            ▼                                                       │
  ┌───────────────────────────────────────────────────────────┐   │
  │ 5. BUILD CATALOG   pipeline/catalog.py (scripts/03_…)       │   │
  │   a. mapping + dedup (0.2") + friends-of-friends (1.0")     │   │
  │   b. centroid refine (2D Gaussian on autocorr)             │   │
  │   c. extract groupdiff LCs at refined pixels               │   │
  │   d. saturation correction (per-pixel ratio model, uncal)  │   │
  │   e. slope correction                                       │◀──┘  astrometry:
  │   f. best-stage selection (10% scatter gate)               │      LW→Gaia DR3,
  │   → master_variable_catalog.h5                             │      SW→LW (WCS only
  └─────────┬─────────────────────────────────────────────────┘      for RA/Dec + xmatch)
            ▼
  ┌───────────────────┐
  │ 6. VIEW / PUBLISH  │  pipeline/server.py (Aladin Lite) ; Zenodo DOI
  └───────────────────┘
```

**Golden rule:** pixel positions are ground truth through stages 2–5. WCS is used
*only* to cross-match SW↔LW detections and to compute the final catalog RA/Dec.

See the paper's reproducibility appendix for the same flow with section
cross-references, and `docs/full_reproduction.md` for the command-by-command
TB-scale procedure.
