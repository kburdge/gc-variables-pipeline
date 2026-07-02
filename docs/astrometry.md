# Astrometric calibration (paper Appendix A)

_Preserved from the original repository README; see the top-level README.md for the overall pipeline._


## Astrometry (Appendix A)

Two-stage absolute astrometry pipeline tying NIRCam positions to the Gaia DR3 reference frame.

### Scripts

| Script | Purpose | Paper section |
|--------|---------|---------------|
| `calibrate_lw_astrometry.py` | LW (nrcblong) to Gaia DR3 alignment | Appendix A.1 |
| `calibrate_sw_astrometry_lw.py` | SW (nrcb1-4) to LW cross-match alignment | Appendix A.2 |
| `build_lw_match_table.py` | Build FITS match table for TOPCAT inspection | Appendix A.1 |
| `run_astrometry_pipeline.sh` | Run full astrometry chain for both targets | Appendix A |

### Method

**Stage 1: LW to Gaia** (`calibrate_lw_astrometry.py`)
- Detect sources in uncal ZF median with DAOStarFinder across a grid of FWHM values (4-30 px)
- For each Gaia source, select the FWHM giving the best (smallest |roundness1|) match
- Filter to |roundness1| < 0.1, IQR clip residuals, compute median shift
- Apply as rigid CRVAL correction preserving JWST SIP distortion
- Terzan 5: 97 matches (G < 17.5), 2.2 mas uncertainty on the mean shift (RA 1.7, Dec 1.5 per axis), 12 mas median residual
- Liller 1: 45-46 matches (G < 18), 1.1 mas uncertainty, 5-6 mas median residual

**Stage 2: SW to LW** (`calibrate_sw_astrometry_lw.py`)
- Cross-match SW and LW zero-frame detections within 0.2"
- 289-631 references per detector, 2.5-6 mas median residuals
- Median shift applied as CRVAL correction

### Stage 1 (LW -> Gaia): PORTED and runnable

```bash
python scripts/06_astrometry_lw_gaia.py --config config/pipeline.yaml
```

Requires the uncal ZF median images and the cached Gaia DR3 tables
(`gaia_{target}.vot`) in `paths.astrometry_dir` (both shipped on Zenodo), plus
calints for the initial WCS. **Validated 2026-07-01: reproduces the published
solution bit-exactly** — N = 97/46/45 matches, identical shifts, and CRVAL
agreement with the shipped `*_wcs_gaia.fits` products to 0.000 mas for all
three fields. The sigma_Gaia values of record: Terzan 5 2.24 mas (total; RA
1.70 / Dec 1.46 per axis), Liller 1 Seg3 1.34 mas, Seg4 1.08 mas.

### Stage 2 (SW -> LW): archival (port pending)

The SW -> LW transfer script is preserved verbatim in [`unported/`](../unported)
(hardcoded paths); its WCS *products* (`{target}_{seg}_{det}_wcs_lw.fits`) are
shipped on Zenodo — place them in `paths.astrometry_dir` and the catalog stage
uses them directly. Original invocation, for reference:

```bash
python unported/calibrate_sw_astrometry_lw.py --target Terzan5 # stage 2 (SW -> LW)
python unported/build_lw_match_table.py --target Terzan5 --seg Segment2  # TOPCAT table
```

### Systematic Error Budget

| Component | Uncertainty |
|-----------|------------|
| Gaia frame tie (LW) | 1-2 mas |
| LW-to-SW transfer | ~0.2 mas |
| Centroid precision | ~3 mas |
| **Total (quadrature)** | **3.2-3.7 mas** |

Validated by PSR J1748-2446A: JWST RA agrees with radio timing to 0.3 mas.

## Environment

- Python 3.12 (miniconda3)
- Key packages: astropy, photutils, numpy, scipy, matplotlib, h5py
