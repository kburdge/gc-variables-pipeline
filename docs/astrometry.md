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
- Terzan 5: 97 matches (G < 17.5), 2.2 mas uncertainty, 12 mas median residual
- Liller 1: 45-46 matches (G < 18), 1.1 mas uncertainty, 5-6 mas median residual

**Stage 2: SW to LW** (`calibrate_sw_astrometry_lw.py`)
- Cross-match SW and LW zero-frame detections within 0.2"
- 289-631 references per detector, 2.5-6 mas median residuals
- Median shift applied as CRVAL correction

### Reference implementation (archival — not runnable from this repo)

The astrometry chain has not been ported to the config-driven `pipeline/`
package; the production scripts are preserved verbatim in [`unported/`](../unported)
as a method record (they contain the author's hardcoded paths). The WCS
*products* they produced (`{target}_{seg}_{det}_wcs_lw.fits` /
`_wcs_gaia.fits`) are shipped on Zenodo — place them in
`paths.astrometry_dir` and the catalog stage uses them directly.

For reference, the original invocation was:

```bash
bash unported/run_astrometry_pipeline.sh        # full chain, both targets
python unported/calibrate_lw_astrometry.py --target Terzan5    # stage 1 (LW -> Gaia)
python unported/calibrate_sw_astrometry_lw.py --target Terzan5 # stage 2 (SW -> LW)
python unported/build_lw_match_table.py --target Terzan5 --seg Segment2  # TOPCAT table
```

### Systematic Error Budget

| Component | Uncertainty |
|-----------|------------|
| Gaia frame tie (LW) | 1-2 mas |
| LW-to-SW transfer | ~0.2 mas |
| Centroid precision | ~3 mas |
| **Total (quadrature)** | **3.2-3.5 mas** |

Validated by PSR J1748-2446A: JWST RA agrees with radio timing to 0.3 mas.

## Environment

- Python 3.12 (miniconda3)
- Key packages: astropy, photutils, numpy, scipy, matplotlib, h5py
