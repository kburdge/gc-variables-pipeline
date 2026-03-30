#!/usr/bin/env python
"""
Fit WCS parameters using UltraNest to minimize chi-squared between
JWST centroid positions and Gaia DR3 positions.

Parameters sampled:
  - dCRVAL1, dCRVAL2: shifts in reference position (arcsec)
  - dPA: rotation angle offset (arcsec, converted to CD matrix perturbation)
  - dScale: fractional scale change (ppm)

Chi-squared uses Gaia positional uncertainties (propagated) + JWST centroid
uncertainties (from FWHM/SNR) added in quadrature.

Uses only sources in the 'good' subfolder if available, otherwise all matches.

Usage:
    python fit_wcs_mcmc.py --det nrcb4 [--target Terzan5] [--use-good]
"""
import numpy as np
import os
import sys
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import ultranest
import ultranest.stepsampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'


def build_wcs_from_params(wcs_base, dcrval1_arcsec, dcrval2_arcsec, dpa_arcsec, dscale_ppm):
    """Create a perturbed WCS from base + parameter offsets."""
    hdr = wcs_base.to_header()

    # Shift
    cos_dec = np.cos(np.radians(hdr['CRVAL2']))
    hdr['CRVAL1'] += dcrval1_arcsec / 3600.0 / cos_dec
    hdr['CRVAL2'] += dcrval2_arcsec / 3600.0

    # Rotation: perturb CD matrix by small angle dpa (in arcsec -> radians)
    dpa_rad = dpa_arcsec / 3600.0 * np.pi / 180.0
    cos_dpa = np.cos(dpa_rad)
    sin_dpa = np.sin(dpa_rad)

    # Scale: multiplicative perturbation
    scale = 1.0 + dscale_ppm * 1e-6

    cd = np.array([[hdr.get('CD1_1', 0), hdr.get('CD1_2', 0)],
                    [hdr.get('CD2_1', 0), hdr.get('CD2_2', 0)]])

    rot = np.array([[cos_dpa, -sin_dpa],
                     [sin_dpa, cos_dpa]])
    cd_new = scale * rot @ cd

    hdr['CD1_1'] = cd_new[0, 0]
    hdr['CD1_2'] = cd_new[0, 1]
    hdr['CD2_1'] = cd_new[1, 0]
    hdr['CD2_2'] = cd_new[1, 1]

    return WCS(hdr)


def main():
    det = 'nrcb4'
    target = 'Terzan5'
    seg = 'Segment2'
    use_good = '--use-good' in sys.argv

    for arg in sys.argv[1:]:
        if arg.startswith('--det='):
            det = arg.split('=')[1]
        elif arg.startswith('--target='):
            target = arg.split('=')[1]

    # Load match table
    tbl_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_match_table.fits'
    tbl = Table.read(tbl_path)
    print(f'Loaded {len(tbl)} matches from {tbl_path}')

    # Filter to 'good' sources if requested
    if use_good:
        good_dir = f'{ASTROM_DIR}/{det}_cutouts/good'
        if os.path.isdir(good_dir):
            good_files = set(os.listdir(good_dir))
            # Match by pixel position encoded in filename
            keep = []
            for i, row in enumerate(tbl):
                ix, iy = int(round(row['jwst_x'])), int(round(row['jwst_y']))
                # Filename pattern: G{mag}_sep{sep}mas_px{x}_{y}.png
                pattern = f'px{ix:04d}_{iy:04d}'
                if any(pattern in f for f in good_files):
                    keep.append(i)
            tbl = tbl[keep]
            print(f'Using {len(tbl)} good sources')
        else:
            print(f'No good/ directory found, using all')

    # Load base WCS (calints)
    import glob
    cal_files = sorted([f for f in glob.glob(f'/data/JWST/{target}/{seg}/*{det}*cal.fits') if 'uncal' not in f])
    wcs_base = WCS(fits.getheader(cal_files[0], 'SCI'))

    # Data arrays
    jx = np.array(tbl['jwst_x']); jy = np.array(tbl['jwst_y'])
    gra = np.array(tbl['gaia_ra']); gdec = np.array(tbl['gaia_dec'])

    # Uncertainties in mas (Gaia + centroid added in quadrature)
    gaia_ra_err = np.array(tbl['gaia_ra_err_mas'])  # mas
    gaia_dec_err = np.array(tbl['gaia_dec_err_mas'])
    centroid_err = np.array(tbl['centroid_err_mas'])  # mas

    sigma_ra = np.sqrt(gaia_ra_err**2 + centroid_err**2)  # mas
    sigma_dec = np.sqrt(gaia_dec_err**2 + centroid_err**2)

    # Floor on uncertainty (confusion limit)
    sigma_ra = np.maximum(sigma_ra, 5.0)  # at least 5 mas
    sigma_dec = np.maximum(sigma_dec, 5.0)

    cos_dec = np.cos(np.radians(np.mean(gdec)))
    n_data = len(tbl)
    print(f'Fitting {n_data} sources')
    print(f'Median σ_RA={np.median(sigma_ra):.1f}mas, σ_Dec={np.median(sigma_dec):.1f}mas')

    # Parameter names and priors
    param_names = ['dCRVAL1_arcsec', 'dCRVAL2_arcsec', 'dPA_arcsec', 'dScale_ppm']

    def prior_transform(cube):
        params = cube.copy()
        params[0] = cube[0] * 0.2 - 0.1       # dCRVAL1: ±100 mas (arcsec)
        params[1] = cube[1] * 0.2 - 0.1       # dCRVAL2: ±100 mas (arcsec)
        params[2] = cube[2] * 7200 - 3600     # dPA: ±1 degree (in arcsec)
        params[3] = cube[3] * 20000 - 10000   # dScale: ±10000 ppm = ±1%
        return params

    def log_likelihood(params):
        dcrval1, dcrval2, dpa, dscale = params
        wcs_test = build_wcs_from_params(wcs_base, dcrval1, dcrval2, dpa, dscale)

        # Project pixel positions to sky
        sky = wcs_test.pixel_to_world(jx, jy)
        pred_ra = sky.ra.deg
        pred_dec = sky.dec.deg

        # Residuals in mas
        dra = (pred_ra - gra) * cos_dec * 3600 * 1000
        ddec = (pred_dec - gdec) * 3600 * 1000

        chi2 = np.sum((dra / sigma_ra)**2 + (ddec / sigma_dec)**2)
        return -0.5 * chi2

    # Run UltraNest
    sampler = ultranest.ReactiveNestedSampler(
        param_names, log_likelihood, prior_transform,
        log_dir=f'{ASTROM_DIR}/ultranest_{det}',
        resume='overwrite',
    )
    result = sampler.run(min_num_live_points=400, min_ess=1000)

    # Extract results
    print(f'\n{"="*60}')
    print(f'RESULTS')
    print(f'{"="*60}')
    for i, name in enumerate(param_names):
        p = result['posterior']['samples'][:, i]
        print(f'  {name}: {np.median(p):+.4f} ± {np.std(p):.4f} '
              f'[{np.percentile(p, 16):.4f}, {np.percentile(p, 84):.4f}]')

    best = result['maximum_likelihood']['point']
    print(f'\nBest fit: dRA={best[0]*1000:.1f}mas, dDec={best[1]*1000:.1f}mas, '
          f'dPA={best[2]:.2f}arcsec, dScale={best[3]:.0f}ppm')

    # Evaluate residuals at best fit
    wcs_best = build_wcs_from_params(wcs_base, *best)
    sky_best = wcs_best.pixel_to_world(jx, jy)
    dra_best = (sky_best.ra.deg - gra) * cos_dec * 3600 * 1000
    ddec_best = (sky_best.dec.deg - gdec) * 3600 * 1000
    resid_best = np.sqrt(dra_best**2 + ddec_best**2)
    chi2_best = np.sum((dra_best/sigma_ra)**2 + (ddec_best/sigma_dec)**2)
    chi2_dof = chi2_best / (2 * n_data - 4)

    print(f'\nResiduals at best fit:')
    print(f'  σ_RA={np.std(dra_best):.1f}mas, σ_Dec={np.std(ddec_best):.1f}mas')
    print(f'  Median sep={np.median(resid_best):.1f}mas, 90th={np.percentile(resid_best, 90):.1f}mas')
    print(f'  χ²/dof = {chi2_dof:.2f} (χ²={chi2_best:.0f}, dof={2*n_data-4})')

    # Save best-fit WCS
    hdr_best = wcs_best.to_header()
    hdr_best['COMMENT'] = f'WCS from UltraNest MCMC fit to {n_data} Gaia matches'
    hdr_best['GAIANMAT'] = n_data
    hdr_best['GAIARESI'] = float(np.median(resid_best))
    med = fits.getdata(f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits')
    fits.writeto(f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_mcmc.fits',
                 med.astype(np.float32), hdr_best, overwrite=True)
    print(f'\nSaved: {ASTROM_DIR}/{target}_{seg}_{det}_wcs_mcmc.fits')

    # Corner plot
    try:
        from ultranest.plot import cornerplot
        cornerplot(result)
        plt.savefig(f'{ASTROM_DIR}/diagnostics/{target}_{seg}_{det}_mcmc_corner.png',
                    dpi=100, bbox_inches='tight')
        plt.close()
    except:
        pass


if __name__ == '__main__':
    main()
