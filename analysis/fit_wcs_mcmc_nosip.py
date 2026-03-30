#!/usr/bin/env python
"""
Fit a pure CD-matrix WCS (no SIP) using UltraNest.
6 parameters: CRVAL1, CRVAL2, CD1_1, CD1_2, CD2_1, CD2_2

This removes SIP degeneracy and lets us directly constrain rotation and scale.

Usage:
    python fit_wcs_mcmc_nosip.py --det nrcb4 --use-good
"""
import numpy as np
import os
import sys
import glob
import json
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import ultranest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'


def main():
    det = 'nrcb4'
    use_good = '--use-good' in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith('--det='): det = arg.split('=')[1]

    # Load match table
    tbl = Table.read(f'{ASTROM_DIR}/Terzan5_Segment2_{det}_match_table.fits')
    print(f'Loaded {len(tbl)} matches')

    if use_good:
        good_dir = f'{ASTROM_DIR}/{det}_cutouts/good'
        good_files = set(os.listdir(good_dir))
        keep = []
        for i, row in enumerate(tbl):
            ix, iy = int(round(row['jwst_x'])), int(round(row['jwst_y']))
            if any(f'px{ix:04d}_{iy:04d}' in f for f in good_files):
                keep.append(i)
        tbl = tbl[keep]
        print(f'Using {len(tbl)} good sources')

    # Load calints WCS as starting point
    cal_files = sorted([f for f in glob.glob(f'/data/JWST/Terzan5/Segment2/*{det}*cal.fits') if 'uncal' not in f])
    hdr_cal = fits.getheader(cal_files[0], 'SCI')

    # Extract the base CD matrix values
    cd0 = np.array([hdr_cal['CD1_1'], hdr_cal['CD1_2'],
                     hdr_cal['CD2_1'], hdr_cal['CD2_2']])
    crval0 = np.array([hdr_cal['CRVAL1'], hdr_cal['CRVAL2']])
    crpix = np.array([hdr_cal['CRPIX1'], hdr_cal['CRPIX2']])

    print(f'Starting CD matrix:')
    print(f'  CD1_1={cd0[0]:.10e}, CD1_2={cd0[1]:.10e}')
    print(f'  CD2_1={cd0[2]:.10e}, CD2_2={cd0[3]:.10e}')
    print(f'  CRVAL=({crval0[0]:.8f}, {crval0[1]:.8f})')

    # Pixel scale from CD matrix
    scale1 = np.sqrt(cd0[0]**2 + cd0[2]**2) * 3600  # arcsec/px
    scale2 = np.sqrt(cd0[1]**2 + cd0[3]**2) * 3600
    pa = np.degrees(np.arctan2(cd0[0], -cd0[2]))
    print(f'  Pixel scale: {scale1*1000:.2f} x {scale2*1000:.2f} mas/px')
    print(f'  PA: {pa:.4f} deg')

    # Data
    jx = np.array(tbl['jwst_x']) - crpix[0]  # relative to CRPIX
    jy = np.array(tbl['jwst_y']) - crpix[1]
    gra = np.array(tbl['gaia_ra'])
    gdec = np.array(tbl['gaia_dec'])

    sigma_ra = np.maximum(np.sqrt(tbl['gaia_ra_err_mas']**2 + tbl['centroid_err_mas']**2), 5.0)
    sigma_dec = np.maximum(np.sqrt(tbl['gaia_dec_err_mas']**2 + tbl['centroid_err_mas']**2), 5.0)

    n_data = len(tbl)
    cos_dec = np.cos(np.radians(np.mean(gdec)))
    print(f'\nFitting {n_data} sources')
    print(f'Median σ: ({np.median(sigma_ra):.1f}, {np.median(sigma_dec):.1f}) mas')

    # Parameters: dCRVAL1 (arcsec), dCRVAL2 (arcsec),
    #             dCD1_1, dCD1_2, dCD2_1, dCD2_2 (fractional perturbations)
    param_names = ['dCRVAL1_as', 'dCRVAL2_as', 'fCD1_1', 'fCD1_2', 'fCD2_1', 'fCD2_2']

    def prior_transform(cube):
        p = cube.copy()
        # Center on known shift from SIP fit, ±10 mas
        p[0] = cube[0] * 0.020 + 0.018     # dCRVAL1: 0.018..0.038 arcsec (~28 ± 10 mas)
        p[1] = cube[1] * 0.020 + 0.038     # dCRVAL2: 0.038..0.058 arcsec (~48 ± 10 mas)
        p[2] = cube[2] * 0.002 - 0.001 + 1.0 # fCD1_1: 1 ± 0.1%
        p[3] = cube[3] * 0.002 - 0.001 + 1.0 # fCD1_2: 1 ± 0.1%
        p[4] = cube[4] * 0.002 - 0.001 + 1.0 # fCD2_1: 1 ± 0.1%
        p[5] = cube[5] * 0.002 - 0.001 + 1.0 # fCD2_2: 1 ± 0.1%
        return p

    def log_likelihood(params):
        dcrval1, dcrval2, f11, f12, f21, f22 = params

        # Build pure CD-matrix WCS (no SIP)
        crval1 = crval0[0] + dcrval1 / 3600.0 / cos_dec
        crval2 = crval0[1] + dcrval2 / 3600.0

        cd11 = cd0[0] * f11
        cd12 = cd0[1] * f12
        cd21 = cd0[2] * f21
        cd22 = cd0[3] * f22

        # Direct pixel-to-sky: RA = CRVAL1 + CD1_1*dx + CD1_2*dy
        pred_ra = crval1 + cd11 * jx + cd12 * jy
        pred_dec = crval2 + cd21 * jx + cd22 * jy

        dra = (pred_ra - gra) * cos_dec * 3600 * 1000  # mas
        ddec = (pred_dec - gdec) * 3600 * 1000

        chi2 = np.sum((dra / sigma_ra)**2 + (ddec / sigma_dec)**2)
        return -0.5 * chi2

    # Run
    run_dir = f'{ASTROM_DIR}/ultranest_{det}_nosip'
    sampler = ultranest.ReactiveNestedSampler(
        param_names, log_likelihood, prior_transform,
        log_dir=run_dir, resume='overwrite',
    )
    result = sampler.run(min_num_live_points=400, min_ess=1000)

    # Extract
    samples = np.loadtxt(f'{run_dir}/chains/equal_weighted_post.txt', skiprows=1)

    print(f'\n{"="*60}')
    print(f'RESULTS (no SIP, pure CD matrix)')
    print(f'{"="*60}')
    for i, name in enumerate(param_names):
        med = np.median(samples[:, i])
        std = np.std(samples[:, i])
        print(f'  {name}: {med:.6f} ± {std:.6f}')

    # Interpret
    dra_mas = np.median(samples[:, 0]) * 1000
    ddec_mas = np.median(samples[:, 1]) * 1000
    f11 = np.median(samples[:, 2]); f12 = np.median(samples[:, 3])
    f21 = np.median(samples[:, 4]); f22 = np.median(samples[:, 5])

    cd_new = np.array([[cd0[0]*f11, cd0[1]*f12], [cd0[2]*f21, cd0[3]*f22]])
    scale1_new = np.sqrt(cd_new[0,0]**2 + cd_new[1,0]**2) * 3600 * 1000
    scale2_new = np.sqrt(cd_new[0,1]**2 + cd_new[1,1]**2) * 3600 * 1000
    pa_new = np.degrees(np.arctan2(cd_new[0,0], -cd_new[1,0]))

    print(f'\n  Shift: ({dra_mas:+.2f}, {ddec_mas:+.2f}) mas')
    print(f'  Scale: {scale1_new:.3f} x {scale2_new:.3f} mas/px (was {scale1*1000:.3f} x {scale2*1000:.3f})')
    print(f'  PA: {pa_new:.5f} deg (was {pa:.5f}, Δ={pa_new-pa:.5f} deg = {(pa_new-pa)*3600:.1f} arcsec)')
    print(f'  CD fractional: ({f11:.6f}, {f12:.6f}, {f21:.6f}, {f22:.6f})')

    # Corner plot
    plot_labels = ['ΔRA (mas)', 'ΔDec (mas)', 'f(CD1_1)', 'f(CD1_2)', 'f(CD2_1)', 'f(CD2_2)']
    sp = samples.copy()
    sp[:, 0] *= 1000; sp[:, 1] *= 1000  # arcsec -> mas

    n_p = 6
    fig, axes = plt.subplots(n_p, n_p, figsize=(16, 16))
    for i in range(n_p):
        for j in range(n_p):
            ax = axes[i, j]
            if i == j:
                ax.hist(sp[:, i], bins=50, color='#1f77b4', alpha=0.7)
                ax.axvline(np.median(sp[:, i]), color='red', lw=1.5)
                ax.set_title(f'{np.median(sp[:,i]):.4f}±{np.std(sp[:,i]):.4f}', fontsize=6)
            elif i > j:
                ax.scatter(sp[:, j], sp[:, i], s=0.3, c='black', alpha=0.1)
            else:
                ax.axis('off')
            if i == n_p-1: ax.set_xlabel(plot_labels[j], fontsize=7)
            if j == 0 and i > 0: ax.set_ylabel(plot_labels[i], fontsize=7)
            ax.tick_params(labelsize=5)

    fig.suptitle(f'nrcb4 MCMC — Pure CD matrix (no SIP), {n_data} good sources', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(f'{ASTROM_DIR}/diagnostics/Terzan5_Segment2_{det}_mcmc_nosip_corner.png',
                dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'\nCorner plot saved')


if __name__ == '__main__':
    main()
