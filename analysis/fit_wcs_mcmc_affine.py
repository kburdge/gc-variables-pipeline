#!/usr/bin/env python
"""
Fit an affine correction on top of the full SIP WCS using UltraNest.

Strategy:
  1. Use the calints WCS (with SIP distortion) to project all pixel positions to sky
  2. Compute residuals vs Gaia in the sky plane (mas)
  3. Fit a 6-parameter affine correction to those residuals:
       correction_RA  = a0 + a1*xn + a2*yn
       correction_Dec = b0 + b1*xn + b2*yn
     where xn, yn are normalized pixel coordinates [-1, 1]

This separates distortion (handled by SIP) from linear corrections (shift, rotation,
scale, shear). No SIP degeneracy because we work in the residual sky plane.

After fitting, the 6 affine parameters are decomposed into physical quantities:
  - Shift (a0, b0)
  - Rotation angle
  - Isotropic scale change
  - Shear

Usage:
    python fit_wcs_mcmc_affine.py --det nrcb4 --use-good
"""
import numpy as np
import os
import sys
import glob
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

    # Load calints WCS with full SIP distortion
    cal_files = sorted([f for f in glob.glob(f'/data/JWST/Terzan5/Segment2/*{det}*cal.fits') if 'uncal' not in f])
    wcs_cal = WCS(fits.getheader(cal_files[0], 'SCI'))
    hdr_cal = fits.getheader(cal_files[0], 'SCI')

    # Print starting WCS info
    cd = np.array([[hdr_cal['CD1_1'], hdr_cal['CD1_2']],
                    [hdr_cal['CD2_1'], hdr_cal['CD2_2']]])
    scale1 = np.sqrt(cd[0,0]**2 + cd[1,0]**2) * 3600 * 1000  # mas/px
    scale2 = np.sqrt(cd[0,1]**2 + cd[1,1]**2) * 3600 * 1000
    pa = np.degrees(np.arctan2(cd[0,0], -cd[1,0]))
    print(f'Calints WCS:')
    print(f'  Pixel scale: {scale1:.3f} x {scale2:.3f} mas/px')
    print(f'  PA: {pa:.5f} deg')

    # Data arrays
    jx = np.array(tbl['jwst_x'])
    jy = np.array(tbl['jwst_y'])
    gra = np.array(tbl['gaia_ra'])
    gdec = np.array(tbl['gaia_dec'])

    sigma_ra = np.maximum(np.sqrt(tbl['gaia_ra_err_mas']**2 + tbl['centroid_err_mas']**2), 5.0)
    sigma_dec = np.maximum(np.sqrt(tbl['gaia_dec_err_mas']**2 + tbl['centroid_err_mas']**2), 5.0)

    n_data = len(tbl)
    cos_dec = np.cos(np.radians(np.mean(gdec)))
    print(f'\nFitting {n_data} sources')
    print(f'Median σ: ({np.median(sigma_ra):.1f}, {np.median(sigma_dec):.1f}) mas')

    # Pre-compute base SIP projection (this is the expensive part, done once)
    print(f'Pre-computing SIP projection...', flush=True)
    sky_base = wcs_cal.pixel_to_world(jx, jy)
    pred_ra_base = sky_base.ra.deg
    pred_dec_base = sky_base.dec.deg

    # Base residuals in mas (JWST_predicted - Gaia)
    resid_ra_base = (pred_ra_base - gra) * cos_dec * 3600 * 1000   # mas
    resid_dec_base = (pred_dec_base - gdec) * 3600 * 1000

    print(f'Base residuals (before any correction):')
    print(f'  ΔRA:  {np.median(resid_ra_base):+.1f} ± {np.std(resid_ra_base):.1f} mas')
    print(f'  ΔDec: {np.median(resid_dec_base):+.1f} ± {np.std(resid_dec_base):.1f} mas')
    print(f'  Total: {np.median(np.sqrt(resid_ra_base**2 + resid_dec_base**2)):.1f} mas')

    # Normalized pixel coordinates for affine model
    xn = (jx - 1024.0) / 1024.0   # [-1, 1] across detector
    yn = (jy - 1024.0) / 1024.0

    # 6 affine parameters:
    # correction_RA  = a0 + a1*xn + a2*yn  (mas)
    # correction_Dec = b0 + b1*xn + b2*yn  (mas)
    param_names = ['a0_RA_mas', 'a1_dRA_dx', 'a2_dRA_dy',
                   'b0_Dec_mas', 'b1_dDec_dx', 'b2_dDec_dy']

    def prior_transform(cube):
        p = cube.copy()
        # Shift: ±100 mas
        p[0] = cube[0] * 200 - 100       # a0: RA shift, ±100 mas
        p[3] = cube[3] * 200 - 100       # b0: Dec shift, ±100 mas
        # Linear terms: ±50 mas across half the detector
        p[1] = cube[1] * 100 - 50        # a1: dRA/dx, ±50 mas per 1024 px
        p[2] = cube[2] * 100 - 50        # a2: dRA/dy, ±50 mas per 1024 px
        p[4] = cube[4] * 100 - 50        # b1: dDec/dx
        p[5] = cube[5] * 100 - 50        # b2: dDec/dy
        return p

    def log_likelihood(params):
        a0, a1, a2, b0, b1, b2 = params

        # Affine correction (mas)
        corr_ra = a0 + a1 * xn + a2 * yn
        corr_dec = b0 + b1 * xn + b2 * yn

        # Corrected residuals = base_residuals - correction
        dra = resid_ra_base - corr_ra
        ddec = resid_dec_base - corr_dec

        chi2 = np.sum((dra / sigma_ra)**2 + (ddec / sigma_dec)**2)
        return -0.5 * chi2

    # Run UltraNest
    run_dir = f'{ASTROM_DIR}/ultranest_{det}_affine'
    sampler = ultranest.ReactiveNestedSampler(
        param_names, log_likelihood, prior_transform,
        log_dir=run_dir, resume='overwrite',
    )
    result = sampler.run(min_num_live_points=400, min_ess=1000)

    # Extract results
    samples = np.loadtxt(f'{run_dir}/chains/equal_weighted_post.txt', skiprows=1)

    print(f'\n{"="*60}')
    print(f'RESULTS (affine correction on SIP WCS)')
    print(f'{"="*60}')
    for i, name in enumerate(param_names):
        med = np.median(samples[:, i])
        std = np.std(samples[:, i])
        lo, hi = np.percentile(samples[:, i], [16, 84])
        print(f'  {name}: {med:+.4f} ± {std:.4f}  [{lo:+.4f}, {hi:+.4f}]')

    # Physical decomposition
    a0 = np.median(samples[:, 0])  # RA shift (mas)
    a1 = np.median(samples[:, 1])  # dRA/dxn (mas per unit xn)
    a2 = np.median(samples[:, 2])  # dRA/dyn
    b0 = np.median(samples[:, 3])  # Dec shift (mas)
    b1 = np.median(samples[:, 4])  # dDec/dxn
    b2 = np.median(samples[:, 5])  # dDec/dyn

    # The affine matrix in the sky plane is:
    #   [[a1, a2],    (mas per normalized pixel unit)
    #    [b1, b2]]
    #
    # Decompose into rotation + scale + shear:
    # For small angles, rotation θ produces:
    #   [[0, -θ], [θ, 0]] * pixel_scale_in_mas
    # Scale s produces:
    #   [[s, 0], [0, s]] * pixel_scale_in_mas
    # But our coordinates are in normalized units, so pixel_scale_in_mas ≈ 31 * 1024 ≈ 31744 mas

    pix_scale_mas = np.mean([scale1, scale2])  # mas/px
    norm_scale = pix_scale_mas * 1024  # mas per normalized unit

    # Rotation: antisymmetric part → θ = (b1 - a2) / (2 * norm_scale)
    dtheta_rad = (b1 - a2) / (2 * norm_scale)
    dtheta_arcsec = np.degrees(dtheta_rad) * 3600

    # Scale: symmetric trace → ds = (a1 + b2) / (2 * norm_scale)
    ds_frac = (a1 + b2) / (2 * norm_scale)
    ds_ppm = ds_frac * 1e6

    # Shear: symmetric off-diagonal → (a2 + b1) / (2 * norm_scale)
    shear = (a2 + b1) / (2 * norm_scale)

    # Anisotropic scale: (a1 - b2) / (2 * norm_scale)
    aniso = (a1 - b2) / (2 * norm_scale)

    # Uncertainties from posterior samples
    dtheta_samples = (samples[:, 4] - samples[:, 2]) / (2 * norm_scale)
    dtheta_as_samples = np.degrees(dtheta_samples) * 3600
    ds_samples = (samples[:, 1] + samples[:, 5]) / (2 * norm_scale) * 1e6

    print(f'\n  PHYSICAL DECOMPOSITION:')
    print(f'  Shift:    ΔRA = {a0:+.2f} ± {np.std(samples[:,0]):.2f} mas')
    print(f'            ΔDec = {b0:+.2f} ± {np.std(samples[:,3]):.2f} mas')
    print(f'  Rotation: {dtheta_arcsec:+.3f} ± {np.std(dtheta_as_samples):.3f} arcsec')
    print(f'            ({dtheta_arcsec*1000:+.1f} ± {np.std(dtheta_as_samples)*1000:.1f} mas)')
    print(f'  Scale:    {ds_ppm:+.1f} ± {np.std(ds_samples):.1f} ppm')
    print(f'            ({ds_frac*100:+.5f}%)')
    print(f'  Shear:    {shear*1e6:+.1f} ppm')
    print(f'  Aniso:    {aniso*1e6:+.1f} ppm')

    # Evaluate residuals at best fit
    a_best = [np.median(samples[:, i]) for i in range(6)]
    corr_ra_best = a_best[0] + a_best[1] * xn + a_best[2] * yn
    corr_dec_best = a_best[3] + a_best[4] * xn + a_best[5] * yn
    dra_corr = resid_ra_base - corr_ra_best
    ddec_corr = resid_dec_base - corr_dec_best
    resid_corr = np.sqrt(dra_corr**2 + ddec_corr**2)
    chi2_corr = np.sum((dra_corr / sigma_ra)**2 + (ddec_corr / sigma_dec)**2)

    print(f'\n  RESIDUALS AFTER AFFINE CORRECTION:')
    print(f'  σ_RA  = {np.std(dra_corr):.1f} mas  (was {np.std(resid_ra_base):.1f})')
    print(f'  σ_Dec = {np.std(ddec_corr):.1f} mas  (was {np.std(resid_dec_base):.1f})')
    print(f'  Median sep = {np.median(resid_corr):.1f} mas  (was {np.median(np.sqrt(resid_ra_base**2 + resid_dec_base**2)):.1f})')
    print(f'  90th pct   = {np.percentile(resid_corr, 90):.1f} mas')
    print(f'  χ²/dof = {chi2_corr/(2*n_data - 6):.2f}  (χ²={chi2_corr:.0f}, dof={2*n_data-6})')

    # Corner plot
    plot_labels = ['ΔRA shift\n(mas)', 'dRA/dx\n(mas/norm)', 'dRA/dy\n(mas/norm)',
                   'ΔDec shift\n(mas)', 'dDec/dx\n(mas/norm)', 'dDec/dy\n(mas/norm)']
    n_p = 6
    fig, axes = plt.subplots(n_p, n_p, figsize=(16, 16))
    for i in range(n_p):
        for j in range(n_p):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=50, color='#1f77b4', alpha=0.7)
                ax.axvline(np.median(samples[:, i]), color='red', lw=1.5)
                ax.set_title(f'{np.median(samples[:,i]):+.3f}±{np.std(samples[:,i]):.3f}', fontsize=6)
            elif i > j:
                ax.scatter(samples[:, j], samples[:, i], s=0.3, c='black', alpha=0.1)
            else:
                ax.axis('off')
            if i == n_p-1: ax.set_xlabel(plot_labels[j], fontsize=7)
            if j == 0 and i > 0: ax.set_ylabel(plot_labels[i], fontsize=7)
            ax.tick_params(labelsize=5)

    fig.suptitle(f'{det} MCMC — Affine correction on SIP WCS, {n_data} good sources\n'
                 f'Shift: ({a0:+.1f}, {b0:+.1f}) mas, Rot: {dtheta_arcsec:+.2f}\", '
                 f'Scale: {ds_ppm:+.0f} ppm',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{ASTROM_DIR}/diagnostics/Terzan5_Segment2_{det}_mcmc_affine_corner.png',
                dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Residual diagnostics plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Before correction
    ax = axes[0, 0]
    ax.scatter(jx, resid_ra_base, s=3, alpha=0.5, c='#1f77b4')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('X pixel'); ax.set_ylabel('ΔRA (mas)')
    ax.set_title('Before: ΔRA vs X')

    ax = axes[0, 1]
    ax.scatter(jy, resid_dec_base, s=3, alpha=0.5, c='#d62728')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('Y pixel'); ax.set_ylabel('ΔDec (mas)')
    ax.set_title('Before: ΔDec vs Y')

    ax = axes[0, 2]
    resid_base_tot = np.sqrt(resid_ra_base**2 + resid_dec_base**2)
    ax.hist(resid_base_tot, bins=25, color='#7f7f7f', alpha=0.5, label='Before')
    ax.hist(resid_corr, bins=25, color='#2ca02c', alpha=0.5, label='After')
    ax.set_xlabel('Total residual (mas)'); ax.set_ylabel('Count')
    ax.set_title('Separation distribution'); ax.legend()

    # After correction
    ax = axes[1, 0]
    ax.scatter(jx, dra_corr, s=3, alpha=0.5, c='#1f77b4')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('X pixel'); ax.set_ylabel('ΔRA (mas)')
    ax.set_title('After: ΔRA vs X')

    ax = axes[1, 1]
    ax.scatter(jy, ddec_corr, s=3, alpha=0.5, c='#d62728')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('Y pixel'); ax.set_ylabel('ΔDec (mas)')
    ax.set_title('After: ΔDec vs Y')

    # Quiver plot of corrected residuals
    ax = axes[1, 2]
    q = ax.quiver(jx, jy, dra_corr, ddec_corr,
                  scale=500, scale_units='width', width=0.003, color='#2ca02c', alpha=0.7)
    ax.quiverkey(q, 0.85, 0.95, 20, '20 mas', coordinates='axes', fontproperties={'size': 8})
    ax.set_xlabel('X pixel'); ax.set_ylabel('Y pixel')
    ax.set_title('Residual vectors after correction')
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')

    fig.suptitle(f'{det} — Affine WCS correction, {n_data} sources\n'
                 f'Residual: {np.std(dra_corr):.1f}/{np.std(ddec_corr):.1f} mas (RA/Dec), '
                 f'median sep = {np.median(resid_corr):.1f} mas',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{ASTROM_DIR}/diagnostics/Terzan5_Segment2_{det}_mcmc_affine_residuals.png',
                dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f'\nPlots saved to {ASTROM_DIR}/diagnostics/')


if __name__ == '__main__':
    main()
