#!/usr/bin/env python
"""
Refine WCS (shift + CD + all SIP coefficients orders 2-5) using UltraNest.

Uses the linearized Jacobian approach:
  - Pre-compute base residuals using full calints SIP WCS
  - Analytically compute sensitivity of each parameter
  - Iterative sigma-clip to remove outliers
  - Run MCMC on the cleaned sample

Usage:
    python fit_wcs_sip_refine.py --det nrcb4 --use-good [--clip N]
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
    n_clip = 25  # default: drop top 25 chi2 outliers
    for arg in sys.argv[1:]:
        if arg.startswith('--det='): det = arg.split('=')[1]
        if arg.startswith('--clip='): n_clip = int(arg.split('=')[1])

    # Load match table + good filter
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

    # Load calints WCS with full SIP
    cal_files = sorted([f for f in glob.glob(f'/data/JWST/Terzan5/Segment2/*{det}*cal.fits') if 'uncal' not in f])
    hdr_cal = fits.getheader(cal_files[0], 'SCI')
    wcs_cal = WCS(hdr_cal)

    cd11 = hdr_cal['CD1_1']; cd12 = hdr_cal['CD1_2']
    cd21 = hdr_cal['CD2_1']; cd22 = hdr_cal['CD2_2']
    crpix1 = hdr_cal['CRPIX1']; crpix2 = hdr_cal['CRPIX2']
    cos_crval2 = np.cos(np.radians(hdr_cal['CRVAL2']))

    # Data
    jx = np.array(tbl['jwst_x']); jy = np.array(tbl['jwst_y'])
    gra = np.array(tbl['gaia_ra']); gdec = np.array(tbl['gaia_dec'])
    cos_dec = np.cos(np.radians(np.mean(gdec)))
    sigma_ra = np.maximum(np.sqrt(tbl['gaia_ra_err_mas']**2 + tbl['centroid_err_mas']**2), 5.0)
    sigma_dec = np.maximum(np.sqrt(tbl['gaia_dec_err_mas']**2 + tbl['centroid_err_mas']**2), 5.0)
    n = len(tbl)
    print(f'Fitting {n} sources')

    # Pre-compute base residuals
    sky_base = wcs_cal.pixel_to_world(jx, jy)
    resid_ra = (sky_base.ra.deg - gra) * cos_dec * 3600 * 1000
    resid_dec = (sky_base.dec.deg - gdec) * 3600 * 1000
    print(f'Base residuals: ΔRA={np.median(resid_ra):+.1f}±{np.std(resid_ra):.1f}, '
          f'ΔDec={np.median(resid_dec):+.1f}±{np.std(resid_dec):.1f} mas')

    u = jx - crpix1; v = jy - crpix2

    # ================================================================
    # Define parameters: shift (2) + CD (4) + order-2 SIP (6) + order-3 SIP (8) = 20
    # ================================================================
    # Build parameters: shift (2) + CD (4) + all SIP orders 2-5 (36) = 42
    # ================================================================
    def build_jacobian(u, v, n, cd11, cd12, cd21, cd22, cos_crval2, cos_dec, hdr_cal):
        """Build parameter list and Jacobian for shift + CD + SIP orders 2-5."""
        params = []

        # Shifts (2)
        params.append(('dCRVAL1_as', np.full(n, 1000.0), np.zeros(n)))
        params.append(('dCRVAL2_as', np.zeros(n), np.full(n, 1000.0)))

        # CD matrix (4)
        params.append(('dCD1_1', u / cos_crval2 * cos_dec * 3600e3, np.zeros(n)))
        params.append(('dCD1_2', v / cos_crval2 * cos_dec * 3600e3, np.zeros(n)))
        params.append(('dCD2_1', np.zeros(n), u * 3600e3))
        params.append(('dCD2_2', np.zeros(n), v * 3600e3))

        # SIP orders 2-5
        for order in range(2, 6):
            for ab in ['A', 'B']:
                for p_exp in range(order + 1):
                    q_exp = order - p_exp
                    key = f'{ab}_{p_exp}_{q_exp}'
                    if key not in hdr_cal:
                        continue
                    upvq = u**p_exp * v**q_exp
                    name = f'd{ab}_{p_exp}_{q_exp}'
                    if ab == 'A':
                        params.append((name,
                                       cd11 * upvq / cos_crval2 * cos_dec * 3600e3,
                                       cd21 * upvq * 3600e3))
                    else:
                        params.append((name,
                                       cd12 * upvq / cos_crval2 * cos_dec * 3600e3,
                                       cd22 * upvq * 3600e3))

        n_params = len(params)
        param_names = [p[0] for p in params]
        J = np.zeros((2 * n, n_params))
        for k, (_, j_ra, j_dec) in enumerate(params):
            J[:n, k] = j_ra
            J[n:, k] = j_dec
        return params, param_names, J, n_params

    params, param_names, J, n_params = build_jacobian(
        u, v, n, cd11, cd12, cd21, cd22, cos_crval2, cos_dec, hdr_cal)
    print(f'{n_params} parameters (shift + CD + SIP orders 2-5)')

    # ================================================================
    # Iterative outlier rejection using LS fit
    # ================================================================
    mask = np.ones(n, dtype=bool)  # start with all sources

    # First pass LS on all sources to identify outliers
    r_vec = np.concatenate([resid_ra, resid_dec])
    w_vec = np.concatenate([1.0 / sigma_ra, 1.0 / sigma_dec])
    Jw = J * w_vec[:, np.newaxis]
    rw = r_vec * w_vec

    U_svd, s_svd, Vt_svd = np.linalg.svd(Jw, full_matrices=False)
    delta_ls = Vt_svd.T @ np.diag(1.0 / s_svd) @ U_svd.T @ rw

    correction = J @ delta_ls
    dra_c = resid_ra - correction[:n]
    ddec_c = resid_dec - correction[n:]
    chi2_per = (dra_c / sigma_ra)**2 + (ddec_c / sigma_dec)**2

    # Drop top N outliers
    if n_clip > 0:
        order_idx = np.argsort(chi2_per)[::-1]
        drop = order_idx[:n_clip]
        mask[drop] = False
        print(f'Dropped {n_clip} outliers (chi2 range {chi2_per[drop[-1]]:.0f}-{chi2_per[drop[0]]:.0f})')

    n_use = mask.sum()
    print(f'Fitting {n_use} sources (dropped {n - n_use})')

    # Rebuild arrays for clean sample
    jx_c = jx[mask]; jy_c = jy[mask]
    resid_ra_c = resid_ra[mask]; resid_dec_c = resid_dec[mask]
    sigma_ra_c = sigma_ra[mask]; sigma_dec_c = sigma_dec[mask]
    u_c = u[mask]; v_c = v[mask]

    # Rebuild Jacobian for clean sample
    params_c, _, J_c, _ = build_jacobian(
        u_c, v_c, n_use, cd11, cd12, cd21, cd22, cos_crval2, cos_dec, hdr_cal)

    # LS on clean sample
    r_vec_c = np.concatenate([resid_ra_c, resid_dec_c])
    w_vec_c = np.concatenate([1.0 / sigma_ra_c, 1.0 / sigma_dec_c])
    Jw_c = J_c * w_vec_c[:, np.newaxis]
    rw_c = r_vec_c * w_vec_c

    chi2_before_c = np.sum(rw_c**2)
    U_c, s_c, Vt_c = np.linalg.svd(Jw_c, full_matrices=False)
    delta_ls_c = Vt_c.T @ np.diag(1.0 / s_c) @ U_c.T @ rw_c
    cov_c = np.linalg.inv(Jw_c.T @ Jw_c)
    sigma_ls_c = np.sqrt(np.diag(cov_c))

    rw_after_c = rw_c - Jw_c @ delta_ls_c
    chi2_ls_c = np.sum(rw_after_c**2)
    dof_ls = 2 * n_use - n_params

    print(f'\nLeast squares on clean sample:')
    print(f'  χ² before: {chi2_before_c:.0f} ({chi2_before_c/(2*n_use):.2f}/datum)')
    print(f'  χ² after:  {chi2_ls_c:.0f} ({chi2_ls_c/dof_ls:.2f}/dof, dof={dof_ls})')
    print()

    # Print LS results
    for k in range(n_params):
        signif = abs(delta_ls_c[k]) / sigma_ls_c[k] if sigma_ls_c[k] > 0 else 0
        name = param_names[k]
        val = delta_ls_c[k]; unc = sigma_ls_c[k]
        if name.startswith('dCRVAL'):
            print(f'  {name:>12s}: {val*1000:+.2f} ± {unc*1000:.2f} mas  ({signif:.1f}σ)')
        elif name.startswith('dCD'):
            cd_key = name[1:]
            current = hdr_cal[cd_key]
            print(f'  {name:>12s}: {val:+.4e} ± {unc:.4e}  ({signif:.1f}σ)  current={current:.6e}')
        else:
            ab = name[1]; p_q = name[3:]
            p_exp, q_exp = int(p_q[0]), int(p_q[2])
            current = hdr_cal.get(f'{ab}_{p_exp}_{q_exp}', 0)
            frac = val / current * 100 if current != 0 else float('inf')
            print(f'  {name:>12s}: {val:+.4e} ± {unc:.4e}  ({signif:.1f}σ)  '
                  f'current={current:+.4e}  Δ={frac:+.1f}%')

    # ================================================================
    # MCMC with UltraNest on clean sample
    # ================================================================
    print(f'\nRunning MCMC ({n_params} params, {n_use} sources)...', flush=True)

    def prior_transform(cube):
        p = np.zeros(n_params)
        for k in range(n_params):
            half_width = max(5 * sigma_ls_c[k], abs(delta_ls_c[k]) * 3)
            p[k] = cube[k] * 2 * half_width + (delta_ls_c[k] - half_width)
        return p

    def log_likelihood(params_vec):
        correction = J_c @ params_vec
        dra = resid_ra_c - correction[:n_use]
        ddec = resid_dec_c - correction[n_use:]
        chi2 = np.sum((dra / sigma_ra_c)**2 + (ddec / sigma_dec_c)**2)
        return -0.5 * chi2

    run_dir = f'{ASTROM_DIR}/ultranest_{det}_sip'
    sampler = ultranest.ReactiveNestedSampler(
        param_names, log_likelihood, prior_transform,
        log_dir=run_dir, resume='overwrite',
    )
    result = sampler.run(min_num_live_points=400, min_ess=1000)

    samples = np.loadtxt(f'{run_dir}/chains/equal_weighted_post.txt', skiprows=1)

    # ================================================================
    # Results
    # ================================================================
    print(f'\n{"="*70}')
    print(f'MCMC RESULTS ({n_params} params, {n_use} sources after clipping {n_clip})')
    print(f'{"="*70}')

    best = np.median(samples, axis=0)
    for k in range(n_params):
        med = best[k]
        std = np.std(samples[:, k])
        signif = abs(med) / std if std > 0 else 0
        name = param_names[k]
        if name.startswith('dCRVAL'):
            print(f'  {name:>12s}: {med*1000:+.2f} ± {std*1000:.2f} mas  ({signif:.1f}σ)')
        elif name.startswith('dCD'):
            cd_key = name[1:]
            current = hdr_cal[cd_key]
            print(f'  {name:>12s}: {med:+.4e} ± {std:.4e}  ({signif:.1f}σ)  '
                  f'Δ={med/current*100:+.4f}%')
        else:
            ab = name[1]; p_q = name[3:]
            p_exp, q_exp = int(p_q[0]), int(p_q[2])
            current = hdr_cal.get(f'{ab}_{p_exp}_{q_exp}', 0)
            frac = med / current * 100 if current != 0 else 0
            print(f'  {name:>12s}: {med:+.4e} ± {std:.4e}  ({signif:.1f}σ)  Δ={frac:+.2f}%')

    # Corrected residuals on CLEAN sample
    correction_c = J_c @ best
    dra_corr_c = resid_ra_c - correction_c[:n_use]
    ddec_corr_c = resid_dec_c - correction_c[n_use:]
    resid_corr_c = np.sqrt(dra_corr_c**2 + ddec_corr_c**2)
    chi2_corr = np.sum((dra_corr_c / sigma_ra_c)**2 + (ddec_corr_c / sigma_dec_c)**2)
    dof = 2 * n_use - n_params

    # Also evaluate on ALL sources (including dropped ones)
    correction_all = J @ best
    dra_corr_all = resid_ra - correction_all[:n]
    ddec_corr_all = resid_dec - correction_all[n:]
    resid_corr_all = np.sqrt(dra_corr_all**2 + ddec_corr_all**2)

    print(f'\n  RESIDUALS (clean sample, {n_use} sources):')
    print(f'  σ_RA  = {np.std(dra_corr_c):.1f} mas  (was {np.std(resid_ra_c):.1f})')
    print(f'  σ_Dec = {np.std(ddec_corr_c):.1f} mas  (was {np.std(resid_dec_c):.1f})')
    print(f'  Median sep = {np.median(resid_corr_c):.1f} mas  (was {np.median(np.sqrt(resid_ra_c**2+resid_dec_c**2)):.1f})')
    print(f'  90th pct   = {np.percentile(resid_corr_c, 90):.1f} mas')
    print(f'  χ²/dof = {chi2_corr/dof:.2f}  (χ²={chi2_corr:.0f}, dof={dof})')

    print(f'\n  ALL sources ({n}):')
    print(f'  σ_RA  = {np.std(dra_corr_all):.1f}, σ_Dec = {np.std(ddec_corr_all):.1f} mas')
    print(f'  Median sep = {np.median(resid_corr_all):.1f} mas')

    # ================================================================
    # Diagnostic plots
    # ================================================================
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    dropped = ~mask

    ax = axes[0, 0]
    ax.scatter(jx_c, resid_ra_c, s=8, alpha=0.6, c='#1f77b4')
    ax.scatter(jx[dropped], resid_ra[dropped], s=15, alpha=0.5, c='red', marker='x', label='dropped')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('X pixel'); ax.set_ylabel('ΔRA (mas)'); ax.set_title('Before: ΔRA vs X')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.scatter(jx_c, resid_dec_c, s=8, alpha=0.6, c='#d62728')
    ax.scatter(jx[dropped], resid_dec[dropped], s=15, alpha=0.5, c='red', marker='x')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('X pixel'); ax.set_ylabel('ΔDec (mas)'); ax.set_title('Before: ΔDec vs X')

    ax = axes[0, 2]
    sc = ax.scatter(jx_c, jy_c, c=resid_dec_c, cmap='RdBu_r', vmin=-70, vmax=30, s=15, edgecolors='k', linewidths=0.2)
    ax.scatter(jx[dropped], jy[dropped], s=30, c='none', edgecolors='red', linewidths=1.5, marker='s')
    plt.colorbar(sc, ax=ax, label='ΔDec (mas)')
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
    ax.set_title('Before: ΔDec on detector')

    ax = axes[0, 3]
    resid_base_c = np.sqrt(resid_ra_c**2 + resid_dec_c**2)
    ax.hist(resid_base_c, bins=25, color='#7f7f7f', alpha=0.6, label=f'Before (med={np.median(resid_base_c):.0f})')
    ax.hist(resid_corr_c, bins=25, color='#2ca02c', alpha=0.6, label=f'After (med={np.median(resid_corr_c):.0f})')
    ax.set_xlabel('Total sep (mas)'); ax.set_ylabel('Count')
    ax.set_title('Separation distribution'); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.scatter(jx_c, dra_corr_c, s=8, alpha=0.6, c='#1f77b4')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('X pixel'); ax.set_ylabel('ΔRA (mas)'); ax.set_title('After: ΔRA vs X')

    ax = axes[1, 1]
    ax.scatter(jx_c, ddec_corr_c, s=8, alpha=0.6, c='#d62728')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.set_xlabel('X pixel'); ax.set_ylabel('ΔDec (mas)'); ax.set_title('After: ΔDec vs X')

    ax = axes[1, 2]
    sc = ax.scatter(jx_c, jy_c, c=ddec_corr_c, cmap='RdBu_r', vmin=-40, vmax=40, s=15, edgecolors='k', linewidths=0.2)
    plt.colorbar(sc, ax=ax, label='ΔDec (mas)')
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
    ax.set_title('After: ΔDec on detector')

    ax = axes[1, 3]
    q = ax.quiver(jx_c, jy_c, dra_corr_c, ddec_corr_c,
                  scale=500, scale_units='width', width=0.003, color='#2ca02c', alpha=0.7)
    ax.quiverkey(q, 0.85, 0.95, 20, '20 mas', coordinates='axes', fontproperties={'size': 8})
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
    ax.set_title('After: residual vectors')

    fig.suptitle(f'{det} — SIP refinement ({n_params} params, {n_use}/{n} sources)\n'
                 f'σ: {np.std(dra_corr_c):.1f}/{np.std(ddec_corr_c):.1f} mas (RA/Dec), '
                 f'median sep: {np.median(resid_corr_c):.1f} mas, '
                 f'χ²/dof={chi2_corr/dof:.2f}',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{ASTROM_DIR}/diagnostics/Terzan5_Segment2_{det}_sip_refine_residuals.png',
                dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nPlots saved')


if __name__ == '__main__':
    main()
