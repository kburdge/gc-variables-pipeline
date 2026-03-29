#!/usr/bin/env python
"""
===========================================================================
 LW (nrcblong) astrometric calibration to Gaia DR3
===========================================================================

 *** DO NOT DELETE THIS FILE ***
 This is the production astrometry calibration script for the LW channel.
 It implements the empirically validated recipe for tying JWST NIRCam
 nrcblong positions to the Gaia DR3 reference frame.

 The LW-aligned WCS produced by this script serves as the absolute
 astrometric reference for all downstream SW alignment (via sw_lw_crossmatch)
 and catalog position computation.

Recipe (documented from iterative development, March 2026):
  1. Source detection on uncal ZF median images using DAOStarFinder
     with a grid of FWHM values (4, 5, 6, 8, 10, 12, 15, 20, 25, 30 px).
     Threshold = 10*sigma, brightest = 500 per FWHM.

  2. For each Gaia source, try ALL FWHM catalogs and pick the one where
     the matched detection has the BEST (smallest |roundness1|). This
     "best-roundness" selection is critical: it finds the FWHM where
     DAOStarFinder's Gaussian model best matches the source profile,
     which varies with brightness due to saturation wings.

  3. Filter to |roundness1| < 0.1 to reject blends and artifacts.

  4. IQR clip (3x) on RA and Dec offsets to remove outliers.

  5. Compute median shift and apply as rigid CRVAL correction,
     preserving the JWST calints SIP distortion model.

Target-specific parameters:
  Terzan 5:  G < 17.5, match radius 0.5", ~97 clean matches
  Liller 1:  G < 18.0, match radius 0.3", ~40-60 clean matches

Key design decisions:
  - Uses uncal ZF medians (not pipeline ZFs which zero saturated pixels)
  - Best-roundness FWHM selection (not fixed tiers by magnitude)
  - No PSF model — pure DAOStarFinder Gaussian approximation
  - Roundness cut |r1| < 0.1 rejects ~80% of detections but keeps
    only well-characterized point sources
  - Shift-only correction (no rotation/scale) — JWST SIP handles distortion

Usage:
    python calibrate_lw_astrometry.py --target Terzan5
    python calibrate_lw_astrometry.py --target Liller1
    python calibrate_lw_astrometry.py --target Terzan5 --plot-only  # just regenerate figures
"""
import numpy as np
import os
import sys
import glob
import argparse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
GAIA_EPOCH = Time('J2016.0')

# ── Observation epochs ──
OBS_EPOCHS = {
    ('Terzan5', 'Segment2'): Time(60786.49, format='mjd'),
    ('Liller1', 'Segment3'): Time(60787.5, format='mjd'),
    ('Liller1', 'Segment4'): Time(60788.97, format='mjd'),
}

# ── Detection parameters ──
FWHM_GRID = [4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
DETECT_THRESHOLD_SIGMA = 10
DETECT_BRIGHTEST = 500
EDGE = 10

# ── Matching parameters (per target) ──
TARGET_CONFIG = {
    'Terzan5': {
        'segments': ['Segment2'],
        'gaia_mag_limit': 17.5,
        'match_radius_arcsec': 0.5,
        'roundness_cut': 0.1,
    },
    'Liller1': {
        'segments': ['Segment3', 'Segment4'],
        'gaia_mag_limit': 18.0,
        'match_radius_arcsec': 0.3,
        'roundness_cut': 0.1,
    },
}


def detect_all_fwhm(zf_image, std):
    """Run DAOStarFinder at each FWHM in the grid. Returns dict of catalogs."""
    median_val = np.median(zf_image[np.isfinite(zf_image)])
    det_cats = {}
    for fwhm in FWHM_GRID:
        finder = DAOStarFinder(fwhm=fwhm, threshold=DETECT_THRESHOLD_SIGMA * std,
                               brightest=DETECT_BRIGHTEST)
        sources = finder(zf_image - median_val)
        if sources is None:
            continue
        xp = np.array(sources['xcentroid'])
        yp = np.array(sources['ycentroid'])
        r1 = np.array(sources['roundness1'])
        r2 = np.array(sources['roundness2'])
        sharp = np.array(sources['sharpness'])
        peak = np.array(sources['peak'])
        mask = (xp > EDGE) & (xp < 2048 - EDGE) & (yp > EDGE) & (yp < 2048 - EDGE)
        det_cats[fwhm] = {
            'x': xp[mask], 'y': yp[mask],
            'r1': r1[mask], 'r2': r2[mask],
            'sharp': sharp[mask], 'peak': peak[mask],
        }
        print(f'    FWHM={fwhm:2d}: {mask.sum()} detections')
    return det_cats


def propagate_gaia(target, obs_epoch):
    """Load Gaia catalog and propagate proper motions to observation epoch."""
    gaia = Table.read(f'{ASTROM_DIR}/gaia_{target}.vot', format='votable')
    dt_yr = (obs_epoch - GAIA_EPOCH).to(u.yr).value

    gaia_ra = np.array(gaia['ra'], dtype=np.float64)
    gaia_dec = np.array(gaia['dec'], dtype=np.float64)
    gaia_pmra = np.array(gaia['pmra'], dtype=np.float64)
    gaia_pmdec = np.array(gaia['pmdec'], dtype=np.float64)
    gaia_gmag = np.array(gaia['phot_g_mean_mag'], dtype=np.float64)
    gaia_ra_err = np.array(gaia['ra_error'], dtype=np.float64)
    gaia_dec_err = np.array(gaia['dec_error'], dtype=np.float64)
    gaia_pmra_err = np.array(gaia['pmra_error'], dtype=np.float64)
    gaia_pmdec_err = np.array(gaia['pmdec_error'], dtype=np.float64)

    has_pm = np.isfinite(gaia_pmra) & np.isfinite(gaia_pmdec)
    cos_dec = np.cos(np.radians(gaia_dec))
    ra_prop = gaia_ra.copy()
    dec_prop = gaia_dec.copy()
    ra_prop[has_pm] += (gaia_pmra[has_pm] * dt_yr / 3600000.0) / cos_dec[has_pm]
    dec_prop[has_pm] += gaia_pmdec[has_pm] * dt_yr / 3600000.0

    gaia_pmra_err[~np.isfinite(gaia_pmra_err)] = 0
    gaia_pmdec_err[~np.isfinite(gaia_pmdec_err)] = 0
    ra_err_prop = np.sqrt(gaia_ra_err**2 + (gaia_pmra_err * dt_yr)**2)
    dec_err_prop = np.sqrt(gaia_dec_err**2 + (gaia_pmdec_err * dt_yr)**2)

    return {
        'ra_prop': ra_prop, 'dec_prop': dec_prop,
        'gmag': gaia_gmag,
        'ra_err': ra_err_prop, 'dec_err': dec_err_prop,
        'dt_yr': dt_yr,
    }


def match_best_roundness(det_cats, wcs_use, gaia_info, cfg, std):
    """
    For each Gaia source, try all FWHM catalogs and pick the one where
    the matched detection has the best (smallest |roundness1|).
    Filter to |r1| < roundness_cut.
    """
    glim = cfg['gaia_mag_limit']
    match_rad = cfg['match_radius_arcsec']
    r1_cut = cfg['roundness_cut']

    bright = gaia_info['gmag'] < glim
    gpx, gpy = wcs_use.world_to_pixel_values(
        gaia_info['ra_prop'][bright], gaia_info['dec_prop'][bright])
    on_det = (gpx > EDGE) & (gpx < 2048 - EDGE) & (gpy > EDGE) & (gpy < 2048 - EDGE)

    gaia_sky = SkyCoord(
        ra=gaia_info['ra_prop'][bright][on_det] * u.deg,
        dec=gaia_info['dec_prop'][bright][on_det] * u.deg)
    gmags_on = gaia_info['gmag'][bright][on_det]
    ra_err_on = gaia_info['ra_err'][bright][on_det]
    dec_err_on = gaia_info['dec_err'][bright][on_det]

    rows = []
    for gi in range(len(gaia_sky)):
        best_r1 = 999
        best_match = None

        for fwhm in FWHM_GRID:
            if fwhm not in det_cats:
                continue
            cat = det_cats[fwhm]
            sky_det = wcs_use.pixel_to_world(cat['x'], cat['y'])
            sc_det = SkyCoord(ra=sky_det.ra, dec=sky_det.dec)
            sep = gaia_sky[gi].separation(sc_det).arcsec
            idx = np.argmin(sep)

            if sep[idx] < match_rad and abs(cat['r1'][idx]) < best_r1:
                best_r1 = abs(cat['r1'][idx])
                cos_d = np.cos(np.radians(float(sc_det[idx].dec.deg)))
                snr = cat['peak'][idx] / std
                centroid_err_mas = (fwhm / (2 * max(snr, 3))) * 63.0

                best_match = {
                    'gaia_ra': float(gaia_sky[gi].ra.deg),
                    'gaia_dec': float(gaia_sky[gi].dec.deg),
                    'gaia_gmag': float(gmags_on[gi]),
                    'gaia_ra_err_mas': float(ra_err_on[gi]),
                    'gaia_dec_err_mas': float(dec_err_on[gi]),
                    'jwst_x': float(cat['x'][idx]),
                    'jwst_y': float(cat['y'][idx]),
                    'dra_mas': (float(sc_det[idx].ra.deg) - float(gaia_sky[gi].ra.deg)) * cos_d * 3600e3,
                    'ddec_mas': (float(sc_det[idx].dec.deg) - float(gaia_sky[gi].dec.deg)) * 3600e3,
                    'sep_mas': sep[idx] * 1000,
                    'best_fwhm': float(fwhm),
                    'roundness1': float(cat['r1'][idx]),
                    'sharpness': float(cat['sharp'][idx]),
                    'peak': float(cat['peak'][idx]),
                    'centroid_err_mas': centroid_err_mas,
                    'era_mas': float(np.sqrt(ra_err_on[gi]**2 + centroid_err_mas**2)),
                    'edec_mas': float(np.sqrt(dec_err_on[gi]**2 + centroid_err_mas**2)),
                }

        if best_match is not None and abs(best_match['roundness1']) < r1_cut:
            rows.append(best_match)

    return rows


def iqr_clip(rows):
    """3x IQR clip on dra_mas and ddec_mas."""
    dra = np.array([r['dra_mas'] for r in rows])
    ddec = np.array([r['ddec_mas'] for r in rows])
    q1r, q3r = np.percentile(dra, [25, 75]); iqr_r = q3r - q1r
    q1d, q3d = np.percentile(ddec, [25, 75]); iqr_d = q3d - q1d
    clip = ((dra >= q1r - 3 * iqr_r) & (dra <= q3r + 3 * iqr_r) &
            (ddec >= q1d - 3 * iqr_d) & (ddec <= q3d + 3 * iqr_d))
    return [r for r, c in zip(rows, clip) if c]


def compute_shift(rows):
    """Compute median shift and statistics from clipped matches."""
    dra = np.array([r['dra_mas'] for r in rows])
    ddec = np.array([r['ddec_mas'] for r in rows])
    n = len(rows)
    shift_ra = np.median(dra)
    shift_dec = np.median(ddec)
    std_ra = np.std(dra)
    std_dec = np.std(ddec)
    unc_ra = std_ra / np.sqrt(n)
    unc_dec = std_dec / np.sqrt(n)
    unc_tot = np.sqrt(unc_ra**2 + unc_dec**2)
    resid = np.sqrt((dra - shift_ra)**2 + (ddec - shift_dec)**2)
    return {
        'shift_ra': shift_ra, 'shift_dec': shift_dec,
        'std_ra': std_ra, 'std_dec': std_dec,
        'unc_ra': unc_ra, 'unc_dec': unc_dec, 'unc_tot': unc_tot,
        'median_resid': np.median(resid), 'p90_resid': np.percentile(resid, 90),
        'n': n,
    }


def make_diagnostic_plot(rows_pre, rows_post, stats, target, seg, out_path):
    """Generate 2x3 pre/post correction diagnostic plot."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for row_idx, (rows, title_prefix) in enumerate([
        (rows_pre, 'Before correction'),
        (rows_post, 'After correction'),
    ]):
        dra = np.array([r['dra_mas'] for r in rows])
        ddec = np.array([r['ddec_mas'] for r in rows])
        gmag = np.array([r['gaia_gmag'] for r in rows])
        era = np.array([r['era_mas'] for r in rows])
        edec = np.array([r['edec_mas'] for r in rows])
        n_p = len(dra)
        med_ra = np.median(dra); med_dec = np.median(ddec)
        std_ra = np.std(dra); std_dec = np.std(ddec)
        unc_ra = std_ra / np.sqrt(n_p); unc_dec = std_dec / np.sqrt(n_p)
        resid = np.sqrt((dra - med_ra)**2 + (ddec - med_dec)**2)

        ax = axes[row_idx, 0]
        ax.errorbar(gmag, dra, yerr=era, fmt='o', ms=4, color='#1f77b4',
                    ecolor='#1f77b4', elinewidth=0.5, capsize=2, alpha=0.7)
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.axhline(med_ra, color='red', lw=1.5,
                    label=f'med={med_ra:+.1f}+/-{unc_ra:.1f}')
        ax.fill_between([12, 20], med_ra - std_ra, med_ra + std_ra,
                        color='red', alpha=0.08)
        ax.set_ylabel(r'$\Delta\alpha\cos\delta$ (mas)')
        ax.set_xlim(12, max(gmag) + 0.5)
        ax.set_title(f'{title_prefix}: RA (N={n_p})')
        ax.legend(fontsize=7)
        if row_idx == 1:
            ax.set_xlabel('Gaia G (mag)')

        ax = axes[row_idx, 1]
        ax.errorbar(gmag, ddec, yerr=edec, fmt='o', ms=4, color='#d62728',
                    ecolor='#d62728', elinewidth=0.5, capsize=2, alpha=0.7)
        ax.axhline(0, color='k', ls='--', lw=0.5)
        ax.axhline(med_dec, color='red', lw=1.5,
                    label=f'med={med_dec:+.1f}+/-{unc_dec:.1f}')
        ax.fill_between([12, 20], med_dec - std_dec, med_dec + std_dec,
                        color='red', alpha=0.08)
        ax.set_ylabel(r'$\Delta\delta$ (mas)')
        ax.set_xlim(12, max(gmag) + 0.5)
        ax.set_title(f'{title_prefix}: Dec')
        ax.legend(fontsize=7)
        if row_idx == 1:
            ax.set_xlabel('Gaia G (mag)')

        ax = axes[row_idx, 2]
        ax.hist(resid, bins=25, color='#7f7f7f', edgecolor='k', alpha=0.8)
        ax.axvline(np.median(resid), color='red', lw=2,
                    label=f'med={np.median(resid):.0f} mas')
        ax.axvline(np.percentile(resid, 90), color='orange', ls='--', lw=1.5,
                    label=f'90th={np.percentile(resid, 90):.0f} mas')
        ax.set_xlabel('Separation (mas)')
        ax.set_ylabel('N')
        ax.set_title(f'{title_prefix}: residuals')
        ax.legend(fontsize=7)

    ss = seg.replace('Segment', 'S')
    fig.suptitle(
        f'{target} {ss} nrcblong — Gaia DR3 alignment\n'
        f'Shift=({stats["shift_ra"]:+.1f},{stats["shift_dec"]:+.1f}) mas, '
        f'unc={stats["unc_tot"]:.1f} mas, '
        f'post-corr median={stats["median_resid"]:.0f} mas',
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def process_segment(target, seg, cfg, plot_only=False):
    """Full LW calibration for one target/segment."""
    det = 'nrcblong'
    obs_epoch = OBS_EPOCHS[(target, seg)]
    ss = seg.replace('Segment', 'S')

    print(f'\n{"=" * 60}')
    print(f'{target}/{seg}/{det}')
    print(f'  G < {cfg["gaia_mag_limit"]}, |r1| < {cfg["roundness_cut"]}, '
          f'match < {cfg["match_radius_arcsec"]}"')
    print(f'{"=" * 60}')

    # Load data
    zf_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
    zf = fits.getdata(zf_path)
    cal_files = sorted([f for f in glob.glob(
        f'/data/JWST/{target}/{seg}/*{det}*cal.fits') if 'uncal' not in f])
    if not cal_files:
        cal_files = sorted(glob.glob(
            f'/data/JWST/{target}/{seg}/calints/*{det}*calints.fits'))
    hdr = fits.getheader(cal_files[0], 'SCI')
    wcs_cal = WCS(hdr)
    mn, md, std = sigma_clipped_stats(zf[np.isfinite(zf)], sigma=3.0)

    # Detect sources
    print('  Detecting sources...')
    det_cats = detect_all_fwhm(zf, std)

    # Propagate Gaia
    gaia_info = propagate_gaia(target, obs_epoch)

    # Pre-correction matching
    print('  Matching (pre-correction)...')
    rows_pre = match_best_roundness(det_cats, wcs_cal, gaia_info, cfg, std)
    rows_pre_c = iqr_clip(rows_pre)
    stats_pre = compute_shift(rows_pre_c)

    print(f'  Pre-correction: {len(rows_pre)} raw -> {stats_pre["n"]} clipped')
    print(f'  Shift: ({stats_pre["shift_ra"]:+.1f}, {stats_pre["shift_dec"]:+.1f}) mas')
    print(f'  Std: ({stats_pre["std_ra"]:.1f}, {stats_pre["std_dec"]:.1f}) mas')
    print(f'  Unc: {stats_pre["unc_tot"]:.1f} mas')

    if not plot_only:
        # Apply correction
        hdr_new = hdr.copy()
        cos_d = np.cos(np.radians(hdr_new['CRVAL2']))
        hdr_new['CRVAL1'] -= stats_pre['shift_ra'] / 1000 / 3600 / cos_d
        hdr_new['CRVAL2'] -= stats_pre['shift_dec'] / 1000 / 3600
        hdr_new['GAIADRA'] = float(stats_pre['shift_ra'] / 1000)
        hdr_new['GAIADDEC'] = float(stats_pre['shift_dec'] / 1000)
        hdr_new['GAIANMAT'] = stats_pre['n']
        hdr_new['GAIARESI'] = float(stats_pre['median_resid'])
        hdr_new['COMMENT'] = (f'Gaia DR3 alignment: dRA={stats_pre["shift_ra"]:+.1f}mas '
                               f'dDec={stats_pre["shift_dec"]:+.1f}mas '
                               f'from {stats_pre["n"]} sources')

        out_wcs = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_gaia.fits'
        fits.writeto(out_wcs, zf.astype(np.float32), hdr_new, overwrite=True)
        print(f'  Saved: {out_wcs}')
        wcs_corr = WCS(hdr_new)
    else:
        wcs_corr = WCS(fits.getheader(
            f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_gaia.fits'))

    # Post-correction matching
    print('  Matching (post-correction)...')
    rows_post = match_best_roundness(det_cats, wcs_corr, gaia_info, cfg, std)
    rows_post_c = iqr_clip(rows_post)
    stats_post = compute_shift(rows_post_c)

    print(f'  Post-correction: {stats_post["n"]} sources')
    print(f'  Median residual: {stats_post["median_resid"]:.0f} mas')
    print(f'  90th pct: {stats_post["p90_resid"]:.0f} mas')

    # Save match table
    tbl = Table(rows_pre_c)
    tbl_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_gaia_match_final.fits'
    tbl.write(tbl_path, overwrite=True)
    print(f'  Match table: {tbl_path} ({len(tbl)} rows)')

    # Diagnostic plot
    diag_path = (f'{ASTROM_DIR}/diagnostics/'
                 f'{target}_{seg}_{det}_gaia_pre_post.png')
    make_diagnostic_plot(rows_pre_c, rows_post_c, stats_pre, target, seg,
                         diag_path)
    print(f'  Diagnostic: {diag_path}')

    return {
        'target': target, 'seg': seg,
        'n': stats_pre['n'],
        'shift_ra': stats_pre['shift_ra'],
        'shift_dec': stats_pre['shift_dec'],
        'unc_tot': stats_pre['unc_tot'],
        'median_resid': stats_post['median_resid'],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate LW (nrcblong) astrometry to Gaia DR3')
    parser.add_argument('--target', required=True, choices=list(TARGET_CONFIG.keys()))
    parser.add_argument('--plot-only', action='store_true',
                        help='Regenerate plots from existing WCS without recomputing')
    args = parser.parse_args()

    target = args.target
    cfg = TARGET_CONFIG[target]

    results = []
    for seg in cfg['segments']:
        r = process_segment(target, seg, cfg, plot_only=args.plot_only)
        results.append(r)

    print(f'\n{"=" * 60}')
    print(f'SUMMARY')
    print(f'{"=" * 60}')
    for r in results:
        ss = r['seg'].replace('Segment', 'S')
        print(f'  {r["target"]} {ss}: N={r["n"]}, '
              f'shift=({r["shift_ra"]:+.1f},{r["shift_dec"]:+.1f}) mas, '
              f'unc={r["unc_tot"]:.1f} mas, '
              f'post-corr median={r["median_resid"]:.0f} mas')


if __name__ == '__main__':
    main()
