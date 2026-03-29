#!/usr/bin/env python
"""
Cross-match SW and LW detections to validate SW astrometry.

Uses the Gaia-corrected LW (nrcblong) WCS as truth, since it has
1.1 mas uncertainty from 208 Gaia matches. Detects stars in both
SW and LW uncal ZF medians, cross-matches by sky position, and
measures residuals as a function of detector position.

This avoids Gaia PM propagation errors and provides many more
reference stars than the G<19 Gaia sample.

Usage:
    python sw_lw_crossmatch.py --sw-det nrcb4
"""
import numpy as np
import os
import sys
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
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


def detect_stars(image, fwhm=5.0, threshold_sigma=10, brightest=1000, edge=20):
    """Detect stars with DAOStarFinder."""
    mn, md, std = sigma_clipped_stats(image[np.isfinite(image)], sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std, brightest=brightest)
    sources = finder(image - md)
    if sources is None:
        return None
    # Edge cut
    xp = np.array(sources['xcentroid'])
    yp = np.array(sources['ycentroid'])
    ny, nx = image.shape
    ok = (xp > edge) & (xp < nx - edge) & (yp > edge) & (yp < ny - edge)
    return sources[ok]


def main():
    sw_det = 'nrcb4'
    for arg in sys.argv[1:]:
        if arg.startswith('--sw-det='): sw_det = arg.split('=')[1]

    target = 'Terzan5'
    seg = 'Segment2'

    # ================================================================
    # Load LW (nrcblong) — our truth reference
    # ================================================================
    lw_zf = fits.getdata(f'{ASTROM_DIR}/{target}_{seg}_nrcblong_uncal_zf_median.fits')
    # Use Gaia-corrected WCS as truth
    lw_wcs_path = f'{ASTROM_DIR}/{target}_{seg}_nrcblong_wcs_gaia.fits'
    lw_wcs = WCS(fits.getheader(lw_wcs_path))

    print(f'LW (nrcblong): {lw_zf.shape}, Gaia-corrected WCS')

    # Detect in LW — FWHM ~3-4 px for F356W at 63 mas/px
    lw_sources = detect_stars(lw_zf, fwhm=4.0, threshold_sigma=10, brightest=2000, edge=10)
    lw_x = np.array(lw_sources['xcentroid'])
    lw_y = np.array(lw_sources['ycentroid'])
    lw_sky = lw_wcs.pixel_to_world(lw_x, lw_y)
    lw_ra = lw_sky.ra.deg
    lw_dec = lw_sky.dec.deg
    print(f'  Detected {len(lw_sources)} LW sources')

    # ================================================================
    # Load SW — what we're testing
    # ================================================================
    sw_zf = fits.getdata(f'{ASTROM_DIR}/{target}_{seg}_{sw_det}_uncal_zf_median.fits')
    # Use calints WCS (the one we're trying to validate/improve)
    cal_files = sorted([f for f in glob.glob(f'/data/JWST/{target}/{seg}/*{sw_det}*cal.fits')
                        if 'uncal' not in f])
    sw_wcs = WCS(fits.getheader(cal_files[0], 'SCI'))

    print(f'SW ({sw_det}): {sw_zf.shape}, calints WCS')

    # Detect in SW — FWHM ~2 px for F200W at 31 mas/px
    sw_sources = detect_stars(sw_zf, fwhm=2.0, threshold_sigma=10, brightest=2000, edge=20)
    sw_x = np.array(sw_sources['xcentroid'])
    sw_y = np.array(sw_sources['ycentroid'])
    sw_sky = sw_wcs.pixel_to_world(sw_x, sw_y)
    sw_ra = sw_sky.ra.deg
    sw_dec = sw_sky.dec.deg
    sw_peak = np.array(sw_sources['peak'])
    print(f'  Detected {len(sw_sources)} SW sources')

    # ================================================================
    # Cross-match SW to LW
    # ================================================================
    sw_sc = SkyCoord(ra=sw_ra * u.deg, dec=sw_dec * u.deg)
    lw_sc = SkyCoord(ra=lw_ra * u.deg, dec=lw_dec * u.deg)

    idx, sep, _ = sw_sc.match_to_catalog_sky(lw_sc)
    matched = sep.arcsec < 0.3  # tight match — both should be well-centroided

    n_match = matched.sum()
    print(f'\n  Cross-matched: {n_match} pairs (< 0.3\")')

    if n_match < 5:
        print('Too few matches!'); return

    # Residuals: SW position - LW position (in mas)
    cos_dec = np.cos(np.radians(np.mean(sw_dec[matched])))
    dra = (sw_ra[matched] - lw_ra[idx[matched]]) * cos_dec * 3600 * 1000  # mas
    ddec = (sw_dec[matched] - lw_dec[idx[matched]]) * 3600 * 1000

    # SW pixel positions of matched sources
    mx = sw_x[matched]
    my = sw_y[matched]
    mpeak = sw_peak[matched]

    resid = np.sqrt(dra**2 + ddec**2)

    print(f'\n  Residuals (SW - LW):')
    print(f'  ΔRA:  {np.median(dra):+.1f} ± {np.std(dra):.1f} mas')
    print(f'  ΔDec: {np.median(ddec):+.1f} ± {np.std(ddec):.1f} mas')
    print(f'  Total: median={np.median(resid):.1f}, 90th={np.percentile(resid, 90):.1f} mas')

    # ================================================================
    # Diagnostic plots
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    ax.scatter(mx, dra, s=5, alpha=0.5, c='#1f77b4')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(np.median(dra), color='red', ls='-', lw=1, label=f'med={np.median(dra):+.1f}')
    ax.set_xlabel('SW X pixel'); ax.set_ylabel('ΔRA (mas)'); ax.set_title('ΔRA vs X')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.scatter(mx, ddec, s=5, alpha=0.5, c='#d62728')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(np.median(ddec), color='red', ls='-', lw=1, label=f'med={np.median(ddec):+.1f}')
    ax.set_xlabel('SW X pixel'); ax.set_ylabel('ΔDec (mas)'); ax.set_title('ΔDec vs X')
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.scatter(my, ddec, s=5, alpha=0.5, c='#d62728')
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(np.median(ddec), color='red', ls='-', lw=1, label=f'med={np.median(ddec):+.1f}')
    ax.set_xlabel('SW Y pixel'); ax.set_ylabel('ΔDec (mas)'); ax.set_title('ΔDec vs Y')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    sc = ax.scatter(mx, my, c=ddec, cmap='RdBu_r', vmin=-50, vmax=50, s=8, edgecolors='k', linewidths=0.1)
    plt.colorbar(sc, ax=ax, label='ΔDec (mas)')
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
    ax.set_title('ΔDec on SW detector')

    ax = axes[1, 1]
    q = ax.quiver(mx, my, dra, ddec, scale=500, scale_units='width', width=0.002, color='#2ca02c', alpha=0.6)
    ax.quiverkey(q, 0.85, 0.95, 20, '20 mas', coordinates='axes', fontproperties={'size': 8})
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
    ax.set_title('Offset vectors (SW - LW)')

    ax = axes[1, 2]
    ax.hist(resid, bins=30, color='#7f7f7f', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(resid), color='red', ls='-', lw=2, label=f'median={np.median(resid):.0f} mas')
    ax.axvline(np.percentile(resid, 90), color='orange', ls='--', lw=1.5, label=f'90th={np.percentile(resid, 90):.0f} mas')
    ax.set_xlabel('Total sep (mas)'); ax.set_ylabel('Count')
    ax.set_title('SW-LW separation'); ax.legend(fontsize=8)

    fig.suptitle(f'{target} {seg} — SW ({sw_det}) vs LW (nrcblong, Gaia-corrected)\n'
                 f'{n_match} matches, ΔRA={np.median(dra):+.1f}±{np.std(dra):.1f}, '
                 f'ΔDec={np.median(ddec):+.1f}±{np.std(ddec):.1f} mas',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{ASTROM_DIR}/diagnostics/{target}_{seg}_{sw_det}_vs_lw_crossmatch.png',
                dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'\nPlot saved')


if __name__ == '__main__':
    main()
