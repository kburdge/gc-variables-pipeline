#!/usr/bin/env python
"""
===========================================================================
 SW detector astrometric calibration via LW cross-match
===========================================================================

 *** DO NOT DELETE THIS FILE ***
 Aligns each SW detector (nrcb1-4) to the Gaia-corrected LW (nrcblong)
 frame by cross-matching stellar detections between SW and LW zero-frame
 median images. Applies a shift-only correction preserving JWST SIP.

 Depends on: calibrate_lw_astrometry.py having been run first to produce
 the Gaia-corrected LW WCS files.

Recipe:
  1. Detect sources in LW uncal ZF median (FWHM=4, thresh=10*sigma)
  2. Project LW detections to sky using Gaia-corrected LW WCS
  3. Detect sources in SW uncal ZF median (FWHM=2, thresh=10*sigma)
  4. Project SW detections to sky using calints WCS
  5. Cross-match within 0.2"
  6. Compute median shift and apply as CRVAL correction

Usage:
    python calibrate_sw_astrometry_lw.py --target Terzan5
    python calibrate_sw_astrometry_lw.py --target Liller1
"""
import numpy as np
import os
import sys
import glob
import argparse
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
SW_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']

TARGETS = {
    'Terzan5': {'segments': ['Segment2']},
    'Liller1': {'segments': ['Segment3', 'Segment4']},
}


def detect_sources(image, fwhm, threshold_sigma, brightest=2000, edge=20):
    mn, md, std = sigma_clipped_stats(image[np.isfinite(image)], sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std, brightest=brightest)
    sources = finder(image - md)
    if sources is None:
        return None, None
    xp = np.array(sources['xcentroid'])
    yp = np.array(sources['ycentroid'])
    ok = (xp > edge) & (xp < 2048 - edge) & (yp > edge) & (yp < 2048 - edge)
    return xp[ok], yp[ok]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=list(TARGETS.keys()))
    args = parser.parse_args()

    target = args.target
    segments = TARGETS[target]['segments']
    results = []

    for seg in segments:
        # Load LW reference (Gaia-corrected)
        lw_zf_path = f'{ASTROM_DIR}/{target}_{seg}_nrcblong_uncal_zf_median.fits'
        lw_wcs_path = f'{ASTROM_DIR}/{target}_{seg}_nrcblong_wcs_gaia.fits'
        if not os.path.exists(lw_wcs_path):
            print(f'  No Gaia-corrected LW WCS for {seg}, skipping')
            continue

        lw_zf = fits.getdata(lw_zf_path)
        lw_wcs = WCS(fits.getheader(lw_wcs_path))
        lw_x, lw_y = detect_sources(lw_zf, fwhm=4.0, threshold_sigma=10, brightest=2000, edge=10)
        lw_sky = lw_wcs.pixel_to_world(lw_x, lw_y)
        lw_ra = lw_sky.ra.deg
        lw_dec = lw_sky.dec.deg
        lw_sc = SkyCoord(ra=lw_ra * u.deg, dec=lw_dec * u.deg)

        print(f'\n{"=" * 60}')
        print(f'{target}/{seg} — LW reference: {len(lw_ra)} sources')

        for det in SW_DETS:
            sw_zf_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
            if not os.path.exists(sw_zf_path):
                continue

            # Load SW with calints WCS
            cal_files = sorted([f for f in glob.glob(
                f'/data/JWST/{target}/{seg}/*{det}*cal.fits') if 'uncal' not in f])
            if not cal_files:
                cal_files = sorted(glob.glob(
                    f'/data/JWST/{target}/{seg}/calints/*{det}*calints.fits'))
            if not cal_files:
                continue

            hdr = fits.getheader(cal_files[0], 'SCI')
            sw_wcs = WCS(hdr)
            sw_zf = fits.getdata(sw_zf_path)
            sw_x, sw_y = detect_sources(sw_zf, fwhm=2.0, threshold_sigma=10,
                                         brightest=2000, edge=20)
            if sw_x is None or len(sw_x) < 10:
                continue

            sw_sky = sw_wcs.pixel_to_world(sw_x, sw_y)
            sw_ra = sw_sky.ra.deg
            sw_dec = sw_sky.dec.deg
            sw_sc = SkyCoord(ra=sw_ra * u.deg, dec=sw_dec * u.deg)

            # Cross-match
            idx, sep, _ = sw_sc.match_to_catalog_sky(lw_sc)
            matched = sep.arcsec < 0.2
            n_match = matched.sum()
            if n_match < 10:
                print(f'  {det}: only {n_match} matches, skipping')
                continue

            jx = sw_x[matched]
            jy = sw_y[matched]
            cos_dec = np.cos(np.radians(np.mean(lw_dec[idx[matched]])))
            dra = (sw_ra[matched] - lw_ra[idx[matched]]) * cos_dec * 3600e3
            ddec = (sw_dec[matched] - lw_dec[idx[matched]]) * 3600e3

            # Median shift
            shift_ra = np.median(dra)
            shift_dec = np.median(ddec)
            dra_c = dra - shift_ra
            ddec_c = ddec - shift_dec
            resid = np.sqrt(dra_c**2 + ddec_c**2)

            print(f'  {det}: {n_match} matches, '
                  f'shift=({shift_ra:+.1f},{shift_dec:+.1f}) mas, '
                  f'med={np.median(resid):.1f} mas, '
                  f'90th={np.percentile(resid, 90):.1f} mas')

            # Save corrected WCS
            hdr_new = hdr.copy()
            cos_d = np.cos(np.radians(hdr_new['CRVAL2']))
            hdr_new['CRVAL1'] -= shift_ra / 1000 / 3600 / cos_d
            hdr_new['CRVAL2'] -= shift_dec / 1000 / 3600
            hdr_new['LWDRA'] = float(shift_ra / 1000)
            hdr_new['LWDDEC'] = float(shift_dec / 1000)
            hdr_new['LWNMAT'] = n_match
            hdr_new['LWRESI'] = float(np.median(resid))
            hdr_new['COMMENT'] = (f'LW-aligned: dRA={shift_ra:.1f}mas '
                                   f'dDec={shift_dec:.1f}mas from {n_match} matches')

            out_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_lw.fits'
            fits.writeto(out_path, sw_zf.astype(np.float32), hdr_new, overwrite=True)

            results.append({
                'target': target, 'seg': seg, 'det': det, 'n': n_match,
                'shift_ra': shift_ra, 'shift_dec': shift_dec,
                'med': np.median(resid), 'p90': np.percentile(resid, 90),
            })

            # Diagnostic plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            ax = axes[0, 0]
            ax.scatter(jx, dra_c, s=3, alpha=0.5, c='#1f77b4')
            ax.axhline(0, color='k', ls='--', lw=0.5)
            ax.set_xlabel('X pixel'); ax.set_ylabel('dRA (mas)')
            ax.set_title('dRA vs X (after shift)')

            ax = axes[0, 1]
            ax.scatter(jx, ddec_c, s=3, alpha=0.5, c='#d62728')
            ax.axhline(0, color='k', ls='--', lw=0.5)
            ax.set_xlabel('X pixel'); ax.set_ylabel('dDec (mas)')
            ax.set_title('dDec vs X (after shift)')

            ax = axes[0, 2]
            ax.scatter(jy, ddec_c, s=3, alpha=0.5, c='#d62728')
            ax.axhline(0, color='k', ls='--', lw=0.5)
            ax.set_xlabel('Y pixel'); ax.set_ylabel('dDec (mas)')
            ax.set_title('dDec vs Y (after shift)')

            ax = axes[1, 0]
            sc = ax.scatter(jx, jy, c=np.sqrt(dra_c**2 + ddec_c**2),
                            cmap='hot_r', vmin=0, vmax=30, s=5)
            plt.colorbar(sc, ax=ax, label='sep (mas)')
            ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
            ax.set_title('Residual magnitude')

            ax = axes[1, 1]
            q = ax.quiver(jx, jy, dra_c, ddec_c, scale=300,
                          scale_units='width', width=0.002, color='green', alpha=0.5)
            ax.quiverkey(q, 0.85, 0.95, 10, '10 mas', coordinates='axes',
                         fontproperties={'size': 8})
            ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')
            ax.set_title('Residual vectors')

            ax = axes[1, 2]
            ax.hist(resid, bins=30, color='#7f7f7f', edgecolor='k', alpha=0.8)
            ax.axvline(np.median(resid), color='red', lw=2,
                       label=f'med={np.median(resid):.1f}')
            ax.axvline(np.percentile(resid, 90), color='orange', ls='--', lw=1.5,
                       label=f'90th={np.percentile(resid, 90):.1f}')
            ax.set_xlabel('Sep (mas)'); ax.legend(fontsize=8)
            ax.set_title('Residuals')

            ss = seg.replace('Segment', 'S')
            fig.suptitle(
                f'{target} {ss} {det} — LW-aligned (shift only)\n'
                f'{n_match} matches, shift=({shift_ra:+.1f},{shift_dec:+.1f}) mas, '
                f'med={np.median(resid):.1f} mas',
                fontsize=11, y=1.02)
            fig.tight_layout()
            fig.savefig(f'{ASTROM_DIR}/diagnostics/{target}_{seg}_{det}_lw_aligned.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

    # Summary
    print(f'\n{"=" * 60}')
    print(f'SUMMARY')
    print(f'{"=" * 60}')
    for r in results:
        ss = r['seg'].replace('Segment', 'S')
        print(f'  {r["target"]} {ss}/{r["det"]}: N={r["n"]:3d}, '
              f'shift=({r["shift_ra"]:+5.1f},{r["shift_dec"]:+5.1f}) mas, '
              f'med={r["med"]:4.1f} mas, 90th={r["p90"]:4.1f} mas')


if __name__ == '__main__':
    main()
