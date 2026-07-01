#!/usr/bin/env python
"""
Build FITS match table for LW (nrcblong) to Gaia DR3 astrometric calibration.

For each Gaia source on the detector, finds the optimal FWHM for DAOStarFinder
by trying a range of values and picking the one that gives the closest match.
Stores all morphology metrics (roundness, sharpness, peak, SNR) for inspection
in TOPCAT.

Recipe:
  1. For each FWHM in a grid (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30):
     run DAOStarFinder on the uncal ZF median with threshold=3*sigma
  2. For each Gaia G<20 source on the detector:
     - Try each FWHM catalog
     - Find the nearest detection within 0.5"
     - Pick the FWHM that gives the smallest separation
  3. Store the match with all morphology stats

Usage:
    python build_lw_match_table.py --target Terzan5 --seg Segment2
    python build_lw_match_table.py --target Liller1 --seg Segment3
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
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
GAIA_EPOCH = Time('J2016.0')

OBS_EPOCHS = {
    ('Terzan5', 'Segment2'): Time(60786.49, format='mjd'),
    ('Liller1', 'Segment3'): Time(60787.5, format='mjd'),
    ('Liller1', 'Segment4'): Time(60788.97, format='mjd'),
}

FWHM_GRID = [4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
DETECT_THRESHOLD_SIGMA = 3
DETECT_BRIGHTEST = 3000
EDGE = 10
MATCH_RADIUS_ARCSEC = 0.5
GAIA_MAG_LIMIT = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--seg', required=True)
    args = parser.parse_args()

    target = args.target
    seg = args.seg
    det = 'nrcblong'
    obs_epoch = OBS_EPOCHS[(target, seg)]

    print(f'{target}/{seg}/{det}')

    # Load uncal ZF median
    zf_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
    zf = fits.getdata(zf_path)
    mn, md, std = sigma_clipped_stats(zf[np.isfinite(zf)], sigma=3.0)
    print(f'  ZF: median={md:.1f}, std={std:.1f}')

    # Load calints WCS
    cal_files = sorted([f for f in glob.glob(f'/data/JWST/{target}/{seg}/*{det}*cal.fits')
                        if 'uncal' not in f])
    if not cal_files:
        cal_files = sorted(glob.glob(f'/data/JWST/{target}/{seg}/calints/*{det}*calints.fits'))
    wcs_cal = WCS(fits.getheader(cal_files[0], 'SCI'))

    # Run DAOStarFinder at each FWHM
    det_catalogs = {}
    for fwhm in FWHM_GRID:
        finder = DAOStarFinder(fwhm=fwhm, threshold=DETECT_THRESHOLD_SIGMA * std,
                               brightest=DETECT_BRIGHTEST)
        sources = finder(zf - md)
        if sources is None:
            continue
        xp = np.array(sources['xcentroid'])
        yp = np.array(sources['ycentroid'])
        r1 = np.array(sources['roundness1'])
        r2 = np.array(sources['roundness2'])
        sharp = np.array(sources['sharpness'])
        peak = np.array(sources['peak'])
        mask = (xp > EDGE) & (xp < 2048 - EDGE) & (yp > EDGE) & (yp < 2048 - EDGE)
        det_catalogs[fwhm] = {
            'x': xp[mask], 'y': yp[mask],
            'r1': r1[mask], 'r2': r2[mask],
            'sharp': sharp[mask], 'peak': peak[mask],
        }
        print(f'  FWHM={fwhm:2d}: {mask.sum()} detections')

    # Load and propagate Gaia
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

    # Propagated errors
    gaia_pmra_err[~np.isfinite(gaia_pmra_err)] = 0
    gaia_pmdec_err[~np.isfinite(gaia_pmdec_err)] = 0
    ra_err_prop = np.sqrt(gaia_ra_err**2 + (gaia_pmra_err * dt_yr)**2)
    dec_err_prop = np.sqrt(gaia_dec_err**2 + (gaia_pmdec_err * dt_yr)**2)

    # Select Gaia sources on detector
    bright = gaia_gmag < GAIA_MAG_LIMIT
    gpx, gpy = wcs_cal.world_to_pixel_values(ra_prop[bright], dec_prop[bright])
    on_det = (gpx > EDGE) & (gpx < 2048 - EDGE) & (gpy > EDGE) & (gpy < 2048 - EDGE)
    gaia_sky = SkyCoord(ra=ra_prop[bright][on_det] * u.deg,
                        dec=dec_prop[bright][on_det] * u.deg)
    gmags_on = gaia_gmag[bright][on_det]
    ra_err_on = ra_err_prop[bright][on_det]
    dec_err_on = dec_err_prop[bright][on_det]

    print(f'  Gaia G<{GAIA_MAG_LIMIT} on detector: {on_det.sum()}')

    # For each Gaia source, find best FWHM match
    rows = []
    for gi in range(len(gaia_sky)):
        best_sep = 999.0
        best_match = None

        for fwhm in FWHM_GRID:
            if fwhm not in det_catalogs:
                continue
            cat = det_catalogs[fwhm]
            sky_det = wcs_cal.pixel_to_world(cat['x'], cat['y'])
            sc_det = SkyCoord(ra=sky_det.ra, dec=sky_det.dec)
            sep = gaia_sky[gi].separation(sc_det).arcsec
            idx = np.argmin(sep)
            if sep[idx] < MATCH_RADIUS_ARCSEC and sep[idx] < best_sep:
                best_sep = sep[idx]
                cos_d = np.cos(np.radians(float(sc_det[idx].dec.deg)))
                best_match = {
                    'gaia_ra': float(gaia_sky[gi].ra.deg),
                    'gaia_dec': float(gaia_sky[gi].dec.deg),
                    'gaia_gmag': float(gmags_on[gi]),
                    'gaia_ra_err_mas': float(ra_err_on[gi]),
                    'gaia_dec_err_mas': float(dec_err_on[gi]),
                    'jwst_ra': float(sc_det[idx].ra.deg),
                    'jwst_dec': float(sc_det[idx].dec.deg),
                    'jwst_x': float(cat['x'][idx]),
                    'jwst_y': float(cat['y'][idx]),
                    'dra_mas': (float(sc_det[idx].ra.deg) - float(gaia_sky[gi].ra.deg)) * cos_d * 3600e3,
                    'ddec_mas': (float(sc_det[idx].dec.deg) - float(gaia_sky[gi].dec.deg)) * 3600e3,
                    'sep_mas': best_sep * 1000,
                    'best_fwhm': float(fwhm),
                    'sharpness': float(cat['sharp'][idx]),
                    'roundness1': float(cat['r1'][idx]),
                    'roundness2': float(cat['r2'][idx]),
                    'peak': float(cat['peak'][idx]),
                }

        if best_match is not None:
            rows.append(best_match)

    tbl = Table(rows)
    out_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_gaia_match_v2.fits'
    tbl.write(out_path, overwrite=True)
    print(f'\nSaved {out_path}: {len(tbl)} matches')
    print(f'  G range: {tbl["gaia_gmag"].min():.1f} - {tbl["gaia_gmag"].max():.1f}')
    print(f'  FWHM range: {tbl["best_fwhm"].min():.0f} - {tbl["best_fwhm"].max():.0f}')
    print(f'  Roundness1 range: {tbl["roundness1"].min():.2f} - {tbl["roundness1"].max():.2f}')


if __name__ == '__main__':
    main()
