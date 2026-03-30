#!/usr/bin/env python
"""
Generate astrometry diagnostic plots for each detector.

For each detector, shows:
  - RA offset (JWST - Gaia) vs Gaia RA, with error bars
  - Dec offset vs Gaia Dec, with error bars
  - Quiver plot of offset vectors on the detector
  - Histogram of total separations before and after correction

Usage:
    python astrometry_diagnostics.py
"""
import numpy as np
import os
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
OUT_DIR = f'{ASTROM_DIR}/diagnostics'
os.makedirs(OUT_DIR, exist_ok=True)

GAIA_EPOCH = Time('J2016.0')
OBS_EPOCHS = {
    ('Liller1', 'Segment3'): Time(60787.5, format='mjd'),
    ('Liller1', 'Segment4'): Time(60788.97, format='mjd'),
    ('Terzan5', 'Segment2'): Time(60786.49, format='mjd'),
}


def load_gaia(target):
    """Load cached Gaia catalog with PM-propagated positions."""
    cache = f'{ASTROM_DIR}/gaia_{target}.vot'
    tbl = Table.read(cache, format='votable')
    return tbl


def detect_and_match(target, seg, det, gaia_tbl, obs_epoch):
    """Detect stars in ZF median, match to Gaia, return matched pairs."""
    # Load ZF median
    # Prefer uncal ZF median (no zeroed saturated pixels)
    zf_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
    if not os.path.exists(zf_path):
        zf_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_zf_median.fits'
    if not os.path.exists(zf_path):
        # Create it
        cube = fits.getdata(f'{BASE}/refs/zeroframes_{target}_{seg}_{det}.fits', memmap=True)
        median_img = np.nanmedian(cube, axis=0)
    else:
        median_img = fits.getdata(zf_path)

    # Load ORIGINAL WCS (before Gaia correction) for comparison
    # We need both old and new to show before/after
    ref_path = f'{BASE}/refs/{target}_{seg}_{det}_ref.fits'
    wcs_corrected = WCS(fits.getheader(ref_path))  # now has Gaia correction applied

    gaia_wcs_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_gaia.fits'
    wcs_gaia = WCS(fits.getheader(gaia_wcs_path))
    dra_applied = fits.getheader(gaia_wcs_path).get('GAIADRA', 0)
    ddec_applied = fits.getheader(gaia_wcs_path).get('GAIADDEC', 0)

    # Detect stars (LW has broader PSF)
    fwhm = 3.5 if det == 'nrcblong' else 2.0
    mean, median_val, std = sigma_clipped_stats(median_img[np.isfinite(median_img)], sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=10 * std, brightest=500)
    sources = finder(median_img - median_val)
    if sources is None or len(sources) < 5:
        return None

    # Get pixel positions and SNR for centroid uncertainty
    xpix = np.array(sources['xcentroid'])
    ypix = np.array(sources['ycentroid'])
    peak = np.array(sources['peak'])
    # Centroid uncertainty ~ FWHM / (2 * SNR) in pixels
    # SNR ~ peak / std
    snr = peak / std
    snr = np.clip(snr, 3, 1000)
    centroid_err_pix = 2.0 / (2.0 * snr)  # FWHM=2 pixels
    # Convert to arcsec (31 mas/pix for SW, 63 mas/pix for LW)
    pixscale = 0.063 if det == 'nrcblong' else 0.031
    centroid_err_arcsec = centroid_err_pix * pixscale

    # Convert pixel to sky using corrected WCS
    jwst_sky = wcs_corrected.pixel_to_world(xpix, ypix)
    jwst_ra = jwst_sky.ra.deg
    jwst_dec = jwst_sky.dec.deg

    # Propagate Gaia proper motions
    dt_yr = (obs_epoch - GAIA_EPOCH).to(u.yr).value
    gaia_ra = np.array(gaia_tbl['ra'], dtype=np.float64)
    gaia_dec = np.array(gaia_tbl['dec'], dtype=np.float64)
    gaia_pmra = np.array(gaia_tbl['pmra'], dtype=np.float64)
    gaia_pmdec = np.array(gaia_tbl['pmdec'], dtype=np.float64)
    gaia_ra_err = np.array(gaia_tbl['ra_error'], dtype=np.float64)  # mas
    gaia_dec_err = np.array(gaia_tbl['dec_error'], dtype=np.float64)  # mas
    gaia_gmag = np.array(gaia_tbl['phot_g_mean_mag'], dtype=np.float64)

    has_pm = np.isfinite(gaia_pmra) & np.isfinite(gaia_pmdec)
    cos_dec = np.cos(np.radians(gaia_dec))
    ra_prop = gaia_ra.copy()
    dec_prop = gaia_dec.copy()
    ra_prop[has_pm] += (gaia_pmra[has_pm] * dt_yr / 3600000.0) / cos_dec[has_pm]
    dec_prop[has_pm] += gaia_pmdec[has_pm] * dt_yr / 3600000.0

    # Propagated position uncertainty (add PM error * dt in quadrature)
    gaia_pmra_err = np.array(gaia_tbl['pmra_error'], dtype=np.float64)
    gaia_pmdec_err = np.array(gaia_tbl['pmdec_error'], dtype=np.float64)
    gaia_pmra_err[~np.isfinite(gaia_pmra_err)] = 0
    gaia_pmdec_err[~np.isfinite(gaia_pmdec_err)] = 0
    gaia_ra_err_prop = np.sqrt(gaia_ra_err**2 + (gaia_pmra_err * dt_yr)**2)  # mas
    gaia_dec_err_prop = np.sqrt(gaia_dec_err**2 + (gaia_pmdec_err * dt_yr)**2)

    gaia_sky = SkyCoord(ra=ra_prop*u.deg, dec=dec_prop*u.deg)

    # Cross-match
    jwst_skycoord = SkyCoord(ra=jwst_ra*u.deg, dec=jwst_dec*u.deg)
    idx, sep, _ = jwst_skycoord.match_to_catalog_sky(gaia_sky)
    good = sep.arcsec < 0.5

    if good.sum() < 3:
        return None

    return {
        'jwst_ra': jwst_ra[good],
        'jwst_dec': jwst_dec[good],
        'jwst_x': xpix[good],
        'jwst_y': ypix[good],
        'jwst_err': centroid_err_arcsec[good],  # arcsec
        'gaia_ra': ra_prop[idx[good]],
        'gaia_dec': dec_prop[idx[good]],
        'gaia_ra_err': gaia_ra_err_prop[idx[good]] / 1000.0,  # arcsec
        'gaia_dec_err': gaia_dec_err_prop[idx[good]] / 1000.0,
        'gaia_gmag': gaia_gmag[idx[good]],
        'sep': sep[good].arcsec,
        'dra_applied': dra_applied,
        'ddec_applied': ddec_applied,
        'n_match': good.sum(),
    }


def plot_detector(target, seg, det, data, out_path):
    """Generate 4-panel diagnostic plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    cos_dec = np.cos(np.radians(np.mean(data['gaia_dec'])))

    # RA and Dec offsets (JWST - Gaia), in mas
    dra_mas = (data['jwst_ra'] - data['gaia_ra']) * cos_dec * 3600 * 1000
    ddec_mas = (data['jwst_dec'] - data['gaia_dec']) * 3600 * 1000

    # Combined uncertainties in mas
    era_mas = np.sqrt((data['jwst_err'] * 1000)**2 + (data['gaia_ra_err'] * 1000)**2)
    edec_mas = np.sqrt((data['jwst_err'] * 1000)**2 + (data['gaia_dec_err'] * 1000)**2)

    # Panel 1: RA offset vs Gaia G mag
    ax = axes[0, 0]
    ax.errorbar(data['gaia_gmag'], dra_mas, yerr=era_mas,
                fmt='o', ms=3, color='#1f77b4', ecolor='#1f77b4',
                elinewidth=0.5, capsize=2, alpha=0.7)
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(np.median(dra_mas), color='red', ls='-', lw=1,
               label=f'median = {np.median(dra_mas):.1f} mas')
    ax.set_xlabel('Gaia G (mag)', fontsize=10)
    ax.set_ylabel('ΔRA (JWST − Gaia) [mas]', fontsize=10)
    ax.set_title('RA offset vs brightness', fontsize=11)
    ax.legend(fontsize=8)

    # Panel 2: Dec offset vs Gaia G mag
    ax = axes[0, 1]
    ax.errorbar(data['gaia_gmag'], ddec_mas, yerr=edec_mas,
                fmt='o', ms=3, color='#d62728', ecolor='#d62728',
                elinewidth=0.5, capsize=2, alpha=0.7)
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(np.median(ddec_mas), color='red', ls='-', lw=1,
               label=f'median = {np.median(ddec_mas):.1f} mas')
    ax.set_xlabel('Gaia G (mag)', fontsize=10)
    ax.set_ylabel('ΔDec (JWST − Gaia) [mas]', fontsize=10)
    ax.set_title('Dec offset vs brightness', fontsize=11)
    ax.legend(fontsize=8)

    # Panel 3: Quiver plot on detector
    ax = axes[1, 0]
    scale = 1000  # mas per arrow length unit
    q = ax.quiver(data['jwst_x'], data['jwst_y'],
                  dra_mas, ddec_mas, scale=scale, scale_units='width',
                  width=0.003, color='#2ca02c', alpha=0.7)
    ax.quiverkey(q, 0.85, 0.95, 50, '50 mas', coordinates='axes', fontproperties={'size': 8})
    ax.set_xlabel('X pixel', fontsize=10)
    ax.set_ylabel('Y pixel', fontsize=10)
    ax.set_title('Offset vectors on detector (after Gaia correction)', fontsize=11)
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048)
    ax.set_aspect('equal')

    # Panel 4: Histogram of total separation
    ax = axes[1, 1]
    total_sep_mas = np.sqrt(dra_mas**2 + ddec_mas**2)
    ax.hist(total_sep_mas, bins=20, color='#7f7f7f', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(total_sep_mas), color='red', ls='-', lw=2,
               label=f'median = {np.median(total_sep_mas):.0f} mas')
    ax.axvline(np.percentile(total_sep_mas, 90), color='orange', ls='--', lw=1.5,
               label=f'90th pct = {np.percentile(total_sep_mas, 90):.0f} mas')
    ax.set_xlabel('Total separation [mas]', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Residual separation distribution', fontsize=11)
    ax.legend(fontsize=8)

    ss = seg.replace('Segment', 'S')
    cluster = 'Liller 1' if target == 'Liller1' else 'Terzan 5'
    fig.suptitle(f'{cluster} {ss} {det} — Gaia DR3 Astrometric Alignment\n'
                 f'{data["n_match"]} matches, applied offset: '
                 f'ΔRA={data["dra_applied"]*1000:.1f} mas, '
                 f'ΔDec={data["ddec_applied"]*1000:.1f} mas',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    for target in ['Liller1', 'Terzan5']:
        gaia_path = f'{ASTROM_DIR}/gaia_{target}.vot'
        if not os.path.exists(gaia_path):
            print(f'No Gaia catalog for {target}, skipping')
            continue
        gaia_tbl = load_gaia(target)
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        for seg in segments:
            obs_epoch = OBS_EPOCHS.get((target, seg))
            if obs_epoch is None: continue

            for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']:
                zf_path = f'{BASE}/refs/zeroframes_{target}_{seg}_{det}.fits'
                if not os.path.exists(zf_path): continue

                print(f'{target}/{seg}/{det}...', end='', flush=True)
                data = detect_and_match(target, seg, det, gaia_tbl, obs_epoch)
                if data is None:
                    print(' too few matches')
                    continue

                out = f'{OUT_DIR}/{target}_{seg}_{det}_astrom.png'
                plot_detector(target, seg, det, data, out)
                print(f' {data["n_match"]} matches, saved')

    print(f'\nDiagnostics in {OUT_DIR}/')


if __name__ == '__main__':
    main()
