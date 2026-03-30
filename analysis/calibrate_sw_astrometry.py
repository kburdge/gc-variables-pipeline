#!/usr/bin/env python
"""
Calibrate SW detector astrometry for Terzan 5 using uncal ZF medians,
calints WCS, adaptive FWHM detection, roundness cuts, and IQR clipping.

Produces diagnostic plots and updated WCS files.
"""
import numpy as np
import glob
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


def run_detector(target, seg, det):
    print(f'\n{"="*60}')
    print(f'{target}/{seg}/{det}')
    print(f'{"="*60}')

    # Load uncal ZF median
    zf_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
    med = fits.getdata(zf_path)
    mn, md, std = sigma_clipped_stats(med[np.isfinite(med)], sigma=3.0)

    # Find calints file for this detector
    cal_files = sorted(glob.glob(f'/data/JWST/{target}/{seg}/*{det}*cal.fits'))
    cal_files = [f for f in cal_files if 'uncal' not in f]
    if not cal_files:
        print('  No calints file found'); return None
    cal_path = cal_files[0]
    wcs_cal = WCS(fits.getheader(cal_path, 'SCI'))
    print(f'  Calints: {cal_path.split("/")[-1]}')

    # Load Gaia
    gaia = Table.read(f'{ASTROM_DIR}/gaia_{target}.vot', format='votable')
    dt_yr = (Time(60786.49, format='mjd') - Time('J2016.0')).to(u.yr).value
    gaia_ra = np.array(gaia['ra'], dtype=np.float64)
    gaia_dec = np.array(gaia['dec'], dtype=np.float64)
    gaia_pmra = np.array(gaia['pmra'], dtype=np.float64)
    gaia_pmdec = np.array(gaia['pmdec'], dtype=np.float64)
    gaia_gmag = np.array(gaia['phot_g_mean_mag'], dtype=np.float64)
    has_pm = np.isfinite(gaia_pmra) & np.isfinite(gaia_pmdec)
    cos_dec = np.cos(np.radians(gaia_dec))
    ra_prop = gaia_ra.copy(); dec_prop = gaia_dec.copy()
    ra_prop[has_pm] += (gaia_pmra[has_pm] * dt_yr / 3600000.0) / cos_dec[has_pm]
    dec_prop[has_pm] += gaia_pmdec[has_pm] * dt_yr / 3600000.0

    # SW: FWHM=5 works best in these crowded fields
    # Bright stars need larger FWHM for saturation wings
    fwhm_tiers = [(18.0, 5.0), (17.0, 6.0), (16.0, 8.0), (15.0, 12.0)]

    edge = 20
    det_tables = {}
    for _, fwhm in fwhm_tiers:
        finder = DAOStarFinder(fwhm=fwhm, threshold=10 * std, brightest=500)
        sources = finder(med - md)
        xp = np.array(sources['xcentroid']); yp = np.array(sources['ycentroid'])
        r1 = np.array(sources['roundness1'])
        mask = (xp > edge) & (xp < 2048 - edge) & (yp > edge) & (yp < 2048 - edge)
        det_tables[fwhm] = (xp[mask], yp[mask], r1[mask])

    # Match to Gaia G < 19
    bright = gaia_gmag < 19
    gaia_sky = SkyCoord(ra=ra_prop[bright] * u.deg, dec=dec_prop[bright] * u.deg)
    gaia_gmag_b = gaia_gmag[bright]

    dra_list = []; ddec_list = []; gmag_list = []; jx_list = []; jy_list = []
    for gi in range(len(gaia_sky)):
        gmag = gaia_gmag_b[gi]
        fwhm_use = 5.0  # default for faint sources
        for mag_thresh, fwhm in fwhm_tiers:
            if gmag < mag_thresh:
                fwhm_use = fwhm
        xp, yp, r1 = det_tables[fwhm_use]
        r_mask = np.abs(r1) < 0.1
        if r_mask.sum() == 0: continue
        sky_det = wcs_cal.pixel_to_world(xp[r_mask], yp[r_mask])
        sc_det = SkyCoord(ra=sky_det.ra, dec=sky_det.dec)
        sep_all = gaia_sky[gi].separation(sc_det).arcsec
        best = np.argmin(sep_all)
        if sep_all[best] < 0.5:
            jwst_sky = wcs_cal.pixel_to_world(xp[r_mask][best], yp[r_mask][best])
            cos_d = np.cos(np.radians(float(jwst_sky.dec.deg)))
            dra_list.append((float(jwst_sky.ra.deg) - ra_prop[bright][gi]) * cos_d * 3600 * 1000)
            ddec_list.append((float(jwst_sky.dec.deg) - dec_prop[bright][gi]) * 3600 * 1000)
            gmag_list.append(gmag)
            jx_list.append(float(xp[r_mask][best]))
            jy_list.append(float(yp[r_mask][best]))

    dra = np.array(dra_list); ddec = np.array(ddec_list)
    gmag_a = np.array(gmag_list); jx = np.array(jx_list); jy = np.array(jy_list)
    print(f'  Raw matches (|r1|<0.05, G<19, sep<0.5\"): {len(dra)}')

    if len(dra) < 5:
        print('  Too few matches'); return None

    # IQR clip
    q1r, q3r = np.percentile(dra, [25, 75]); iqr_r = q3r - q1r
    q1d, q3d = np.percentile(ddec, [25, 75]); iqr_d = q3d - q1d
    clip = ((dra >= q1r - 3 * iqr_r) & (dra <= q3r + 3 * iqr_r) &
            (ddec >= q1d - 3 * iqr_d) & (ddec <= q3d + 3 * iqr_d))
    n_clip = (~clip).sum()

    dra_c = dra[clip]; ddec_c = ddec[clip]; gmag_c = gmag_a[clip]
    jx_c = jx[clip]; jy_c = jy[clip]
    n = len(dra_c)
    resid_c = np.sqrt(dra_c ** 2 + ddec_c ** 2)

    mean_ra = np.mean(dra_c); mean_dec = np.mean(ddec_c)
    std_ra = np.std(dra_c); std_dec = np.std(ddec_c)
    unc_ra = std_ra / np.sqrt(n); unc_dec = std_dec / np.sqrt(n)
    unc_tot = np.sqrt(unc_ra ** 2 + unc_dec ** 2)

    print(f'  After IQR clip: {n} sources (removed {n_clip})')
    print(f'  Mean: ({mean_ra:+.1f}, {mean_dec:+.1f}) mas')
    print(f'  Std: ({std_ra:.1f}, {std_dec:.1f}) mas')
    print(f'  Unc on mean: ({unc_ra:.2f}, {unc_dec:.2f}) mas, total={unc_tot:.2f} mas')
    print(f'  Median sep: {np.median(resid_c):.0f} mas, 90th: {np.percentile(resid_c, 90):.0f} mas')

    # Apply correction to WCS and save
    hdr = fits.getheader(cal_path, 'SCI').copy()
    cos_d = np.cos(np.radians(hdr['CRVAL2']))
    hdr['CRVAL1'] -= mean_ra / 1000 / 3600 / cos_d
    hdr['CRVAL2'] -= mean_dec / 1000 / 3600
    hdr['COMMENT'] = f'Gaia correction: dRA={mean_ra:.1f}mas dDec={mean_dec:.1f}mas from {n} sources'
    hdr['GAIADRA'] = float(mean_ra / 1000)
    hdr['GAIADDEC'] = float(mean_dec / 1000)
    hdr['GAIANMAT'] = n
    hdr['GAIARESI'] = float(np.median(resid_c))

    out_wcs = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_gaia.fits'
    fits.writeto(out_wcs, med.astype(np.float32), hdr, overwrite=True)

    # Diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(gmag_c, dra_c, s=8, c='#1f77b4', alpha=0.6)
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(mean_ra, color='red', ls='-', lw=1.5,
               label=f'mean = {mean_ra:+.1f} ± {unc_ra:.1f} mas')
    ax.fill_between([12, 20], mean_ra - std_ra, mean_ra + std_ra,
                    color='red', alpha=0.1, label=f'1σ = {std_ra:.1f} mas')
    ax.set_xlabel('Gaia G (mag)'); ax.set_ylabel('ΔRA [mas]')
    ax.set_title('RA offset'); ax.legend(fontsize=7); ax.set_xlim(12, 19.5)

    ax = axes[0, 1]
    ax.scatter(gmag_c, ddec_c, s=8, c='#d62728', alpha=0.6)
    ax.axhline(0, color='black', ls='--', lw=0.5)
    ax.axhline(mean_dec, color='red', ls='-', lw=1.5,
               label=f'mean = {mean_dec:+.1f} ± {unc_dec:.1f} mas')
    ax.fill_between([12, 20], mean_dec - std_dec, mean_dec + std_dec,
                    color='red', alpha=0.1, label=f'1σ = {std_dec:.1f} mas')
    ax.set_xlabel('Gaia G (mag)'); ax.set_ylabel('ΔDec [mas]')
    ax.set_title('Dec offset'); ax.legend(fontsize=7); ax.set_xlim(12, 19.5)

    ax = axes[1, 0]
    q = ax.quiver(jx_c, jy_c, dra_c, ddec_c,
                  scale=500, scale_units='width', width=0.003, color='#2ca02c', alpha=0.7)
    ax.quiverkey(q, 0.85, 0.95, 20, '20 mas', coordinates='axes', fontproperties={'size': 8})
    ax.set_xlabel('X pixel'); ax.set_ylabel('Y pixel')
    ax.set_title('Offset vectors')
    ax.set_xlim(0, 2048); ax.set_ylim(0, 2048); ax.set_aspect('equal')

    ax = axes[1, 1]
    ax.hist(resid_c, bins=25, color='#7f7f7f', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(resid_c), color='red', ls='-', lw=2,
               label=f'median = {np.median(resid_c):.0f} mas')
    ax.axvline(np.percentile(resid_c, 90), color='orange', ls='--', lw=1.5,
               label=f'90th pct = {np.percentile(resid_c, 90):.0f} mas')
    ax.set_xlabel('Total separation [mas]'); ax.set_ylabel('Count')
    ax.set_title('Separation distribution'); ax.legend(fontsize=8)

    ss = seg.replace('Segment', 'S')
    fig.suptitle(f'{target} {ss} {det} — Calints WCS, |r1|<0.05, G<19, 3×IQR\n'
                 f'{n} sources, mean: ({mean_ra:+.1f}, {mean_dec:+.1f}) mas, '
                 f'unc: {unc_tot:.2f} mas',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_fig = f'{ASTROM_DIR}/diagnostics/{target}_{seg}_{det}_calint_morphcut_astrom.png'
    fig.savefig(out_fig, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return {
        'det': det, 'n': n, 'mean_ra': mean_ra, 'mean_dec': mean_dec,
        'std_ra': std_ra, 'std_dec': std_dec,
        'unc_ra': unc_ra, 'unc_dec': unc_dec, 'unc_tot': unc_tot,
        'median_sep': np.median(resid_c),
    }


if __name__ == '__main__':
    results = []
    for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']:
        r = run_detector('Terzan5', 'Segment2', det)
        if r: results.append(r)

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    for r in results:
        print(f'  {r["det"]}: N={r["n"]:3d}, '
              f'mean=({r["mean_ra"]:+5.1f},{r["mean_dec"]:+5.1f})mas, '
              f'std=({r["std_ra"]:4.1f},{r["std_dec"]:4.1f})mas, '
              f'unc={r["unc_tot"]:.2f}mas, '
              f'med_sep={r["median_sep"]:.0f}mas')
