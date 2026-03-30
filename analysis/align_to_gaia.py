#!/usr/bin/env python
"""
Astrometric alignment of JWST NIRCam ZF images to Gaia DR3.

For each detector/segment:
1. Create median ZF image (bright unsaturated stars)
2. Query Gaia DR3 for stars in the field
3. Propagate Gaia positions to JWST observation epoch (J2025.3)
4. Detect bright stars in ZF median image
5. Cross-match JWST detections to propagated Gaia positions
6. Compute and apply WCS correction (shift + rotation)

Output: corrected WCS reference files in astrometry/ directory.

Usage:
    python align_to_gaia.py [--target Liller1] [--det nrcb1]
"""
import numpy as np
import os
import sys
import time
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.table import Table
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
os.makedirs(ASTROM_DIR, exist_ok=True)

# Observation epochs (J2025.3 for all)
OBS_EPOCHS = {
    ('Liller1', 'Segment3'): Time(60787.5, format='mjd'),
    ('Liller1', 'Segment4'): Time(60788.97, format='mjd'),
    ('Terzan5', 'Segment2'): Time(60786.49, format='mjd'),
}

# FOV centers (not cluster centers!) and search radii
CLUSTER_INFO = {
    'Liller1': {'ra': 263.350, 'dec': -33.390, 'radius_arcmin': 3.0},
    'Terzan5': {'ra': 267.020, 'dec': -24.782, 'radius_arcmin': 3.0},
}

GAIA_EPOCH = Time('J2016.0')


def make_zf_median(target, seg, det):
    """Load uncal ZF median (no masked pixels — saturated stars peak at 65k)."""
    # Prefer uncal ZF median (built by build_uncal_zf_median.py)
    uncal_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
    if os.path.exists(uncal_path):
        return fits.getdata(uncal_path), WCS(fits.getheader(uncal_path))

    # Fallback to pipeline ZF median
    pipe_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_zf_median.fits'
    if os.path.exists(pipe_path):
        print(f'  WARNING: using pipeline ZF (saturated pixels may be zeroed)')
        return fits.getdata(pipe_path), WCS(fits.getheader(pipe_path))

    return None, None


def query_gaia(target, obs_epoch):
    """Query Gaia DR3 and propagate to observation epoch."""
    from astroquery.gaia import Gaia

    info = CLUSTER_INFO[target]
    cache_path = f'{ASTROM_DIR}/gaia_{target}.vot'

    if os.path.exists(cache_path):
        tbl = Table.read(cache_path, format='votable')
        print(f'  Loaded cached Gaia: {len(tbl)} sources')
    else:
        print(f'  Querying Gaia DR3 for {target}...', flush=True)
        Gaia.ROW_LIMIT = 50000  # default is 50!
        coord = SkyCoord(ra=info['ra']*u.deg, dec=info['dec']*u.deg)
        radius = info['radius_arcmin'] * u.arcmin

        job = Gaia.cone_search_async(coord, radius=radius)
        tbl = job.get_results()
        tbl.write(cache_path, format='votable', overwrite=True)
        print(f'  Got {len(tbl)} Gaia sources')

    # Propagate proper motions to observation epoch
    dt_yr = (obs_epoch - GAIA_EPOCH).to(u.yr).value
    print(f'  Propagating PM by {dt_yr:.2f} yr (Gaia J2016.0 → obs {obs_epoch.jyear:.3f})')

    has_pm = np.isfinite(tbl['pmra']) & np.isfinite(tbl['pmdec'])
    ra_prop = np.array(tbl['ra'], dtype=np.float64)
    dec_prop = np.array(tbl['dec'], dtype=np.float64)

    # Propagate: Δra = pmra * dt / cos(dec), Δdec = pmdec * dt
    cos_dec = np.cos(np.radians(dec_prop))
    ra_prop[has_pm] += (tbl['pmra'][has_pm] * dt_yr / 3600000.0) / cos_dec[has_pm]
    dec_prop[has_pm] += tbl['pmdec'][has_pm] * dt_yr / 3600000.0

    tbl['ra_prop'] = ra_prop
    tbl['dec_prop'] = dec_prop
    tbl['g_mag'] = tbl['phot_g_mean_mag']

    n_pm = has_pm.sum()
    print(f'  {n_pm}/{len(tbl)} have proper motions')

    return tbl


def detect_stars(image, wcs, n_brightest=500, det='nrcb1'):
    """Detect bright stars in ZF median image."""
    fwhm = 3.5 if det == 'nrcblong' else 2.0  # LW PSF is broader
    mean, median, std = sigma_clipped_stats(image[np.isfinite(image)], sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=10 * std, brightest=n_brightest)
    sources = finder(image - median)
    if sources is None:
        return None

    # Convert pixel to sky coordinates
    sky = wcs.pixel_to_world(sources['xcentroid'], sources['ycentroid'])
    sources['ra'] = sky.ra.deg
    sources['dec'] = sky.dec.deg

    print(f'  Detected {len(sources)} stars')
    return sources


def match_and_solve(jwst_sources, gaia_sources, wcs_orig, max_sep_arcsec=2.0, refine_sep=0.5):
    """Cross-match JWST detections to Gaia and compute WCS correction.

    Uses iterative matching: first pass with wide tolerance to find bulk offset,
    then refine with tighter tolerance. Returns offset, residual stats, and
    individual match info for documentation.
    """
    jwst_sky = SkyCoord(ra=jwst_sources['ra']*u.deg, dec=jwst_sources['dec']*u.deg)
    gaia_sky = SkyCoord(ra=gaia_sources['ra_prop']*u.deg,
                        dec=gaia_sources['dec_prop']*u.deg)

    # First pass: wide match
    idx, sep, _ = jwst_sky.match_to_catalog_sky(gaia_sky)
    good = sep.arcsec < max_sep_arcsec
    n_match = good.sum()
    print(f'  Pass 1: {n_match}/{len(jwst_sky)} matched (< {max_sep_arcsec}")')

    if n_match < 3:
        print(f'  Too few matches for alignment')
        return None

    # Compute median offset
    dra = (gaia_sky[idx[good]].ra - jwst_sky[good].ra).to(u.arcsec).value
    ddec = (gaia_sky[idx[good]].dec - jwst_sky[good].dec).to(u.arcsec).value
    cos_dec = np.cos(np.radians(np.median(jwst_sky[good].dec.deg)))
    dra *= cos_dec  # correct for cos(dec)

    dra_med = np.median(dra)
    ddec_med = np.median(ddec)

    # Second pass: apply bulk offset and re-match with tighter tolerance
    jwst_ra_corr = jwst_sources['ra'] + dra_med / 3600.0 / cos_dec
    jwst_dec_corr = jwst_sources['dec'] + ddec_med / 3600.0
    jwst_sky2 = SkyCoord(ra=jwst_ra_corr*u.deg, dec=jwst_dec_corr*u.deg)

    idx2, sep2, _ = jwst_sky2.match_to_catalog_sky(gaia_sky)
    good2 = sep2.arcsec < refine_sep  # tighter after bulk correction
    n_match2 = good2.sum()

    if n_match2 >= 3:
        # Refine offset with tight matches
        dra2 = (gaia_sky[idx2[good2]].ra - jwst_sky[good2].ra).to(u.arcsec).value * cos_dec
        ddec2 = (gaia_sky[idx2[good2]].dec - jwst_sky[good2].dec).to(u.arcsec).value
        dra_final = np.median(dra2)
        ddec_final = np.median(ddec2)

        # Residuals after correction
        resid_ra = dra2 - dra_final
        resid_dec = ddec2 - ddec_final
        resid_total = np.sqrt(resid_ra**2 + resid_dec**2)
        resid_median = np.median(resid_total) * 1000  # mas

        # Individual match info
        gaia_mags = np.array(gaia_sources['g_mag'])[idx2[good2]]
        match_seps = sep2[good2].arcsec

        print(f'  Pass 2: {n_match2} matches (< 0.5" after bulk correction)')
        print(f'  Final offset: ΔRA={dra_final:.4f}", ΔDec={ddec_final:.4f}" '
              f'(total={np.sqrt(dra_final**2 + ddec_final**2):.4f}")')
        print(f'  Fit residual: {resid_median:.1f} mas median '
              f'({np.percentile(resid_total*1000, 90):.1f} mas 90th pct)')
        print(f'  Gaia mag range of matches: {np.nanmin(gaia_mags):.1f} - {np.nanmax(gaia_mags):.1f}')
    else:
        # Fall back to first-pass offset
        dra_final = dra_med
        ddec_final = ddec_med
        n_match2 = n_match
        resid_median = np.std(dra) * 1000
        print(f'  Pass 2 failed ({n_match2} matches), using pass 1 offset')

    return {
        'dra': dra_final,
        'ddec': ddec_final,
        'n_match': n_match2,
        'residual_mas': resid_median,
        'total_offset': np.sqrt(dra_final**2 + ddec_final**2),
    }


def apply_offset(wcs_orig, result, target, seg, det, median_img=None):
    """Apply astrometric offset to WCS and save with full provenance."""
    hdr = wcs_orig.to_header()

    # Shift CRVAL
    cos_dec = np.cos(np.radians(hdr['CRVAL2']))
    hdr['CRVAL1'] += result['dra'] / 3600.0 / cos_dec
    hdr['CRVAL2'] += result['ddec'] / 3600.0

    # Add provenance
    obs_epoch = OBS_EPOCHS.get((target, seg))
    epoch_str = f'{obs_epoch.jyear:.4f}' if obs_epoch else 'unknown'
    hdr['COMMENT'] = f'WCS refined using {result["n_match"]} Gaia DR3 matches to ZF detections'
    hdr['COMMENT'] = f'Gaia positions propagated from J2016.0 to J{epoch_str}'
    hdr['COMMENT'] = f'Offset: dRA={result["dra"]:.4f}" dDec={result["ddec"]:.4f}"'
    hdr['COMMENT'] = f'Fit residual: {result["residual_mas"]:.1f} mas median'
    hdr['COMMENT'] = f'Aligned by align_to_gaia.py on {time.strftime("%Y-%m-%d %H:%M")}'
    hdr['GAIADRA'] = (result['dra'], 'Gaia alignment RA offset (arcsec)')
    hdr['GAIADDEC'] = (result['ddec'], 'Gaia alignment Dec offset (arcsec)')
    hdr['GAIANMAT'] = (result['n_match'], 'Number of Gaia matches used')
    hdr['GAIARESI'] = (result['residual_mas'], 'Median fit residual (mas)')

    wcs_new = WCS(hdr)

    # Save WCS-only ref
    out_wcs = f'{ASTROM_DIR}/{target}_{seg}_{det}_wcs_gaia.fits'
    data = median_img if median_img is not None else np.zeros((2048, 2048), dtype=np.float32)
    fits.writeto(out_wcs, data.astype(np.float32), hdr, overwrite=True)
    print(f'  Saved: {out_wcs}')
    return wcs_new


def main():
    single_target = None
    single_det = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--target='):
            single_target = arg.split('=')[1]
        elif arg == '--target' and i < len(sys.argv) - 1:
            single_target = sys.argv[i + 1]
        elif arg.startswith('--det='):
            single_det = arg.split('=')[1]
        elif arg == '--det' and i < len(sys.argv) - 1:
            single_det = sys.argv[i + 1]

    t0 = time.time()
    results = []

    for target in ['Liller1', 'Terzan5']:
        if single_target and target != single_target: continue
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        # Query Gaia once per target
        obs_epoch = OBS_EPOCHS[(target, segments[0])]
        gaia = query_gaia(target, obs_epoch)

        for seg in segments:
            obs_epoch = OBS_EPOCHS[(target, seg)]

            for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']:
                if single_det and det != single_det: continue

                print(f'\n=== {target}/{seg}/{det} ===')

                # Make ZF median
                zf_path = f'{BASE}/refs/zeroframes_{target}_{seg}_{det}.fits'
                if not os.path.exists(zf_path):
                    print(f'  No ZF cube, skipping'); continue

                median_img, wcs = make_zf_median(target, seg, det)

                # Detect stars in ZF median
                jwst_sources = detect_stars(median_img, wcs, det=det)
                if jwst_sources is None or len(jwst_sources) < 5:
                    print(f'  Too few detections, skipping'); continue

                # Filter Gaia to bright stars (G < 18) for reliable matches
                bright_mask = np.array(gaia['g_mag'], dtype=np.float64) < 18.0
                gaia_bright = gaia[bright_mask]

                # LW needs wider refine tolerance due to crowding
                refine = 1.0 if det == 'nrcblong' else 0.5

                # Match to Gaia
                result = match_and_solve(jwst_sources, gaia_bright, wcs, refine_sep=refine)
                if result is None: continue

                # Apply offset
                wcs_new = apply_offset(wcs, result, target, seg, det, median_img)

                result['target'] = target
                result['seg'] = seg
                result['det'] = det
                results.append(result)

    # Summary
    print(f'\n{"="*60}')
    print(f'ASTROMETRY SUMMARY')
    print(f'{"="*60}')
    for r in results:
        print(f'  {r["target"]}/{r["seg"]}/{r["det"]}: '
              f'ΔRA={r["dra"]:+.4f}" ΔDec={r["ddec"]:+.4f}" '
              f'(total={r["total_offset"]:.4f}", '
              f'{r["n_match"]} matches, {r["residual_mas"]:.1f} mas residual)')

    print(f'\nDone in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
