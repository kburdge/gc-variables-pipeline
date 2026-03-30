#!/usr/bin/env python
"""
Cross-match master variable catalog to Chandra MAVERIC X-ray catalog.

Uses Gaia-corrected positions from master_variable_catalog.h5.
Match radius: 0.5"

Output:
  - Prints match summary
  - Saves match table to catalogs/chandra_crossmatch.fits
  - Adds chandra_match group to master_variable_catalog.h5

Usage:
    python cross_match_chandra.py
"""
import numpy as np
import os
import h5py
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
CHANDRA = f'{BASE}/Chandra/Terzan5_Chandra.fits'
MATCH_RADIUS = 0.5  # arcsec


def main():
    # Load Chandra catalog
    chandra = Table.read(CHANDRA, hdu=1)
    chandra_ra = np.array(chandra['RAJ2000'], dtype=np.float64)
    chandra_dec = np.array(chandra['DEJ2000'], dtype=np.float64)
    chandra_sky = SkyCoord(ra=chandra_ra*u.deg, dec=chandra_dec*u.deg)
    print(f'Chandra MAVERIC: {len(chandra)} sources')

    # Load master catalog
    h5 = h5py.File(CATALOG, 'r')

    # Only Terzan 5 for now (that's where the MAVERIC data is)
    target = 'Terzan5'
    srcs = h5[f'{target}/sources'][:]
    cat_ra = np.array(srcs['ra'], dtype=np.float64)
    cat_dec = np.array(srcs['dec'], dtype=np.float64)
    cat_sky = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg)
    print(f'Master catalog ({target}): {len(srcs)} sources')

    # Cross-match
    idx_chandra, sep, _ = cat_sky.match_to_catalog_sky(chandra_sky)
    matched = sep.arcsec < MATCH_RADIUS

    n_matched = matched.sum()
    print(f'\nMatched: {n_matched}/{len(srcs)} within {MATCH_RADIUS}"')
    print(f'Unmatched IR variables: {(~matched).sum()}')

    # Also check reverse: how many Chandra sources have IR counterparts?
    idx_ir, sep_rev, _ = chandra_sky.match_to_catalog_sky(cat_sky)
    chandra_matched = sep_rev.arcsec < MATCH_RADIUS
    print(f'Chandra sources with IR variable counterpart: {chandra_matched.sum()}/{len(chandra)}')
    print(f'Chandra sources without IR variable: {(~chandra_matched).sum()}')

    # Get period info
    ps = h5[f'period_search/{target}'][:]

    # Build match table
    print(f'\n{"="*100}')
    print(f'{"obj":>7s} {"SNR":>5s} {"Sep":>6s} {"Chandra_name":>22s} {"err":>5s} '
          f'{"Cts":>7s} {"Lx":>9s} {"Model":>8s} {"LS_P":>8s} {"LS_sig":>7s}')
    print(f'{"="*100}')

    matches = []
    for i in np.where(matched)[0]:
        mid = int(srcs[i]['master_id'])
        snr = float(srcs[i]['snr']) if 'snr' in srcs.dtype.names else float(srcs[i]['best_snr'])
        ci = idx_chandra[i]
        cname = chandra[ci]['CXOU']
        if isinstance(cname, bytes): cname = cname.decode()
        cname = cname.strip()
        cerr = float(chandra[ci]['errc'])
        cts = float(chandra[ci]['NCt0_5-8'])
        try:
            lx = float(chandra[ci]['plLum'])
        except:
            lx = 0.0
        model = chandra[ci]['Best']
        if isinstance(model, bytes): model = model.decode()
        model = model.strip()

        # Get best period
        ps_rows = ps[ps['master_id'] == mid]
        if len(ps_rows) > 0:
            best_ps = ps_rows[np.argmax(ps_rows['ls_significance'])]
            ls_p = float(best_ps['ls_period_min'])
            ls_sig = float(best_ps['ls_significance'])
        else:
            ls_p = 0.0; ls_sig = 0.0

        sep_val = sep[i].arcsec

        print(f'obj{mid:04d} {snr:5.1f} {sep_val:5.3f}" {cname:>22s} {cerr:5.3f}" '
              f'{cts:7.1f} {lx:9.2e} {model:>8s} {ls_p:7.1f}m {ls_sig:6.0f}')

        matches.append({
            'master_id': mid, 'snr': snr, 'separation': sep_val,
            'chandra_name': cname, 'chandra_err': cerr,
            'chandra_ra': chandra_ra[ci], 'chandra_dec': chandra_dec[ci],
            'counts_05_8': cts, 'lx_pl': lx, 'best_model': model,
            'ls_period_min': ls_p, 'ls_significance': ls_sig,
        })

    h5.close()

    # Save match table
    if matches:
        out_tbl = Table(matches)
        out_path = f'{BASE}/catalogs/chandra_crossmatch_terzan5.fits'
        out_tbl.write(out_path, overwrite=True)
        print(f'\nSaved: {out_path}')

    # Summary stats
    if matches:
        seps = [m['separation'] for m in matches]
        print(f'\nMatch statistics:')
        print(f'  Median separation: {np.median(seps)*1000:.0f} mas')
        print(f'  Mean separation: {np.mean(seps)*1000:.0f} mas')
        print(f'  Max separation: {np.max(seps)*1000:.0f} mas')
        n_periodic = sum(1 for m in matches if m['ls_period_min'] > 0 and m['ls_period_min'] < 700)
        print(f'  With significant period: {n_periodic}/{len(matches)}')


if __name__ == '__main__':
    main()
