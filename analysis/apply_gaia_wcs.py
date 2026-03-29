#!/usr/bin/env python
"""
Apply Gaia-aligned WCS to reference images and update master catalog positions.

1. Copy Gaia-aligned WCS from astrometry/ into the ref files (updating in-place)
2. Recompute master catalog RA/Dec using the corrected WCS
3. Document everything

Usage:
    python apply_gaia_wcs.py [--dry-run]
"""
import numpy as np
import os
import sys
import re
import json
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'

with open(f'{BASE}/catalogs/master_source_mapping.json') as f:
    MAPPING = json.load(f)


def main():
    dry_run = '--dry-run' in sys.argv

    # Step 1: Update WCS in reference images
    print('=== Step 1: Updating WCS in reference images ===')
    updated_refs = []

    for wcs_file in sorted(os.listdir(ASTROM_DIR)):
        if not wcs_file.endswith('_wcs_gaia.fits'):
            continue
        # Parse: {target}_{seg}_{det}_wcs_gaia.fits
        m = re.match(r'(\w+)_(Segment\d+)_(nrcb\w+)_wcs_gaia\.fits$', wcs_file)
        if not m:
            continue
        target, seg, det = m.group(1), m.group(2), m.group(3)

        # Read the Gaia-aligned WCS
        gaia_hdr = fits.getheader(os.path.join(ASTROM_DIR, wcs_file))
        gaia_wcs = WCS(gaia_hdr)

        # Update the main reference image
        ref_path = f'{BASE}/refs/{target}_{seg}_{det}_ref.fits'
        if os.path.exists(ref_path):
            if not dry_run:
                with fits.open(ref_path, mode='update') as hdul:
                    # Update WCS keywords
                    for key in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                                'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                                'CDELT1', 'CDELT2', 'CTYPE1', 'CTYPE2']:
                        if key in gaia_hdr:
                            hdul[0].header[key] = gaia_hdr[key]
                    # Add provenance
                    hdul[0].header['COMMENT'] = f'WCS updated from Gaia DR3 alignment ({wcs_file})'
                    hdul[0].header['GAIADRA'] = gaia_hdr.get('GAIADRA', 0.0)
                    hdul[0].header['GAIADDEC'] = gaia_hdr.get('GAIADDEC', 0.0)
                    hdul[0].header['GAIANMAT'] = gaia_hdr.get('GAIANMAT', 0)
                    hdul[0].header['GAIARESI'] = gaia_hdr.get('GAIARESI', 0.0)
                    hdul.flush()
            updated_refs.append(f'{target}/{seg}/{det}')
            offset = np.sqrt(gaia_hdr.get('GAIADRA', 0)**2 + gaia_hdr.get('GAIADDEC', 0)**2)
            print(f'  {target}/{seg}/{det}_ref.fits: offset={offset:.3f}" '
                  f'({gaia_hdr.get("GAIANMAT", 0)} matches, '
                  f'{gaia_hdr.get("GAIARESI", 0):.1f} mas residual)')

    print(f'\n  Updated {len(updated_refs)} reference images')

    # Step 2: Recompute master catalog positions
    print('\n=== Step 2: Updating master catalog positions ===')

    h5 = h5py.File(CATALOG, 'r+' if not dry_run else 'r')

    n_updated = 0
    for target in ['Liller1', 'Terzan5']:
        if target not in h5:
            continue
        srcs = h5[f'{target}/sources'][:]
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        old_ra = srcs['ra'].copy()
        old_dec = srcs['dec'].copy()

        for i, src in enumerate(srcs):
            mid = int(src['master_id'])

            # Find the detection detector from mapping
            sw_det = None
            best_seg = None
            if mid < len(MAPPING):
                for dk, dv in MAPPING[mid].get('detections', {}).items():
                    if 'ramp' in dk and '_LW' not in dk:
                        dm = re.search(r'(nrcb[1-4])', dv.get('filename', ''))
                        if dm:
                            sw_det = dm.group(1)
                            # Determine segment from key
                            if 'Segment3' in dk:
                                best_seg = 'Segment3'
                            elif 'Segment4' in dk:
                                best_seg = 'Segment4'
                            elif 'Segment2' in dk:
                                best_seg = 'Segment2'
                            break

            # For single-segment targets, default to their segment
            if sw_det is not None and best_seg is None:
                if target == 'Terzan5':
                    best_seg = 'Segment2'
                elif target == 'Liller1':
                    best_seg = segments[0]  # default to first segment

            if sw_det is None or best_seg is None:
                continue

            # Get OLD WCS (original ref file) and NEW WCS (Gaia-aligned)
            wcs_gaia_file = os.path.join(ASTROM_DIR,
                                          f'{target}_{best_seg}_{sw_det}_wcs_gaia.fits')
            if not os.path.exists(wcs_gaia_file):
                continue

            ref_path = f'{BASE}/refs/{target}_{best_seg}_{sw_det}_ref.fits'
            if not os.path.exists(ref_path):
                continue

            wcs_old = WCS(fits.getheader(ref_path))
            wcs_new = WCS(fits.getheader(wcs_gaia_file))

            # Old RA/Dec → pixel (using original WCS) → new RA/Dec (using Gaia WCS)
            old_sky = SkyCoord(ra=float(src['ra'])*u.deg,
                               dec=float(src['dec'])*u.deg)
            px, py = wcs_old.world_to_pixel(old_sky)
            new_sky = wcs_new.pixel_to_world(float(px), float(py))

            srcs[i]['ra'] = new_sky.ra.deg
            srcs[i]['dec'] = new_sky.dec.deg
            n_updated += 1

        # Compute statistics
        dra = (srcs['ra'] - old_ra) * np.cos(np.radians(np.mean(srcs['dec']))) * 3600
        ddec = (srcs['dec'] - old_dec) * 3600
        total = np.sqrt(dra**2 + ddec**2)
        changed = total > 0.0001  # > 0.1 mas

        print(f'  {target}: {changed.sum()}/{len(srcs)} positions updated')
        if changed.sum() > 0:
            print(f'    Median shift: {np.median(total[changed])*1000:.1f} mas')
            print(f'    Max shift: {np.max(total[changed])*1000:.1f} mas')

        # Write back
        if not dry_run:
            h5[f'{target}/sources'][:] = srcs

    h5.close()

    # Step 3: Update source mapping JSON with corrected positions
    print(f'\n=== Step 3: Updating source mapping ===')
    if not dry_run:
        h5 = h5py.File(CATALOG, 'r')
        for target in ['Liller1', 'Terzan5']:
            if target not in h5: continue
            srcs = h5[f'{target}/sources'][:]
            for src in srcs:
                mid = int(src['master_id'])
                if mid < len(MAPPING):
                    MAPPING[mid]['ra'] = float(src['ra'])
                    MAPPING[mid]['dec'] = float(src['dec'])
        h5.close()

        with open(f'{BASE}/catalogs/master_source_mapping.json', 'w') as f:
            json.dump(MAPPING, f, indent=2)
        print('  Updated master_source_mapping.json')

    print(f'\n=== Done ===')
    action = 'Would update' if dry_run else 'Updated'
    print(f'{action} {len(updated_refs)} ref images and {n_updated} source positions')


if __name__ == '__main__':
    main()
