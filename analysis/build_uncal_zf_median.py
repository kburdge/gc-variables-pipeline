#!/usr/bin/env python
"""
Build ZF median images from uncal ZEROFRAME extensions.
These have no pixel masking — saturated pixels show ~65535 DN instead of 0.
Better for astrometric centroiding of bright stars.

Usage:
    python build_uncal_zf_median.py
"""
import numpy as np
import glob
import os
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
ASTROM_DIR = f'{BASE}/astrometry'
os.makedirs(ASTROM_DIR, exist_ok=True)

TARGETS = {
    'Liller1': ['Segment3', 'Segment4'],
    'Terzan5': ['Segment2'],
}
DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']


def main():
    for target, segments in TARGETS.items():
        for seg in segments:
            for det in DETS:
                uncals = sorted(glob.glob(f'/data/JWST/{target}/{seg}/*{det}*uncal.fits'))
                if not uncals:
                    print(f'{target}/{seg}/{det}: no uncal files')
                    continue

                out_path = f'{ASTROM_DIR}/{target}_{seg}_{det}_uncal_zf_median.fits'
                if os.path.exists(out_path):
                    print(f'{target}/{seg}/{det}: already exists')
                    continue

                print(f'{target}/{seg}/{det}: {len(uncals)} files...', end='', flush=True)

                # Collect all zeroframes
                all_zf = []
                for uf in uncals:
                    with fits.open(uf) as hdul:
                        if 'ZEROFRAME' in [h.name for h in hdul]:
                            zf = hdul['ZEROFRAME'].data.astype(np.float32)
                            # Stack all integrations
                            for i in range(zf.shape[0]):
                                all_zf.append(zf[i])

                if not all_zf:
                    print(' no ZEROFRAME data')
                    continue

                all_zf = np.array(all_zf)
                print(f' {all_zf.shape[0]} frames...', end='', flush=True)

                # Median
                median_img = np.median(all_zf, axis=0)

                # Get WCS from ref file
                ref_path = f'{BASE}/refs/{target}_{seg}_{det}_ref.fits'
                hdr = WCS(fits.getheader(ref_path)).to_header()

                fits.writeto(out_path, median_img, hdr, overwrite=True)

                # Stats
                n_sat = (median_img > 60000).sum()
                print(f' done (max={median_img.max():.0f}, {n_sat} pixels > 60k)')

    print('\nDone.')


if __name__ == '__main__':
    main()
