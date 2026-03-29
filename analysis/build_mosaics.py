#!/usr/bin/env python
"""Build mosaic FITS files combining all 4 SW detectors for Aladin Lite viewing.

Creates 4 mosaic files (2 segments x 2 image types) at 2x downsampled resolution
(~0.062"/px) using reproject. Output files are ~18 MB each.

Usage:
    python build_mosaics.py
"""
import os
import sys
import time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from reproject import reproject_interp
import astropy.units as u
import warnings
warnings.filterwarnings('ignore')

REFS_DIR = '/data/Globulars_Pipeline/refs'
JWST_DIR = '/data/JWST/Liller1'
OUTPUT_DIR = '/data/Globulars_Pipeline/mosaics'
TARGET = 'Liller1'
DETECTORS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
I2D_OBS = {'Segment3': '003', 'Segment4': '004'}

# Native pixel scale (~0.031"/px)
OUTPUT_RESOLUTION = 0.031 * u.arcsec


def build_mosaic(input_list, output_path):
    """Build mosaic from list of (data, WCS) tuples.

    Uses reproject to combine 4 detectors into a single image on a common
    TAN WCS grid at 2x downsampled resolution.
    """
    # Create HDU list for reproject
    hdus = []
    for data, wcs in input_list:
        header = wcs.to_header()
        header['NAXIS'] = 2
        header['NAXIS1'] = data.shape[1]
        header['NAXIS2'] = data.shape[0]
        hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
        hdus.append(hdu)

    # Find optimal output WCS at 2x downsampled resolution
    wcs_out, shape_out = find_optimal_celestial_wcs(
        hdus, resolution=OUTPUT_RESOLUTION)
    print(f'  Output shape: {shape_out[1]}x{shape_out[0]}, '
          f'pixel scale: {abs(wcs_out.wcs.cdelt[0])*3600:.4f}"/px')

    # Reproject and coadd
    t0 = time.time()
    array, footprint = reproject_and_coadd(
        hdus, wcs_out, shape_out,
        reproject_function=reproject_interp)
    dt = time.time() - t0
    print(f'  Reprojection: {dt:.1f}s')

    # Replace NaN (outside footprint) with 0
    array = np.nan_to_num(array, nan=0.0)

    # Save
    header = wcs_out.to_header()
    out_hdu = fits.PrimaryHDU(data=array.astype(np.float32), header=header)
    out_hdu.writeto(output_path, overwrite=True)

    sz = os.path.getsize(output_path) / 1e6
    print(f'  Wrote {output_path} ({sz:.1f} MB)')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seg in ['Segment3', 'Segment4']:
        print(f'\n=== {seg} ===')

        # Autocorrelation mosaic
        print('Building autocorrelation mosaic...')
        inputs = []
        for det in DETECTORS:
            ac_path = f'{REFS_DIR}/{TARGET}_{seg}_{det}_autocorr.fits'
            data = fits.getdata(ac_path)
            wcs = WCS(fits.getheader(ac_path))
            inputs.append((data, wcs))
        build_mosaic(inputs, f'{OUTPUT_DIR}/{TARGET}_{seg}_autocorr_mosaic.fits')

        # i2d mosaic
        print('Building i2d mosaic...')
        obs = I2D_OBS[seg]
        inputs = []
        for det in DETECTORS:
            path = f'{JWST_DIR}/{seg}/jw05381{obs}001_02101_00001_{det}_i2d.fits'
            with fits.open(path) as hdul:
                data = hdul['SCI'].data.astype(np.float32)
                wcs = WCS(hdul['SCI'].header)
                inputs.append((data, wcs))
        build_mosaic(inputs, f'{OUTPUT_DIR}/{TARGET}_{seg}_i2d_mosaic.fits')

    print('\nDone!')


if __name__ == '__main__':
    main()
