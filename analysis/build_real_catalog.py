#!/usr/bin/env python
"""
Build HDF5 catalog of user-sorted REAL variable stars.

Creates 12 tables (one per target/segment/mode/channel combination).
Each source includes:
  - Detection lightcurve from extraction HDF5
  - Forced photometry from the counterpart segment/channel cubes
  - Source metadata (RA, Dec, SNR, period, LS significance, amplitude)

Tables:
  Liller1:  Segment{3,4} x {ramp,zf} x {SW,LW}  = 8 tables
  Terzan5:  Segment2     x {ramp,zf} x {SW,LW}  = 4 tables

Usage:
    python build_real_catalog.py
"""
import os, re, time, sys
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
REFS_DIR = f'{BASE}/refs'
DIAG_DIR = f'{BASE}/diagnostics'
EXTRACT_DIR = f'{BASE}/extraction'
OUTPUT_PATH = f'{BASE}/catalogs/real_variable_catalog.h5'
AP_RADIUS = 1.5
CUTOUT_HALF = 2
SW_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW_DETS = ['nrcblong']

# Folder name mapping
FOLDER_MAP = {
    ('Liller1', 'Segment3', 'ramp', 'sw'): 'Liller1_ramp_Segment3',
    ('Liller1', 'Segment3', 'ramp', 'lw'): 'Liller1_ramp_Segment3_LW',
    ('Liller1', 'Segment3', 'zf', 'sw'):   'Liller1_zf_Segment3',
    ('Liller1', 'Segment3', 'zf', 'lw'):   'Liller1_zf_Segment3_LW',
    ('Liller1', 'Segment4', 'ramp', 'sw'): 'Liller1_ramp_Segment4',
    ('Liller1', 'Segment4', 'ramp', 'lw'): 'Liller1_ramp_Segment4_LW',
    ('Liller1', 'Segment4', 'zf', 'sw'):   'Liller1_zf_Segment4',
    ('Liller1', 'Segment4', 'zf', 'lw'):   'Liller1_zf_Segment4_LW',
    ('Terzan5', 'Segment2', 'ramp', 'sw'): 'Terzan5_ramp',
    ('Terzan5', 'Segment2', 'ramp', 'lw'): 'Terzan5_ramp_LW',
    ('Terzan5', 'Segment2', 'zf', 'sw'):   'Terzan5_zf',
    ('Terzan5', 'Segment2', 'zf', 'lw'):   'Terzan5_zf_LW',
}


def build_aperture_mask():
    ap = CircularAperture([(CUTOUT_HALF, CUTOUT_HALF)], r=AP_RADIUS)
    mask = ap.to_mask(method='exact')[0]
    return mask.to_image((2 * CUTOUT_HALF + 1, 2 * CUTOUT_HALF + 1)).astype(np.float32)


def parse_diagnostic_filename(fname):
    """Parse SNR, src_id, detector, period, LS, amplitude, RA, Dec from filename."""
    m = re.match(
        r'SNR(\d+\.\d+)_src(\d+)_(nrc\w+)_P(\d+)min_LS(\d+)_amp([\d.]+)_'
        r'([\d.]+)_([-\d.]+)\.png$', fname)
    if not m:
        return None
    return {
        'snr': float(m.group(1)),
        'src_id': int(m.group(2)),
        'det': m.group(3),
        'period_min': float(m.group(4)),
        'ls_sig': float(m.group(5)),
        'amplitude': float(m.group(6)),
        'ra': float(m.group(7)),
        'dec': float(m.group(8)),
    }


def load_real_sources(target, segment, mode, channel):
    """Load REAL sources from diagnostic folder."""
    folder_key = (target, segment, mode, channel)
    folder_name = FOLDER_MAP.get(folder_key)
    if not folder_name:
        return []

    real_dir = os.path.join(DIAG_DIR, folder_name, 'REAL')
    if not os.path.isdir(real_dir):
        return []

    sources = []
    for fname in os.listdir(real_dir):
        if not fname.endswith('.png'):
            continue
        info = parse_diagnostic_filename(fname)
        if info:
            info['filename'] = fname
            sources.append(info)

    sources.sort(key=lambda x: -x['snr'])
    return sources


def forced_photometry_cube(cube, px, py, ap_mask, h=CUTOUT_HALF):
    """Extract aperture photometry lightcurve from cube at (px, py)."""
    nf, ny, nx = cube.shape
    ix, iy = int(round(px)), int(round(py))
    if ix - h < 0 or ix + h + 1 > nx or iy - h < 0 or iy + h + 1 > ny:
        return np.full(nf, np.nan, dtype=np.float32)
    cutout = np.array(cube[:, iy-h:iy+h+1, ix-h:ix+h+1], dtype=np.float32)
    np.nan_to_num(cutout, nan=0.0, copy=False)
    return np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))


def get_pixel_position(ra, dec, wcs):
    """Convert RA/Dec to pixel using WCS."""
    sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
    px, py = wcs.world_to_pixel(sky)
    return float(px), float(py)


def process_table(target, segment, mode, channel, ap_mask):
    """
    Build one catalog table. For each REAL source:
      - Store detection lightcurve
      - Store forced photometry from counterpart cubes
    """
    sources = load_real_sources(target, segment, mode, channel)
    if not sources:
        return None

    table_key = f'{target}/{segment}/{mode}_{channel}'
    print(f'\n{"="*60}')
    print(f'  {table_key}: {len(sources)} REAL sources')
    print(f'{"="*60}')

    dets = SW_DETS if channel == 'sw' else LW_DETS
    other_channel = 'lw' if channel == 'sw' else 'sw'
    other_dets = LW_DETS if channel == 'sw' else SW_DETS

    if target == 'Liller1':
        all_segments = ['Segment3', 'Segment4']
    else:
        all_segments = ['Segment2']

    # Load WCS for all detectors
    wcs_map = {}
    for det in dets + other_dets:
        ref_path = f'{REFS_DIR}/{target}_{segment}_{det}_ref.fits'
        if os.path.exists(ref_path):
            wcs_map[(segment, det)] = WCS(fits.getheader(ref_path))
        # Also load WCS for other segments
        for other_seg in all_segments:
            if other_seg == segment:
                continue
            ref_path2 = f'{REFS_DIR}/{target}_{other_seg}_{det}_ref.fits'
            if os.path.exists(ref_path2):
                wcs_map[(other_seg, det)] = WCS(fits.getheader(ref_path2))

    # Match sources to extraction HDF5 to get detection lightcurve
    h5_cache = {}
    for det in dets:
        h5_path = f'{EXTRACT_DIR}/{target}/{segment}/{det}_{mode}.h5'
        if os.path.exists(h5_path):
            h5_cache[(segment, mode, det)] = h5py.File(h5_path, 'r')

    # Build source metadata array
    n = len(sources)
    dt = np.dtype([
        ('source_id', 'i4'),
        ('ra', 'f8'), ('dec', 'f8'),
        ('detector', 'S10'),
        ('snr', 'f4'),
        ('period_min', 'f4'),
        ('ls_sig', 'f4'),
        ('amplitude', 'f4'),
        ('filename', 'S200'),
    ])
    src_arr = np.zeros(n, dtype=dt)
    for i, s in enumerate(sources):
        src_arr[i]['source_id'] = i
        src_arr[i]['ra'] = s['ra']
        src_arr[i]['dec'] = s['dec']
        src_arr[i]['detector'] = s['det']
        src_arr[i]['snr'] = s['snr']
        src_arr[i]['period_min'] = s['period_min']
        src_arr[i]['ls_sig'] = s['ls_sig']
        src_arr[i]['amplitude'] = s['amplitude']
        src_arr[i]['filename'] = s['filename']

    # Now extract lightcurves:
    # For each source, we need forced photometry from:
    #   1. Same segment, same channel (detection LC from HDF5)
    #   2. Other segment, same channel (forced from cube)
    #   3. Same segment, other channel (forced from cube)
    #   4. Other segment, other channel (forced from cube)

    lc_datasets = {}

    # Define all cube combinations we need
    cube_combos = []
    for seg in all_segments:
        for ch, ch_dets, cube_mode in [(channel, dets, mode), (other_channel, other_dets, mode)]:
            for det in ch_dets:
                if cube_mode == 'ramp':
                    cube_path = f'{REFS_DIR}/groupdiffs_{target}_{seg}_{det}.fits'
                else:
                    cube_path = f'{REFS_DIR}/zeroframes_{target}_{seg}_{det}.fits'
                cube_combos.append((seg, cube_mode, ch, det, cube_path))

    for seg, cmode, ch, det, cube_path in cube_combos:
        if not os.path.exists(cube_path):
            continue

        lc_key = f'{seg}/{cmode}_{ch}/{det}'
        print(f'  Extracting {lc_key}...', end=' ', flush=True)
        t0 = time.time()

        cube = fits.getdata(cube_path, memmap=True)
        nf = cube.shape[0]

        # Get times
        h5_time_path = f'{EXTRACT_DIR}/{target}/{seg}/{det}_{cmode}.h5'
        if os.path.exists(h5_time_path):
            with h5py.File(h5_time_path, 'r') as tf:
                times = tf['times'][:]
        else:
            times = np.arange(nf, dtype=np.float64)

        # Get WCS for this det/seg
        wcs = wcs_map.get((seg, det))

        flux = np.full((n, nf), np.nan, dtype=np.float32)
        for i, s in enumerate(sources):
            if wcs is not None:
                px, py = get_pixel_position(s['ra'], s['dec'], wcs)
            else:
                continue
            flux[i] = forced_photometry_cube(cube, px, py, ap_mask)

        lc_datasets[lc_key] = (flux, times)
        print(f'{nf} frames, {time.time()-t0:.1f}s')

    # Close HDF5s
    for f in h5_cache.values():
        f.close()

    return {
        'sources': src_arr,
        'lightcurves': lc_datasets,
        'table_key': table_key,
    }


def main():
    t0_total = time.time()
    ap_mask = build_aperture_mask()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Define all 12 tables
    tables_config = []
    for target in ['Liller1', 'Terzan5']:
        segs = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']
        for seg in segs:
            for mode in ['ramp', 'zf']:
                for channel in ['sw', 'lw']:
                    tables_config.append((target, seg, mode, channel))

    # Process all tables
    all_tables = {}
    for target, seg, mode, channel in tables_config:
        result = process_table(target, seg, mode, channel, ap_mask)
        if result:
            all_tables[result['table_key']] = result

    # Write HDF5
    print(f'\n{"="*60}')
    print(f'  Writing {OUTPUT_PATH}')
    print(f'{"="*60}')

    with h5py.File(OUTPUT_PATH, 'w') as f:
        f.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['pipeline_version'] = '3.0'
        f.attrs['description'] = 'User-sorted REAL variable star catalog with cross-channel forced photometry'

        for table_key, result in all_tables.items():
            grp = f.create_group(table_key)
            grp.create_dataset('sources', data=result['sources'])
            grp.attrs['n_sources'] = len(result['sources'])

            for lc_key, (flux, times) in result['lightcurves'].items():
                nf = flux.shape[1]
                grp.create_dataset(f'lightcurves/{lc_key}',
                                   data=flux, chunks=(1, nf),
                                   compression='gzip', compression_opts=4)
                grp.create_dataset(f'times/{lc_key}', data=times)

    fsize = os.path.getsize(OUTPUT_PATH) / 1e6
    elapsed = time.time() - t0_total
    print(f'\nDone: {OUTPUT_PATH} ({fsize:.1f} MB) in {elapsed/60:.1f} min')

    # Summary
    print(f'\nTables:')
    for key in sorted(all_tables.keys()):
        n = len(all_tables[key]['sources'])
        n_lc = len(all_tables[key]['lightcurves'])
        print(f'  {key}: {n} sources, {n_lc} LC datasets')


if __name__ == '__main__':
    main()
