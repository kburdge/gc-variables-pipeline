#!/usr/bin/env python
"""Build HDF5 variable star catalog for the web viewer (v3).

Reads sources from pipeline extraction HDF5 files (autocorr-detected),
applies variability filter, and extracts forced photometry from cubes.

Sources come from:
  - extraction/{target}/{segment}/{det}_{ramp,zf}.h5

Outputs a single viewer HDF5 per target with lightcurves + source metadata.

Usage:
    python build_catalog_v3.py --target Terzan5 [--config pipeline.yaml]
    python build_catalog_v3.py --target Liller1
"""
import os
import sys
import time
import argparse
import numpy as np
import h5py
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
import warnings
warnings.filterwarnings('ignore')

DEFAULT_REFS_DIR = '/data/Globulars_Pipeline/refs'
DEFAULT_OUTPUT_DIR = '/data/Globulars_Pipeline/catalogs'
DEFAULT_EXTRACTION_DIR = '/data/Globulars_Pipeline/extraction'
DETECTORS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
LW_DETECTORS = ['nrcblong']
AP_RADIUS = 1.5
CUTOUT_HALF = 2  # 5x5 cutout

TARGETS = {
    'Liller1': {
        'segments': ['Segment3', 'Segment4'],
    },
    'Terzan5': {
        'segments': ['Segment2'],
    },
}


def build_aperture_mask():
    """Pre-compute fractional aperture mask (5x5) for r=1.5px."""
    ap = CircularAperture([(CUTOUT_HALF, CUTOUT_HALF)], r=AP_RADIUS)
    mask = ap.to_mask(method='exact')[0]
    return mask.to_image((2 * CUTOUT_HALF + 1, 2 * CUTOUT_HALF + 1)).astype(np.float32)


def load_sources_from_extraction(target, extraction_dir, refs_dir, wcs_override=None):
    """Load sources from pipeline extraction HDF5 files.

    Each extraction file has autocorr-detected positions (px, py) and
    RA/Dec computed from the ref.fits WCS at pipeline run time.

    If wcs_override is provided (dict of det -> WCS), recompute RA/Dec
    from pixel positions using the new WCS.
    """
    target_cfg = TARGETS[target]
    segments = target_cfg['segments']
    sources = []
    source_id = 0

    for seg in segments:
        for mode in ['ramp', 'zf']:
            for det in DETECTORS:
                h5_path = f'{extraction_dir}/{target}/{seg}/{det}_{mode}.h5'
                if not os.path.exists(h5_path):
                    continue

                with h5py.File(h5_path, 'r') as f:
                    src = f['sources'][:]
                    det_method = f.attrs.get('detection_method', 'unknown')

                # Accept all sources — filtering is done downstream or via config
                # If criteria field is populated, skip 'rejected'; otherwise accept all
                for s in src:
                    criteria = s['criteria'].decode().strip() if hasattr(s['criteria'], 'decode') else str(s['criteria'])
                    if criteria == 'rejected':
                        continue

                    px, py = float(s['px']), float(s['py'])

                    # Recompute RA/Dec from pixel position if new WCS provided
                    if wcs_override and det in wcs_override:
                        sky = wcs_override[det].pixel_to_world(px, py)
                        ra, dec = float(sky.ra.deg), float(sky.dec.deg)
                    else:
                        ra, dec = float(s['ra']), float(s['dec'])

                    sources.append({
                        'source_id': source_id,
                        'ra': ra, 'dec': dec,
                        'px': px, 'py': py,
                        'detector': det,
                        'source_type': mode,
                        'segment': seg,
                        'detection_type': criteria,
                        'period_min': float(s['best_period_min']),
                        'ls_sig': float(s['ls_significance']),
                        'amplitude': float(s['amplitude']),
                        'autocorr': float(s['autocorr']),
                        'det_snr': float(s['det_snr']),
                        'source_file': '',
                    })
                    source_id += 1

                print(f"  {seg}/{mode}/{det}: {len(src)} total, "
                      f"{sum(1 for s in src if s['criteria'] != b'rejected' and s['criteria'] != b'')} passed ({det_method})")

    # Also load LW
    for seg in segments:
        for mode in ['ramp', 'zf']:
            for det in LW_DETECTORS:
                h5_path = f'{extraction_dir}/{target}/{seg}/{det}_{mode}.h5'
                if not os.path.exists(h5_path):
                    continue
                with h5py.File(h5_path, 'r') as f:
                    src = f['sources'][:]
                for s in src:
                    criteria = s['criteria'].decode().strip() if hasattr(s['criteria'], 'decode') else str(s['criteria'])
                    if criteria == 'rejected':
                        continue
                    sources.append({
                        'source_id': source_id,
                        'ra': float(s['ra']), 'dec': float(s['dec']),
                        'px': float(s['px']), 'py': float(s['py']),
                        'detector': det,
                        'source_type': mode,
                        'segment': seg,
                        'detection_type': criteria,
                        'period_min': float(s['best_period_min']),
                        'ls_sig': float(s['ls_significance']),
                        'amplitude': float(s['amplitude']),
                        'autocorr': float(s['autocorr']),
                        'det_snr': float(s['det_snr']),
                        'source_file': '',
                    })
                    source_id += 1
                print(f"  {seg}/{mode}/{det}: loaded")

    return sources


def extract_forced_photometry(sources, target, segments, refs_dir, ap_mask):
    """Extract forced photometry from ramp and ZF cubes."""
    print(f"\nPhase 2: Extracting forced photometry...")
    t0_total = time.time()
    lc_data = {}
    h = CUTOUT_HALF
    all_dets = DETECTORS + LW_DETECTORS

    for det in all_dets:
        det_sources = [s for s in sources if s['detector'] == det]
        if not det_sources:
            continue
        n_src = len(det_sources)
        print(f"\n  {det}: {n_src} sources")

        for seg in segments:
            for mode in ['ramp', 'zf']:
                if mode == 'ramp':
                    cube_path = f'{refs_dir}/groupdiffs_{target}_{seg}_{det}.fits'
                else:
                    cube_path = f'{refs_dir}/zeroframes_{target}_{seg}_{det}.fits'

                if not os.path.exists(cube_path):
                    print(f"    {seg}/{mode}: MISSING")
                    continue

                t0 = time.time()
                with fits.open(cube_path, memmap=True) as hdul:
                    cube = hdul[0].data
                    nf, ny, nx = cube.shape
                    time_mjd = np.array(hdul['DIFF_TIMES'].data['MID_BARY_MJD'],
                                        dtype=np.float64)

                    flux = np.full((n_src, nf), np.nan, dtype=np.float32)
                    pxs = np.array([s['px'] for s in det_sources])
                    pys = np.array([s['py'] for s in det_sources])
                    sort_order = np.argsort(np.round(pys).astype(int))

                    for si in sort_order:
                        if np.isnan(pxs[si]) or np.isnan(pys[si]):
                            continue
                        ix, iy = int(round(pxs[si])), int(round(pys[si]))
                        if (ix - h < 0 or ix + h + 1 > nx or
                                iy - h < 0 or iy + h + 1 > ny):
                            continue
                        cutout = np.array(cube[:, iy - h:iy + h + 1,
                                               ix - h:ix + h + 1],
                                          dtype=np.float32)
                        np.nan_to_num(cutout, nan=0.0, copy=False)
                        flux[si] = np.sum(cutout * ap_mask[np.newaxis, :, :],
                                          axis=(1, 2))

                lc_data[(seg, mode, det)] = (flux, time_mjd, det_sources)
                print(f"    {seg}/{mode}: {nf} frames, {time.time()-t0:.1f}s")

    print(f"\n  Total: {time.time()-t0_total:.1f}s")
    return lc_data


def write_hdf5(sources, lc_data, target, segments, output_path):
    """Write viewer catalog HDF5."""
    print(f"\nPhase 3: Writing {output_path}...")
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    has_multi_seg = len(segments) > 1

    with h5py.File(output_path, 'w') as f:
        n = len(sources)

        fields = [
            ('source_id', 'i4'),
            ('ra', 'f8'), ('dec', 'f8'),
            ('px', 'f4'), ('py', 'f4'),
            ('detector', 'S8'),
            ('source_type', 'S4'),
            ('detection_type', 'S20'),
            ('period_min', 'f4'),
            ('ls_sig', 'f4'),
            ('max_amplitude', 'f4'),
            ('center_autocorr', 'f4'),
            ('det_snr', 'f4'),
        ]

        dt = np.dtype(fields)
        src_arr = np.zeros(n, dtype=dt)
        for i, s in enumerate(sources):
            src_arr[i]['source_id'] = s['source_id']
            src_arr[i]['ra'] = s['ra']
            src_arr[i]['dec'] = s['dec']
            src_arr[i]['px'] = s['px']
            src_arr[i]['py'] = s['py']
            src_arr[i]['detector'] = s['detector']
            src_arr[i]['source_type'] = s['source_type']
            src_arr[i]['detection_type'] = s['detection_type']
            src_arr[i]['period_min'] = s['period_min']
            src_arr[i]['ls_sig'] = s['ls_sig']
            src_arr[i]['max_amplitude'] = s['amplitude']
            src_arr[i]['center_autocorr'] = s['autocorr']
            src_arr[i]['det_snr'] = s['det_snr']

        f.create_dataset('sources', data=src_arr)

        # Detector indices
        for det in DETECTORS + LW_DETECTORS:
            ids = [s['source_id'] for s in sources if s['detector'] == det]
            if ids:
                f.create_dataset(f'detector_indices/{det}',
                                 data=np.array(ids, dtype=np.int32))

        # Lightcurves and times
        for (seg, mode, det), (flux, time_mjd, _) in lc_data.items():
            f.create_dataset(f'times/{seg}/{mode}/{det}_mjd', data=time_mjd)
            nf = flux.shape[1]
            f.create_dataset(f'lightcurves/{seg}/{mode}/{det}',
                             data=flux, chunks=(1, nf),
                             compression='gzip', compression_opts=4)

        # Attributes
        f.attrs['target'] = target
        f.attrs['segments'] = segments
        f.attrs['detectors'] = DETECTORS + LW_DETECTORS
        f.attrs['ap_radius'] = AP_RADIUS
        f.attrs['n_sources'] = n
        f.attrs['source_origin'] = 'extraction_h5'
        n_ramp = sum(1 for s in sources if s['source_type'] == 'ramp')
        f.attrs['n_ramp'] = n_ramp
        f.attrs['n_zf'] = n - n_ramp
        f.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['pipeline_version'] = '3.1'

    fsize = os.path.getsize(output_path) / 1e6
    print(f"  Written {output_path} ({fsize:.1f} MB) in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Build v3 viewer catalog')
    parser.add_argument('--target', required=True, choices=list(TARGETS.keys()))
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--extraction-dir', type=str, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument('--use-lw-wcs', action='store_true',
                        help='Recompute SW RA/Dec using LW-aligned WCS')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    target = args.target
    target_cfg = TARGETS[target]
    segments = target_cfg['segments']

    refs_dir = DEFAULT_REFS_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        refs_dir = cfg.get('paths', {}).get('refs_dir', refs_dir)
        output_dir = cfg.get('paths', {}).get('catalogs_dir', output_dir)

    output_path = args.output or f'{output_dir}/{target.lower()}_variable_star_catalog.h5'

    print("=" * 60)
    print(f"Building {target} Viewer Catalog (v3.1)")
    print(f"  Segments: {segments}")
    print(f"  Extraction: {args.extraction_dir}")
    print(f"  Refs: {refs_dir}")
    print(f"  Output: {output_path}")
    print(f"  LW WCS override: {args.use_lw_wcs}")
    print("=" * 60)

    # Optionally load LW-aligned WCS for SW RA/Dec recomputation
    wcs_override = None
    if args.use_lw_wcs:
        wcs_override = {}
        astrom_dir = '/data/Globulars_Pipeline/astrometry'
        for seg in segments:
            for det in DETECTORS:
                wcs_path = f'{astrom_dir}/{target}_{seg}_{det}_wcs_lw.fits'
                if os.path.exists(wcs_path):
                    wcs_override[det] = WCS(fits.getheader(wcs_path))
                    print(f"  Loaded LW-aligned WCS for {det}")
        if not wcs_override:
            print("  WARNING: No LW-aligned WCS files found, using extraction RA/Dec")
            wcs_override = None

    print(f"\nPhase 1: Loading sources from extraction HDF5...")
    sources = load_sources_from_extraction(
        target, args.extraction_dir, refs_dir, wcs_override=wcs_override)
    n_ramp = sum(1 for s in sources if s['source_type'] == 'ramp')
    n_zf = len(sources) - n_ramp
    print(f"\n  Total: {len(sources)} sources ({n_ramp} ramp, {n_zf} ZF)")

    ap_mask = build_aperture_mask()

    if not args.dry_run:
        lc_data = extract_forced_photometry(sources, target, segments,
                                            refs_dir, ap_mask)
        write_hdf5(sources, lc_data, target, segments, output_path)

    print(f"\nDONE: {len(sources)} sources ({n_ramp} ramp, {n_zf} ZF)")


if __name__ == '__main__':
    main()
