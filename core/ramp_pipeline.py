#!/usr/bin/env python3
"""
JWST NIRCam Ramp Analysis Pipeline (Integrated with Zeroframes)

This script processes JWST ramp files to:
1. Create group-differenced cubes with barycentric timestamps
2. Generate variance reference images for source detection (from ramps or calints)
3. Detect sources and perform aperture photometry
4. Run Lomb-Scargle and BLS periodogram analysis
5. Save lightcurve plots with power spectra and CSV files
6. Generate FITS cube movies (group diffs, diff-of-diffs, calints, etc.)
7. Process zeroframes with crossmatch catalog support for robust source detection

Updated features:
- Significance-based filename sorting (like DOLPHOT pipeline)
- LS and BLS power spectra in plots
- Parallelized source processing with fork method
- Pixel coordinates in filenames for easy lookup
- Integrated zeroframe processing with crossmatch catalog support
- Output organized by target/segment/detector
- Error bars in lightcurve plots

Usage:
    # Standard ramp processing
    python ramp_pipeline.py --target Liller1 [--segment Segment3] [--detectors nrcb1 nrcb2 ...]
    
    # Include zeroframe processing (uses crossmatch catalog by default)
    python ramp_pipeline.py --target Terzan5 --segment Segment1 --zeroframes
    
    # Specify custom crossmatch catalog
    python ramp_pipeline.py --target Terzan5 --segment Segment2 --zeroframes \\
        --zeroframe-catalog zeroframe_sources/sources_Terzan5_Segment2_nrcb1.fits
    
    # Use calints for reference image
    python ramp_pipeline.py --target Liller1 --ref-source calints
    
    # Generate FITS cube movies
    python ramp_pipeline.py --target Liller1 --make-movies
"""
import os
import glob
import argparse
import hashlib
import numpy as np
import h5py
import yaml
from astropy.io import fits
from astropy.timeseries import LombScargle, BoxLeastSquares
from astropy.wcs import WCS
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import sys

from jwst_utils import (
    DATA_ROOT, TARGETS, DETECTORS,
    get_ramp_files, get_calints_files, get_detector_from_filename,
    detect_sources, positions_to_skycoords,
    clip_outliers_iqr, compute_string_length_ratio,
    save_lightcurve_csv, make_2d_wcs_header, make_3d_header,
    create_diff_cube, save_fits_cube, save_2d_fits,
    create_reference_from_calints, create_autocorr_reference,
    create_calints_cube, extract_zeroframes,
    load_psf_kernel, iterative_psf_detect, merge_detection_lists,
    fast_psf_detect
)
from source_tracker import SourceTracker, NULL_TRACKER, load_track_file

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'data_root': DATA_ROOT,
    
    # Local output directories (will be created in current working directory)
    'ref_dir': 'refs',
    'cube_dir': 'cubes',
    'out_dir': 'lightcurves_ramp',
    'bad_dir': 'lightcurves_ramp_bad',
    # Reference source: 'ramps' or 'calints' or path to custom file
    'ref_source': 'calints',
    'ref_file': None,  # Custom reference file path (overrides ref_source)
    
    # Whether to generate movie cubes
    'make_movies': True,
    
    # Source detection
    'detection_threshold': 4.0,  # sigma threshold for DAOStarFinder (per-frame)

    # PSF-matched filter detection (iterative subtraction)
    'psf_detect': False,         # Enable PSF detection (union with DAOStarFinder)
    'psf_path': '/data/Globulars_Pipeline/PSFs/PSF_NIRCam_in_flight_opd_filter_F200W.fits',
    'psf_kernel_size': 21,       # PSF kernel extraction size (pixels)
    'psf_threshold': 3.0,        # Detection threshold (sigma)
    'psf_min_separation': 1,     # Min peak separation (px); 1 = 3x3 max filter
    'psf_n_iter': 3,             # Detect-subtract iterations

    # Autocorrelation detection: use calints lag-1 autocorrelation as detection ref
    # instead of var/mean. Achieves 100% recovery with 3× fewer detections.
    'autocorr_detect': False,
    'autocorr_sigma': 3.0,      # fast_psf_detect threshold on autocorrelation image

    # Temporal STD detection (ramp cube nanstd along time axis)
    'ramp_temporal_std': False,  # Also detect on temporal STD of group-diff cube
    'ramp_tstd_sigma': 5.0,     # fast_psf_detect threshold on temporal STD image

    # Photometry
    'ap_radius': 1.5,

    # Annulus background subtraction (off by default)
    'use_annulus': False,
    'annulus_inner': 4.0,
    'annulus_outer': 10.0,
    
    # Outlier rejection
    'clip_chunk_size': 18,
    'sl_ratio_thresh': 100,
    
    # Lomb-Scargle
    'freq_low_hr': 1/12,   # 12 hr max period
    'freq_high_hr': 60,     # 1 min min period
    'freq_n_points': 5000,
    
    # BLS settings
    'bls_min_period_hr': 0.5/60,  # 0.5 min
    'bls_max_period_hr': 6.0,     # 6 hr
    'bls_duration_min': 0.5,      # Transit duration in minutes
    'bls_duration_max': 30.0,
    
    # Binning for plots
    'bin_size': 9,
    
    # Zeroframe mode settings
    'zeroframe_mode': False,
    'zeroframe_out_dir': 'lightcurves_zeroframe',
    'zeroframe_bad_dir': 'lightcurves_zeroframe_bad',
    'zeroframe_clip_chunk_size': 1,  # Minimal clipping for zeroframes
    'zeroframe_bin_size': 1,
    'zeroframe_freq_low_hr': 1/12,   # 12 hr max period
    'zeroframe_freq_high_hr': 3,     # 20 min min period (60/20 = 3 cycles/hr)
    
    # Crossmatch catalog settings (for zeroframe source detection)
    'use_crossmatch_catalog': True,
    'crossmatch_out_dir': 'zeroframe_sources',
    'crossmatch_fwhm': 2.0,          # FWHM for source detection (pixels)
    'crossmatch_threshold': 5.0,     # Detection threshold (sigma)
    'crossmatch_radius': 0.1,        # Cross-match radius (arcsec)
    'crossmatch_min_detections': 30, # Min frames for robust detection (default 30)
    
    # Parallelization
    'n_workers': None,  # None = cpu_count - 1
}


def load_pipeline_config(yaml_path):
    """Load pipeline.yaml and merge into CONFIG dict.

    Maps the YAML structure into the flat CONFIG dict expected by the pipeline.
    Only overrides keys that are present in the YAML; everything else keeps
    its default value from CONFIG.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    c = CONFIG  # shorthand

    # Store config hash for provenance
    with open(yaml_path, 'rb') as f:
        c['_config_hash'] = hashlib.sha256(f.read()).hexdigest()[:16]
    c['_config_path'] = yaml_path

    # Detection
    det = cfg.get('detection', {})
    if det.get('method') == 'autocorr':
        c['autocorr_detect'] = True
    c['autocorr_sigma'] = det.get('ramp_sigma', c['autocorr_sigma'])
    c['zf_detect_sigma'] = det.get('zf_sigma', 3.0)
    c['psf_kernel_size'] = det.get('psf_kernel_size', c['psf_kernel_size'])
    c['psf_min_separation'] = det.get('min_separation', c['psf_min_separation'])
    c['crossmatch_fwhm'] = det.get('fwhm', c.get('crossmatch_fwhm', 2.0))
    c['crossmatch_radius'] = det.get('crossmatch_radius', c.get('crossmatch_radius', 0.1))
    c['crossmatch_min_detections'] = det.get('crossmatch_min_detections', c.get('crossmatch_min_detections', 30))

    # Paths
    paths = cfg.get('paths', {})
    if paths.get('psf_f200w'):
        c['psf_path'] = paths['psf_f200w']
    if paths.get('refs_dir'):
        c['ref_dir'] = paths['refs_dir']
    if paths.get('data_root'):
        c['data_root'] = paths['data_root']
    if paths.get('extraction_dir'):
        c['extraction_dir'] = paths['extraction_dir']
    if paths.get('catalogs_dir'):
        c['catalogs_dir'] = paths['catalogs_dir']
    if paths.get('diagnostics_dir'):
        c['diagnostics_dir'] = paths['diagnostics_dir']

    # Photometry
    phot = cfg.get('photometry', {})
    c['ap_radius'] = phot.get('aperture_radius', c['ap_radius'])
    c['use_annulus'] = phot.get('use_annulus', c['use_annulus'])
    c['annulus_inner'] = phot.get('annulus_inner', c['annulus_inner'])
    c['annulus_outer'] = phot.get('annulus_outer', c['annulus_outer'])

    # Clipping
    clip = cfg.get('clipping', {})
    ramp_clip = clip.get('ramp', {})
    c['clip_chunk_size'] = ramp_clip.get('chunk_size', c['clip_chunk_size'])
    zf_clip = clip.get('zf', {})
    c['zeroframe_clip_chunk_size'] = zf_clip.get('chunk_size', c.get('zeroframe_clip_chunk_size', 4))

    # Period search
    ps = cfg.get('period_search', {})
    ls = ps.get('lomb_scargle', {})
    c['freq_low_hr'] = ls.get('freq_min_cph', c['freq_low_hr'])
    c['freq_high_hr'] = ls.get('freq_max_cph', c['freq_high_hr'])
    c['freq_n_points'] = ls.get('n_points', c['freq_n_points'])
    bls_cfg = ps.get('bls', {})
    c['bls_min_period_hr'] = bls_cfg.get('period_min_hr', c['bls_min_period_hr'])
    c['bls_max_period_hr'] = bls_cfg.get('period_max_hr', c['bls_max_period_hr'])
    c['bls_duration_min'] = bls_cfg.get('duration_min_min', c['bls_duration_min'])
    c['bls_duration_max'] = bls_cfg.get('duration_max_min', c['bls_duration_max'])
    zf_ps = ps.get('zf', {})
    c['zeroframe_freq_low_hr'] = zf_ps.get('freq_min_cph', c.get('zeroframe_freq_low_hr', c['freq_low_hr']))
    c['zeroframe_freq_high_hr'] = zf_ps.get('freq_max_cph', c.get('zeroframe_freq_high_hr', 3))
    # Binning
    binning = cfg.get('binning', {})
    c['bin_size'] = binning.get('ramp', c['bin_size'])
    c['zeroframe_bin_size'] = binning.get('zf', c.get('zeroframe_bin_size', 1))

    # Processing
    proc = cfg.get('processing', {})
    if proc.get('n_workers') is not None:
        c['n_workers'] = proc['n_workers']
    if 'skip_plots' in proc:
        c['skip_plots'] = proc['skip_plots']
    if 'output_hdf5' in proc:
        c['output_hdf5'] = proc['output_hdf5']

    return c


def write_extraction_hdf5(results, positions, skycoords, sig_vals,
                          times_hr, times_mjd, target, segment, detector,
                          mode, config):
    """Write all extracted lightcurves and metrics to a single HDF5 file.

    This replaces the per-source CSV+PNG output. Stores flux arrays and a
    compound source statistics table in one file per segment/detector/mode.
    """
    extraction_dir = config.get('extraction_dir',
                                os.path.join(config.get('_v3_dir', '.'), 'extraction'))
    out_dir = os.path.join(extraction_dir, target, segment)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{detector}_{mode}.h5')

    n_sources = len(results)
    n_frames = len(times_hr)

    # Build source statistics table
    dt = np.dtype([
        ('source_id', 'i4'),
        ('px', 'f4'), ('py', 'f4'),
        ('ra', 'f8'), ('dec', 'f8'),
        ('det_snr', 'f4'),
        ('best_period_min', 'f4'),
        ('ls_significance', 'f4'),
        ('bls_significance', 'f4'),
        ('median_flux', 'f4'),
        ('amplitude', 'f4'),
        ('rms', 'f4'),
        ('autocorr', 'f4'),
        ('chi2_binned', 'f4'),
        ('excess_scatter', 'f4'),
        ('drift_ratio', 'f4'),
        ('drift_numerator', 'f4'),
        ('drift_denominator', 'f4'),
        ('scramble_ratio', 'f4'),
        ('rms_improvement', 'f4'),
        ('directionality', 'f4'),
        ('n_points', 'i4'),
        ('criteria', 'S32'),
    ])

    src_arr = np.zeros(n_sources, dtype=dt)
    flux_raw = np.full((n_sources, n_frames), np.nan, dtype=np.float32)
    flux_clipped = np.full((n_sources, n_frames), np.nan, dtype=np.float32)

    # Sort results by source_id
    results_by_idx = {r['idx']: r for r in results}

    for i in range(n_sources):
        r = results_by_idx.get(i, {})
        src_arr[i]['source_id'] = i
        src_arr[i]['px'] = positions[i][0] if i < len(positions) else np.nan
        src_arr[i]['py'] = positions[i][1] if i < len(positions) else np.nan
        src_arr[i]['ra'] = float(skycoords[i].ra.deg) if i < len(skycoords) else np.nan
        src_arr[i]['dec'] = float(skycoords[i].dec.deg) if i < len(skycoords) else np.nan
        src_arr[i]['det_snr'] = sig_vals[i] if i < len(sig_vals) else np.nan

        src_arr[i]['best_period_min'] = r.get('best_P_min', np.nan)
        src_arr[i]['ls_significance'] = r.get('ls_significance', np.nan)
        src_arr[i]['bls_significance'] = r.get('bls_significance', np.nan)
        src_arr[i]['n_points'] = r.get('n_points', 0)
        src_arr[i]['criteria'] = r.get('filter_criteria', '')

        # Lightcurve metrics (computed in analyze_source_worker)
        src_arr[i]['median_flux'] = r.get('median_flux', np.nan)
        src_arr[i]['amplitude'] = r.get('amplitude', np.nan)
        src_arr[i]['rms'] = r.get('rms', np.nan)
        src_arr[i]['autocorr'] = r.get('autocorr_lag1', np.nan)
        src_arr[i]['chi2_binned'] = r.get('chi2_binned', np.nan)
        src_arr[i]['excess_scatter'] = r.get('excess_scatter', np.nan)
        src_arr[i]['drift_ratio'] = r.get('drift_ratio', np.nan)
        src_arr[i]['drift_numerator'] = r.get('drift_numerator', np.nan)
        src_arr[i]['drift_denominator'] = r.get('drift_denominator', np.nan)
        src_arr[i]['scramble_ratio'] = r.get('scramble_ratio', np.nan)
        src_arr[i]['rms_improvement'] = r.get('rms_improvement', np.nan)
        src_arr[i]['directionality'] = r.get('directionality', np.nan)

        # Raw flux: the full lightcurve before clipping
        y_raw = r.get('y_raw')
        if y_raw is not None and len(y_raw) == n_frames:
            flux_raw[i, :] = y_raw

        # Clipped flux: after IQR clipping (NaN for removed points)
        y = r.get('y')
        t = r.get('t')
        if y is not None and t is not None and len(y) > 0:
            # Map clipped times back to frame indices
            t_full = np.asarray(times_hr, dtype=np.float64)
            for j, tv in enumerate(t):
                idx = np.argmin(np.abs(t_full - tv))
                flux_clipped[i, idx] = y[j]

    with h5py.File(out_path, 'w') as f:
        f.attrs['target'] = target
        f.attrs['segment'] = segment
        f.attrs['detector'] = detector
        f.attrs['mode'] = mode
        f.attrs['n_sources'] = n_sources
        f.attrs['n_frames'] = n_frames
        f.attrs['detection_method'] = config.get('autocorr_detect', False) and 'autocorr' or 'calints'
        det_sigma = config.get('autocorr_sigma', 3.0) if mode == 'ramp' else config.get('zf_detect_sigma', 5.0)
        f.attrs['detection_sigma'] = det_sigma
        f.attrs['aperture_radius'] = config.get('ap_radius', 1.5)
        f.attrs['pipeline_version'] = '3.0'
        f.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['config_hash'] = config.get('_config_hash', '')

        f.create_dataset('times', data=np.asarray(times_hr, dtype=np.float64))
        f['times'].attrs['unit'] = 'hr'
        f.create_dataset('times_mjd', data=np.asarray(times_mjd, dtype=np.float64))
        f['times_mjd'].attrs['unit'] = 'MJD'

        f.create_dataset('sources', data=src_arr)

        f.create_dataset('flux', data=flux_raw,
                         chunks=(1, n_frames), compression='gzip',
                         compression_opts=4)
        f['flux'].attrs['unit'] = 'ADU'

        f.create_dataset('flux_clipped', data=flux_clipped,
                         chunks=(1, n_frames), compression='gzip',
                         compression_opts=4)

    fsize = os.path.getsize(out_path) / 1e6
    print(f"  HDF5: {out_path} ({fsize:.1f} MB, {n_sources} sources)")
    return out_path


# ============================================================================
# Global variables for parallel processing (set before fork)
# ============================================================================
_global_cube = None
_global_times = None
_global_freqs = None
_global_config = None
_global_tracker = NULL_TRACKER


def compute_significance(power, peak_power):
    """Compute significance of peak as (peak - median) / MAD."""
    if not np.isfinite(power).any() or np.isnan(peak_power):
        return np.nan
    median_power = np.nanmedian(power)
    mad = np.nanmedian(np.abs(power - median_power))
    if mad == 0:
        return np.nan
    return (peak_power - median_power) / mad


# ============================================================================
# Step 1: Create Group-Differenced Cube with Barycentric Times
# ============================================================================
def create_groupdiff_cube(target, segment, detector, config=CONFIG):
    """
    Create group-differenced cube from ramp files.
    Uses barycentric midpoint times from GROUP table.
    """
    ramp_files = get_ramp_files(target, segment, detector, config['data_root'])
    if not ramp_files:
        print(f"  No ramp files found for {target}/{segment}/{detector}")
        return None
    
    print(f"  Found {len(ramp_files)} ramp files")
    
    diff_frames = []
    diff_times = []
    header = None
    
    for fn in ramp_files:
        with fits.open(fn) as hdul:
            sci = hdul['SCI'].data.astype(np.float32)
            groupdq = hdul['GROUPDQ'].data
            pixeldq = hdul['PIXELDQ'].data
            grp_tab = hdul['GROUP'].data
            
            if header is None:
                header = hdul['SCI'].header.copy()
            
            n_int, n_grp, ny, nx = sci.shape
            static_bad = (pixeldq != 0)
            
            for i_int in range(n_int):
                ramp = sci[i_int]
                dq = groupdq[i_int]
                
                diffs = ramp[1:] - ramp[:-1]
                mask = (static_bad[None, ...] | 
                       (dq[:-1] > 10) | (dq[1:] > 10))
                diffs[mask] = np.nan
                
                sel = ((grp_tab['integration_number'] == (i_int+1)) & 
                       (grp_tab['group_number'] >= 1) & 
                       (grp_tab['group_number'] <= n_grp))
                this_int = grp_tab[sel]
                order_g = np.argsort(this_int['group_number'])
                bary_end = this_int['bary_end_time'][order_g]
                
                for k in range(n_grp - 1):
                    t_mid = 0.5 * (bary_end[k] + bary_end[k+1])
                    diff_times.append(t_mid)
                
                diff_frames.extend(diffs)
    
    cube = np.stack(diff_frames, axis=0)
    times = np.array(diff_times, dtype=np.float64)
    
    os.makedirs(config['ref_dir'], exist_ok=True)
    
    hdr = make_3d_header(header, cube.shape[0], cube.shape[1], cube.shape[2])
    hdr['TARGET'] = target
    hdr['SEGMENT'] = segment
    hdr['DETECTOR'] = detector
    
    primary_hdu = fits.PrimaryHDU(data=cube, header=hdr)
    col = fits.Column(name='MID_BARY_MJD', format='D', unit='d', array=times)
    time_hdu = fits.BinTableHDU.from_columns([col], name='DIFF_TIMES')
    
    outname = os.path.join(config['ref_dir'], f'groupdiffs_{target}_{segment}_{detector}.fits')
    fits.HDUList([primary_hdu, time_hdu]).writeto(outname, overwrite=True)
    print(f"  Wrote {outname} ({cube.shape[0]} frames)")
    
    return outname, cube, times, header


# ============================================================================
# Step 2: Create Reference Image for Source Detection
# ============================================================================
def create_reference_image(target, segment, detector, config=CONFIG):
    """Create variance/mean reference image for source detection."""
    ref_source = config.get('ref_source', 'calints')
    
    if ref_source == 'calints':
        return create_reference_from_calints_wrapper(target, segment, detector, config)
    else:
        return create_reference_from_ramps(target, segment, detector, config)


def create_reference_from_ramps(target, segment, detector, config):
    """Create variance/mean reference image from ramp zero-frames."""
    ramp_files = get_ramp_files(target, segment, detector, config['data_root'])
    if not ramp_files:
        print(f"  No ramp files found for {target}/{segment}/{detector}")
        return None
    
    zf_list = []
    header = None
    
    for fn in ramp_files:
        with fits.open(fn) as hdul:
            if 'ZEROFRAME' not in hdul:
                print(f"  Warning: No ZEROFRAME in {fn}")
                continue
            zf = hdul['ZEROFRAME'].data.astype(np.float32)
            if header is None and 'SCI' in hdul:
                header = hdul['SCI'].header.copy()
        zf_list.extend(zf[1:])
    
    if not zf_list:
        print(f"  No zeroframes found")
        return None
    
    cube = np.stack(zf_list, axis=0)
    print(f"  Collected {cube.shape[0]} zero-frames from ramps")
    
    var_frame = np.nanvar(cube, axis=0)
    mean_frame = np.nanmean(cube, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ref_img = var_frame / mean_frame
    ref_img[~np.isfinite(ref_img)] = 0.0
    
    ny, nx = ref_img.shape
    hdr = make_2d_wcs_header(header, ny, nx, extname='VAR_OVER_MEAN')
    hdr['TARGET'] = target
    hdr['SEGMENT'] = segment
    hdr['DETECTOR'] = detector
    hdr['REFTYPE'] = 'ramps'
    
    os.makedirs(config['ref_dir'], exist_ok=True)
    outname = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_ref.fits')
    save_2d_fits(ref_img, outname, hdr, description='Variance/Mean from ramp zeroframes')
    
    return outname


def create_reference_from_calints_wrapper(target, segment, detector, config):
    """Create variance/mean reference image from calints files."""
    calints_files = get_calints_files(target, segment, detector, config['data_root'])
    if not calints_files:
        print(f"  No calints files found for {target}/{segment}/{detector}")
        print(f"  Falling back to ramps...")
        return create_reference_from_ramps(target, segment, detector, config)
    
    os.makedirs(config['ref_dir'], exist_ok=True)
    outname = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_ref.fits')
    
    ref_img, hdr = create_reference_from_calints(calints_files, outname)
    
    if ref_img is not None and hdr is not None:
        hdr['TARGET'] = target
        hdr['SEGMENT'] = segment
        hdr['DETECTOR'] = detector
    
    return outname


# ============================================================================
# Zeroframe Cube Building
# ============================================================================
def build_zeroframe_cube(target, segment, detector, config=CONFIG):
    """
    Build zero-frame cube and extract timestamps from INT_TIMES table.
    
    Returns: cube (n_frames, ny, nx), times_mjd, header
    """
    print(f"    DEBUG build_zeroframe_cube: data_root={config['data_root']}")
    ramp_files = get_ramp_files(target, segment, detector, config['data_root'])
    print(f"    DEBUG build_zeroframe_cube: Found {len(ramp_files)} ramp files")
    if not ramp_files:
        return None, None, None
    
    zf_list = []
    time_list = []
    header = None
    
    for fn in ramp_files:
        with fits.open(fn) as hdul:
            if 'ZEROFRAME' not in hdul:
                continue
            zf = hdul['ZEROFRAME'].data.astype(np.float32)
            tmid = hdul['INT_TIMES'].data['int_mid_MJD_UTC']
            
            if header is None and 'SCI' in hdul:
                header = hdul['SCI'].header.copy()
        
        # Drop first zero-frame of each integration
        for i in range(1, zf.shape[0]):
            zf_list.append(zf[i])
            time_list.append(tmid[i])
    
    if not zf_list:
        return None, None, None
    
    cube = np.stack(zf_list, axis=0)
    times_mjd = np.array(time_list, dtype=np.float64)
    
    return cube, times_mjd, header



# ============================================================================
# Step 3: Generate FITS Cube Movies
# ============================================================================
def generate_movies(target, segment, detector, config=CONFIG):
    """Generate various FITS cube movies for visualization."""
    print(f"\n  Generating FITS cube movies for {detector}...")
    
    cube_dir = os.path.join(config['cube_dir'], target, segment)
    os.makedirs(cube_dir, exist_ok=True)
    
    ramp_files = get_ramp_files(target, segment, detector, config['data_root'])
    if ramp_files:
        result = create_groupdiff_cube(target, segment, detector, config)
        if result:
            _, groupdiff_cube, times, header = result
            
            groupdiff_path = os.path.join(cube_dir, f'{detector}_groupdiffs.fits')
            save_fits_cube(groupdiff_cube, groupdiff_path, header,
                          description='Group differences', times=times)
            
            if groupdiff_cube.shape[0] > 1:
                diff_of_diffs = create_diff_cube(groupdiff_cube)
                diff_times = 0.5 * (times[:-1] + times[1:]) if len(times) > 1 else None
                diff_of_diffs_path = os.path.join(cube_dir, f'{detector}_diff_of_groupdiffs.fits')
                save_fits_cube(diff_of_diffs, diff_of_diffs_path, header,
                              description='Diff of group differences', times=diff_times)
    
    calints_files = get_calints_files(target, segment, detector, config['data_root'])
    if calints_files:
        print(f"  Processing {len(calints_files)} calints files...")
        calints_cube, calints_times, calints_hdr = create_calints_cube(
            calints_files, subtract_background=True)
        
        if calints_cube is not None:
            calints_path = os.path.join(cube_dir, f'{detector}_calints_bgsub.fits')
            save_fits_cube(calints_cube, calints_path, calints_hdr,
                          description='Background-subtracted calints', times=calints_times)
            
            if calints_cube.shape[0] > 1:
                diff_of_calints = create_diff_cube(calints_cube)
                diff_times = 0.5 * (calints_times[:-1] + calints_times[1:]) if calints_times is not None else None
                diff_calints_path = os.path.join(cube_dir, f'{detector}_diff_of_calints.fits')
                save_fits_cube(diff_of_calints, diff_calints_path, calints_hdr,
                              description='Diff of calints', times=diff_times)
    
    if ramp_files:
        print(f"  Extracting zeroframes...")
        zf_cube, zf_times, zf_hdr = extract_zeroframes(ramp_files, drop_first=True)
        
        if zf_cube is not None:
            zf_mean = np.nanmean(zf_cube, axis=0)
            zf_cube_sub = zf_cube - zf_mean[np.newaxis, ...]
            zf_cube_sub = np.nan_to_num(zf_cube_sub, nan=0.0)
            
            zf_path = os.path.join(cube_dir, f'{detector}_zeroframes_bgsub.fits')
            save_fits_cube(zf_cube_sub, zf_path, zf_hdr,
                          description='Background-subtracted zeroframes', times=zf_times)
            
            if zf_cube_sub.shape[0] > 1:
                diff_of_zf = create_diff_cube(zf_cube_sub)
                diff_times = 0.5 * (zf_times[:-1] + zf_times[1:]) if zf_times is not None else None
                diff_zf_path = os.path.join(cube_dir, f'{detector}_diff_of_zeroframes.fits')
                save_fits_cube(diff_of_zf, diff_zf_path, zf_hdr,
                              description='Diff of zeroframes', times=diff_times)
    
    print(f"  Movie cubes written to {cube_dir}/")


# ============================================================================
# Step 4: Source Analysis (Parallelized)
# ============================================================================
def analyze_source_worker(args):
    """Worker function for parallel source analysis."""
    global _global_cube, _global_times, _global_freqs, _global_config, _global_tracker

    (idx, position, ra_deg, dec_deg) = args
    config = _global_config
    cube = _global_cube
    times_hr = _global_times
    freqs = _global_freqs
    tracker = _global_tracker
    is_tracked = tracker.is_tracking(ra_deg, dec_deg)

    cx, cy = position

    result = {
        'idx': idx,
        'pix_x': float(cx),
        'pix_y': float(cy),
        'ra_deg': ra_deg,
        'dec_deg': dec_deg,
        'n_points': 0,
        'best_f_1phr': np.nan,
        'best_P_hr': np.nan,
        'best_P_min': np.nan,
        'peak_power': np.nan,
        'ls_significance': np.nan,
        'bls_significance': np.nan,
        'sl_ratio': np.nan,
        't': None,
        'y': None,
        'ls_power': None,
        'bls_power': None,
        'bls_periods': None,
        'n_10sigma': 0,
    }

    # Get photometry for this source
    nframes, ny, nx = cube.shape

    if config['use_annulus']:
        max_radius = int(np.ceil(config['annulus_outer'])) + 1
    else:
        max_radius = int(np.ceil(config['ap_radius'])) + 1

    x0 = max(0, int(cx - max_radius))
    x1 = min(nx, int(cx + max_radius + 1))
    y0 = max(0, int(cy - max_radius))
    y1 = min(ny, int(cy + max_radius + 1))

    local_x = np.arange(x0, x1) - cx
    local_y = np.arange(y0, y1) - cy
    xx, yy = np.meshgrid(local_x, local_y)
    dist = np.sqrt(xx**2 + yy**2)

    ap_mask = dist <= config['ap_radius']
    n_ap = np.sum(ap_mask)

    if n_ap == 0:
        if is_tracked:
            tracker.log(ra_deg, dec_deg, 'PHOTOMETRY',
                        f"REJECTED: n_ap=0 (no aperture pixels), idx={idx}")
        return result

    cutout = cube[:, y0:y1, x0:x1].astype(np.float64)

    # Exclude pixels that are NaN in ANY frame so the effective aperture
    # is constant across all frames.
    any_nan = np.any(np.isnan(cutout), axis=0)        # (ny_cut, nx_cut)
    n_nan_excluded = int(np.sum(any_nan & ap_mask))
    good_ap = ap_mask & ~any_nan
    n_ap_eff = int(np.sum(good_ap))

    if n_ap_eff == 0:
        if is_tracked:
            tracker.log(ra_deg, dec_deg, 'PHOTOMETRY',
                        f"REJECTED: all {n_ap} aperture pixels have NaN "
                        f"in at least one frame, idx={idx}")
        return result

    cutout_clean = np.nan_to_num(cutout, nan=0.0)
    ap_flux = np.sum(cutout_clean[:, good_ap], axis=1)

    if config['use_annulus']:
        ann_mask = (dist >= config['annulus_inner']) & (dist <= config['annulus_outer'])
        n_ann = np.sum(ann_mask)
        if n_ann > 0:
            ann_values = cutout[:, ann_mask]
            ann_median = np.nanmedian(ann_values, axis=1)
            flux_raw = ap_flux - ann_median * n_ap_eff
        else:
            flux_raw = ap_flux
    else:
        flux_raw = ap_flux

    if not np.isfinite(flux_raw).any():
        if is_tracked:
            tracker.log(ra_deg, dec_deg, 'PHOTOMETRY',
                        f"REJECTED: no finite flux values, idx={idx}")
        return result

    if is_tracked:
        tracker.log(ra_deg, dec_deg, 'PHOTOMETRY',
                    f"idx={idx}, px=({cx:.1f}, {cy:.1f}), n_ap={n_ap}, "
                    f"r={config['ap_radius']}px, "
                    f"NaN_excluded={n_nan_excluded}/{n_ap}, "
                    f"n_ap_eff={n_ap_eff}, "
                    f"flux_mean={np.nanmean(flux_raw):.1f}, "
                    f"flux_std={np.nanstd(flux_raw):.1f}, "
                    f"n_frames={nframes}")

    # Outlier clipping
    zeroframe_mode = config.get('zeroframe_mode', False)
    n_raw = len(flux_raw)

    if zeroframe_mode:
        clip_chunk = config.get('zeroframe_clip_chunk_size', 1)
    else:
        clip_chunk = config['clip_chunk_size']
    y, t = clip_outliers_iqr(flux_raw, times_hr, clip_chunk)

    if is_tracked:
        n_after = len(y)
        tracker.log(ra_deg, dec_deg, 'CLIP',
                    f"{n_after}/{n_raw} points kept ({n_raw - n_after} removed)")

    n_points = len(y)
    result['n_points'] = n_points
    result['t'] = t
    result['y'] = y
    result['y_raw'] = flux_raw  # full raw flux for HDF5 output

    # Compute metrics early so they're available even for rejected sources
    med = float(np.nanmedian(y))
    result['median_flux'] = med
    if med != 0:
        p10, p90 = np.nanpercentile(y, [10, 90])
        result['amplitude'] = float((p90 - p10) / abs(med))
    result['rms'] = float(np.nanstd(y))
    # Lag-1 autocorrelation
    if len(y) > 2:
        ym = y - np.mean(y)
        var = np.sum(ym ** 2)
        if var > 0:
            result['autocorr_lag1'] = float(np.sum(ym[:-1] * ym[1:]) / var)

    if zeroframe_mode:
        min_points = 30
    else:
        min_points = 500

    # Lomb-Scargle (skip if too few points for meaningful periodogram)
    if n_points >= min_points:
        try:
            ls = LombScargle(t, y, fit_mean=True, center_data=True)
            ls_power = ls.power(freqs)
            result['ls_power'] = ls_power

            if np.isfinite(ls_power).any():
                bi = int(np.nanargmax(ls_power))
                best_f = float(freqs[bi])
                best_P = 1.0 / best_f
                best_P_min = best_P * 60.0
                peak_power = float(ls_power[bi])
                ls_significance = compute_significance(ls_power, peak_power)

                result['best_f_1phr'] = best_f
                result['best_P_hr'] = best_P
                result['best_P_min'] = best_P_min
                result['peak_power'] = peak_power
                result['ls_significance'] = ls_significance

                if is_tracked:
                    tracker.log(ra_deg, dec_deg, 'LS',
                                f"freq={best_f:.4f} cyc/hr, P={best_P_min:.2f} min, "
                                f"power={peak_power:.4f}, sig={ls_significance:.1f}")
        except Exception as e:
            if is_tracked:
                tracker.log(ra_deg, dec_deg, 'LS', f"EXCEPTION: {e}")

    # BLS
    try:
        min_period = config['bls_min_period_hr']
        max_period = config['bls_max_period_hr']
        duration_min = config['bls_duration_min'] / 60.0  # to hours
        duration_max = config['bls_duration_max'] / 60.0

        bls = BoxLeastSquares(t, y)
        bls_periods = np.linspace(min_period, max_period, 1000)
        bls_result = bls.power(bls_periods, duration=[duration_min, duration_max])
        bls_power = bls_result.power
        result['bls_power'] = bls_power
        result['bls_periods'] = bls_periods

        if np.isfinite(bls_power).any():
            bls_peak = float(np.nanmax(bls_power))
            bls_significance = compute_significance(bls_power, bls_peak)
            result['bls_significance'] = bls_significance

            if is_tracked:
                tracker.log(ra_deg, dec_deg, 'BLS', f"sig={bls_significance:.1f}")
    except Exception as e:
        if is_tracked:
            tracker.log(ra_deg, dec_deg, 'BLS', f"EXCEPTION: {e}")

    # String-length ratio
    if n_points >= config['bin_size']:
        result['sl_ratio'] = float(compute_string_length_ratio(y, config['bin_size']))
        if is_tracked:
            tracker.log(ra_deg, dec_deg, 'SL_RATIO',
                        f"sl_ratio={result['sl_ratio']:.1f}")

    # Attach tracker entries for parent to collect (fork-safe)
    if is_tracked:
        result['_tracker_entries'] = {
            label: list(entries)
            for label, entries in tracker._entries.items()
            if entries
        }

    return result


def analyze_source_worker_zf(args):
    """
    Worker function for parallel zeroframe source analysis.
    Same as analyze_source_worker but includes n_10sigma from crossmatch catalog.
    """
    global _global_cube, _global_times, _global_freqs, _global_config
    
    # Unpack with n_10sigma
    (idx, position, ra_deg, dec_deg, n_10sigma) = args
    
    # Call the regular worker with 4-tuple
    result = analyze_source_worker((idx, position, ra_deg, dec_deg))
    
    # Add n_10sigma to result
    result['n_10sigma'] = n_10sigma
    
    return result


def generate_source_plot(result, target, segment, detector, out_dir, bad_dir, config, freqs=None, debug=False):
    """Generate plot for a single source with LS and BLS power spectra.

    When config['skip_plots'] is True, saves only the CSV (no PNG).
    """

    t = result['t']
    y = result['y']
    y_err = result.get('y_err')  # Optional error bars
    ls_power = result.get('ls_power')
    bls_power = result.get('bls_power')
    bls_periods = result.get('bls_periods')

    if t is None or y is None:
        return None

    if len(t) == 0 or len(y) == 0:
        return None

    # Determine output directory
    sl_ratio = result.get('sl_ratio')
    is_bad = (sl_ratio is not None and np.isfinite(sl_ratio) and
              sl_ratio > config['sl_ratio_thresh'])
    save_dir = bad_dir if is_bad else out_dir

    # Source tracker: log routing decision
    tracker = config.get('_tracker', NULL_TRACKER)
    if tracker.is_tracking(result['ra_deg'], result['dec_deg']):
        dest = 'bad_dir' if save_dir == bad_dir else 'out_dir'
        tracker.log(result['ra_deg'], result['dec_deg'], 'ROUTE',
                    f"Routed to {dest}: sl_ratio={sl_ratio}, "
                    f"sl_thresh={config['sl_ratio_thresh']}")

    # Build filename
    ra_str = f"{result['ra_deg']:.6f}"
    dec_str = f"{result['dec_deg']:+.6f}"
    ls_sig = result.get('ls_significance', np.nan)
    sig_val = ls_sig if np.isfinite(ls_sig) else 0.0
    period_min = result.get('best_P_min', 0.0)
    if not np.isfinite(period_min):
        period_min = 0.0

    # For zeroframe mode, use n_10sigma as primary sort key
    zeroframe_mode = config.get('zeroframe_mode', False)
    if zeroframe_mode:
        n_10sigma = result.get('n_10sigma', 0)
        # Format: n10sig_NNN_sigXXX_PYYY_srcZZZZ_px_py_ra_dec
        basename = f"n10sig{n_10sigma:04d}_sig{sig_val:06.1f}_P{period_min:06.1f}min_src{result['idx']:04d}_px{result['pix_x']:.1f}_py{result['pix_y']:.1f}_{ra_str}_{dec_str}"
    else:
        # Standard ramp format
        sig_str_fn = f"{sig_val:08.2f}"
        basename = f"sig{sig_str_fn}_P{period_min:07.2f}min_src{result['idx']:04d}_px{result['pix_x']:.2f}_py{result['pix_y']:.2f}_{ra_str}_{dec_str}"

    png_path = os.path.join(save_dir, f"{basename}.png")

    # Save CSV
    csv_path = os.path.join(save_dir, f"{basename}.csv")
    save_lightcurve_csv(t, y, csv_path)

    # Skip matplotlib plot if requested
    if config.get('skip_plots'):
        # Write a zero-byte PNG marker
        open(png_path, 'a').close()
        return png_path

    if debug:
        print(f"      DEBUG: Saving to {png_path}")
        sys.stdout.flush()

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_lc, ax_phase = axes[0]
    ax_ls, ax_bls = axes[1]
    
    # Panel 1: Lightcurve with error bars
    ax_lc.scatter(t, y, s=4, c='black', alpha=0.6, label='Data')
    
    # Add error bars if available
    if y_err is not None and len(y_err) == len(t):
        ax_lc.errorbar(t, y, yerr=y_err, fmt='none', ecolor='gray', alpha=0.3, 
                       capsize=0, elinewidth=0.5, zorder=0)
    
    bin_size = config['bin_size']
    n_bins = len(y) // bin_size
    if n_bins > 0:
        t_bin = np.array([t[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)])
        y_bin = np.array([y[i*bin_size:(i+1)*bin_size].mean() for i in range(n_bins)])
        ax_lc.plot(t_bin, y_bin, '-r', lw=1.5, label=f'Binned({bin_size})')
    
    ax_lc.set_xlabel('Time (hr)')
    ax_lc.set_ylabel('Flux')
    pix_str = f"Pix: ({result['pix_x']:.2f}, {result['pix_y']:.2f})"

    # Add n_10sigma to title for zeroframes
    if zeroframe_mode:
        n_10sigma = result.get('n_10sigma', 0)
        ax_lc.set_title(f"{target} {segment} {detector.upper()} src{result['idx']:03d}\n{pix_str} | N(10σ)={n_10sigma}")
    else:
        ax_lc.set_title(f"{target} {segment} {detector.upper()} src{result['idx']:03d}\n{pix_str}")
    ax_lc.legend(loc='best')
    ax_lc.grid(alpha=0.3)
    
    # Panel 2: Phase-folded
    best_P = result.get('best_P_hr', np.nan)
    if not np.isnan(best_P) and best_P > 0:
        phase = (t % best_P) / best_P
        ax_phase.scatter(phase, y, s=4, c='black', alpha=0.6)
        ax_phase.set_xlabel('Phase')
        ax_phase.set_ylabel('Flux')
        best_P_min = result.get('best_P_min', best_P * 60)
        ax_phase.set_title(f"P={best_P_min:.2f} min")
        ax_phase.grid(alpha=0.3)
    else:
        ax_phase.text(0.5, 0.5, 'No period found', ha='center', va='center', transform=ax_phase.transAxes)
        ax_phase.set_title('Phase-folded')
    
    # Panel 3: LS Periodogram
    if freqs is None:
        freqs = np.linspace(config['freq_low_hr'], config['freq_high_hr'], config['freq_n_points'])
    
    if ls_power is not None and len(ls_power) > 0:
        periods_min = 60.0 / freqs  # Convert to minutes
        ax_ls.plot(periods_min, ls_power, 'b-', lw=0.8)
        ax_ls.set_xlabel('Period (min)')
        ax_ls.set_ylabel('LS Power')
        ax_ls.set_xscale('log')
        ax_ls.set_title(f"Lomb-Scargle (sig={ls_sig:.1f})" if np.isfinite(ls_sig) else "Lomb-Scargle")
        ax_ls.grid(alpha=0.3)
        
        # Mark best period
        best_P_min = result.get('best_P_min', np.nan)
        if not np.isnan(best_P_min):
            ax_ls.axvline(best_P_min, color='r', ls='--', alpha=0.7, label=f"P={best_P_min:.1f}min")
            ax_ls.legend(loc='best')
    else:
        ax_ls.text(0.5, 0.5, 'No LS data', ha='center', va='center', transform=ax_ls.transAxes)
        ax_ls.set_title('Lomb-Scargle')
    
    # Panel 4: BLS Periodogram
    if bls_power is not None and bls_periods is not None and len(bls_power) > 0:
        bls_periods_min = bls_periods * 60.0  # Convert to minutes
        ax_bls.plot(bls_periods_min, bls_power, 'g-', lw=0.8)
        ax_bls.set_xlabel('Period (min)')
        ax_bls.set_ylabel('BLS Power')
        ax_bls.set_xscale('log')
        bls_sig = result.get('bls_significance', np.nan)
        ax_bls.set_title(f"Box Least Squares (sig={bls_sig:.1f})" if np.isfinite(bls_sig) else "Box Least Squares")
        ax_bls.grid(alpha=0.3)
    else:
        ax_bls.text(0.5, 0.5, 'No BLS data', ha='center', va='center', transform=ax_bls.transAxes)
        ax_bls.set_title('Box Least Squares')
    
    # Suptitle with key info
    sig_str = f"{ls_sig:.1f}" if np.isfinite(ls_sig) else "N/A"
    sl_str = f"{sl_ratio:.1f}" if sl_ratio is not None and np.isfinite(sl_ratio) else "N/A"
    
    fig.suptitle(f"RA={ra_str} Dec={dec_str}  |  LS_sig={sig_str}  SL={sl_str}  N={result['n_points']}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    
    if debug:
        # Verify file was created
        if os.path.exists(png_path):
            print(f"      DEBUG: File created successfully: {os.path.getsize(png_path)} bytes")
        else:
            print(f"      DEBUG: FILE NOT CREATED!")
        sys.stdout.flush()

    return png_path


# ============================================================================
# Step 5: Full Analysis Pipeline for One Detector
# ============================================================================
def process_detector(target, segment, detector, config=CONFIG):
    """Run full analysis pipeline for one detector."""
    global _global_cube, _global_times, _global_freqs, _global_config
    
    print(f"\n=== Processing {target}/{segment}/{detector.upper()} ===")
    sys.stdout.flush()
    
    # Generate movies if requested
    if config.get('make_movies', False):
        generate_movies(target, segment, detector, config)
    
    # Paths (local)
    # Use custom ref file if provided, otherwise use standard path
    if config.get('ref_file') and os.path.exists(config['ref_file']):
        ref_path = config['ref_file']
        print(f"  Using custom reference: {ref_path}")
    else:
        ref_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_ref.fits')
    
    cube_path = os.path.join(config['ref_dir'], f'groupdiffs_{target}_{segment}_{detector}.fits')
    
    # Check/create files
    if not os.path.exists(ref_path):
        print(f"  Reference not found, creating (source: {config.get('ref_source', 'calints')})...")
        create_reference_image(target, segment, detector, config)
        # Update ref_path to the newly created file
        ref_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_ref.fits')
    
    if not os.path.exists(cube_path):
        print(f"  Cube not found, creating...")
        create_groupdiff_cube(target, segment, detector, config)
    
    if not os.path.exists(ref_path) or not os.path.exists(cube_path):
        print(f"  Skipping {detector}: missing files")
        return
    
    # Load reference and WCS
    print(f"  Loading reference image...")
    sys.stdout.flush()
    with fits.open(ref_path) as hd:
        ref_img = hd[0].data.astype(float)
        wcs = WCS(hd[0].header)

    # Source tracker
    tracker = config.get('_tracker', NULL_TRACKER)
    sources_table = None

    # ── Autocorrelation detection mode ──
    # Uses calints lag-1 autocorrelation as detection reference.
    # Shot noise is white (autocorr ≈ 0), variable stars have correlated
    # flux changes (autocorr >> 0). Achieves 100% recovery with 3× fewer
    # detections than var/mean on v13 matched catalog.
    if config.get('autocorr_detect', False):
        autocorr_path = os.path.join(config['ref_dir'],
                                     f'{target}_{segment}_{detector}_autocorr.fits')
        if not os.path.exists(autocorr_path):
            print(f"  Computing calints autocorrelation reference...")
            sys.stdout.flush()
            calints_files = get_calints_files(target, segment, detector, config['data_root'])
            if calints_files:
                create_autocorr_reference(calints_files, autocorr_path)
            else:
                print(f"  WARNING: No calints files, falling back to var/mean detection")

        if os.path.exists(autocorr_path):
            with fits.open(autocorr_path) as hd:
                autocorr_img = hd[0].data.astype(float)
            det_img = autocorr_img
            det_label = "autocorrelation"
        else:
            det_img = ref_img
            det_label = "var/mean (fallback)"
    else:
        det_img = ref_img
        det_label = "var/mean"

    # ── Source detection ──
    # PSF-matched filter on the detection image (fast_psf_detect)
    psf_path = config.get('psf_path')
    psf_kernel = None
    if psf_path and os.path.exists(psf_path):
        psf_kernel = load_psf_kernel(psf_path, size=config.get('psf_kernel_size', 21))

    if config.get('autocorr_detect', False) and psf_kernel is not None:
        # Autocorrelation mode: use fast_psf_detect on autocorrelation image
        ac_sigma = config.get('autocorr_sigma', 3.0)
        positions, sig_vals = fast_psf_detect(
            det_img, psf_kernel,
            threshold_sigma=ac_sigma, min_separation=1)
        print(f"  Autocorr PSF detect @ {ac_sigma}σ: {len(positions)} detections")
    else:
        # Standard mode: DAOStarFinder on var/mean
        det_thresh = config.get('detection_threshold', 4.0)
        det_fwhm = config.get('crossmatch_fwhm', 2.0)
        if tracker.active:
            positions, sig_vals, sources_table = detect_sources(
                det_img, fwhm=det_fwhm, threshold_sigma=det_thresh, return_table=True)
        else:
            positions, sig_vals = detect_sources(
                det_img, fwhm=det_fwhm, threshold_sigma=det_thresh)
        print(f"  DAOStarFinder on {det_label}: {len(positions)} detections")

    n_initial = len(positions)

    # PSF-matched filter detection (iterative, on var/mean ref — merged with above)
    if config.get('psf_detect', False) and psf_kernel is not None:
        print(f"  Running iterative PSF detection on var/mean ref...")
        sys.stdout.flush()
        pos_psf, sig_psf = iterative_psf_detect(
            ref_img, psf_kernel,
            threshold_sigma=config.get('psf_threshold', 3.0),
            min_separation=config.get('psf_min_separation', 1),
            n_iter=config.get('psf_n_iter', 3),
            max_sources_per_iter=config.get('psf_max_sources_per_iter', 50000))
        n_psf = len(pos_psf)
        print(f"  PSF filter: {n_psf} detections")

        positions, sig_vals = merge_detection_lists(
            positions, sig_vals, pos_psf, sig_psf, dedup_radius=2.5)
        n_new = len(positions) - n_initial
        print(f"  Merged: {len(positions)} total ({n_new} new from PSF)")
        if n_new > 0:
            sources_table = None

    if len(positions) == 0:
        print(f"  No sources detected in {detector}")
        return

    # Source tracker: check which tracked sources were detected
    if tracker.active:
        tracker.log_all('INFO', f"=== {segment} / {detector} ===")
        tracker.log_all('DETECT', f"Total detections: {len(positions)}")
        tracker.check_detection(positions, sig_vals, wcs,
                                sources_table=sources_table)

    skycoords = positions_to_skycoords(positions, wcs)
    print(f"  Detected {len(positions)} sources")
    sys.stdout.flush()

    # Load cube and times
    print(f"  Loading cube...")
    sys.stdout.flush()
    with fits.open(cube_path) as hd:
        cube = hd[0].data
        mjd_mid = hd['DIFF_TIMES'].data['MID_BARY_MJD']
    
    # Convert to native byte order if needed
    if cube.dtype.byteorder == '>' or (cube.dtype.byteorder == '=' and sys.byteorder == 'big'):
        print(f"  Converting cube to native byte order...")
        cube = cube.astype(np.float32, copy=True)
    else:
        cube = np.ascontiguousarray(cube, dtype=np.float32)
    
    times_hr = (mjd_mid - mjd_mid[0]) * 24.0
    
    # Use zeroframe-specific frequency range if in zeroframe mode
    zeroframe_mode = config.get('zeroframe_mode', False)
    if zeroframe_mode:
        freq_low = config.get('zeroframe_freq_low_hr', config['freq_low_hr'])
        freq_high = config.get('zeroframe_freq_high_hr', config['freq_high_hr'])
        print(f"  Zeroframe mode: searching periods {60/freq_high:.1f} - {60/freq_low:.1f} min")
    else:
        freq_low = config['freq_low_hr']
        freq_high = config['freq_high_hr']
    freqs = np.linspace(freq_low, freq_high, config['freq_n_points'])
    
    print(f"  Cube shape: {cube.shape}, {len(times_hr)} timestamps")
    sys.stdout.flush()

    # ── Temporal STD detection (merged with existing detections) ──
    if config.get('ramp_temporal_std', False) and not zeroframe_mode:
        tstd_path = os.path.join(config['ref_dir'],
                                 f'{target}_{segment}_{detector}_temporal_std.fits')
        if os.path.exists(tstd_path):
            print(f"  Loading existing temporal STD: {os.path.basename(tstd_path)}")
            temporal_std = fits.getdata(tstd_path).astype(np.float32)
        else:
            # Compute drop-2-outlier robust temporal STD from loaded cube
            print(f"  Computing drop-2-outlier temporal STD from {cube.shape[0]}-frame cube...")
            sys.stdout.flush()
            median_frame = np.nanmedian(cube, axis=0)
            residuals = np.abs(cube - median_frame[np.newaxis, :, :])
            yy, xx = np.mgrid[:cube.shape[1], :cube.shape[2]]
            worst1 = np.argmax(residuals, axis=0)
            residuals_copy = residuals.copy()
            residuals_copy[worst1, yy, xx] = 0
            worst2 = np.argmax(residuals_copy, axis=0)
            del residuals_copy
            tmask = np.ones(cube.shape, dtype=bool)
            tmask[worst1, yy, xx] = False
            tmask[worst2, yy, xx] = False
            del worst1, worst2, yy, xx
            cube_clipped = cube.copy()
            cube_clipped[~tmask] = np.nan
            del tmask
            temporal_std = np.nanstd(cube_clipped, axis=0).astype(np.float32)
            temporal_std[~np.isfinite(temporal_std)] = 0
            del cube_clipped, median_frame, residuals
            # Save with WCS from calints ref
            tstd_hdr = fits.Header()
            with fits.open(ref_path) as hd_ref:
                for key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2',
                            'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2', 'RADESYS']:
                    if key in hd_ref[0].header:
                        tstd_hdr[key] = hd_ref[0].header[key]
            tstd_hdr['BUNIT'] = 'DN'
            tstd_hdr['NFRAMES'] = cube.shape[0]
            tstd_hdr['DESCRIP'] = f'Temporal STD of {cube.shape[0]}-frame ramp cube (drop-2)'
            fits.PrimaryHDU(data=temporal_std, header=tstd_hdr).writeto(
                tstd_path, overwrite=True)
            print(f"  Saved temporal STD: {os.path.basename(tstd_path)}")

        # Detect on temporal STD with fast PSF matched filter
        tstd_sigma = config.get('ramp_tstd_sigma', 5.0)
        psf_path_cfg = config.get('psf_path')
        if psf_path_cfg and os.path.exists(psf_path_cfg):
            tstd_psf_kernel = load_psf_kernel(
                psf_path_cfg, size=config.get('psf_kernel_size', 21))
            pos_tstd, sig_tstd = fast_psf_detect(
                temporal_std, tstd_psf_kernel,
                threshold_sigma=tstd_sigma, min_separation=1)
            n_tstd = len(pos_tstd)
            print(f"  Temporal STD: {n_tstd} detections at {tstd_sigma}σ")
            if n_tstd > 0:
                n_before = len(positions)
                positions, sig_vals = merge_detection_lists(
                    positions, sig_vals, pos_tstd, sig_tstd, dedup_radius=2.5)
                n_new = len(positions) - n_before
                print(f"  Merged: {len(positions)} total ({n_new} new from temporal STD)")
                # Re-compute sky coordinates for merged list
                skycoords = positions_to_skycoords(positions, wcs)
                sources_table = None  # invalidated by merge
            del pos_tstd, sig_tstd
        else:
            print(f"  WARNING: temporal STD enabled but PSF file not found: {psf_path_cfg}")
        del temporal_std

    # Setup output directories (only needed when NOT using HDF5 output)
    hdf5_only = config.get('output_hdf5', False)
    if not hdf5_only:
        out_dir = os.path.join(config['out_dir'], target, segment, detector)
        bad_dir = os.path.join(config['bad_dir'], target, segment, detector)
        for d in [out_dir, bad_dir]:
            if os.path.exists(d):
                import shutil
                shutil.rmtree(d)
                print(f"  Cleaned old output: {d}")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)
    
    # Set global variables for workers (inherited by fork)
    # Snapshot tracker entries before fork so workers start with empty entries
    # (prevents double-counting pre-fork entries like DETECT)
    global _global_tracker
    pre_fork_entries = tracker.snapshot_entries() if tracker.active else None
    _global_cube = cube
    _global_times = times_hr
    _global_freqs = freqs
    _global_config = config
    _global_tracker = tracker
    
    # Prepare work items
    work_items = []
    for idx in range(len(positions)):
        ra_deg = float(skycoords[idx].ra.deg)
        dec_deg = float(skycoords[idx].dec.deg)
        work_items.append((idx, positions[idx], ra_deg, dec_deg))
    
    # Process sources
    n_workers = config.get('n_workers')
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"  Processing {len(work_items)} sources with {n_workers} workers...")
    sys.stdout.flush()
    
    # Force fork method
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    start_time = time.time()
    results = []

    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(analyze_source_worker, work_items, chunksize=10)):
            results.append(result)

            # Collect tracker entries from forked worker
            if '_tracker_entries' in result:
                tracker.merge_worker_entries(result.pop('_tracker_entries'))

            if (i + 1) % 100 == 0 or (i + 1) == len(work_items):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(work_items)} - {rate:.0f} src/s")
                sys.stdout.flush()
    
    elapsed_total = time.time() - start_time
    print(f"  Processing complete: {elapsed_total:.1f}s ({len(results)/elapsed_total:.0f} src/s)")

    # Restore pre-fork tracker entries
    if pre_fork_entries is not None:
        tracker.restore_entries(pre_fork_entries)

    # Write extraction HDF5 (primary output path)
    if hdf5_only:
        write_extraction_hdf5(
            results, positions, skycoords, sig_vals,
            times_hr, mjd_mid, target, segment, detector,
            mode='ramp', config=config)
    else:
        # Legacy CSV/PNG output path
        print(f"  Generating outputs for {len(results)} sources{' (skip-plots: CSV only)' if config.get('skip_plots') else ''}...")
        sys.stdout.flush()

        n_plotted = 0
        n_failed = 0
        n_no_data = 0
        n_empty = 0
        for result in results:
            t = result.get('t')
            y = result.get('y')

            if t is None or y is None:
                n_no_data += 1
                continue
            if len(t) == 0 or len(y) == 0:
                n_empty += 1
                continue

            debug_this = (n_plotted < 5) and not config.get('skip_plots')

            try:
                png_path = generate_source_plot(result, target, segment, detector, out_dir, bad_dir, config, freqs=freqs, debug=debug_this)
                if png_path:
                    n_plotted += 1
                else:
                    n_failed += 1
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"    Error plotting source {result['idx']}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()

            total_processed = n_plotted + n_failed + n_no_data + n_empty
            if total_processed % 50 == 0:
                print(f"    Processed {total_processed}/{len(results)}: plotted={n_plotted}, failed={n_failed}, no_data={n_no_data}, empty={n_empty}")
                sys.stdout.flush()

        print(f"  Summary: plotted={n_plotted}, failed={n_failed}, no_data={n_no_data}, empty={n_empty}")
        print(f"  Saved {n_plotted} lightcurve outputs for {detector}")

    # Flush tracker entries for this segment/detector
    tracker.flush(section_header=f"=== {segment} / {detector} (ramp_pipeline) ===")

    # Clear globals
    _global_cube = None
    _global_times = None
    _global_freqs = None
    _global_tracker = NULL_TRACKER


def deduplicate_sources(sources, dedup_radius_arcsec=0.3):
    """
    Remove duplicate sources that are within dedup_radius_arcsec of each other.
    Keeps the source with the higher peak_max value.

    Args:
        sources: List of source dicts with 'ra', 'dec', 'peak_max', etc.
        dedup_radius_arcsec: Separation threshold in arcseconds

    Returns:
        filtered_sources: List of unique sources
        n_duplicates: Number of duplicate sources removed
    """
    from astropy.coordinates import SkyCoord

    if len(sources) <= 1:
        return sources, 0

    # Create SkyCoord array
    coords = SkyCoord(ra=[s['ra'] for s in sources]*u.deg,
                     dec=[s['dec'] for s in sources]*u.deg)

    keep = np.ones(len(sources), dtype=bool)
    dedup_radius = dedup_radius_arcsec * u.arcsec

    for i in range(len(sources)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(sources)):
            if not keep[j]:
                continue

            sep = coords[i].separation(coords[j])
            if sep < dedup_radius:
                # Keep the one with higher peak_max
                if sources[i]['peak_max'] >= sources[j]['peak_max']:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    filtered_sources = [s for i, s in enumerate(sources) if keep[i]]
    n_duplicates = len(sources) - len(filtered_sources)

    return filtered_sources, n_duplicates


def process_detector_zeroframes(target, segment, detector, config=CONFIG):
    """Run zeroframe analysis pipeline for one detector."""
    global _global_cube, _global_times, _global_freqs, _global_config
    
    print(f"\n=== Processing ZEROFRAMES {target}/{segment}/{detector.upper()} ===")
    sys.stdout.flush()

    positions = None
    skycoords = None
    catalog_info = None

    # Build zeroframe cube FIRST (needed for temporal STD detection)
    print(f"  Building zeroframe cube...")
    sys.stdout.flush()
    cube, times_mjd, header = build_zeroframe_cube(target, segment, detector, config)
    
    if cube is None:
        print(f"  Failed to build zeroframe cube")
        return
    
    # Convert to native byte order if needed
    if cube.dtype.byteorder == '>' or (cube.dtype.byteorder == '=' and sys.byteorder == 'big'):
        print(f"  Converting cube to native byte order...")
        cube = cube.astype(np.float32, copy=True)
    else:
        cube = np.ascontiguousarray(cube, dtype=np.float32)

    # Save zeroframe cube to disk (mirrors groupdiff cube format)
    os.makedirs(config['ref_dir'], exist_ok=True)
    zf_cube_path = os.path.join(config['ref_dir'], f'zeroframes_{target}_{segment}_{detector}.fits')
    if header is not None:
        zf_hdr = make_3d_header(header, cube.shape[0], cube.shape[1], cube.shape[2])
    else:
        zf_hdr = fits.Header()
    zf_hdr['TARGET'] = target
    zf_hdr['SEGMENT'] = segment
    zf_hdr['DETECTOR'] = detector
    zf_hdr['CUBETYPE'] = 'ZEROFRAME'
    zf_primary = fits.PrimaryHDU(data=cube, header=zf_hdr)
    zf_time_col = fits.Column(name='MID_BARY_MJD', format='D', unit='d', array=times_mjd)
    zf_time_hdu = fits.BinTableHDU.from_columns([zf_time_col], name='DIFF_TIMES')
    fits.HDUList([zf_primary, zf_time_hdu]).writeto(zf_cube_path, overwrite=True)
    print(f"  Saved zeroframe cube: {zf_cube_path} ({cube.shape[0]} frames)")
    sys.stdout.flush()

    times_hr = (times_mjd - times_mjd[0]) * 24.0
    freqs = np.linspace(config['zeroframe_freq_low_hr'], config['zeroframe_freq_high_hr'], config['freq_n_points'])

    print(f"  Cube shape: {cube.shape}, {len(times_hr)} timestamps, {times_hr[-1]:.2f} hr span")
    sys.stdout.flush()

    # --- Source detection ---
    # Load WCS from ramp ref (preferred) or ZF ref
    ramp_ref_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_ref.fits')
    if os.path.exists(ramp_ref_path):
        with fits.open(ramp_ref_path) as hd:
            wcs = WCS(hd[0].header)
    else:
        print(f"  Warning: ramp ref not found, using ZF WCS")
        zf_ref_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_zf_ref.fits')
        if os.path.exists(zf_ref_path):
            with fits.open(zf_ref_path) as hd:
                wcs = WCS(hd[0].header)
        else:
            wcs = None

    psf_path = config.get('psf_path', '/data/Globulars_Pipeline/PSFs/PSF_NIRCam_in_flight_opd_filter_F200W.fits')
    psf_kernel = load_psf_kernel(psf_path)

    if config.get('autocorr_detect', False):
        # Autocorrelation detection: compute lag-1 autocorrelation from ZF cube
        print(f"  Computing lag-1 autocorrelation from {cube.shape[0]}-frame ZF cube...")
        sys.stdout.flush()
        zf_mean = np.nanmean(cube, axis=0)
        zf_residuals = cube - zf_mean[np.newaxis, :, :]
        zf_cross = np.nansum(zf_residuals[:-1] * zf_residuals[1:], axis=0)
        zf_var = np.nansum(zf_residuals ** 2, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            zf_autocorr = np.where(zf_var > 0, zf_cross / zf_var, 0.0)
        zf_autocorr = np.clip(zf_autocorr, -1.0, 1.0).astype(np.float32)
        zf_autocorr[~np.isfinite(zf_autocorr)] = 0.0
        del zf_mean, zf_residuals, zf_cross, zf_var
        print(f"  ZF autocorrelation: median={np.nanmedian(zf_autocorr):.4f}, "
              f"range=[{np.nanmin(zf_autocorr):.4f}, {np.nanmax(zf_autocorr):.4f}]")

        # Save autocorrelation ref
        ac_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_zf_autocorr.fits')
        ac_hdr = fits.Header()
        if wcs is not None:
            for key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2',
                        'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2', 'RADESYS']:
                try:
                    ac_hdr[key] = wcs.to_header()[key]
                except KeyError:
                    pass
        ac_hdr['NFRAMES'] = cube.shape[0]
        ac_hdr['DESCRIP'] = f'Lag-1 autocorrelation of {cube.shape[0]}-frame ZF cube'
        fits.PrimaryHDU(data=zf_autocorr, header=ac_hdr).writeto(ac_path, overwrite=True)
        print(f"  Saved ZF autocorrelation ref: {ac_path}")

        det_img = zf_autocorr
        det_label = "ZF autocorrelation"
        del zf_autocorr
    else:
        # Temporal STD detection (original approach)
        print(f"  Computing drop-2-outlier robust STD for detection...")
        sys.stdout.flush()
        median_frame = np.nanmedian(cube, axis=0)
        residuals = np.abs(cube - median_frame[np.newaxis, :, :])
        yy, xx = np.mgrid[:cube.shape[1], :cube.shape[2]]
        worst1 = np.argmax(residuals, axis=0)
        residuals_copy = residuals.copy()
        residuals_copy[worst1, yy, xx] = 0
        worst2 = np.argmax(residuals_copy, axis=0)
        tmask = np.ones(cube.shape, dtype=bool)
        tmask[worst1, yy, xx] = False
        tmask[worst2, yy, xx] = False
        cube_clipped = cube.copy()
        cube_clipped[~tmask] = np.nan
        temporal_std = np.nanstd(cube_clipped, axis=0).astype(np.float32)
        temporal_std[~np.isfinite(temporal_std)] = 0
        del median_frame, residuals, residuals_copy, cube_clipped, tmask, worst1, worst2, yy, xx

        # Save temporal STD reference image
        tstd_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_zf_temporal_std.fits')
        tstd_hdr = fits.Header()
        if wcs is not None:
            for key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2',
                        'CD2_1', 'CD2_2', 'CTYPE1', 'CTYPE2', 'RADESYS']:
                try:
                    tstd_hdr[key] = wcs.to_header()[key]
                except KeyError:
                    pass
        tstd_hdr['BUNIT'] = 'DN'
        tstd_hdr['NFRAMES'] = cube.shape[0]
        tstd_hdr['DESCRIP'] = f'Temporal STD of {cube.shape[0]}-frame ZF cube'
        fits.PrimaryHDU(data=temporal_std, header=tstd_hdr).writeto(tstd_path, overwrite=True)
        print(f"  Saved temporal STD ref: {tstd_path}")

        det_img = temporal_std
        det_label = "temporal STD"
        del temporal_std

    # Detect sources with fast PSF matched filter
    zf_detect_sigma = config.get('zf_detect_sigma', 5.0)
    print(f"  Detecting sources on {det_label} at {zf_detect_sigma}σ...")
    sys.stdout.flush()
    positions, sig_vals = fast_psf_detect(det_img, psf_kernel,
                                          threshold_sigma=zf_detect_sigma,
                                          min_separation=1)
    del det_img
    if len(positions) == 0:
        print(f"  No sources detected in {detector}")
        return

    print(f"  Detected {len(positions)} sources at {zf_detect_sigma}σ")

    # Assign sky coordinates using ramp WCS
    if wcs is None:
        ramp_ref_path = os.path.join(config['ref_dir'], f'{target}_{segment}_{detector}_ref.fits')
        if os.path.exists(ramp_ref_path):
            with fits.open(ramp_ref_path) as hd:
                wcs = WCS(hd[0].header)
    if wcs is not None:
        skycoords = positions_to_skycoords(positions, wcs)
    else:
        print(f"  Warning: no WCS available for sky coordinate assignment")
        return

    print(f"  Processing {len(positions)} sources")
    sys.stdout.flush()

    # Setup output directories (only needed when NOT using HDF5 output)
    hdf5_only = config.get('output_hdf5', False)
    if not hdf5_only:
        out_dir = os.path.join(config['zeroframe_out_dir'], target, segment, detector)
        bad_dir = os.path.join(config['zeroframe_bad_dir'], target, segment, detector)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)
    
    # Create a zeroframe-specific config copy
    zf_config = config.copy()
    zf_config['zeroframe_mode'] = True
    zf_config['bin_size'] = config.get('zeroframe_bin_size', 1)
    
    # Set global variables for workers
    _global_cube = cube
    _global_times = times_hr
    _global_freqs = freqs
    _global_config = zf_config
    
    # Get n_10sigma array (from catalog_info if available, else zeros)
    if catalog_info is not None and 'n_10sigma' in catalog_info:
        n_10sigma_arr = catalog_info['n_10sigma']
        print(f"  Using N_10SIGMA from catalog (max={np.max(n_10sigma_arr)}, sources with >0: {np.sum(n_10sigma_arr > 0)})")
    else:
        n_10sigma_arr = np.zeros(len(positions), dtype=int)
        print(f"  Warning: No N_10SIGMA data available, using zeros")
    
    # Prepare work items (include n_10sigma)
    work_items = []
    for idx in range(len(positions)):
        ra_deg = float(skycoords[idx].ra.deg)
        dec_deg = float(skycoords[idx].dec.deg)
        n_10sig = int(n_10sigma_arr[idx]) if idx < len(n_10sigma_arr) else 0
        work_items.append((idx, positions[idx], ra_deg, dec_deg, n_10sig))
    
    # Process sources
    n_workers = config.get('n_workers')
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    print(f"  Processing {len(work_items)} sources with {n_workers} workers...")
    sys.stdout.flush()
    
    # Force fork method
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    
    start_time = time.time()
    results = []

    with mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(analyze_source_worker_zf, work_items, chunksize=10)):
            results.append(result)

            if (i + 1) % 100 == 0 or (i + 1) == len(work_items):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{len(work_items)} - {rate:.0f} src/s")
                sys.stdout.flush()

    elapsed_total = time.time() - start_time
    print(f"  Processing complete: {elapsed_total:.1f}s ({len(results)/elapsed_total:.0f} src/s)")

    # Write extraction HDF5 (primary output path)
    if hdf5_only:
        write_extraction_hdf5(
            results, positions, skycoords, sig_vals,
            times_hr, times_mjd, target, segment, detector,
            mode='zf', config=config)
    else:
        # Legacy CSV/PNG output path
        skip_plots = config.get('skip_plots', False)
        print(f"  Generating outputs for {len(results)} sources{' (CSV only, --skip-plots)' if skip_plots else ''}...")
        sys.stdout.flush()

        n_plotted = 0
        n_failed = 0
        n_no_data = 0
        n_empty = 0
        for result in results:
            t = result.get('t')
            y = result.get('y')

            if t is None or y is None:
                n_no_data += 1
                continue
            if len(t) == 0 or len(y) == 0:
                n_empty += 1
                continue

            debug_this = (n_plotted < 5) and not skip_plots

            try:
                out_path = generate_source_plot(result, target, segment, detector, out_dir, bad_dir, zf_config, freqs=freqs, debug=debug_this)
                if out_path:
                    n_plotted += 1
                else:
                    n_failed += 1
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"    Error plotting source {result['idx']}: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()

            total_processed = n_plotted + n_failed + n_no_data + n_empty
            if total_processed % 5000 == 0:
                print(f"    Processed {total_processed}/{len(results)}: saved={n_plotted}, failed={n_failed}")
                sys.stdout.flush()

        print(f"  Summary: saved={n_plotted}, failed={n_failed}, no_data={n_no_data}, empty={n_empty}")
        print(f"  Saved {n_plotted} zeroframe lightcurve files for {detector}")

    # Clear globals
    _global_cube = None
    _global_times = None
    _global_freqs = None


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='JWST Ramp Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all segments and detectors for Liller1
    python ramp_pipeline.py --target Liller1
    
    # Process specific segment and detector
    python ramp_pipeline.py --target Terzan5 --segment Segment1 --detectors nrcb1 nrcb2
    
    # Use calints for reference image instead of ramp zeroframes
    python ramp_pipeline.py --target Liller1 --ref-source calints
    
    # Use a custom reference image (e.g., pixel periodogram significance map)
    python ramp_pipeline.py --target Terzan5 --segment Segment2 --detectors nrcb2 \
        --ref-file refs/groupdiffs_Terzan5_Segment2_nrcb2_pixelLS_significance.fits
    
    # Generate FITS cube movies for visualization
    python ramp_pipeline.py --target Liller1 --segment Segment3 --make-movies
    
    # Customize photometry settings
    python ramp_pipeline.py --target Liller1 --ap-radius 2.0 --annulus-inner 5.0 --annulus-outer 12.0
    
    # Also process zeroframes
    python ramp_pipeline.py --target Terzan5 --segment Segment2 --zeroframes

    # Use crossmatch catalog for zeroframes (default, recommended)
    python ramp_pipeline.py --target Terzan5 --segment Segment2 --zeroframes
    
    # Specify explicit crossmatch catalog
    python ramp_pipeline.py --target Terzan5 --segment Segment2 --zeroframes \\
        --zeroframe-catalog zeroframe_sources/sources_Terzan5_Segment2_nrcb1.fits
    
    # Fall back to reference image detection for zeroframes
    python ramp_pipeline.py --target Terzan5 --segment Segment2 --zeroframes --no-crossmatch
        """
    )
    parser.add_argument('--target', required=True, choices=list(TARGETS.keys()),
                        help='Target name')
    parser.add_argument('--segment', default=None,
                        help='Segment name (default: all segments for target)')
    parser.add_argument('--detectors', nargs='+', default=DETECTORS,
                        help='Detectors to process')
    parser.add_argument('--ref-source', choices=['ramps', 'calints'], default='calints',
                        help='Source for reference image: ramps (zeroframes) or calints')
    parser.add_argument('--ref-file', type=str, default=None,
                        help='Path to custom reference FITS file (overrides --ref-source)')
    parser.add_argument('--make-movies', action='store_true',
                        help='Generate FITS cube movies for visualization')
    parser.add_argument('--skip-cube', action='store_true',
                        help='Skip cube creation (use existing)')
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help=f'Data root directory (default: {DATA_ROOT})')
    parser.add_argument('--ap-radius', type=float, default=1.5,
                        help='Aperture radius in pixels (default: 1.5)')
    parser.add_argument('--annulus', action='store_true',
                        help='Enable annulus background subtraction (default: simple aperture)')
    parser.add_argument('--annulus-inner', type=float, default=4.0,
                        help='Inner annulus radius (default: 4.0)')
    parser.add_argument('--annulus-outer', type=float, default=10.0,
                        help='Outer annulus radius (default: 10.0)')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')
    parser.add_argument('--detection-threshold', type=float, default=4.0,
                        help='Source detection sigma threshold (default: 4.0)')
    parser.add_argument('--fwhm', type=float, default=2.0,
                        help='FWHM for DAOStarFinder detection in pixels (default: 2.0 for F200W, use ~3.5 for F356W)')
    parser.add_argument('--psf-detect', action='store_true',
                        help='Enable PSF-matched filter detection with iterative '
                             'subtraction (union with DAOStarFinder)')
    parser.add_argument('--psf-path', type=str,
                        default='/data/Globulars_Pipeline/PSFs/PSF_NIRCam_in_flight_opd_filter_F200W.fits',
                        help='Path to WebbPSF FITS file for PSF detection')
    parser.add_argument('--psf-threshold', type=float, default=3.0,
                        help='PSF detection threshold in sigma (default: 3.0)')
    parser.add_argument('--psf-min-separation', type=int, default=1,
                        help='Min peak separation for PSF detection in px (default: 1)')
    parser.add_argument('--psf-n-iter', type=int, default=3,
                        help='PSF detect-subtract iterations (default: 3)')
    parser.add_argument('--autocorr-detect', action='store_true',
                        help='Use calints lag-1 autocorrelation as detection reference '
                             'instead of var/mean (brightness-independent, better selectivity)')
    parser.add_argument('--autocorr-sigma', type=float, default=3.0,
                        help='Autocorrelation detection threshold in sigma (default: 3.0)')
    parser.add_argument('--ramp-temporal-std', action='store_true',
                        help='Also detect on temporal STD of ramp group-diff cube '
                             '(merged with DAOStarFinder/PSF detections)')
    parser.add_argument('--ramp-tstd-sigma', type=float, default=5.0,
                        help='Temporal STD detection threshold in sigma (default: 5.0)')
    parser.add_argument('--min-detections', type=int, default=30,
                        help='Min frames for crossmatch catalog detection (default: 30)')
    parser.add_argument('--out-dir', type=str, default='lightcurves_ramp',
                        help='Output directory for lightcurves (default: lightcurves_ramp)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Skip lightcurve plot generation (analysis only)')
    parser.add_argument('--zeroframes', action='store_true',
                        help='Also process zeroframes (separate output)')
    parser.add_argument('--zeroframe-only', action='store_true',
                        help='Process ONLY zeroframes, skip ramp (group-diff) processing')
    
    # Zeroframe-specific options
    parser.add_argument('--zeroframe-out-dir', type=str, default='lightcurves_zeroframe',
                        help='Output directory for zeroframe lightcurves (default: lightcurves_zeroframe)')
    parser.add_argument('--zeroframe-catalog', type=str, default=None,
                        help='Path to crossmatch source catalog for zeroframes')
    parser.add_argument('--no-crossmatch', action='store_true',
                        help='Disable crossmatch catalog, use reference image detection for zeroframes')
    parser.add_argument('--regenerate-catalog', action='store_true',
                        help='Force regeneration of crossmatch catalog (even if one exists)')

    # v3 pipeline options
    parser.add_argument('--config', type=str, default=None,
                        help='Path to pipeline.yaml config file (v3). Overrides CLI defaults.')
    parser.add_argument('--output-hdf5', action='store_true',
                        help='Write extraction HDF5 file instead of per-source CSVs')

    # Source tracking for debugging (see source_tracker.py for full docs)
    parser.add_argument('--track-sources', type=str, default=None,
                        help='Track specific sources for debugging. Path to file with '
                             '"RA Dec Label" rows, or inline "RA,Dec,Label;RA,Dec,Label"')
    parser.add_argument('--track-radius', type=float, default=0.5,
                        help='Match radius for source tracking in arcseconds (default: 0.5)')
    parser.add_argument('--track-output', type=str, default='tracker_logs',
                        help='Output directory for tracker log files (default: tracker_logs)')

    args = parser.parse_args()

    # Load YAML config first (if provided), then CLI args override
    if args.config:
        print(f"Loading config: {args.config}")
        load_pipeline_config(args.config)

    # CLI overrides output_hdf5
    if args.output_hdf5:
        CONFIG['output_hdf5'] = True

    CONFIG['data_root'] = args.data_root
    CONFIG['ref_source'] = args.ref_source
    CONFIG['ref_file'] = args.ref_file
    CONFIG['make_movies'] = args.make_movies
    CONFIG['ap_radius'] = args.ap_radius
    CONFIG['use_annulus'] = args.annulus
    CONFIG['annulus_inner'] = args.annulus_inner
    CONFIG['annulus_outer'] = args.annulus_outer
    CONFIG['n_workers'] = args.n_workers
    CONFIG['skip_plots'] = args.skip_plots
    CONFIG['detection_threshold'] = args.detection_threshold
    CONFIG['crossmatch_fwhm'] = args.fwhm
    CONFIG['psf_detect'] = args.psf_detect
    CONFIG['psf_path'] = args.psf_path
    CONFIG['psf_threshold'] = args.psf_threshold
    CONFIG['psf_min_separation'] = args.psf_min_separation
    CONFIG['psf_n_iter'] = args.psf_n_iter
    CONFIG['autocorr_detect'] = args.autocorr_detect
    CONFIG['autocorr_sigma'] = args.autocorr_sigma
    CONFIG['ramp_temporal_std'] = args.ramp_temporal_std
    CONFIG['ramp_tstd_sigma'] = args.ramp_tstd_sigma
    CONFIG['crossmatch_min_detections'] = args.min_detections
    CONFIG['out_dir'] = args.out_dir
    CONFIG['bad_dir'] = args.out_dir.replace('lightcurves_ramp', 'lightcurves_ramp_bad') if 'lightcurves_ramp' in args.out_dir else args.out_dir + '_bad'

    # Source tracker
    if args.track_sources:
        track_sources = load_track_file(args.track_sources)
        CONFIG['_tracker'] = SourceTracker(
            track_sources,
            match_radius_arcsec=args.track_radius,
            output_dir=args.track_output)
        print(f"\nSource tracking ENABLED: {len(track_sources)} source(s), "
              f"radius={args.track_radius}\"")
        for ra, dec, label in track_sources:
            print(f"  Tracking: {label} at RA={ra:.6f} Dec={dec:.6f}")
        print()
    else:
        CONFIG['_tracker'] = NULL_TRACKER

    # Zeroframe options
    CONFIG['zeroframe_out_dir'] = args.zeroframe_out_dir
    CONFIG['use_crossmatch_catalog'] = not args.no_crossmatch
    CONFIG['zeroframe_source_catalog'] = args.zeroframe_catalog
    CONFIG['regenerate_catalog'] = args.regenerate_catalog
    
    # Get segments to process
    if args.segment:
        segments = [args.segment]
    else:
        segments = TARGETS[args.target]['segments']
    
    print(f"Target: {args.target}")
    print(f"Segments: {segments}")
    print(f"Detectors: {args.detectors}")
    if CONFIG['ref_file']:
        print(f"Reference file: {CONFIG['ref_file']} (custom)")
    else:
        print(f"Reference source: {CONFIG['ref_source']}")
    print(f"Make movies: {CONFIG['make_movies']}")
    print(f"Data root: {CONFIG['data_root']}")
    if CONFIG['use_annulus']:
        print(f"Aperture: r={CONFIG['ap_radius']}px with annulus ({CONFIG['annulus_inner']}-{CONFIG['annulus_outer']}px)")
    else:
        print(f"Aperture: r={CONFIG['ap_radius']}px (simple, no annulus)")
    if CONFIG['psf_detect']:
        print(f"PSF detection: ENABLED (threshold={CONFIG['psf_threshold']}σ, "
              f"min_sep={CONFIG['psf_min_separation']}px, "
              f"n_iter={CONFIG['psf_n_iter']})")
    print(f"Process zeroframes: {args.zeroframes}")
    if args.zeroframes:
        print(f"  Zeroframe source: {'crossmatch catalog' if CONFIG['use_crossmatch_catalog'] else 'reference image'}")
        if args.zeroframe_catalog:
            print(f"  Catalog path: {args.zeroframe_catalog}")
    print()
    
    for segment in segments:
        print(f"\n{'='*60}")
        print(f"Processing {args.target} / {segment}")
        print(f"{'='*60}")
        
        for det in args.detectors:
            # Process ramps (group differences) unless zeroframe-only
            if not args.zeroframe_only:
                process_detector(args.target, segment, det, CONFIG)

            # Also process zeroframes if requested
            if args.zeroframes or args.zeroframe_only:
                process_detector_zeroframes(args.target, segment, det, CONFIG)


if __name__ == '__main__':
    main()
