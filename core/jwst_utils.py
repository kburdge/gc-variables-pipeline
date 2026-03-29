#!/usr/bin/env python3
"""
Shared utilities for JWST NIRCam time-series analysis.

Updated to work with /data/JWST directory structure:
    /data/JWST/
    ├── Liller1/
    │   ├── Segment3/detector1_output/*_ramp.fits
    │   ├── Segment4/detector1_output/*_ramp.fits
    │   └── ...
    └── Terzan5/
        ├── Segment1/detector1_output/*_ramp.fits
        ├── Segment2/detector1_output/*_ramp.fits
        └── ...
"""
import os
import glob
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime

try:
    from photutils.detection import DAOStarFinder
    from photutils.aperture import CircularAperture, aperture_photometry
except ImportError:
    from photutils.detection.finders import DAOStarFinder
    from photutils.aperture import CircularAperture
    from photutils.photometry import aperture_photometry


# =============================================================================
# Data Directory Configuration
# =============================================================================
DATA_ROOT = '/data/JWST'

# Define available targets and their segments
TARGETS = {
    'Liller1': {
        'segments': ['Segment3', 'Segment4'],
        'ramp_subdir': 'detector1_output',
    },
    'Terzan5': {
        'segments': ['Segment1', 'Segment2'],
        'ramp_subdir': 'detector1_output',
    },
}

# Standard detectors
DETECTORS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']


def get_ramp_files(target, segment=None, detector=None, data_root=DATA_ROOT):
    """
    Get list of ramp files for a given target/segment/detector.
    
    Args:
        target: Target name ('Liller1' or 'Terzan5')
        segment: Segment name (e.g., 'Segment3') or None for all segments
        detector: Detector name (e.g., 'nrcb1') or None for all detectors
        data_root: Root data directory
    
    Returns: sorted list of ramp file paths
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}. Available: {list(TARGETS.keys())}")
    
    tgt_info = TARGETS[target]
    segments = [segment] if segment else tgt_info['segments']
    detectors = [detector] if detector else DETECTORS
    
    ramp_files = []
    for seg in segments:
        seg_dir = os.path.join(data_root, target, seg, tgt_info['ramp_subdir'])
        if not os.path.isdir(seg_dir):
            continue
        for det in detectors:
            pattern = os.path.join(seg_dir, f'*_{det}_ramp.fits')
            ramp_files.extend(glob.glob(pattern))
    
    return sorted(ramp_files)


def get_calints_files(target, segment=None, detector=None, data_root=DATA_ROOT):
    """
    Get list of calints files for a given target/segment/detector.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    
    tgt_info = TARGETS[target]
    segments = [segment] if segment else tgt_info['segments']
    detectors = [detector] if detector else DETECTORS
    
    calints_files = []
    for seg in segments:
        calints_dir = os.path.join(data_root, target, seg, 'calints')
        if not os.path.isdir(calints_dir):
            continue
        for det in detectors:
            pattern = os.path.join(calints_dir, f'*_{det}_calints.fits')
            calints_files.extend(glob.glob(pattern))
    
    return sorted(calints_files)


# =============================================================================
# Image Processing Utilities
# =============================================================================

def heal_nans(image):
    """Replace isolated NaN pixels with mean of neighbors."""
    img = image.copy()
    ny, nx = img.shape
    ys, xs = np.where(np.isnan(img))
    for y, x in zip(ys, xs):
        y0, y1 = max(y-1, 0), min(y+2, ny)
        x0, x1 = max(x-1, 0), min(x+2, nx)
        win = img[y0:y1, x0:x1]
        if np.count_nonzero(np.isnan(win)) == 1:
            vals = win[~np.isnan(win)]
            if vals.size:
                img[y, x] = vals.mean()
    return img


def detect_sources(image, fwhm=2.0, threshold_sigma=4.0, clip_sigma=3.0,
                    return_table=False):
    """
    Detect sources using DAOStarFinder with dual-FWHM detection.

    Runs DAOStarFinder twice (fwhm=1.0 and fwhm=2.0) to recover both compact
    and resolved sources. Results are merged, with duplicates within 1.5px
    consolidated (keeping the higher-sigma detection).

    Relaxed sharphi=1.5, roundlo=-1.5, roundhi=1.5 (from defaults 1.0, -1.0, 1.0).

    Returns: positions (Nx2), sigma_values, sorted by decreasing sigma
             If return_table=True, also returns the sorted DAOStarFinder table.
    """
    from astropy.table import vstack as table_vstack

    img = heal_nans(image)
    mean, median, std = sigma_clipped_stats(img, sigma=clip_sigma)
    img_sub = img - median

    # Dual-FWHM detection: run with a compact FWHM and the requested fwhm.
    # The compact pass uses fwhm/2 (clamped to >= 1.0) to find sub-resolution
    # or compact sources without going below Nyquist.
    small_fwhm = max(1.0, fwhm / 2.0)
    fwhm_list = [small_fwhm, fwhm] if small_fwhm != fwhm else [fwhm]
    all_sources = []
    for fw in fwhm_list:
        finder = DAOStarFinder(fwhm=fw, threshold=threshold_sigma * std,
                               sharplo=0.2, sharphi=1.5, roundlo=-1.5, roundhi=1.5)
        det = finder(img_sub)
        if det is not None and len(det) > 0:
            all_sources.append(det)

    if not all_sources:
        empty = (np.empty((0, 2)), np.array([]))
        return (*empty, None) if return_table else empty

    if len(all_sources) == 1:
        sources = all_sources[0]
    else:
        sources = table_vstack(all_sources)

    # Consolidate duplicates within 2.5px (keep higher peak).
    # Different FWHM kernels shift centroids by up to ~2px for the same source.
    dedup_radius = 2.5
    if len(sources) > 1:
        x = np.array(sources['xcentroid'])
        y = np.array(sources['ycentroid'])
        peaks = np.array(sources['peak'])
        keep = np.ones(len(sources), dtype=bool)

        # Sort by peak descending so we keep brighter detections
        order_by_peak = np.argsort(peaks)[::-1]
        for i, idx in enumerate(order_by_peak):
            if not keep[idx]:
                continue
            for jdx in order_by_peak[i+1:]:
                if not keep[jdx]:
                    continue
                dist = np.sqrt((x[idx] - x[jdx])**2 + (y[idx] - y[jdx])**2)
                if dist < dedup_radius:
                    keep[jdx] = False

        sources = sources[keep]

    if len(sources) == 0:
        empty = (np.empty((0, 2)), np.array([]))
        return (*empty, None) if return_table else empty

    sig_vals = sources['peak'] / std
    positions = np.vstack([sources['xcentroid'], sources['ycentroid']]).T
    order = np.argsort(sig_vals)[::-1]

    if return_table:
        return positions[order], sig_vals[order], sources[order]
    return positions[order], sig_vals[order]


def load_psf_kernel(psf_path, size=21):
    """
    Load NIRCam PSF and extract a detector-sampled kernel for matched filtering.

    Parameters
    ----------
    psf_path : str
        Path to WebbPSF FITS file (must have 'DET_SAMP' extension).
    size : int
        Side length of extracted kernel in pixels (odd).

    Returns
    -------
    psf_kernel : 2D array, normalized to unit sum.
    """
    with fits.open(psf_path) as hd:
        psf_full = hd['DET_SAMP'].data.astype(np.float64)
    cy, cx = psf_full.shape[0] // 2, psf_full.shape[1] // 2
    half = size // 2
    psf_kernel = psf_full[cy-half:cy+half+1, cx-half:cx+half+1]
    psf_kernel /= np.sum(psf_kernel)
    return psf_kernel


def iterative_psf_detect(image, psf_kernel, threshold_sigma=3.0,
                         min_separation=1, n_iter=3,
                         max_sources_per_iter=5000):
    """
    Iterative PSF-subtraction source detection.

    Uses a matched-filter (FFT convolution with the real PSF) to build an
    S/N map, finds local maxima, then subtracts scaled PSFs from the
    residual image and repeats.  Sources near bright neighbors that are
    hidden under the neighbor's wings are revealed after subtraction.

    Parameters
    ----------
    image : 2D array
        Reference image (e.g. var/mean map).
    psf_kernel : 2D array
        Normalized PSF kernel (from load_psf_kernel).
    threshold_sigma : float
        Detection threshold in units of the background noise.
    min_separation : int
        Minimum peak separation in pixels.  The local-maximum filter
        has size ``2*min_separation + 1``.  Use 1 for crowded fields.
    n_iter : int
        Number of detect-subtract iterations.
    max_sources_per_iter : int
        Cap on sources subtracted per iteration (brightest first).

    Returns
    -------
    positions : (N, 2) float array — (x, y) pixel positions.
    snr_vals  : (N,) float array — matched-filter S/N at detection.
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import fftconvolve

    mean_bg, median_bg, std_bg = sigma_clipped_stats(image, sigma=3.0)
    residual = image.astype(np.float64) - median_bg
    psf_norm = np.sqrt(np.sum(psf_kernel**2))

    half = psf_kernel.shape[0] // 2
    ny, nx = image.shape

    all_positions = []
    all_snr = []

    for it in range(n_iter):
        filtered = fftconvolve(residual, psf_kernel, mode='same')
        noise_filtered = std_bg * psf_norm
        snr_map = filtered / (noise_filtered + 1e-10)

        data_max = maximum_filter(snr_map, size=min_separation * 2 + 1)
        is_peak = (snr_map == data_max) & (snr_map > threshold_sigma)
        peak_y, peak_x = np.where(is_peak)
        peak_snr = snr_map[peak_y, peak_x]

        if len(peak_snr) == 0:
            break

        order = np.argsort(peak_snr)[::-1]
        peak_x = peak_x[order][:max_sources_per_iter]
        peak_y = peak_y[order][:max_sources_per_iter]
        peak_snr = peak_snr[order][:max_sources_per_iter]

        # Record new detections (skip duplicates near previous iterations)
        for px, py, snr in zip(peak_x, peak_y, peak_snr):
            if len(all_positions) > 0:
                prev = np.array(all_positions)
                dists = np.sqrt((prev[:, 0] - px)**2 +
                                (prev[:, 1] - py)**2)
                if np.min(dists) < min_separation:
                    continue
            all_positions.append([float(px), float(py)])
            all_snr.append(float(snr))

        # Subtract PSFs from residual (brightest first)
        psf_sum_sq = np.sum(psf_kernel**2)
        for px, py in zip(peak_x, peak_y):
            amplitude = filtered[py, px] / psf_sum_sq

            y0 = max(0, py - half)
            y1 = min(ny, py + half + 1)
            x0 = max(0, px - half)
            x1 = min(nx, px + half + 1)

            ky0 = y0 - (py - half)
            ky1 = psf_kernel.shape[0] - ((py + half + 1) - y1)
            kx0 = x0 - (px - half)
            kx1 = psf_kernel.shape[1] - ((px + half + 1) - x1)

            residual[y0:y1, x0:x1] -= amplitude * psf_kernel[ky0:ky1, kx0:kx1]

    if len(all_positions) == 0:
        return np.empty((0, 2)), np.array([])

    positions = np.array(all_positions)
    snr_vals = np.array(all_snr)

    # Sort by decreasing S/N
    order = np.argsort(snr_vals)[::-1]
    return positions[order], snr_vals[order]


def fast_psf_detect(image, psf_kernel, threshold_sigma=5.0, min_separation=1):
    """
    Single-pass PSF matched filter detection — O(N log N) via FFT.

    Unlike iterative_psf_detect, this skips iterative subtraction and the
    O(n²) duplicate check, making it orders of magnitude faster (~1 sec
    vs ~500+ sec for dense fields with 50K+ sources).

    Parameters
    ----------
    image : 2D array
        Reference image (e.g. var/mean map).
    psf_kernel : 2D array
        Normalized PSF kernel (from load_psf_kernel).
    threshold_sigma : float
        Detection threshold in matched-filter S/N units.
    min_separation : int
        Minimum peak separation in pixels (local-max filter size).

    Returns
    -------
    positions : (N, 2) float array — (x, y) pixel positions.
    snr_vals  : (N,) float array — matched-filter S/N at detection.
    """
    from scipy.ndimage import maximum_filter
    from scipy.signal import fftconvolve

    # Mask extreme outliers (hot pixels, bad columns) before noise estimation.
    # Var/mean images can have ~20K pixels with values >1000 from artifacts,
    # which inflate sigma_clipped_stats std from ~5 to ~22, burying real sources.
    img_clean = image.copy()
    p995 = np.nanpercentile(img_clean, 99.5)
    img_clean[(img_clean > p995) | (img_clean < -p995)] = np.nan
    mean_bg, median_bg, std_bg = sigma_clipped_stats(img_clean, sigma=3.0)
    # Replace NaN/Inf with median before convolution (NaN propagates through FFT)
    residual = np.nan_to_num(image.astype(np.float64) - median_bg,
                             nan=0.0, posinf=0.0, neginf=0.0)
    psf_norm = np.sqrt(np.sum(psf_kernel**2))

    filtered = fftconvolve(residual, psf_kernel, mode='same')
    noise_filtered = std_bg * psf_norm
    snr_map = filtered / (noise_filtered + 1e-10)

    data_max = maximum_filter(snr_map, size=min_separation * 2 + 1)
    is_peak = (snr_map == data_max) & (snr_map > threshold_sigma)
    peak_y, peak_x = np.where(is_peak)
    peak_snr = snr_map[peak_y, peak_x]

    if len(peak_snr) == 0:
        return np.empty((0, 2)), np.array([])

    order = np.argsort(peak_snr)[::-1]
    positions = np.column_stack([peak_x[order].astype(float),
                                 peak_y[order].astype(float)])
    return positions, peak_snr[order]


def merge_detection_lists(pos_a, sig_a, pos_b, sig_b, dedup_radius=2.5):
    """
    Merge two detection lists (union), deduplicating within dedup_radius.

    When two detections overlap, the one with higher significance is kept.

    Parameters
    ----------
    pos_a, pos_b : (N,2) arrays of (x, y)
    sig_a, sig_b : (N,) significance arrays
    dedup_radius : float
        Merge radius in pixels.

    Returns
    -------
    positions : (M, 2) array
    sig_vals  : (M,) array
    """
    if len(pos_a) == 0:
        return pos_b, sig_b
    if len(pos_b) == 0:
        return pos_a, sig_a

    from scipy.spatial import cKDTree

    # Concatenate both lists
    positions = np.vstack([pos_a, pos_b])
    sig_vals = np.concatenate([sig_a, sig_b])

    # Deduplicate using KDTree: process brightest first, kill neighbors
    keep = np.ones(len(positions), dtype=bool)
    order = np.argsort(sig_vals)[::-1]
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=dedup_radius, output_type='ndarray')

    # For each pair, mark the fainter one for removal
    # Build adjacency: for each source, which others are within radius
    if len(pairs) > 0:
        # Process in brightness order: brightest survives
        for idx in order:
            if not keep[idx]:
                continue
            # Find all neighbors within radius
            neighbors = tree.query_ball_point(positions[idx], r=dedup_radius)
            for jdx in neighbors:
                if jdx != idx and keep[jdx] and sig_vals[jdx] <= sig_vals[idx]:
                    keep[jdx] = False

    positions = positions[keep]
    sig_vals = sig_vals[keep]
    order = np.argsort(sig_vals)[::-1]
    return positions[order], sig_vals[order]


def positions_to_skycoords(positions, wcs):
    """Convert pixel positions to SkyCoord objects."""
    world = wcs.all_pix2world(positions, 0)
    return SkyCoord(ra=world[:, 0]*u.deg, dec=world[:, 1]*u.deg)


# =============================================================================
# Time Series Utilities
# =============================================================================

def clip_outliers_iqr(data, times, chunk_size=18, iqr_factor=2.0):
    """
    Remove outliers using IQR clipping in chunks.
    
    Returns: clipped_data, clipped_times
    """
    n = len(data)
    n_chunks = n // chunk_size
    if n_chunks < 1:
        return data, times
    
    mask = np.ones(n, dtype=bool)
    for b in range(n_chunks):
        start, end = b * chunk_size, (b + 1) * chunk_size
        seg = data[start:end]
        q1, q3 = np.percentile(seg, [25, 75])
        iqr = q3 - q1
        good = (seg >= q1 - iqr_factor*iqr) & (seg <= q3 + iqr_factor*iqr)
        mask[start:end] = good
    
    return data[mask], times[mask]


def compute_string_length_ratio(raw_data, bin_size=9):
    """
    Compute string-length ratio (raw SL / binned SL).
    High values indicate noise-dominated lightcurves.
    """
    n = len(raw_data)
    n_bins = n // bin_size
    if n_bins < 2:
        return np.nan
    
    trimmed = raw_data[:n_bins * bin_size]
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    
    raw_sl = np.abs(np.diff(trimmed)).sum()
    bin_sl = np.abs(np.diff(binned)).sum()
    
    return raw_sl / bin_sl if bin_sl > 0 else np.inf


def save_lightcurve_csv(time, flux, filepath):
    """Save lightcurve as CSV file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    data = np.column_stack([time, flux])
    np.savetxt(filepath, data, delimiter=',', header='time_hr,flux', 
               fmt='%.7f', comments='')


# =============================================================================
# FITS Header Utilities
# =============================================================================

def make_2d_wcs_header(sci_header, ny, nx, extname='DATA'):
    """
    Create a clean 2D WCS header from a SCI extension header.
    Strips 3D/4D keywords and sets correct dimensions.
    """
    hdr = sci_header.copy()
    
    # Remove higher-dimension keywords (3 and above)
    for i in range(3, 10):
        for kw in [f'NAXIS{i}', f'CRPIX{i}', f'CRVAL{i}', f'CDELT{i}', 
                   f'CTYPE{i}', f'CUNIT{i}', f'CD{i}_{i}', f'PC{i}_{i}']:
            hdr.pop(kw, None)
    
    # Remove cross-term WCS keywords
    for i in range(1, 10):
        for j in range(3, 10):
            hdr.pop(f'CD{i}_{j}', None)
            hdr.pop(f'CD{j}_{i}', None)
            hdr.pop(f'PC{i}_{j}', None)
            hdr.pop(f'PC{j}_{i}', None)
    
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['WCSAXES'] = 2
    hdr['EXTNAME'] = extname
    
    return hdr


def make_3d_header(sci_header, nz, ny, nx):
    """
    Create a clean 3D header from a SCI extension header.
    Strips 4D+ keywords and sets correct dimensions.
    """
    hdr = sci_header.copy()
    
    # Remove higher-dimension keywords (4 and above)
    for i in range(4, 10):
        for kw in [f'NAXIS{i}', f'CRPIX{i}', f'CRVAL{i}', f'CDELT{i}', 
                   f'CTYPE{i}', f'CUNIT{i}', f'CD{i}_{i}', f'PC{i}_{i}']:
            hdr.pop(kw, None)
    
    # Remove cross-term WCS keywords for dimensions 4+
    for i in range(1, 10):
        for j in range(4, 10):
            hdr.pop(f'CD{i}_{j}', None)
            hdr.pop(f'CD{j}_{i}', None)
            hdr.pop(f'PC{i}_{j}', None)
            hdr.pop(f'PC{j}_{i}', None)
    
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = nx
    hdr['NAXIS2'] = ny
    hdr['NAXIS3'] = nz
    
    return hdr


def get_detector_from_filename(filename):
    """Extract detector name from JWST filename."""
    basename = os.path.basename(filename)
    for det in DETECTORS:
        if f'_{det}_' in basename:
            return det
    return None


# =============================================================================
# FITS Cube Movie Utilities
# =============================================================================

def create_diff_cube(cube):
    """
    Create difference cube (frame[i+1] - frame[i]).
    
    Args:
        cube: 3D array (n_frames, ny, nx)
    
    Returns: diff_cube (n_frames-1, ny, nx)
    """
    return cube[1:] - cube[:-1]


def save_fits_cube(cube, filepath, header=None, description=None, times=None):
    """
    Save a 3D cube as a FITS file with optional time extension.
    
    Args:
        cube: 3D numpy array (n_frames, ny, nx)
        filepath: Output path
        header: Optional FITS header (will be modified for 3D)
        description: Description to add to header
        times: Optional array of timestamps (MJD)
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    if header is None:
        header = fits.Header()
        hdr = header
    else:
        # Use proper 3D header cleaning
        hdr = make_3d_header(header, cube.shape[0], cube.shape[1], cube.shape[2])
    
    if description:
        hdr['DESCRIPT'] = description
    hdr['DATE'] = datetime.utcnow().isoformat()
    
    hdu_list = [fits.PrimaryHDU(data=cube.astype(np.float32), header=hdr)]
    
    # Add time extension if provided
    if times is not None:
        col = fits.Column(name='MJD', format='D', unit='d', array=times)
        time_hdu = fits.BinTableHDU.from_columns([col], name='TIMES')
        hdu_list.append(time_hdu)
    
    fits.HDUList(hdu_list).writeto(filepath, overwrite=True)
    print(f"  Wrote {filepath} ({cube.shape[0]} frames)")


def save_2d_fits(image, filepath, header=None, description=None):
    """
    Save a 2D image as a FITS file.
    
    Args:
        image: 2D numpy array (ny, nx)
        filepath: Output path
        header: Optional FITS header
        description: Description to add to header
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    if header is None:
        header = fits.Header()
    else:
        header = header.copy()
    
    # Update header for 2D
    header['NAXIS'] = 2
    header['NAXIS1'] = image.shape[1]
    header['NAXIS2'] = image.shape[0]
    # Remove 3D keywords if present
    for kw in ['NAXIS3', 'CRPIX3', 'CRVAL3', 'CDELT3', 'CTYPE3']:
        header.pop(kw, None)
    
    if description:
        header['DESCRIPT'] = description
    header['DATE'] = datetime.utcnow().isoformat()
    
    fits.PrimaryHDU(data=image.astype(np.float32), header=header).writeto(
        filepath, overwrite=True)
    print(f"  Wrote {filepath}")


# =============================================================================
# Calints-based Reference Creation
# =============================================================================

def create_reference_from_calints(calints_files, output_path, header=None, 
                                   mask_nans=True):
    """
    Create variance/mean reference image from calints files.
    Based on create_ref_variability.py approach.
    
    Args:
        calints_files: List of calints FITS files
        output_path: Where to save reference image
        header: Optional header to use (otherwise from first file)
        mask_nans: If True, zero out pixels that were ever NaN
    
    Returns: reference image array, header
    """
    if not calints_files:
        print("  No calints files provided")
        return None, None
    
    data_cubes = []
    headers = []
    
    for fn in sorted(calints_files):
        with fits.open(fn) as hdul:
            # calints data is typically in extension 1 (SCI)
            if 'SCI' in hdul:
                data = hdul['SCI'].data.astype(np.float32)
                hdr = hdul['SCI'].header
            else:
                data = hdul[1].data.astype(np.float32)
                hdr = hdul[1].header
        data_cubes.append(data)
        headers.append(hdr)
    
    # Sanity check: consistent shapes
    shapes = {d.shape for d in data_cubes}
    if len(shapes) != 1:
        raise ValueError(f"Input cubes differ in shape: {shapes}")
    
    # Concatenate all frames
    big_cube = np.concatenate(data_cubes, axis=0)
    print(f"  Combined {len(calints_files)} files -> {big_cube.shape[0]} total frames")
    
    # Build mask of any-NaN pixels
    had_nan = np.isnan(big_cube).any(axis=0)

    # Compute per-pixel variance and mean
    var_frame = np.nanvar(big_cube, axis=0)
    mean_frame = np.nanmean(big_cube, axis=0)

    # Form ratio image (variance / mean)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_frame = var_frame / mean_frame

    # Clean up non-finite values
    ratio_frame[~np.isfinite(ratio_frame)] = 0.0

    # For pixels that had NaNs: instead of setting to 0 (which flattens PSF cores),
    # interpolate from neighbors to preserve PSF structure for detection
    if mask_nans and np.sum(had_nan) > 0:
        # Use brightest neighbor interpolation to maintain PSF peak structure
        ys, xs = np.where(had_nan)
        ny, nx = ratio_frame.shape
        for y, x in zip(ys, xs):
            y0, y1 = max(y-1, 0), min(y+2, ny)
            x0, x1 = max(x-1, 0), min(x+2, nx)
            win = ratio_frame[y0:y1, x0:x1]
            valid = win[~had_nan[y0:y1, x0:x1]]
            if valid.size > 0:
                # Use max of neighbors to preserve PSF peaks
                ratio_frame[y, x] = valid.max()
            else:
                ratio_frame[y, x] = 0.0
    
    # Prepare header
    if header is None:
        out_hdr = headers[0].copy()
    else:
        out_hdr = header.copy()
    
    out_hdr = make_2d_wcs_header(out_hdr, ratio_frame.shape[0], ratio_frame.shape[1],
                                  extname='VAR_OVER_MEAN')
    out_hdr['REFTYPE'] = 'calints'
    out_hdr['NFILES'] = len(calints_files)
    out_hdr['NFRAMES'] = big_cube.shape[0]
    
    # Save
    save_2d_fits(ratio_frame, output_path, out_hdr, 
                 description='Variance/Mean from calints')
    
    return ratio_frame, out_hdr


def create_autocorr_reference(calints_files, output_path, header=None):
    """
    Create lag-1 temporal autocorrelation reference image from calints.

    For each pixel, computes:
        autocorr = sum(r[t] * r[t+1]) / sum(r[t]^2)
    where r[t] = flux[t] - mean(flux).

    This is brightness-independent: shot noise is white (autocorr ≈ 0),
    while real variable stars have temporally correlated flux changes
    (autocorr >> 0). Achieves 100% recovery at 3σ on v13 matched catalog
    with 3.2× fewer total detections than var/mean.

    Args:
        calints_files: List of calints FITS files
        output_path: Where to save reference image
        header: Optional header to use

    Returns: autocorrelation image array, header
    """
    if not calints_files:
        print("  No calints files provided")
        return None, None

    data_cubes = []
    headers = []

    for fn in sorted(calints_files):
        with fits.open(fn) as hdul:
            if 'SCI' in hdul:
                data = hdul['SCI'].data.astype(np.float32)
                hdr = hdul['SCI'].header
            else:
                data = hdul[1].data.astype(np.float32)
                hdr = hdul[1].header
        data_cubes.append(data)
        headers.append(hdr)

    big_cube = np.concatenate(data_cubes, axis=0)
    print(f"  Combined {len(calints_files)} files -> {big_cube.shape[0]} frames for autocorrelation")

    # Compute per-pixel lag-1 autocorrelation
    mean = np.nanmean(big_cube, axis=0)
    residuals = big_cube - mean[np.newaxis, :, :]
    cross = np.nansum(residuals[:-1] * residuals[1:], axis=0)
    var = np.nansum(residuals ** 2, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        autocorr = np.where(var > 0, cross / var, 0.0)
    autocorr = np.clip(autocorr, -1.0, 1.0)
    autocorr[~np.isfinite(autocorr)] = 0.0

    print(f"  Autocorrelation: median={np.nanmedian(autocorr):.4f}, "
          f"range=[{np.nanmin(autocorr):.4f}, {np.nanmax(autocorr):.4f}]")

    # Prepare header
    if header is None:
        out_hdr = headers[0].copy()
    else:
        out_hdr = header.copy()

    out_hdr = make_2d_wcs_header(out_hdr, autocorr.shape[0], autocorr.shape[1],
                                  extname='AUTOCORR_LAG1')
    out_hdr['REFTYPE'] = 'calints_autocorr'
    out_hdr['NFILES'] = len(calints_files)
    out_hdr['NFRAMES'] = big_cube.shape[0]

    save_2d_fits(autocorr, output_path, out_hdr,
                 description='Lag-1 autocorrelation from calints')

    return autocorr, out_hdr


def create_calints_cube(calints_files, subtract_background=True):
    """
    Create combined cube from calints files.
    
    Args:
        calints_files: List of calints FITS files
        subtract_background: If True, subtract per-pixel nanmean
    
    Returns: cube (n_frames, ny, nx), times_mjd, header
    """
    if not calints_files:
        return None, None, None
    
    data_cubes = []
    time_list = []
    headers = []
    
    for fn in sorted(calints_files):
        with fits.open(fn) as hdul:
            if 'SCI' in hdul:
                data = hdul['SCI'].data.astype(np.float32)
                hdr = hdul['SCI'].header
            else:
                data = hdul[1].data.astype(np.float32)
                hdr = hdul[1].header
            
            # Get timestamps from INT_TIMES if available
            if 'INT_TIMES' in hdul:
                tmid = hdul['INT_TIMES'].data['int_mid_MJD_UTC']
                time_list.extend(tmid)
            else:
                # Generate placeholder times
                n_int = data.shape[0]
                time_list.extend([np.nan] * n_int)
        
        data_cubes.append(data)
        headers.append(hdr)
    
    # Concatenate
    big_cube = np.concatenate(data_cubes, axis=0)
    times = np.array(time_list, dtype=np.float64)
    
    if subtract_background:
        # Compute per-pixel nanmean and subtract
        mean_frame = np.nanmean(big_cube, axis=0)
        big_cube = big_cube - mean_frame[np.newaxis, ...]
        # Replace remaining NaNs with 0
        big_cube = np.nan_to_num(big_cube, nan=0.0)
    
    return big_cube, times, headers[0]


# =============================================================================
# Zeroframe Utilities
# =============================================================================

def extract_zeroframes(ramp_files, drop_first=True):
    """
    Extract zeroframes from ramp files.
    
    Args:
        ramp_files: List of ramp FITS files
        drop_first: If True, drop first frame of each integration
    
    Returns: cube (n_frames, ny, nx), times_mjd, header
    """
    if not ramp_files:
        return None, None, None
    
    zf_list = []
    time_list = []
    header = None
    
    for fn in sorted(ramp_files):
        with fits.open(fn) as hdul:
            if 'ZEROFRAME' not in hdul:
                print(f"  Warning: No ZEROFRAME in {fn}")
                continue
            
            zf = hdul['ZEROFRAME'].data.astype(np.float32)
            
            if header is None and 'SCI' in hdul:
                header = hdul['SCI'].header.copy()
            
            # Get timestamps
            if 'INT_TIMES' in hdul:
                tmid = hdul['INT_TIMES'].data['int_mid_MJD_UTC']
            else:
                tmid = np.full(zf.shape[0], np.nan)
            
            start_idx = 1 if drop_first else 0
            for i in range(start_idx, zf.shape[0]):
                zf_list.append(zf[i])
                time_list.append(tmid[i] if i < len(tmid) else np.nan)
    
    if not zf_list:
        return None, None, None
    
    cube = np.stack(zf_list, axis=0)
    times = np.array(time_list, dtype=np.float64)
    
    return cube, times, header
