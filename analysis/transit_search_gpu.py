#!/usr/bin/env python
"""
GPU-accelerated blind single-transit search on EVERY PIXEL.
Uses CuPy for:
  1. 2D convolution with circular aperture kernel (aperture photometry at every pixel)
  2. 1D boxcar sliding search along time axis

Processes each groupdiff cube (972 x 2048 x 2048) in row-chunks.
"""
import os, glob, time
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve as gpu_convolve
from cupyx.scipy.ndimage import uniform_filter1d as gpu_uniform_filter1d
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
OUT_DIR = f'{BASE}/diagnostics/transit_candidates_gpu'
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds
SIG_THRESHOLD = 12.0
MAX_DEPTH = 0.05
MIN_DEPTH = 0.002
MAX_DURATION_HR = 3.0
MIN_DURATION_HR = 10.0 / 60.0
CHUNK_ROWS = 64       # rows per GPU chunk (balance memory vs speed)
DEDUP_RADIUS = 5.0    # pixels

# Build circular aperture kernel (r=1.5 px) — 5x5
_y, _x = np.mgrid[-2:3, -2:3]
AP_KERNEL = ((_x**2 + _y**2) <= 1.5**2).astype(np.float32)
AP_KERNEL_GPU = cp.asarray(AP_KERNEL)


def search_chunk_gpu(ap_cube_chunk, times_gpu, widths, max_width_reject=None):
    """
    Search a chunk of the APERTURE-CONVOLVED cube on GPU.
    ap_cube_chunk: (nframes, nrows, ncols) CuPy array — already aperture-summed
    Returns: list of candidate dicts with (row_local, col, depth, etc.)
    """
    nframes, nrows, ncols = ap_cube_chunk.shape
    npix = nrows * ncols

    # Reshape to (npix, nframes)
    lcs = ap_cube_chunk.transpose(1, 2, 0).reshape(npix, nframes)

    # Median-normalize
    medians = cp.median(lcs, axis=1, keepdims=True)
    valid = (medians.ravel() > 0)
    lcs_norm = cp.where(medians > 0, lcs / medians, cp.ones_like(lcs))

    # Asymmetric isolated-outlier clipping (2 iterations).
    # Cosmic rays are positive spikes — clip aggressively upward (3 sigma),
    # gently downward (10 sigma) to preserve eclipses.
    for _clip_iter in range(2):
        local_base = gpu_uniform_filter1d(lcs_norm, size=5, axis=1, mode='nearest')
        resid = lcs_norm - local_base  # positive = spike, negative = dip
        abs_resid = cp.abs(resid)
        global_mad = cp.median(abs_resid, axis=1, keepdims=True) * 1.4826
        global_mad = cp.where(global_mad > 0, global_mad, cp.float32(1e-6))
        # Clip positive spikes at 3 sigma, negative at 10 sigma
        spike_mask = (resid > 3.0 * global_mad) | (resid < -10.0 * global_mad)
        lcs_norm = cp.where(spike_mask, local_base, lcs_norm)

    # MAD per pixel (after clipping)
    mad = cp.median(cp.abs(lcs_norm - 1.0), axis=1) * 1.4826
    mad = cp.where(mad > 0, mad, cp.std(lcs_norm, axis=1))
    valid = valid & (mad > 0)

    best_sig = cp.zeros(npix, dtype=cp.float32)
    best_width = cp.zeros(npix, dtype=cp.int32)
    best_center = cp.zeros(npix, dtype=cp.int32)

    for w in widths:
        if w >= nframes - 10:
            continue
        rm = gpu_uniform_filter1d(lcs_norm.astype(cp.float32), size=w, axis=1, mode='nearest')
        start, end = w, nframes - w
        if start >= end:
            continue
        dips = 1.0 - rm[:, start:end]
        sig = dips / (mad[:, None] / cp.sqrt(float(w)))

        bi = cp.argmax(sig, axis=1)
        best_val = sig[cp.arange(npix), bi]
        improved = best_val > best_sig
        best_sig = cp.where(improved, best_val, best_sig)
        best_width = cp.where(improved, w, best_width)
        best_center = cp.where(improved, bi + start, best_center)

    mask = valid & (best_sig >= SIG_THRESHOLD)
    if mask.sum() == 0:
        return []

    # Pull to CPU
    idxs = cp.where(mask)[0].get()
    sigs = best_sig[mask].get()
    centers = best_center[mask].get()
    ws = best_width[mask].get()
    lcs_cpu = lcs_norm[mask].get()

    times_cpu = times_gpu.get()
    candidates = []
    for k in range(len(idxs)):
        px_idx = int(idxs[k])
        row_local = px_idx // ncols
        col = px_idx % ncols
        w = int(ws[k])
        center = int(centers[k])
        ts = center - w // 2
        te = center + w // 2

        lc = lcs_cpu[k]
        out_mask = np.ones(nframes, dtype=bool)
        out_mask[ts:te] = False
        depth = float(np.median(lc[out_mask]) - np.median(lc[ts:te]))
        duration_hr = float(times_cpu[min(te, nframes-1)] - times_cpu[ts])

        if not (MIN_DEPTH <= depth <= MAX_DEPTH):
            continue
        if not (MIN_DURATION_HR <= duration_hr <= MAX_DURATION_HR):
            continue
        # Reject fits that hit the max allowed width (likely drifts, not transits)
        if max_width_reject is not None and w >= max_width_reject:
            continue

        # Quality: binned LC stays in plot range
        n_bins = max(1, nframes // 9)
        f_bin = lc[:n_bins*9].reshape(n_bins, 9).mean(axis=1)
        margin = max(0.03, depth * 3)
        frac_ok = np.mean((f_bin >= 1 - margin) & (f_bin <= 1 + margin * 0.7))
        if frac_ok < 0.85:
            continue

        candidates.append({
            'row': row_local, 'col': col,
            'depth': depth, 'duration_hr': duration_hr,
            'significance': float(sigs[k]),
            'width': w, 'ts': ts, 'te': te,
            'lc_norm': lc,
        })

    return candidates


def process_cube(target, segment, det):
    """Process one groupdiff cube: aperture-convolve on GPU, then boxcar search."""
    cube_path = f'{BASE}/refs/groupdiffs_{target}_{segment}_{det}.fits'
    ref_path = f'{BASE}/refs/{target}_{segment}_{det}_ref.fits'
    if not os.path.exists(cube_path):
        print(f'  SKIP: {cube_path} not found')
        return []

    cube = fits.getdata(cube_path, memmap=True)
    nframes, ny, nx = cube.shape

    # Times
    import h5py
    h5_path = f'{BASE}/extraction/{target}/{segment}/{det}_ramp.h5'
    if not os.path.exists(h5_path):
        return []
    with h5py.File(h5_path, 'r') as f:
        times = f['times'][:]

    wcs = WCS(fits.getheader(ref_path)) if os.path.exists(ref_path) else None

    cadence_hr = float(np.median(np.diff(times)))
    min_w = max(5, int(MIN_DURATION_HR / cadence_hr))
    max_w = min(nframes // 3, int(MAX_DURATION_HR / cadence_hr))
    if min_w >= max_w:
        return []
    widths = np.unique(np.geomspace(min_w, max_w, 20).astype(int)).tolist()

    times_gpu = cp.asarray(times, dtype=cp.float32)
    all_candidates = []

    for row_start in range(0, ny, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, ny)
        nrows_chunk = row_end - row_start

        # Load chunk: (nframes, nrows_chunk, nx)
        chunk_np = cube[:, row_start:row_end, :].astype(np.float32)

        # Aperture-convolve each frame on GPU
        # We need border pixels from adjacent rows for the convolution,
        # but for simplicity we'll just convolve within the chunk
        # (edge rows lose ~2px accuracy, acceptable)
        ap_chunk = np.empty_like(chunk_np)
        for i in range(nframes):
            frame_gpu = cp.asarray(chunk_np[i])
            conv = gpu_convolve(frame_gpu, AP_KERNEL_GPU, mode='constant', cval=0.0)
            ap_chunk[i] = conv.get()
            del frame_gpu, conv

        # Now search the aperture-convolved chunk
        ap_chunk_gpu = cp.asarray(ap_chunk)
        cands = search_chunk_gpu(ap_chunk_gpu, times_gpu, widths,
                                     max_width_reject=widths[-1])

        for c in cands:
            c['row'] += row_start
            if wcs is not None:
                sky = wcs.pixel_to_world(float(c['col']), float(c['row']))
                c['ra'] = float(sky.ra.deg)
                c['dec'] = float(sky.dec.deg)
            else:
                c['ra'] = c['dec'] = 0.0

        all_candidates.extend(cands)
        del ap_chunk_gpu, ap_chunk
        cp.get_default_memory_pool().free_all_blocks()

    # Spatial dedup
    if len(all_candidates) > 1:
        coords = np.array([[c['col'], c['row']] for c in all_candidates])
        sigs_arr = np.array([c['significance'] for c in all_candidates])
        order = np.argsort(-sigs_arr)
        tree = cKDTree(coords)
        kept = np.ones(len(all_candidates), dtype=bool)
        for i in order:
            if not kept[i]:
                continue
            for j in tree.query_ball_point(coords[i], DEDUP_RADIUS):
                if j != i and kept[j]:
                    kept[j] = False
        pre = len(all_candidates)
        all_candidates = [c for c, k in zip(all_candidates, kept) if k]
        print(f'  Dedup: {pre} -> {len(all_candidates)}')

    # Store times for plotting
    for c in all_candidates:
        c['times'] = times

    return all_candidates


def plot_candidate(cand, target, segment, det, out_dir):
    """Save diagnostic PNG."""
    depth_pct = cand['depth'] * 100
    dur_min = cand['duration_hr'] * 60
    sig = cand['significance']
    times = cand['times']
    lc = cand['lc_norm']

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.scatter(times, lc, s=0.5, c='black', alpha=0.3, rasterized=True)

    n = len(times)
    n_bins = max(1, n // 9); n_use = n_bins * 9
    t_bin = times[:n_use].reshape(n_bins, 9).mean(axis=1)
    f_bin = lc[:n_use].reshape(n_bins, 9).mean(axis=1)
    ax.plot(t_bin, f_bin, color='#1a3a6b', lw=2)

    model = np.ones(n)
    ts_t = times[cand['ts']]
    te_t = times[min(cand['te'], n-1)]
    in_transit = (times >= ts_t) & (times <= te_t)
    model[in_transit] = 1.0 - cand['depth']
    ax.plot(times, model, color='black', lw=2, ls='--')

    margin = max(0.03, cand['depth'] * 3)
    ax.set_ylim(1 - margin, 1 + margin * 0.7)
    ax.set_xlabel('Time (hours)', fontsize=9)
    ax.set_ylabel('Normalized Flux', fontsize=9)
    ax.set_title(f'Depth={depth_pct:.1f}%  Dur={dur_min:.0f}min  Sig={sig:.0f}  '
                 f'px=({cand["col"]},{cand["row"]})  {det}  {target}/{segment}',
                 fontsize=9)

    fig.tight_layout()
    fname = (f'depth{depth_pct:05.2f}_dur{dur_min:03.0f}_sig{sig:05.1f}_'
             f'{det}_{target}_{segment}_'
             f'{cand["ra"]:.5f}_{cand["dec"]:.5f}.png')
    fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    t0 = time.time()
    total_pixels = 0
    total_candidates = 0

    configs = []
    for target in ['Liller1', 'Terzan5']:
        segs = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']
        for seg in segs:
            for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']:
                configs.append((target, seg, det))

    for target, segment, det in configs:
        print(f'\n{target}/{segment}/{det}:', flush=True)
        t1 = time.time()
        candidates = process_cube(target, segment, det)
        elapsed = time.time() - t1
        npix = 2048 * 2048
        print(f'  {len(candidates)} candidates from {npix/1e6:.1f}M pixels ({elapsed:.0f}s)')

        for c in candidates:
            plot_candidate(c, target, segment, det, OUT_DIR)

        total_pixels += npix
        total_candidates += len(candidates)

    elapsed_total = time.time() - t0
    print(f'\n=== DONE ===')
    print(f'Total pixels: {total_pixels/1e6:.1f}M')
    print(f'Total candidates: {total_candidates}')
    print(f'Elapsed: {elapsed_total/60:.1f} min')
    print(f'Output: {OUT_DIR}/')
