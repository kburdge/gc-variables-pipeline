#!/usr/bin/env python
"""
Blind single-transit search across all SW ramp extractions.
Slides boxcar templates of various widths, finds the deepest significant dip.
Saves candidates above threshold to diagnostics/transit_candidates/.
"""
import os, sys, glob, time
import numpy as np
import h5py
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/data/Globulars_Pipeline'
OUT_DIR = f'{BASE}/diagnostics/transit_candidates'
os.makedirs(OUT_DIR, exist_ok=True)

# Thresholds
SIG_THRESHOLD = 12.0   # boxcar fit significance
MAX_DEPTH = 0.05       # 5% — skip deep eclipsing binaries
MIN_DEPTH = 0.002      # 0.2% — skip noise
MAX_DURATION_HR = 3.0  # max transit duration
MIN_DURATION_HR = 10.0 / 60.0  # min 10 minutes

def fit_boxcar_transit(times, flux_norm, n_widths=20):
    """Slide boxcar, find deepest dip. Returns dict or None."""
    n = len(flux_norm)
    if n < 100:
        return None
    # Compute frame cadence and set width limits from duration limits
    cadence_hr = np.median(np.diff(times))
    min_width = max(5, int(MIN_DURATION_HR / cadence_hr))
    max_width = min(n // 3, int(MAX_DURATION_HR / cadence_hr))
    if min_width >= max_width:
        return None
    widths = np.unique(np.geomspace(min_width, max_width, n_widths).astype(int))
    
    mad = np.median(np.abs(flux_norm - 1.0)) * 1.4826
    if mad <= 0:
        mad = np.std(flux_norm)
    if mad <= 0:
        return None
    
    best_sig = 0
    best_result = None
    
    for w in widths:
        if w >= n - 10:
            continue
        running_mean = uniform_filter1d(flux_norm, size=w, mode='nearest')
        half = w // 2
        # Only consider positions where full boxcar fits inside the window
        start = w  # full box must start at index >= w//2, so center >= w
        end = n - w  # full box must end at index <= n - w//2, so center <= n - w
        if start >= end:
            continue
        dips = 1.0 - running_mean[start:end]
        sig = dips / (mad / np.sqrt(w))

        bi = np.argmax(sig)
        if sig[bi] > best_sig:
            best_sig = sig[bi]
            center = bi + start
            ts = center - w // 2
            te = center + w // 2
            
            out_mask = np.ones(n, dtype=bool)
            out_mask[ts:te] = False
            depth = np.median(flux_norm[out_mask]) - np.median(flux_norm[ts:te])
            duration_hr = times[min(te, n-1)] - times[ts]
            
            model = np.ones(n)
            model[ts:te] = 1.0 - depth
            
            best_result = {
                'depth': depth, 'duration_hr': duration_hr,
                'significance': best_sig, 'width_frames': w,
                'model': model,
            }
    
    return best_result


def process_h5(h5_path, target, segment, det):
    """Process all sources in one extraction HDF5."""
    f = h5py.File(h5_path, 'r')
    srcs = f['sources'][:]
    times = f['times'][:]
    n_src = len(srcs)
    
    candidates = []
    
    for i in range(n_src):
        flux = f['flux_clipped'][i, :]
        good = (flux != 0) & np.isfinite(flux)
        t, fl = times[good], flux[good]
        
        if len(t) < 100:
            continue
        
        med = np.median(fl)
        if med <= 0:
            continue
        flux_norm = fl / med
        
        fit = fit_boxcar_transit(t, flux_norm)
        if fit is None:
            continue
        
        if (fit['significance'] >= SIG_THRESHOLD and
                MIN_DEPTH <= fit['depth'] <= MAX_DEPTH and
                MIN_DURATION_HR <= fit['duration_hr'] <= MAX_DURATION_HR):
            # Check that binned lightcurve stays within plot range
            # (rejects sources with big outliers / bad baselines)
            n_pts = len(flux_norm)
            n_bins = max(1, n_pts // 9); n_use = n_bins * 9
            f_bin = flux_norm[:n_use].reshape(n_bins, 9).mean(axis=1)
            margin = max(0.03, fit['depth'] * 3)
            ylo, yhi = 1 - margin, 1 + margin * 0.7
            frac_in_range = np.mean((f_bin >= ylo) & (f_bin <= yhi))
            if frac_in_range < 0.85:
                continue

            candidates.append({
                'idx': i,
                'ra': float(srcs[i]['ra']),
                'dec': float(srcs[i]['dec']),
                'det_snr': float(srcs[i]['det_snr']),
                'times': t,
                'flux_norm': flux_norm,
                'flux_raw': f['flux'][i, good] / med,
                'fit': fit,
            })
    
    f.close()
    return candidates


def plot_candidate(cand, target, segment, det, out_dir):
    """Save a diagnostic PNG for one candidate."""
    fit = cand['fit']
    depth_pct = fit['depth'] * 100
    dur_min = fit['duration_hr'] * 60
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.scatter(cand['times'], cand['flux_raw'], s=0.5, c='black', alpha=0.3, rasterized=True)
    
    # Binned
    t, f = cand['times'], cand['flux_norm']
    n_bins = max(1, len(t) // 9); n_use = n_bins * 9
    t_bin = t[:n_use].reshape(n_bins, 9).mean(axis=1)
    f_bin = f[:n_use].reshape(n_bins, 9).mean(axis=1)
    ax.plot(t_bin, f_bin, color='#1a3a6b', lw=2)
    ax.plot(t, fit['model'], color='black', lw=2, ls='--')
    
    margin = max(0.03, fit['depth'] * 3)
    ax.set_ylim(1 - margin, 1 + margin * 0.7)
    ax.set_xlabel('Time (hours)', fontsize=9)
    ax.set_ylabel('Normalized Flux', fontsize=9)
    ax.set_title(f'Depth={depth_pct:.1f}%  Dur={dur_min:.0f}min  '
                 f'Sig={fit["significance"]:.0f}  SNR={cand["det_snr"]:.1f}  '
                 f'{det}  {target}/{segment}', fontsize=9)
    
    fig.tight_layout()
    fname = (f'depth{depth_pct:05.2f}_sig{fit["significance"]:05.1f}_'
             f'{det}_{target}_{segment}_'
             f'{cand["ra"]:.5f}_{cand["dec"]:.5f}.png')
    fig.savefig(os.path.join(out_dir, fname), dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fname


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    total_sources = 0
    total_candidates = 0
    
    h5_files = sorted(glob.glob(f'{BASE}/extraction/*/*/nrcb[1234]_ramp.h5'))
    
    for h5_path in h5_files:
        # Parse target/segment/det from path
        parts = h5_path.split('/')
        target = parts[-3]
        segment = parts[-2]
        det = parts[-1].replace('_ramp.h5', '')
        
        with h5py.File(h5_path, 'r') as f:
            n_src = f['sources'].shape[0]
        
        print(f'{target}/{segment}/{det}: {n_src} sources...', end=' ', flush=True)
        t1 = time.time()
        
        candidates = process_h5(h5_path, target, segment, det)
        
        elapsed = time.time() - t1
        print(f'{len(candidates)} candidates ({elapsed:.0f}s, {n_src/max(1,elapsed):.0f} src/s)')
        
        for cand in candidates:
            plot_candidate(cand, target, segment, det, OUT_DIR)
        
        total_sources += n_src
        total_candidates += len(candidates)
    
    elapsed_total = time.time() - t0
    print(f'\n=== DONE ===')
    print(f'Total sources: {total_sources}')
    print(f'Total candidates: {total_candidates}')
    print(f'Elapsed: {elapsed_total/60:.1f} min')
    print(f'Output: {OUT_DIR}/')
