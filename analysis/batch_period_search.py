#!/usr/bin/env python
"""
Batch period search for all sources: LS + alias phase folds.

For each source, combines all available segments (amplitude-matched),
runs Lomb-Scargle with proper frequency grid, and generates alias
comparison plots.

Usage:
    python batch_period_search.py [--target Terzan5] [--source 6]
"""
import os, sys, argparse
import numpy as np
import h5py
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
OUT_DIR = f'{BASE}/diagnostics/period_search'
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS = {
    'Terzan5': {'segments': ['Segment1', 'Segment2']},
    'Liller1': {'segments': ['Segment3', 'Segment4']},
}


def get_seg_lc(h5, target, mid, seg, ch, mjd_refs):
    """Get a single segment lightcurve in MJD."""
    bs_tbl = h5[f'best_stage/{target}'][:]
    bs_row = bs_tbl[bs_tbl['master_id'] == mid]
    if len(bs_row) == 0:
        return None, None
    bs_row = bs_row[0]
    col = f'{seg}_{ch}'
    if col not in bs_tbl.dtype.names:
        return None, None
    stage = bs_row[col].decode()
    if stage == 'special':
        stage = 'special_reduction'
    path = f'{stage}/{target}/{mid}/{seg}_{ch}'
    if path not in h5:
        path = f'groupdiff/{target}/{mid}/{seg}_{ch}'
        if path not in h5:
            return None, None
    t_hr = h5[path]['times'][:]
    f_n = h5[path]['flux_norm'][:]
    mjd_ref = h5[path].attrs.get('mjd_ref', None)
    if mjd_ref is None:
        mjd_ref = mjd_refs.get((target, seg), None)
    if mjd_ref is None:
        return None, None
    mjd = mjd_ref + t_hr / 24.0
    so = np.argsort(mjd)
    return mjd[so], f_n[so]


def bin_lc(t, f, bs=9):
    n = len(t) // bs
    if n < 1:
        return t, f
    return np.array([np.mean(t[i*bs:(i+1)*bs]) for i in range(n)]), \
           np.array([np.median(f[i*bs:(i+1)*bs]) for i in range(n)])


def amplitude_match_and_combine(seg_data):
    """Amplitude-match segments and combine.
    seg_data: list of (t, f) tuples. Uses last segment as reference."""
    valid = [(t, f) for t, f in seg_data if t is not None and len(t) > 20]
    if len(valid) == 0:
        return None, None
    if len(valid) == 1:
        return valid[0]

    # Use last segment as amplitude reference
    t_ref, f_ref = valid[-1]
    tb_ref, fb_ref = bin_lc(t_ref, f_ref, bs=9)
    pp_ref = fb_ref.max() - fb_ref.min()
    med_ref = np.median(f_ref)

    all_t, all_f = [], []
    for i, (t_s, f_s) in enumerate(valid):
        if i == len(valid) - 1:
            # Reference segment — no scaling
            all_t.extend(t_s.tolist())
            all_f.extend(f_s.tolist())
        else:
            tb_s, fb_s = bin_lc(t_s, f_s, bs=9)
            pp_s = fb_s.max() - fb_s.min()
            if pp_s > 0.001:
                scale = pp_ref / pp_s
            else:
                scale = 1.0
            f_scaled = (f_s - np.median(f_s)) * scale + med_ref
            all_t.extend(t_s.tolist())
            all_f.extend(f_scaled.tolist())

    t = np.array(all_t)
    f = np.array(all_f)
    so = np.argsort(t)
    return t[so], f[so]


def run_period_search(t, f):
    """Run LS with proper frequency grid. Returns freqs, power, periods_min."""
    t_min = (t - t[0]) * 24 * 60
    baseline = t_min[-1] - t_min[0]

    df = 1.0 / (baseline * 3.0)
    fmin = 2.0 / baseline
    fmax = 1.0 / 20.0  # 20 min
    nf = int(np.ceil((fmax - fmin) / df))
    if nf < 10:
        return None, None, None, baseline
    freqs = np.linspace(fmin, fmax, nf)
    power = LombScargle(t_min, f).power(freqs)
    periods_min = 1.0 / freqs
    return freqs, power, periods_min, baseline


def find_aliases(power, periods_min, best_period, window=60):
    """Find alias peaks near best period."""
    mask = (periods_min > best_period - window) & (periods_min < best_period + window)
    if mask.sum() < 10:
        return [], []
    local_power = power[mask]
    local_periods = periods_min[mask]
    pks, _ = find_peaks(local_power, height=0.1 * local_power.max(), distance=5)
    if len(pks) == 0:
        return [], []
    pk_periods = local_periods[pks]
    pk_powers = local_power[pks]
    order = np.argsort(pk_periods)
    return pk_periods[order], pk_powers[order]


def make_alias_plot(target, mid, ch, seg_times, seg_fluxes, seg_labels,
                    t_combined, f_combined, aliases, out_path):
    """Make alias comparison plot with segment color coding + 2P fold.

    seg_times/seg_fluxes/seg_labels: per-segment data for color coding.
    t_combined/f_combined: amplitude-matched combined data.
    aliases: list of (period_min, ls_power).
    """
    n_al = len(aliases)
    if n_al == 0:
        return
    # n_al alias rows + 1 row for 2P fold of best
    n_rows = n_al + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 2.5 * n_rows), squeeze=False)

    seg_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    best_power = max(pw for _, pw in aliases)
    best_period_min = aliases[np.argmax([pw for _, pw in aliases])][0]

    for ai, (period_min, pw) in enumerate(aliases):
        period_day = period_min / (24 * 60)
        is_best = pw >= best_power - 0.001

        ax = axes[ai, 0]
        # Color by segment
        for si, (ts, fs, label) in enumerate(zip(seg_times, seg_fluxes, seg_labels)):
            if ts is None or len(ts) == 0:
                continue
            phase = ((ts - t_combined[0]) / period_day) % 1.0
            c = seg_colors[si % len(seg_colors)]
            ax.scatter(phase, fs, s=0.3, alpha=0.15, c=c)
            ax.scatter(phase + 1, fs, s=0.3, alpha=0.15, c=c)

        # Binned from combined
        phase_all = ((t_combined - t_combined[0]) / period_day) % 1.0
        bins = np.linspace(0, 1, 50)
        for wrap in [0, 1]:
            for i in range(len(bins)-1):
                m = (phase_all >= bins[i]) & (phase_all < bins[i+1])
                if m.sum() > 3:
                    ax.plot([(bins[i]+bins[i+1])/2 + wrap],
                           [np.median(f_combined[m])], 'o', color='k', ms=3)

        ax.set_xlim(0, 2)
        ax.set_ylabel('Flux', fontsize=8)
        ax.tick_params(labelsize=7)

        tag = 'BEST' if is_best else ''
        ax.text(0.02, 0.95,
                f'P={period_min:.2f} min = {period_min/60:.4f} hr\nLS={pw:.3f} {tag}',
                transform=ax.transAxes, fontsize=7, va='top',
                fontweight='bold' if is_best else 'normal',
                bbox=dict(boxstyle='round',
                          facecolor='yellow' if is_best else 'white', alpha=0.8))

    # Last row: fold at 2 * best period (ellipsoidal check)
    ax = axes[n_al, 0]
    period_2p = best_period_min * 2
    period_day_2p = period_2p / (24 * 60)
    for si, (ts, fs, label) in enumerate(zip(seg_times, seg_fluxes, seg_labels)):
        if ts is None or len(ts) == 0:
            continue
        phase = ((ts - t_combined[0]) / period_day_2p) % 1.0
        c = seg_colors[si % len(seg_colors)]
        ax.scatter(phase, fs, s=0.3, alpha=0.15, c=c, label=label if si < 4 else None)
        ax.scatter(phase + 1, fs, s=0.3, alpha=0.15, c=c)
    phase_all = ((t_combined - t_combined[0]) / period_day_2p) % 1.0
    bins = np.linspace(0, 1, 80)
    for wrap in [0, 1]:
        for i in range(len(bins)-1):
            m = (phase_all >= bins[i]) & (phase_all < bins[i+1])
            if m.sum() > 3:
                ax.plot([(bins[i]+bins[i+1])/2 + wrap],
                       [np.median(f_combined[m])], 'o', color='k', ms=3)
    ax.set_xlim(0, 2)
    ax.set_xlabel('Phase', fontsize=9)
    ax.set_ylabel('Flux', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.text(0.02, 0.95,
            f'2P={period_2p:.2f} min = {period_2p/60:.4f} hr\n(ellipsoidal check)',
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='#e0ffe0', alpha=0.8))

    fig.suptitle(f'{target} obj{mid} {ch}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default=None)
    parser.add_argument('--source', type=int, default=None)
    args = parser.parse_args()

    h5 = h5py.File(CATALOG, 'r')

    # Build MJD refs from raw time arrays
    mjd_refs = {}
    for target in TARGETS:
        if target not in h5:
            continue
        for seg in TARGETS[target]['segments']:
            for det in ['nrcb4', 'nrcb1', 'nrcblong']:
                path = f'{target}/times/{seg}/ramp/{det}'
                if path in h5:
                    mjd_refs[(target, seg)] = float(h5[path][0])
                    break
        # Also check root attrs
        for seg in TARGETS[target]['segments']:
            key = f'mjd_ref_{target}_{seg}'
            if key in h5.attrs:
                mjd_refs[(target, seg)] = float(h5.attrs[key])

    print(f'MJD refs: {mjd_refs}')

    targets_to_run = [args.target] if args.target else list(TARGETS.keys())
    n_done = 0

    for target in targets_to_run:
        if target not in h5:
            continue
        srcs = h5[f'{target}/sources'][:]
        segments = TARGETS[target]['segments']

        target_dir = f'{OUT_DIR}/{target}'
        os.makedirs(target_dir, exist_ok=True)

        for src in srcs:
            mid = int(src['master_id'])
            if args.source is not None and mid != args.source:
                continue

            # Prefer SW; fall back to LW only if no SW data
            ch_to_use = None
            for ch_try in ['SW', 'LW']:
                has_data = False
                for seg in segments:
                    t_s, f_s = get_seg_lc(h5, target, mid, seg, ch_try, mjd_refs)
                    if t_s is not None and len(t_s) > 20:
                        has_data = True
                        break
                if has_data:
                    ch_to_use = ch_try
                    break
            if ch_to_use is None:
                continue

            ch = ch_to_use

            # Get all segments (keeping per-segment data for color coding)
            seg_times = []
            seg_fluxes = []
            seg_labels = []
            seg_data = []
            for seg in segments:
                t_s, f_s = get_seg_lc(h5, target, mid, seg, ch, mjd_refs)
                seg_data.append((t_s, f_s))
                seg_times.append(t_s)
                seg_fluxes.append(f_s)
                seg_labels.append(seg.replace('Segment', 'S'))

            # Amplitude match and combine
            t, f = amplitude_match_and_combine(seg_data)
            if t is None or len(t) < 50:
                continue

            # Also amplitude-match the per-segment arrays for plotting
            valid_seg = [(i, ts, fs) for i, (ts, fs) in enumerate(seg_data)
                         if ts is not None and len(ts) > 20]
            if len(valid_seg) > 1:
                ref_idx = valid_seg[-1][0]
                ref_t, ref_f = seg_data[ref_idx]
                tb_ref, fb_ref = bin_lc(ref_t, ref_f, bs=9)
                pp_ref = fb_ref.max() - fb_ref.min()
                med_ref = np.median(ref_f)
                for vi, ts, fs in valid_seg[:-1]:
                    tb_s, fb_s = bin_lc(ts, fs, bs=9)
                    pp_s = fb_s.max() - fb_s.min()
                    scale = pp_ref / pp_s if pp_s > 0.001 else 1.0
                    seg_fluxes[vi] = (fs - np.median(fs)) * scale + med_ref

            # Period search
            freqs, power, periods_min, baseline = run_period_search(t, f)
            if freqs is None:
                continue

            best_idx = np.argmax(power)
            best_period = periods_min[best_idx]
            best_power_val = power[best_idx]

            if best_power_val < 0.05:
                continue

            # Find aliases
            pk_periods, pk_powers = find_aliases(power, periods_min, best_period)
            if len(pk_periods) == 0:
                continue

            # Pick best + 3 on each side
            best_sorted = np.argmin(np.abs(pk_periods - best_period))
            start = max(0, best_sorted - 3)
            end = min(len(pk_periods), best_sorted + 4)
            aliases = list(zip(pk_periods[start:end], pk_powers[start:end]))

            # Make plot
            out_path = f'{target_dir}/obj{mid:04d}_{ch}.png'
            make_alias_plot(target, mid, ch, seg_times, seg_fluxes, seg_labels,
                           t, f, aliases, out_path)

            n_done += 1
            if n_done % 50 == 0:
                print(f'  {n_done} done...', flush=True)

    h5.close()
    print(f'\nDone: {n_done} plots in {OUT_DIR}/')


if __name__ == '__main__':
    main()
