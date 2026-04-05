#!/usr/bin/env python
"""
Generate classification thumbnails for all sources in the master catalog.

Each thumbnail shows all available lightcurves at their best correction stage,
similar to the website display. Output goes to diagnostics/classification/.

Usage:
    python build_classification_thumbnails.py [--target Liller1] [--source 623]
"""
import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
OUT_DIR = f'{BASE}/diagnostics/classification'


def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    for b in range(nc):
        s, e = b*cs, (b+1)*cs; sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75]); r = q3 - q1
        m[s:e] = (sg >= q1 - iq*r) & (sg <= q3 + iq*r)
    return f[m], t[m]


def bin_lc(t, f, bs=9):
    n = len(t) // bs
    if n < 1: return t.tolist(), f.tolist()
    tb = [float(np.mean(t[i*bs:(i+1)*bs])) for i in range(n)]
    fb = [float(np.median(f[i*bs:(i+1)*bs])) for i in range(n)]
    return tb, fb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default=None)
    parser.add_argument('--source', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    h5 = h5py.File(CATALOG, 'r')

    targets = [args.target] if args.target else ['Liller1', 'Terzan5']

    n_plotted = 0
    for target in targets:
        if target not in h5:
            continue
        srcs = h5[f'{target}/sources'][:]
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        # Load best_stage table
        best_stages = {}
        if 'best_stage' in h5 and target in h5['best_stage']:
            bs_tbl = h5[f'best_stage/{target}'][:]
            for row in bs_tbl:
                mid = int(row['master_id'])
                for col in bs_tbl.dtype.names:
                    if col == 'master_id':
                        continue
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2:
                        best_stages[(mid, parts[0], parts[1])] = row[col].decode()

        for src in srcs:
            mid = int(src['master_id'])
            if args.source is not None and mid != args.source:
                continue
            ra = float(src['ra'])
            dec = float(src['dec'])
            snr = float(src['best_snr'])
            row_idx = int(np.where(srcs['master_id'] == mid)[0][0])

            # Collect all lightcurves to plot
            lcs = {}  # label -> (t, f, color, is_binned)

            # Best ramp LCs from correction stages
            for seg in segments:
                seg_short = seg.replace('Segment', 'S')
                for ch in ['SW', 'LW']:
                    stage = best_stages.get((mid, seg, ch), 'groupdiff')
                    if stage == 'special':
                        stage_group = 'special_reduction'
                    else:
                        stage_group = stage

                    path = f'{stage_group}/{target}/{mid}/{seg}_{ch}'
                    if path in h5:
                        t = h5[path]['times'][:].astype(np.float64)
                        f = h5[path]['flux_norm'][:].astype(np.float64)
                        if len(t) > 10:
                            det_label = 'nrcblong' if ch == 'LW' else 'SW'
                            stage_short = stage.replace('_corrected', '').replace('_', '+')
                            label = f'{seg_short} ramp {det_label} ({stage_short})'
                            color = '#d62728' if ch == 'SW' else '#1f77b4'
                            lcs[label] = (t, f, color)

            # ZF lightcurves from raw table
            lc_grp = h5[f'{target}/lightcurves']
            time_grp = h5[f'{target}/times']
            for seg in segments:
                if seg not in lc_grp:
                    continue
                if 'zf' not in lc_grp[seg]:
                    continue
                seg_short = seg.replace('Segment', 'S')
                for det in lc_grp[seg]['zf']:
                    ch = 'LW' if det == 'nrcblong' else 'SW'
                    flux = lc_grp[f'{seg}/zf/{det}'][row_idx, :].astype(np.float64)
                    times = time_grp[f'{seg}/zf/{det}'][:].astype(np.float64)
                    valid = np.isfinite(flux) & (flux != 0)
                    if valid.sum() < 5:
                        continue
                    fv, tv = flux[valid], times[valid]
                    tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
                    fv_c, tv_c = clip_iqr(fv, tv_hr, cs=4)
                    fv_c, tv_c = clip_iqr(fv_c, tv_c, cs=4)
                    if len(fv_c) < 5:
                        continue
                    fn = fv_c / np.median(fv_c)
                    label = f'{seg_short} ZF {det}'
                    color = '#9467bd' if ch == 'SW' else '#17becf'
                    lcs[label] = (tv_c, fn, color)

            if not lcs:
                continue

            # Sort: ramp before ZF, SW before LW, Seg3 before Seg4
            def sort_key(label):
                priority = 0
                if 'ZF' in label:
                    priority += 10
                if 'LW' in label or 'nrcblong' in label:
                    priority += 1
                if 'S4' in label or 'S2' in label:
                    priority += 0  # keep segment order
                return (priority, label)

            sorted_labels = sorted(lcs.keys(), key=sort_key)

            # Layout: 2 columns (SW left, LW right), rows by seg+mode
            sw_labels = [l for l in sorted_labels if 'nrcblong' not in l and 'LW' not in l.split('(')[0]]
            lw_labels = [l for l in sorted_labels if 'nrcblong' in l or ('LW' in l.split('(')[0] and 'nrcblong' not in l)]
            # Fix: just split by channel
            sw_labels = [l for l in sorted_labels if 'nrcblong' not in l]
            lw_labels = [l for l in sorted_labels if 'nrcblong' in l]

            n_rows = max(len(sw_labels), len(lw_labels), 1)
            fig, axes = plt.subplots(n_rows, 2, figsize=(14, 2.5 * n_rows), squeeze=False)

            for col, labels in enumerate([sw_labels, lw_labels]):
                for ri, label in enumerate(labels):
                    ax = axes[ri, col]
                    t, f, color = lcs[label]
                    ax.scatter(t, f, s=0.5, c=color, alpha=0.3, rasterized=True)
                    # Binned overlay
                    bs = 4 if 'ZF' in label else 9
                    if len(t) > bs * 2:
                        tb, fb = bin_lc(t, f, bs=bs)
                        ax.plot(tb, fb, '-', color=color, lw=1, alpha=0.8)
                    std = np.std(f)
                    amp_pct = std * 100
                    ax.set_title(f'{label}  (n={len(f)}, amp={amp_pct:.1f}%)', fontsize=8)
                    ax.tick_params(labelsize=7)
                    if ri == len(labels) - 1:
                        ax.set_xlabel('Time (hr)', fontsize=8)

                # Blank remaining rows
                for ri in range(len(labels), n_rows):
                    axes[ri, col].set_visible(False)

            # Coordinate string
            ra_h = int(ra / 15)
            ra_m = int((ra / 15 - ra_h) * 60)
            ra_s = ((ra / 15 - ra_h) * 60 - ra_m) * 60
            dec_sign = '-' if dec < 0 else '+'
            dec_abs = abs(dec)
            dec_d = int(dec_abs)
            dec_m = int((dec_abs - dec_d) * 60)
            dec_sec = ((dec_abs - dec_d) * 60 - dec_m) * 60
            coord_str = f'{ra_h:02d}:{ra_m:02d}:{ra_s:05.2f} {dec_sign}{dec_d:02d}:{dec_m:02d}:{dec_sec:04.1f}'

            fig.suptitle(f'{target}  obj{mid:04d}  SNR={snr:.1f}  ({coord_str})',
                         fontsize=11, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            out_path = f'{OUT_DIR}/{target}_obj{mid:04d}_snr{snr:05.1f}.png'
            fig.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

            n_plotted += 1
            if n_plotted % 100 == 0:
                print(f'  {n_plotted} plotted...', flush=True)

    h5.close()
    print(f'Done: {n_plotted} thumbnails in {OUT_DIR}/')


if __name__ == '__main__':
    main()
