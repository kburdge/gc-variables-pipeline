#!/usr/bin/env python
"""
Generate paper figure: side-by-side Segment 1 processing stages for two sources.

Layout: 6 rows x 2 columns
  Rows: (a) raw, (b) sat corr, (c) IQR clip, (d) slope corr, (e) stitched, (f) Seg2 ref
  Cols: Object 6 (left), Object 115 (right)

Usage:
    python paper_seg1_figure.py
"""
import os
import sys
import json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from paper_seg1_stages import (extract_source_stages, get_seg2_lc,
                                plot_exposure_blocks, clip_iqr)

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'

ROW_LABELS = [
    '(a) Raw group-difference photometry',
    '(b) After saturation correction',
    '(c) After per-exposure IQR clip',
    '(d) After integration slope correction',
    '(e) Normalized + slope-aware stitch',
    '(f) Segment 2 reference',
]
STAGE_KEYS = ['raw_hr', 'sat_corr_hr', 'iqr_clip', 'slope_corr']


def main():
    sources = [6, 115]

    h5 = h5py.File(CATALOG, 'r')
    srcs = h5['Terzan5/sources'][:]

    with open(f'{BASE}/catalogs/master_source_mapping.json') as f:
        mapping = json.load(f)
    mapping_by_mid = {m['master_id']: m for m in mapping}

    # Extract data for both sources
    all_stages = {}
    all_seg2 = {}
    all_det = {}
    for mid in sources:
        row = srcs[srcs['master_id'] == mid][0]
        det = row['detector'].decode()
        m = mapping_by_mid[mid]
        ra, dec = m['ra'], m['dec']
        ch = 'LW' if det == 'nrcblong' else 'SW'
        print(f'Extracting obj{mid} ({det})...')
        all_stages[mid] = extract_source_stages(mid, ra, dec, det)
        seg2_t, seg2_f = get_seg2_lc(h5, mid, ch)
        all_seg2[mid] = (seg2_t, seg2_f)
        all_det[mid] = det
    h5.close()

    # Build figure
    fig, axes = plt.subplots(6, 2, figsize=(16, 18))

    for col, mid in enumerate(sources):
        stages = all_stages[mid]
        seg2_t, seg2_f = all_seg2[mid]
        det = all_det[mid]
        ch = 'LW' if det == 'nrcblong' else 'SW'
        rejected = stages.get('rejected', set())

        n_exp = len(stages['raw_hr'])
        cmap = plt.cm.turbo
        exp_colors = [cmap(i / max(n_exp - 1, 1)) for i in range(n_exp)]

        # Shared x range for rows 0-3
        all_times = []
        for key in STAGE_KEYS:
            for b in stages[key]:
                if b is not None:
                    all_times.extend(b[0].tolist())
        if all_times:
            t_lo, t_hi = min(all_times), max(all_times)
            t_pad = (t_hi - t_lo) * 0.02
            shared_xlim = (t_lo - t_pad, t_hi + t_pad)
        else:
            shared_xlim = None

        # Rows 0-3: per-exposure stages
        for row_i, key in enumerate(STAGE_KEYS):
            ax = axes[row_i, col]
            plot_exposure_blocks(ax, stages[key], exp_colors, rejected,
                                s=0.3, alpha=1.0)
            if shared_xlim:
                ax.set_xlim(shared_xlim)
            ax.tick_params(labelsize=7)
            if row_i < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (hr)', fontsize=9)
            if col == 0:
                ax.set_ylabel('Flux (ADU)', fontsize=9)

        # Row 4: stitched
        ax = axes[4, col]
        t_s, f_s = stages['stitched']
        ax.scatter(t_s, f_s, s=0.3, c='k', alpha=1.0, rasterized=True)
        ax.set_xlabel('Time (hr)', fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel('Normalized Flux', fontsize=9)

        # Row 5: Seg2 reference
        ax = axes[5, col]
        if seg2_t is not None:
            ax.scatter(seg2_t, seg2_f, s=0.3, c='k', alpha=1.0, rasterized=True)
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='gray')
        ax.set_xlabel('Time (hr)', fontsize=9)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel('Normalized Flux', fontsize=9)

        # Column title
        axes[0, col].set_title(f'Object {mid} ({ch}, {det})', fontsize=12,
                               fontweight='bold', pad=8)

    # Row labels as left-side text
    for row_i, label in enumerate(ROW_LABELS):
        # Place label inside the left panel, top-left corner
        axes[row_i, 0].text(0.02, 0.96, label, transform=axes[row_i, 0].transAxes,
                            fontsize=8, fontweight='bold', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor='none', alpha=0.8))
        # Mirror in right panel
        axes[row_i, 1].text(0.02, 0.96, label, transform=axes[row_i, 1].transAxes,
                            fontsize=8, fontweight='bold', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor='none', alpha=0.8))

    fig.suptitle('Terzan 5 Segment 1: Dithered Extraction Pipeline',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    out_path = f'{BASE}/diagnostics/paper_seg1_cleaning_figure.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    out_pdf = out_path.replace('.png', '.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved: {out_path}')
    print(f'Saved: {out_pdf}')


if __name__ == '__main__':
    main()
