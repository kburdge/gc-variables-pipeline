#!/usr/bin/env python
"""
Build master_source_mapping.json from the manually curated cross_match folders.

Ground truth:
  - Each obj* folder = spatial group
  - If no subfolders: all PNGs = one source
  - If numbered subfolders (1, 2, 3...): each subfolder = one unique source
  - bad/ folder = unreliable detections, excluded
  - dedup/ folder = near bright stars, excluded

For each unique source, the position is from the highest-SNR detection's filename.
Pixel positions are looked up from the extraction H5 files.

Usage:
    python build_mapping_from_crossmatch.py
"""
import os
import re
import json
import numpy as np
import h5py
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CROSS_DIR = f'{BASE}/diagnostics/cross_match'
EXTRACT_DIR = f'{BASE}/extraction'
OUTPUT = f'{BASE}/catalogs/master_source_mapping.json'

TARGETS = {'Liller1': ['Segment3', 'Segment4'], 'Terzan5': ['Segment2']}
ALL_DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']


def parse_diag_filename(fn):
    """Parse folder_SNR...png filename into components."""
    # Format: {folder}_SNR{snr}_src{id}_{det}_P{period}min_LS{ls}_amp{amp}_{ra}_{dec}.png
    m = re.match(
        r'(.+?)_SNR(\d+\.\d+)_src(\d+)_(nrc\w+)_P(\d+)min_LS(\d+)_amp([\d.]+)_'
        r'([\d.]+)_([-\d.]+)\.png$', fn)
    if not m:
        return None
    return {
        'diag_folder': m.group(1),
        'snr': float(m.group(2)),
        'src_id': int(m.group(3)),
        'det': m.group(4),
        'period': float(m.group(5)),
        'ls_sig': float(m.group(6)),
        'amplitude': float(m.group(7)),
        'ra': float(m.group(8)),
        'dec': float(m.group(9)),
    }


def infer_target(folder_name):
    if 'Terzan5' in folder_name:
        return 'Terzan5'
    return 'Liller1'


def main():
    # Load extraction data for pixel lookup
    ext_data = {}
    for target, segs in TARGETS.items():
        for seg in segs:
            for det in ALL_DETS:
                for mode in ['ramp', 'zf']:
                    path = f'{EXTRACT_DIR}/{target}/{seg}/{det}_{mode}.h5'
                    if os.path.exists(path):
                        with h5py.File(path, 'r') as f:
                            ext_data[(target, seg, det, mode)] = f['sources'][:]

    # Collect all unique sources from cross_match folders
    sources = []

    for obj_dir_name in sorted(os.listdir(CROSS_DIR)):
        if not obj_dir_name.startswith('obj'):
            continue

        obj_path = os.path.join(CROSS_DIR, obj_dir_name)
        if not os.path.isdir(obj_path):
            continue

        contents = os.listdir(obj_path)
        subdirs = sorted([c for c in contents if os.path.isdir(os.path.join(obj_path, c))])
        top_pngs = [c for c in contents if c.endswith('.png') and c != 'summary.png']

        if subdirs:
            # Each subfolder is a unique source
            for subdir in subdirs:
                sub_path = os.path.join(obj_path, subdir)
                pngs = [f for f in os.listdir(sub_path) if f.endswith('.png')]
                if pngs:
                    sources.append({
                        'obj_folder': obj_dir_name,
                        'sub_folder': subdir,
                        'pngs': pngs,
                    })
        else:
            # All top-level PNGs are one source
            if top_pngs:
                sources.append({
                    'obj_folder': obj_dir_name,
                    'sub_folder': None,
                    'pngs': top_pngs,
                })

    print(f'Found {len(sources)} unique sources from cross_match folders')

    # Build mapping entries
    mapping = []

    for src in sources:
        detections = {}
        best_snr = 0
        best_ra = 0
        best_dec = 0
        target = None

        for fn in src['pngs']:
            info = parse_diag_filename(fn)
            if info is None:
                continue

            if target is None:
                target = infer_target(info['diag_folder'])

            # Find pixel position from extraction H5
            det = info['det']
            fn_ra = info['ra']
            fn_dec = info['dec']
            fn_ls = info['ls_sig']

            # Determine segment and mode from diag_folder
            diag_folder = info['diag_folder']
            mode = 'zf' if '_zf_' in diag_folder or diag_folder.endswith('_zf') else 'ramp'
            seg = None
            for s in TARGETS.get(target, []):
                if s in diag_folder:
                    seg = s
                    break
            if seg is None:
                seg = TARGETS[target][0]  # default

            # Look up pixel from extraction
            px, py = 0, 0
            ext_key = (target, seg, det, mode)
            if ext_key not in ext_data:
                # Try ramp if zf not found
                ext_key = (target, seg, det, 'ramp')
            if ext_key in ext_data:
                ext = ext_data[ext_key]
                dist = np.sqrt((ext['ra'] - fn_ra)**2 + (ext['dec'] - fn_dec)**2)
                candidates = np.where(dist < 0.001)[0]
                if len(candidates) == 0:
                    candidates = [np.argmin(dist)]
                best_ext = candidates[0]
                ls_field = 'ls_significance' if 'ls_significance' in ext.dtype.names else 'ls_sig'
                for c in candidates:
                    if abs(ext[c][ls_field] - fn_ls) < 1:
                        best_ext = c
                        break
                px = int(ext[best_ext]['px'])
                py = int(ext[best_ext]['py'])

            detections[diag_folder] = {
                'snr': info['snr'],
                'filename': fn,
                'det': det,
                'segment': seg,
                'px': px,
                'py': py,
            }

            if info['snr'] > best_snr:
                best_snr = info['snr']
                best_ra = fn_ra
                best_dec = fn_dec

        if not detections or target is None:
            continue

        mapping.append({
            'target': target,
            'ra': best_ra,
            'dec': best_dec,
            'best_snr': best_snr,
            'n_folders': len(detections),
            'detections': detections,
            'obj_folder': src['obj_folder'],
            'sub_folder': src['sub_folder'],
        })

    # Sort by target then SNR
    mapping.sort(key=lambda m: (-{'Liller1': 0, 'Terzan5': 1}[m['target']], -m['best_snr']))

    # Assign master IDs
    for i, m in enumerate(mapping):
        m['master_id'] = i

    # Exclude bad entries
    bad_dir = os.path.join(CROSS_DIR, 'bad')
    bad_files = set()
    if os.path.isdir(bad_dir):
        bad_files = set(os.listdir(bad_dir))

    # Check if any source has ALL its detections in bad
    n_excluded = 0
    clean_mapping = []
    for m in mapping:
        all_bad = all(dv['filename'] in bad_files for dv in m['detections'].values())
        if all_bad:
            n_excluded += 1
        else:
            clean_mapping.append(m)

    # Re-assign IDs
    for i, m in enumerate(clean_mapping):
        m['master_id'] = i

    print(f'Excluded {n_excluded} bad sources')
    print(f'Final: {len(clean_mapping)} master entries')

    # Count by target
    by_target = {}
    for m in clean_mapping:
        by_target.setdefault(m['target'], 0)
        by_target[m['target']] += 1
    for t, n in by_target.items():
        print(f'  {t}: {n}')

    # Check obj0341
    for m in clean_mapping:
        if abs(m['ra'] - 267.02137) < 0.001 and abs(m['dec'] - (-24.77741)) < 0.001:
            print(f'\nobj0341 (17:48:05.13): master_id={m["master_id"]}, '
                  f'ra={m["ra"]:.5f}, dec={m["dec"]:.5f}')
            break
    else:
        print('\nWARNING: obj0341 NOT FOUND!')

    # Save
    with open(OUTPUT, 'w') as f:
        json.dump(clean_mapping, f, indent=2)
    print(f'\nSaved {OUTPUT}')


if __name__ == '__main__':
    main()
