#!/usr/bin/env python
"""Post-process catalog HDF5 files: apply IQR clipping to all stored
lightcurves so the HDF5 contains final cleaned data.

Clipped points are set to 0 (the existing convention for invalid/OOB data).
The catalog server can then serve lightcurves as-is without re-clipping.

Supports both:
  - master_variable_catalog.h5 (structure: {target}/lightcurves/{seg}/{mode}/{det})
  - real_variable_catalog.h5 (structure: {target}/{seg}/{table}/lightcurves/{seg}/{mode_ch}/{det})

Usage:
    python clean_catalog_lcs.py [--dry-run] [--catalog master|real|both]
"""
import numpy as np
import h5py
import time
import sys
import os

CATALOG_DIR = '/data/Globulars_Pipeline/catalogs'
MASTER_CATALOG = f'{CATALOG_DIR}/master_variable_catalog.h5'
REAL_CATALOG = f'{CATALOG_DIR}/real_variable_catalog.h5'

CHUNK_SIZE_RAMP = 18
CHUNK_SIZE_ZF = 4
IQR_FACTOR = 2.0


def clip_iqr_inplace(flux, chunk_size, iqr_factor):
    """IQR clip a 1D flux array. Returns number of points clipped.
    Sets clipped points to 0."""
    valid = np.isfinite(flux) & (flux != 0)
    if valid.sum() < chunk_size:
        return 0
    idx = np.where(valid)[0]
    fv = flux[idx]
    n = len(fv)
    n_chunks = n // chunk_size
    if n_chunks < 1:
        return 0
    clipped = 0
    for b in range(n_chunks):
        s, e = b * chunk_size, (b + 1) * chunk_size
        seg = fv[s:e]
        q1, q3 = np.percentile(seg, [25, 75])
        iqr = q3 - q1
        bad = (seg < q1 - iqr_factor * iqr) | (seg > q3 + iqr_factor * iqr)
        for j in np.where(bad)[0]:
            flux[idx[s + j]] = 0.0
            clipped += 1
    return clipped


def is_ramp(mode_str):
    """Determine if a mode string indicates ramp data (vs ZF)."""
    return 'ramp' in mode_str.lower()


def clean_dataset(ds, chunk_size, dry_run):
    """Apply IQR clipping to a 2D (n_src, n_frames) dataset."""
    n_src, n_frames = ds.shape
    n_clipped = 0
    n_valid = 0
    for i in range(n_src):
        flux = ds[i, :].astype(np.float64)
        nv = np.count_nonzero(flux)
        if nv < 10:
            continue
        n_valid += nv
        nc = clip_iqr_inplace(flux, chunk_size, IQR_FACTOR)
        n_clipped += nc
        if nc > 0 and not dry_run:
            ds[i, :] = flux.astype(np.float32)
    return n_clipped, n_valid


def clean_master(dry_run):
    """Clean master_variable_catalog.h5."""
    if not os.path.exists(MASTER_CATALOG):
        print(f'  Master catalog not found: {MASTER_CATALOG}')
        return 0, 0

    print(f'\n=== Master catalog ===')
    h5 = h5py.File(MASTER_CATALOG, 'r+' if not dry_run else 'r')
    total_clipped = 0
    total_points = 0

    for target in ['Liller1', 'Terzan5']:
        if target not in h5:
            continue
        lc_grp = h5[f'{target}/lightcurves']
        for seg in lc_grp:
            for mode in lc_grp[seg]:
                chunk_size = CHUNK_SIZE_RAMP if is_ramp(mode) else CHUNK_SIZE_ZF
                for det in lc_grp[seg][mode]:
                    ds = lc_grp[f'{seg}/{mode}/{det}']
                    nc, nv = clean_dataset(ds, chunk_size, dry_run)
                    total_clipped += nc
                    total_points += nv
                    pct = nc / nv * 100 if nv > 0 else 0
                    print(f'  {target}/{seg}/{mode}/{det}: '
                          f'{ds.shape[0]} src, {nc}/{nv} clipped ({pct:.1f}%)')

    h5.close()
    return total_clipped, total_points


def clean_real(dry_run):
    """Clean real_variable_catalog.h5."""
    if not os.path.exists(REAL_CATALOG):
        print(f'  Real catalog not found: {REAL_CATALOG}')
        return 0, 0

    print(f'\n=== Real catalog ===')
    h5 = h5py.File(REAL_CATALOG, 'r+' if not dry_run else 'r')
    total_clipped = 0
    total_points = 0

    for target in ['Liller1', 'Terzan5']:
        if target not in h5:
            continue
        for seg in h5[target]:
            seg_grp = h5[f'{target}/{seg}']
            for table in seg_grp:
                tbl_grp = seg_grp[table]
                if 'lightcurves' not in tbl_grp:
                    continue
                lc_grp = tbl_grp['lightcurves']
                for lc_seg in lc_grp:
                    for mode_ch in lc_grp[lc_seg]:
                        chunk_size = CHUNK_SIZE_RAMP if is_ramp(mode_ch) else CHUNK_SIZE_ZF
                        for det in lc_grp[f'{lc_seg}/{mode_ch}']:
                            ds = lc_grp[f'{lc_seg}/{mode_ch}/{det}']
                            if len(ds.shape) != 2:
                                continue
                            nc, nv = clean_dataset(ds, chunk_size, dry_run)
                            total_clipped += nc
                            total_points += nv
                            pct = nc / nv * 100 if nv > 0 else 0
                            print(f'  {target}/{seg}/{table}/{lc_seg}/{mode_ch}/{det}: '
                                  f'{ds.shape[0]} src, {nc}/{nv} clipped ({pct:.1f}%)')

    h5.close()
    return total_clipped, total_points


def main():
    dry_run = '--dry-run' in sys.argv
    which = 'both'
    for arg in sys.argv[1:]:
        if arg.startswith('--catalog='):
            which = arg.split('=')[1]
        elif arg in ('master', 'real', 'both') and not arg.startswith('-'):
            which = arg

    t0 = time.time()
    tc, tp = 0, 0

    if which in ('master', 'both'):
        c, p = clean_master(dry_run)
        tc += c; tp += p

    if which in ('real', 'both'):
        c, p = clean_real(dry_run)
        tc += c; tp += p

    pct = tc / tp * 100 if tp > 0 else 0
    action = 'Would clip' if dry_run else 'Clipped'
    print(f'\n{action} {tc}/{tp} points ({pct:.1f}%) in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
