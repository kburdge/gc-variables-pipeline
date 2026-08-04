#!/usr/bin/env python
"""
Extract dithered Segment1 lightcurves for all Terzan5 sources and add to master catalog.

For each exposure:
  1. Read uncal ramps, compute group differences
  2. Use calints WCS to find each source's pixel position
  3. Extract aperture photometry
Then stitch exposures using slope-continuity matching and apply standard cleaning.

Usage:
    python add_segment1_terzan5.py [--dry-run]
"""
import os
import sys
import glob
import time
import json
import numpy as np
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAperture
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
SEG1_DIR = '/data/JWST/Terzan5/Segment1'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING = f'{BASE}/catalogs/master_source_mapping.json'

H = 2
ap = CircularAperture([(H, H)], r=1.5)
ap_mask = ap.to_mask(method='exact')[0].to_image((2*H+1, 2*H+1)).astype(np.float32)
AP_AREA = float(np.sum(ap_mask))
TGROUP_HR = 21.47354 / 3600.0
BKG_ANNULUS = {'SW': (10, 16), 'LW': (5, 8)}


GATE_THRESH = 0.9  # require 10% improvement for correction stages


def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    # If there's a remainder, merge it with the last chunk
    last_chunk = nc - 1 if n % cs != 0 and nc > 0 else nc
    for b in range(nc):
        s = b * cs
        e = (b+1)*cs if b < last_chunk else n  # extend last chunk to include remainder
        sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75]); r = q3 - q1
        m[s:e] = (sg >= q1 - iq*r) & (sg <= q3 + iq*r)
    return f[m], t[m]


def apply_bkg_rescale(fn, med_raw, bkg_offset):
    source_med = med_raw - bkg_offset
    if source_med <= 0 or bkg_offset <= 0:
        return fn
    return (fn * med_raw - bkg_offset) / source_med


def find_integrations(t):
    if len(t) < 3: return []
    dt = np.diff(t)
    med_dt = np.median(dt[dt > 0])
    breaks = np.where(dt > 2 * med_dt)[0] + 1
    boundaries = np.concatenate([[0], breaks, [len(t)]])
    return [(int(boundaries[i]), int(boundaries[i+1]))
            for i in range(len(boundaries) - 1)
            if boundaries[i+1] - boundaries[i] >= 3]


def integration_scatter(t, f):
    ints = find_integrations(t)
    if len(ints) < 3: return np.inf
    return np.mean([np.std(f[s:e]) for s, e in ints])


def apply_slope_correction(t, f):
    ints = find_integrations(t)
    if len(ints) < 10: return None
    slopes = []; med_fluxes = []
    for s, e in ints:
        x = np.arange(e - s, dtype=float)
        slopes.append(np.polyfit(x, f[s:e], 1)[0])
        med_fluxes.append(np.median(f[s:e]))
    slopes = np.array(slopes); med_fluxes = np.array(med_fluxes)
    try:
        poly = np.polyfit(med_fluxes, slopes, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None
    pred_slopes = np.polyval(poly, med_fluxes)
    f_corr = f.copy()
    for i, (s, e) in enumerate(ints):
        x = np.arange(e - s, dtype=float)
        f_corr[s:e] -= pred_slopes[i] * (x - x.mean())
    return f_corr / np.median(f_corr)


def correct_integration_slopes_raw(t, f):
    """Remove intra-integration saturation ramps in raw ADU space.
    Same as apply_slope_correction but stays in raw ADU (no normalization)."""
    ints = find_integrations(t)
    if len(ints) < 3:
        return f.copy()
    slopes = []; med_fluxes = []
    for s, e in ints:
        x = np.arange(e - s, dtype=float)
        slopes.append(np.polyfit(x, f[s:e], 1)[0])
        med_fluxes.append(np.median(f[s:e]))
    slopes = np.array(slopes); med_fluxes = np.array(med_fluxes)
    try:
        poly = np.polyfit(med_fluxes, slopes, 2)
    except:
        return f.copy()
    pred_slopes = np.polyval(poly, med_fluxes)
    f_corr = f.copy()
    for i, (s, e) in enumerate(ints):
        x = np.arange(e - s, dtype=float)
        f_corr[s:e] -= pred_slopes[i] * (x - x.mean())
    return f_corr  # raw ADU, no normalization


def slope_stitch(blocks):
    """Curvature-aware stitching in raw ADU space (v3, 2026-07-16).

    Fits a QUADRATIC to each observation block (centered time for numerical
    conditioning) and predicts the expected flux change across the gap using
    the blocks' ENDPOINT values and ENDPOINT derivatives. The previous linear
    version (_slope_stitch_v2_linear) mis-set the joins near light-curve
    extrema, where the slope changes sign within a block: a whole-block linear
    fit biases the endpoint estimates and the extrapolated slope, producing
    jumps at the joins nearest minima. Validated on 30 PHOEBE-modeled sources:
    median stitch-offset error 11.1 -> 7.2 ppt (synthetic, known offsets);
    on real Seg1 data chi2/dof vs fixed models improves for 5/6 test sources
    (e.g. #29: 3.21 -> 2.59, #55: 1.12 -> 0.91).
    Only the flat-field offset is corrected; real astrophysical variation
    (including curvature) is preserved.

    blocks: list of (t_hr, flux_ADU) tuples, one per exposure (None if no data).
    Returns: (t_hr_all, fn_normalized, global_med) or None.
    """
    valid_blocks = [b for b in blocks if b is not None]
    if not valid_blocks:
        return None

    # Work in raw ADU space
    offsets = np.zeros(len(blocks))
    models = []
    for b in blocks:
        if b is None:
            models.append(None)
            continue
        if len(b[0]) < 10:
            models.append(None)
            continue
        tc = b[0] - b[0].mean()          # centered time: quad term is
        p = np.polyfit(tc, b[1], 2)      # ill-conditioned on raw hours
        d = np.polyder(p)
        models.append({
            'sl_end': np.polyval(d, tc[-1]),
            'sl_start': np.polyval(d, tc[0]),
            'val_end': np.polyval(p, tc[-1]),
            'val_start': np.polyval(p, tc[0]),
            't_end': b[0][-1],
            't_start': b[0][0],
        })

    for i in range(1, len(blocks)):
        if blocks[i] is None or models[i] is None:
            continue
        prev = None
        for j in range(i-1, -1, -1):
            if blocks[j] is not None and models[j] is not None:
                prev = j
                break
        if prev is None:
            continue

        pm, cm = models[prev], models[i]
        dt_gap = cm['t_start'] - pm['t_end']
        avg_slope = (pm['sl_end'] + cm['sl_start']) / 2.0
        expected_start = pm['val_end'] + offsets[prev] + avg_slope * dt_gap
        offsets[i] = expected_start - cm['val_start']

    # No detrending — real long-term variability should be preserved.
    # The 4-IQR exposure rejection + per-exposure clipping keep
    # the cumulative drift small enough without detrending.

    # Concatenate with offsets applied
    all_t, all_f = [], []
    for i, b in enumerate(blocks):
        if b is None:
            continue
        all_t.append(b[0])
        all_f.append(b[1] + offsets[i])

    t_cat = np.concatenate(all_t)
    f_cat = np.concatenate(all_f)
    so = np.argsort(t_cat)
    t_cat, f_cat = t_cat[so], f_cat[so]

    # Normalize to median=1
    global_med = np.median(f_cat)
    if global_med <= 0:
        return None
    return t_cat, f_cat / global_med, global_med


def _slope_stitch_v1_legacy(blocks):
    """Legacy edge-bin stitching (kept for reference)."""
    BIN_SIZE = 9
    N_EDGE = 3
    all_f = np.concatenate([b[1] for b in blocks if b is not None])
    global_med = np.median(all_f)
    if global_med <= 0: return None
    norm_blocks = [(b[0], b[1]/global_med) if b else None for b in blocks]
    def get_edge_info(t, f):
        n_bins = len(t) // BIN_SIZE
        if n_bins < 2 * N_EDGE: return None
        tb = np.array([np.mean(t[i*BIN_SIZE:(i+1)*BIN_SIZE]) for i in range(n_bins)])
        fb = np.array([np.median(f[i*BIN_SIZE:(i+1)*BIN_SIZE]) for i in range(n_bins)])
        p_start = np.polyfit(tb[:N_EDGE], fb[:N_EDGE], 1)
        p_end = np.polyfit(tb[-N_EDGE:], fb[-N_EDGE:], 1)
        return {
            'slope_start': p_start[0], 'slope_end': p_end[0],
            'val_start': np.polyval(p_start, tb[0]),
            'val_end': np.polyval(p_end, tb[-1]),
            't_start': tb[0], 't_end': tb[-1],
        }

    edge_info = [get_edge_info(b[0], b[1]) if b is not None else None
                 for b in norm_blocks]

    # Compute offsets
    offsets = np.zeros(len(norm_blocks))
    for i in range(1, len(norm_blocks)):
        if norm_blocks[i] is None or edge_info[i] is None:
            continue
        prev = None
        for j in range(i-1, -1, -1):
            if norm_blocks[j] is not None and edge_info[j] is not None:
                prev = j
                break
        if prev is None:
            continue
        pi = edge_info[prev]
        ci = edge_info[i]
        dt = ci['t_start'] - pi['t_end']
        predicted = pi['val_end'] + offsets[prev] + pi['slope_end'] * dt
        offsets[i] = predicted - ci['val_start']

    # Apply and concatenate
    all_t, all_f = [], []
    for i, b in enumerate(norm_blocks):
        if b is None:
            continue
        all_t.append(b[0])
        all_f.append(b[1] + offsets[i])

    if not all_t:
        return None

    t_cat = np.concatenate(all_t)
    f_cat = np.concatenate(all_f)
    so = np.argsort(t_cat)
    return t_cat[so], f_cat[so], global_med


def main():
    dry_run = '--dry-run' in sys.argv
    t0_global = time.time()

    with open(MAPPING) as f:
        mapping = json.load(f)

    ter5_sources = [(i, m) for i, m in enumerate(mapping) if m['target'] == 'Terzan5']
    print(f'{len(ter5_sources)} Terzan5 sources')

    # Detectors to process
    DETS = ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']

    # For each detector, find which sources land on it (using Seg2 as reference)
    # We'll use the catalog RA/Dec to project onto each Seg1 exposure
    source_ras = np.array([m['ra'] for _, m in ter5_sources])
    source_decs = np.array([m['dec'] for _, m in ter5_sources])
    source_mids = np.array([m['master_id'] for _, m in ter5_sources])

    # Find all uncal + calints files per detector
    det_files = {}
    for det in DETS:
        uncals = sorted(glob.glob(f'{SEG1_DIR}/*{det}_uncal.fits'))
        calints = sorted(glob.glob(f'{SEG1_DIR}/*{det}_calints.fits'))
        if len(uncals) == len(calints) and len(uncals) > 0:
            det_files[det] = list(zip(uncals, calints))
            print(f'  {det}: {len(uncals)} exposures')

    # Extract group-diff photometry: result[mid][det] = list of (t_hr, flux) per exposure
    # Process one detector at a time, one exposure at a time
    raw_data = {}  # mid -> {det -> [block_or_None per exposure]}
    bkg_data = {}  # mid -> {det -> bkg_per_pixel}

    for det in DETS:
        if det not in det_files:
            continue
        ch = 'LW' if det == 'nrcblong' else 'SW'
        r_in, r_out = BKG_ANNULUS[ch]
        n_exp = len(det_files[det])

        print(f'\nProcessing {det} ({n_exp} exposures)...')
        t1 = time.time()

        # For background: accumulate per-source background estimates across exposures
        bkg_accum = {}  # mid -> list of per-pixel medians

        for ei, (uncal_path, calints_path) in enumerate(det_files[det]):
            # Get WCS from calints
            wcs = WCS(fits.getheader(calints_path, 'SCI'), naxis=2)
            try:
                int_times = fits.getdata(calints_path, 'INT_TIMES')
                times_bjd = int_times['int_mid_BJD_TDB']
            except:
                continue

            # Project all sources
            px_all, py_all = wcs.world_to_pixel_values(source_ras, source_decs)
            px_all = np.array(px_all, dtype=np.float64)
            py_all = np.array(py_all, dtype=np.float64)

            # Filter to on-detector sources
            on_det = (px_all >= H) & (px_all < 2048-H) & (py_all >= H) & (py_all < 2048-H)
            on_idx = np.where(on_det)[0]

            if len(on_idx) == 0:
                continue

            # Build aperture positions for all on-detector sources
            positions = np.column_stack([px_all[on_idx], py_all[on_idx]])
            apertures = CircularAperture(positions, r=1.5)

            # Also build background annuli
            from photutils.aperture import CircularAnnulus
            bkg_annuli = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

            # Read uncal
            uncal_data = fits.getdata(uncal_path, 'SCI')  # (n_ints, n_groups, ny, nx)
            n_ints, n_groups, ny, nx = uncal_data.shape
            n_gd = n_groups - 1

            # Initialize per-source storage for this exposure
            n_on = len(on_idx)
            # flux array: (n_on, n_ints * n_gd)
            flux_all = np.zeros((n_on, n_ints * n_gd), dtype=np.float64)
            group_all = np.zeros(n_ints * n_gd, dtype=np.int32)
            time_all = np.zeros(n_ints * n_gd, dtype=np.float64)
            bkg_vals = np.zeros(n_on, dtype=np.float64)
            n_bkg = 0

            from photutils.aperture import aperture_photometry
            for i_int in range(n_ints):
                ramp = uncal_data[i_int].astype(np.float64)
                for g in range(n_gd):
                    gd_frame = ramp[g+1] - ramp[g]
                    idx_flat = i_int * n_gd + g

                    phot = aperture_photometry(gd_frame.astype(np.float32), apertures)
                    flux_all[:, idx_flat] = phot['aperture_sum'].value
                    group_all[idx_flat] = g

                    t_off = (g - n_gd/2.0 + 0.75) * TGROUP_HR  # flux-weighted mid-exposure of diff g
                    time_all[idx_flat] = float(times_bjd[i_int]) + t_off / 24.0

                # Background from first group-diff
                gd0 = (ramp[1] - ramp[0]).astype(np.float32)
                bkg_phot = aperture_photometry(gd0, bkg_annuli)
                bkg_area = bkg_annuli.area
                bkg_per_px = bkg_phot['aperture_sum'].value / bkg_area
                bkg_vals += bkg_per_px
                n_bkg += 1

            if n_bkg > 0:
                bkg_vals /= n_bkg

            # Per-source: apply groupdiff-based saturation correction,
            # then store corrected lightcurve
            for ji, si in enumerate(on_idx):
                mid = int(source_mids[si])
                if mid not in raw_data:
                    raw_data[mid] = {}
                if det not in raw_data[mid]:
                    raw_data[mid][det] = [None] * n_exp

                fl = flux_all[ji]
                valid = np.isfinite(fl) & (fl > 0)
                if valid.sum() < 10:
                    continue

                # Saturation correction: compute per-group median ratio to g0
                g0_med = np.median(fl[group_all == 0])
                if g0_med <= 0:
                    continue
                fl_corr = fl.copy()
                for g in range(n_gd):
                    gmask = group_all == g
                    g_med = np.median(fl[gmask & valid])
                    ratio = g_med / g0_med if g0_med > 0 else 1.0
                    if ratio > 0.05:
                        fl_corr[gmask] = fl[gmask] / ratio
                    else:
                        fl_corr[gmask] = np.nan  # too saturated

                valid_corr = np.isfinite(fl_corr) & (fl_corr > 0)
                if valid_corr.sum() > 5:
                    raw_data[mid][det][ei] = (time_all[valid_corr], fl_corr[valid_corr])

                bkg_accum.setdefault(mid, []).append(float(bkg_vals[ji]))

        # Compute static bkg per source
        for mid, vals in bkg_accum.items():
            if mid not in bkg_data:
                bkg_data[mid] = {}
            bkg_data[mid][det] = float(np.median(vals)) * AP_AREA

        print(f'  Done in {time.time()-t1:.0f}s')

    # Now stitch + clean for each source/det
    print(f'\nStitching and cleaning...')

    # Collect results: mid -> {ch -> (t_hr, fn)}
    results = {}  # mid -> {'SW': (t, f, med), 'LW': (t, f, med)}
    bkg_scalars = {}  # (mid, ch) -> (med_raw, bkg_offset), recorded not applied

    for mid in source_mids:
        mid = int(mid)
        if mid not in raw_data:
            continue

        for det in DETS:
            ch = 'LW' if det == 'nrcblong' else 'SW'
            if det not in raw_data.get(mid, {}):
                continue

            blocks = raw_data[mid][det]

            # Convert times to hours
            all_valid_t = []
            for b in blocks:
                if b is not None:
                    all_valid_t.extend(b[0].tolist())
            if not all_valid_t:
                continue
            t0 = min(all_valid_t)
            hr_blocks = []
            for b in blocks:
                if b is None:
                    hr_blocks.append(None)
                else:
                    hr_blocks.append(((b[0] - t0) * 24.0, b[1]))

            # Per-exposure: IQR clip + integration slope correction in raw ADU,
            # then normalize each block to its median.
            # Normalizing first means the stitcher measures fractional slopes
            # which are independent of flat-field level — more consistent stitching.
            blocks_corrected = []
            for b in hr_blocks:
                if b is None:
                    blocks_corrected.append(None)
                    continue
                t_b, f_b = b
                # IQR clip in raw ADU
                f_bc, t_bc = clip_iqr(f_b, t_b)
                f_bc, t_bc = clip_iqr(f_bc, t_bc)
                # Integration slope correction in raw ADU
                f_corr = correct_integration_slopes_raw(t_bc, f_bc)
                # Normalize to this block's median
                med_b = np.median(f_corr)
                if med_b <= 0:
                    blocks_corrected.append(None)
                    continue
                blocks_corrected.append((t_bc, f_corr / med_b))

            # 4-IQR rejection of anomalous exposures (e.g., source near detector edge)
            block_raw_meds = []
            for b in hr_blocks:
                if b is None: block_raw_meds.append(None)
                else: block_raw_meds.append(np.median(b[1]) if len(b[1]) > 0 else None)
            valid_meds = [m for m in block_raw_meds if m is not None and m > 0]
            if len(valid_meds) >= 3:
                q1, q3 = np.percentile(valid_meds, [25, 75])
                iqr_val = q3 - q1
                lo, hi = q1 - 4*iqr_val, q3 + 4*iqr_val
                for bi in range(len(blocks_corrected)):
                    if block_raw_meds[bi] is not None and (block_raw_meds[bi] < lo or block_raw_meds[bi] > hi):
                        blocks_corrected[bi] = None

            if not any(b is not None for b in blocks_corrected):
                continue
            result = slope_stitch(blocks_corrected)
            if result is None:
                continue

            t_hr, fn, med_raw = result

            # IQR clip the stitched result
            fn_c, t_c = clip_iqr(fn, t_hr)
            fn_c, t_c = clip_iqr(fn_c, t_c)

            if len(fn_c) < 20:
                continue

            # Background: MEASURED but NOT applied (2026-07-22). We serve raw
            # aperture photometry and let lightcurve models carry dilution as a
            # free third-light term bounded by dilution > 0; subtracting an
            # annulus estimate can over- OR under-correct the blend. The
            # scalars are recorded for anyone who wants to impose a prior.
            bkg_off = bkg_data.get(mid, {}).get(det, 0.0)

            # This is the "groupdiff" (corrected+stitched) baseline
            sc_gd = integration_scatter(t_c, fn_c)
            stages = {'groupdiff': (t_c, fn_c, sc_gd)}
            # NOTE: `med_raw` here is ~1.0 -- each dither block is normalized to
            # its own median before stitching, so the DN scale is gone by this
            # point (this is also why apply_bkg_rescale was always a silent
            # no-op for Segment 1: med_raw - bkg < 0). For the supplementary
            # dilution scalars we therefore record the median of the per-block
            # RAW aperture medians, which is on the same DN scale as bkg_off.
            _raw_meds = [m for m in block_raw_meds if m is not None and m > 0]
            if _raw_meds:
                bkg_scalars[(mid, ch)] = (float(np.median(_raw_meds)), float(bkg_off))

            # Second-pass slope correction on the stitched LC
            f_sl = apply_slope_correction(t_c, fn_c)
            if f_sl is not None:
                f_sl_c, t_sl_c = clip_iqr(f_sl, t_c)
                f_sl_c, t_sl_c = clip_iqr(f_sl_c, t_sl_c)
                sc_sl = integration_scatter(t_sl_c, f_sl_c)
                if sc_sl < GATE_THRESH * sc_gd:
                    stages['slope_corrected'] = (t_sl_c, f_sl_c, sc_sl)

            # Pick best stage
            candidates = {k: v for k, v in stages.items() if k != 'groupdiff'}
            if candidates:
                best = min(candidates, key=lambda k: candidates[k][2])
            else:
                best = 'groupdiff'

            if mid not in results:
                results[mid] = {}
            # Keep best det for SW (most points)
            if ch in results[mid]:
                if len(fn_c) <= len(results[mid][ch][0][1]):
                    continue
            results[mid][ch] = (stages, best, med_raw, t0)

    print(f'Sources with Segment1 data:')
    n_sw = sum(1 for r in results.values() if 'SW' in r)
    n_lw = sum(1 for r in results.values() if 'LW' in r)
    print(f'  SW: {n_sw}, LW: {n_lw}')

    if dry_run:
        print('Dry run — not writing to HDF5')
        return

    # Write to a COPY of the HDF5 (don't take down the server)
    import shutil
    CATALOG_TMP = CATALOG.replace('.h5', '_seg1_tmp.h5')
    print(f'\nCopying catalog to temp file...')
    shutil.copy2(CATALOG, CATALOG_TMP)

    print(f'Writing Segment1 to {CATALOG_TMP}...')
    h5 = h5py.File(CATALOG_TMP, 'r+')

    # Write all correction stages for Segment1
    n_written = {s: 0 for s in ['groupdiff', 'slope_corrected']}
    for mid, channels in results.items():
        for ch, (stages, best, med_raw, t0_mjd) in channels.items():
            for stage_name, (t_s, f_s, sc_s) in stages.items():
                path = f'{stage_name}/Terzan5/{mid}/Segment1_{ch}'
                if path in h5:
                    del h5[path]
                grp = h5.require_group(f'{stage_name}/Terzan5/{mid}')
                sg = grp.create_group(f'Segment1_{ch}')
                sg.create_dataset('times', data=t_s.astype(np.float64))
                sg.create_dataset('flux_norm', data=f_s.astype(np.float64))
                sg.attrs['scatter'] = float(sc_s)
                sg.attrs['mjd_ref'] = float(t0_mjd)
                n_written[stage_name] = n_written.get(stage_name, 0) + 1

    # Record the Segment-1 background scalars in background_level (measured,
    # NOT applied to the flux -- see the note at the groupdiff stage).
    blp = 'background_level/Terzan5'
    if blp in h5 and bkg_scalars:
        bl = h5[blp][:]
        names = bl.dtype.names
        n_set = 0
        for i, row in enumerate(bl):
            mid_i = int(row['master_id'])
            for ch in ('SW', 'LW'):
                sc = bkg_scalars.get((mid_i, ch))
                if sc is None:
                    continue
                med_raw_v, bkg_v = sc
                key = f'Segment1_{ch}'
                if f'{key}_med_raw_dn' in names:
                    bl[i][f'{key}_med_raw_dn'] = med_raw_v
                    bl[i][f'{key}_bkg_dn'] = bkg_v
                    bl[i][f'{key}_rescale'] = (med_raw_v / (med_raw_v - bkg_v)
                                               if (med_raw_v - bkg_v) > 0 and bkg_v > 0
                                               else 1.0)
                    n_set += 1
        attrs = dict(h5[blp].attrs)
        del h5[blp]
        d = h5.create_dataset(blp, data=bl)
        for k, v in attrs.items():
            d.attrs[k] = v
        print(f'Recorded Segment1 background scalars for {n_set} source/channels')

    # Update best_stage table to include Segment1
    bs_old = h5['best_stage/Terzan5'][:]
    old_fields = list(bs_old.dtype.names)
    new_fields = [('master_id', 'i4')]
    for col in old_fields:
        if col != 'master_id':
            new_fields.append((col, 'S20'))
    if 'Segment1_SW' not in old_fields:
        new_fields.append(('Segment1_SW', 'S20'))
    if 'Segment1_LW' not in old_fields:
        new_fields.append(('Segment1_LW', 'S20'))

    dt = np.dtype(new_fields)
    arr = np.zeros(len(bs_old), dtype=dt)
    for col in old_fields:
        arr[col] = bs_old[col]

    # Fill Segment1 best stages
    for i, row in enumerate(arr):
        mid = int(row['master_id'])
        for ch_col in ['Segment1_SW', 'Segment1_LW']:
            ch = ch_col.split('_')[1]
            if mid in results and ch in results[mid]:
                arr[i][ch_col] = results[mid][ch][1]  # best stage name
            else:
                arr[i][ch_col] = 'groupdiff'

    del h5['best_stage/Terzan5']
    h5.create_dataset('best_stage/Terzan5', data=arr)

    h5.close()

    # Atomic swap
    CATALOG_PREV = CATALOG.replace('.h5', '_prev.h5')
    print(f'Swapping: {CATALOG_TMP} -> {CATALOG}')
    if os.path.exists(CATALOG_PREV):
        os.remove(CATALOG_PREV)
    os.rename(CATALOG, CATALOG_PREV)
    os.rename(CATALOG_TMP, CATALOG)

    dt_total = time.time() - t0_global
    print(f'\nDone in {dt_total/60:.1f} min')
    print(f'Written stages: {n_written}')
    n_slope = n_written.get('slope_corrected', 0)
    print(f'  groupdiff: {n_written["groupdiff"]}, slope_corrected: {n_slope}')
    print(f'Restart catalog server to pick up changes.')


if __name__ == '__main__':
    main()
