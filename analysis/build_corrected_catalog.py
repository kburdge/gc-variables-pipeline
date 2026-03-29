#!/usr/bin/env python
"""
Build complete corrected lightcurve catalog.

For each source/seg/ch, tries all correction strategies and stores results:
  - groupdiff: IQR-clipped aperture photometry (already in main lightcurves table)
  - sat_corrected: v8 ratio correction from uncal files + 2nd IQR clip
  - slope_corrected: flux-dependent slope removal on original groupdiff LC
  - sat_slope: slope correction applied to sat-corrected LC
  - special_reduction: custom extractions (managed separately)
  - ZF lightcurves: already in main lightcurves table

Selection logic (10% integration scatter improvement required at each step):
  1. Start with groupdiff as baseline
  2. If sat_corrected < 0.9 * groupdiff: sat_corrected is a candidate
  3. If slope_corrected < 0.9 * groupdiff: slope_corrected is a candidate
  4. If sat_slope < 0.9 * groupdiff: sat_slope is a candidate
  5. Pick the candidate with lowest scatter
  6. special_reduction always wins if present

Stores a best_stage/{target} table mapping each source to its best stage per seg/ch.
The server reads this table to decide what to serve.

Usage:
    python build_corrected_catalog.py [--source 624] [--dry-run]
"""
import numpy as np
import glob
import os
import re
import sys
import json
import time
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import median_filter
from photutils.aperture import CircularAperture
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
MAPPING_PATH = f'{BASE}/catalogs/master_source_mapping.json'
OUT_DIR = f'{BASE}/diagnostics/master_thumbnails/new_test'
os.makedirs(OUT_DIR, exist_ok=True)

TGROUP_HR = 21.47354 / 3600.0
H = 2
RATIO_THRESH = 0.05
TEMPORAL_WINDOW = 7
CLIP_SIGMA = 3.0
GATE_THRESH = 0.9  # require 10% improvement

ap = CircularAperture([(H, H)], r=1.5)
ap_mask = ap.to_mask(method='exact')[0].to_image((2*H+1, 2*H+1)).astype(np.float32)

STAGE_NAMES = ['groupdiff', 'sat_corrected', 'slope_corrected', 'sat_slope', 'special']


# ── Utilities ──────────────────────────────────────────────────────────────

def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    for b in range(nc):
        s, e = b*cs, (b+1)*cs; sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75]); r = q3 - q1
        m[s:e] = (sg >= q1 - iq*r) & (sg <= q3 + iq*r)
    return f[m], t[m]


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


# ── Saturation correction (batch per detector) ────────────────────────────

def sat_correct_det(target, seg, det, sources_on_det):
    uncals = sorted(glob.glob(f'/data/JWST/{target}/{seg}/*{det}*uncal.fits'))
    if not uncals: return {}
    fb = None
    for uf in uncals:
        try:
            it = fits.getdata(uf, 'INT_TIMES')
            fb = it['int_start_BJD_TDB'].min(); break
        except: continue
    if fb is None: return {}

    allpx = set(); spx = {}
    for s in sources_on_det:
        pl = []
        for dy in range(-H, H+1):
            for dx in range(-H, H+1):
                w = ap_mask[dy+H, dx+H]
                if w < 0.01: continue
                py, px = s['iy']+dy, s['ix']+dx
                if 0 <= py < 2048 and 0 <= px < 2048:
                    allpx.add((py, px)); pl.append((py, px, w))
        spx[s['mid']] = pl

    all_ramps = {k: [] for k in allpx}; t0s = []
    for uf in uncals:
        d = fits.getdata(uf, 'SCI'); ni, ng = d.shape[0], d.shape[1]
        try:
            it = fits.getdata(uf, 'INT_TIMES'); tsb = it['int_start_BJD_TDB']
        except: continue
        for i in range(ni):
            t0s.append((float(tsb[i]) - fb) * 24.)
            for (py, px) in allpx:
                all_ramps[(py, px)].append(d[i, :, py, px].astype(np.float64))
    nint = len(t0s)
    if nint == 0: return {}
    ng = all_ramps[next(iter(allpx))][0].shape[0]; max_gd = ng - 1

    pix_info = {}
    for (py, px) in allpx:
        ramps = all_ramps[(py, px)]
        all_gds = [r[1:] - r[:-1] for r in ramps]
        g0v = np.array([gds[0] for gds in all_gds])
        g0_clean = g0v.copy(); pos = g0_clean > 0
        if pos.sum() > TEMPORAL_WINDOW:
            g0_filled = np.where(pos, g0_clean, np.median(g0_clean[pos]))
            g0_rm = median_filter(g0_filled, size=TEMPORAL_WINDOW)
            resid = np.abs(g0_clean - g0_rm)
            gmad = np.median(resid[pos]) * 1.4826
            if gmad > 0: g0_clean[resid > CLIP_SIGMA * gmad] = np.nan
        max_g = 0; rm = {}
        for g in range(1, max_gd):
            giv = np.array([gds[g] for gds in all_gds])
            vl2 = np.isfinite(g0_clean) & (g0_clean > 0) & (giv > 0)
            if vl2.sum() < 10: break
            r = giv[vl2] / g0_clean[vl2]; mr = np.median(r)
            if mr < 0.03: break
            poly = np.polyfit(g0_clean[vl2], r, 2)
            pred = np.polyval(poly, g0_clean[vl2])
            if np.std(r - pred) / mr > RATIO_THRESH: break
            rm[g] = poly; max_g = g
        pix_info[(py, px)] = {'max_g': max_g, 'rm': rm, 'g0v': g0_clean, 'all_gds': all_gds}

    n_slots = nint * max_gd
    time_grid = np.zeros(n_slots)
    for ii in range(nint):
        for g in range(max_gd):
            time_grid[ii*max_gd+g] = t0s[ii] + (g+0.5) * TGROUP_HR

    results = {}
    for s in sources_on_det:
        mid = s['mid']
        pix_contribs = []
        for (py, px, ap_w) in spx[mid]:
            info = pix_info.get((py, px))
            if info is None: continue
            mg = info['max_g']; rm_d = info['rm']
            g0v = info['g0v']; all_gds_px = info['all_gds']
            pix_vals = np.full((nint, max_gd), np.nan)
            for ii in range(nint):
                g0 = g0v[ii]
                if not np.isfinite(g0) or g0 <= 0: continue
                gds = all_gds_px[ii]
                for g in range(max_gd):
                    if g == 0: pix_vals[ii, g] = g0
                    elif g <= mg and g in rm_d:
                        pr = np.polyval(rm_d[g], g0)
                        if pr > 0.03: pix_vals[ii, g] = gds[g] / pr
            valid = np.isfinite(pix_vals) & (pix_vals > 0)
            if valid.sum() < 20: continue
            pmed = np.nanmedian(pix_vals[valid])
            if pmed <= 0: continue
            pix_contribs.append((pix_vals / pmed, ap_w))
        if not pix_contribs: continue

        ws = np.zeros(n_slots); wa = np.zeros(n_slots)
        for idx in range(n_slots):
            ii = idx // max_gd; g = idx % max_gd
            vals = []; wts = []
            for (cn, w) in pix_contribs:
                if np.isfinite(cn[ii, g]) and cn[ii, g] > 0:
                    vals.append(cn[ii, g]); wts.append(w)
            if len(vals) >= 3:
                vals = np.array(vals); wts = np.array(wts)
                med = np.median(vals); mad = np.median(np.abs(vals - med)) * 1.4826
                good = np.abs(vals - med) < CLIP_SIGMA * mad if mad > 0 else np.ones(len(vals), bool)
                ws[idx] = np.sum(vals[good] * wts[good]); wa[idx] = np.sum(wts[good])
            elif vals:
                ws[idx] = sum(v*wt for v, wt in zip(vals, wts)); wa[idx] = sum(wts)
        vc = wa > 0
        if vc.sum() < 50: continue
        cf = ws[vc] / wa[vc]; cfn = cf / np.median(cf); tc = time_grid[vc]
        so = np.argsort(tc)
        cfn_c, tc_c = clip_iqr(cfn[so], tc[so])
        cfn_c, tc_c = clip_iqr(cfn_c, tc_c)
        if len(cfn_c) >= 20:
            results[mid] = (tc_c, cfn_c)
    return results


# ── Original LC extraction ─────────────────────────────────────────────────

def extract_original(target, seg, det, ix, iy):
    cube_path = f'{BASE}/refs/groupdiffs_{target}_{seg}_{det}.fits'
    h5_path = f'{BASE}/extraction/{target}/{seg}/{det}_ramp.h5'
    if not os.path.exists(cube_path) or not os.path.exists(h5_path): return None
    cube = fits.getdata(cube_path, memmap=True)
    with h5py.File(h5_path, 'r') as f: times = f['times'][:]
    nf, ny, nx = cube.shape
    if ix-H < 0 or ix+H+1 > nx or iy-H < 0 or iy+H+1 > ny: return None
    cutout = np.array(cube[:, iy-H:iy+H+1, ix-H:ix+H+1], dtype=np.float32)
    np.nan_to_num(cutout, nan=0., copy=False)
    flux = np.sum(cutout * ap_mask[np.newaxis, :, :], axis=(1, 2))
    valid = np.isfinite(flux) & (flux != 0)
    if valid.sum() < 50: return None
    fn = flux[valid] / np.median(flux[valid])
    so = np.argsort(times[valid])
    fn_c, t_c = clip_iqr(fn[so], times[valid][so])
    fn_c, t_c = clip_iqr(fn_c, t_c)
    return t_c, fn_c


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    single_source = None
    dry_run = '--dry-run' in sys.argv
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--source='):
            single_source = int(arg.split('=')[1])
        elif arg == '--source' and i < len(sys.argv) - 1:
            single_source = int(sys.argv[i + 1])

    t0_global = time.time()

    # Load metadata
    h5 = h5py.File(CATALOG, 'r')
    all_sources = {}
    for target in ['Liller1', 'Terzan5']:
        if target not in h5: continue
        srcs = h5[f'{target}/sources'][:]
        for i, src in enumerate(srcs):
            mid = int(src['master_id'])
            if single_source is not None and mid != single_source: continue
            all_sources[mid] = {
                'target': target, 'ra': float(src['ra']), 'dec': float(src['dec']),
                'idx': i, 'snr': float(src['best_snr']),
            }
    h5.close()

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    for mid, info in all_sources.items():
        sw_det = None
        if mid < len(mapping):
            for dk, dv in mapping[mid].get('detections', {}).items():
                fn = dv.get('filename', '')
                if 'ramp' in dk and '_LW' not in dk:
                    m = re.search(r'(nrcb[1-4])', fn)
                    if m: sw_det = m.group(1); break
        info['sw_det'] = sw_det

    print(f'{len(all_sources)} sources')

    # ── Phase 1: Extract originals + sat corrections ──
    originals = {}
    sat_corr = {}

    for target in ['Liller1', 'Terzan5']:
        mids = [m for m, info in all_sources.items() if info['target'] == target]
        if not mids: continue
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']
        for seg in segments:
            for det in ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong']:
                ch = 'LW' if det == 'nrcblong' else 'SW'
                rp = f'{BASE}/refs/{target}_{seg}_{det}_ref.fits'
                if not os.path.exists(rp): continue
                wcs = WCS(fits.getheader(rp))
                sources_on_det = []
                for mid in mids:
                    info = all_sources[mid]
                    if ch == 'SW' and info.get('sw_det') and info['sw_det'] != det:
                        continue
                    sky = SkyCoord(ra=info['ra']*u.deg, dec=info['dec']*u.deg)
                    px, py = wcs.world_to_pixel(sky)
                    ix, iy = int(round(float(px))), int(round(float(py)))
                    if ix < H or ix >= 2048-H or iy < H or iy >= 2048-H:
                        continue
                    sources_on_det.append({'mid': mid, 'ix': ix, 'iy': iy})
                    orig = extract_original(target, seg, det, ix, iy)
                    if orig is not None:
                        originals[(target, mid, seg, ch)] = orig
                if not sources_on_det: continue
                t1 = time.time()
                corr = sat_correct_det(target, seg, det, sources_on_det)
                for mid_c, lc in corr.items():
                    sat_corr[(target, mid_c, seg, ch)] = lc
                print(f'  {target}/{seg}/{det}({ch}): {len(sources_on_det)} src, '
                      f'{len(corr)} sat-corrected ({time.time()-t1:.0f}s)', flush=True)

    # ── Phase 2: Compute all strategies, pick best ──
    # Results: key -> {stage_name: (t, f, sc), 'best': stage_name}
    all_results = {}
    stats = {s: 0 for s in STAGE_NAMES}

    for mid, info in all_sources.items():
        target = info['target']
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']
        for seg in segments:
            for ch in ['SW', 'LW']:
                key = (target, mid, seg, ch)
                stages = {}

                # Groupdiff (baseline)
                if key in originals:
                    t_o, f_o = originals[key]
                    sc_o = integration_scatter(t_o, f_o)
                    stages['groupdiff'] = (t_o, f_o, sc_o)
                else:
                    continue  # no data at all

                # Sat-corrected
                if key in sat_corr:
                    t_s, f_s = sat_corr[key]
                    sc_s = integration_scatter(t_s, f_s)
                    if sc_s < GATE_THRESH * sc_o:
                        stages['sat_corrected'] = (t_s, f_s, sc_s)

                # Slope-only (on original)
                f_sl = apply_slope_correction(t_o, f_o)
                if f_sl is not None:
                    sc_sl = integration_scatter(t_o, f_sl)
                    if sc_sl < GATE_THRESH * sc_o:
                        stages['slope_corrected'] = (t_o, f_sl, sc_sl)

                # Sat + slope
                if key in sat_corr:
                    t_s, f_s = sat_corr[key]
                    f_ss = apply_slope_correction(t_s, f_s)
                    if f_ss is not None:
                        sc_ss = integration_scatter(t_s, f_ss)
                        if sc_ss < GATE_THRESH * sc_o:
                            stages['sat_slope'] = (t_s, f_ss, sc_ss)

                # Pick best (lowest scatter among candidates, excluding groupdiff)
                candidates = {k: v for k, v in stages.items() if k != 'groupdiff'}
                if candidates:
                    best_stage = min(candidates, key=lambda k: candidates[k][2])
                else:
                    best_stage = 'groupdiff'

                # Check for special_reduction (always wins)
                # (We don't recompute these — they're already in the HDF5)

                stages['best'] = best_stage
                all_results[key] = stages
                stats[best_stage] += 1

    print(f'\nBest stage distribution:')
    for s in STAGE_NAMES[:-1]:
        print(f'  {s}: {stats.get(s, 0)}')
    total = sum(stats.values())
    print(f'  Total: {total}')

    if dry_run:
        print('Dry run — not writing to HDF5.')
        return

    # ── Phase 3: Write to HDF5 ──
    print(f'\nWriting to HDF5...')
    h5 = h5py.File(CATALOG, 'r+')

    # Clear old corrected groups (we'll rewrite)
    for grp_name in ['sat_corrected', 'slope_corrected', 'sat_slope']:
        if grp_name in h5:
            del h5[grp_name]

    # Also migrate old 'corrected' -> keep special_reduction untouched
    if 'corrected' in h5:
        del h5['corrected']

    n_written = {s: 0 for s in STAGE_NAMES}

    for (target, mid, seg, ch), stages in all_results.items():
        best = stages['best']

        # Write each passing stage
        for stage_name in ['sat_corrected', 'slope_corrected', 'sat_slope']:
            if stage_name in stages:
                t_lc, f_lc, sc = stages[stage_name]
                path = f'{stage_name}/{target}/{mid}/{seg}_{ch}'
                grp = h5.require_group(f'{stage_name}/{target}/{mid}')
                sg = grp.create_group(f'{seg}_{ch}')
                sg.create_dataset('times', data=t_lc.astype(np.float64))
                sg.create_dataset('flux_norm', data=f_lc.astype(np.float64))
                sg.attrs['scatter'] = float(sc)
                n_written[stage_name] += 1

    # Build best_stage table per target
    for target in ['Liller1', 'Terzan5']:
        t_mids = [m for m, info in all_sources.items() if info['target'] == target]
        if not t_mids: continue
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        # Build dtype dynamically
        fields = [('master_id', 'i4')]
        for seg in segments:
            for ch_name in ['SW', 'LW']:
                fields.append((f'{seg}_{ch_name}', 'S20'))
        dt = np.dtype(fields)
        arr = np.zeros(len(t_mids), dtype=dt)

        for i, mid in enumerate(sorted(t_mids)):
            arr[i]['master_id'] = mid
            for seg in segments:
                for ch_name in ['SW', 'LW']:
                    key = (target, mid, seg, ch_name)
                    # Check special_reduction first
                    sp_path = f'special_reduction/{target}/{mid}/{seg}_{ch_name}'
                    if sp_path in h5:
                        stage = 'special'
                    elif key in all_results:
                        stage = all_results[key]['best']
                    else:
                        stage = 'groupdiff'
                    arr[i][f'{seg}_{ch_name}'] = stage

        path = f'best_stage/{target}'
        if path in h5: del h5[path]
        h5.require_group('best_stage')
        h5.create_dataset(path, data=arr)

    h5.close()

    print(f'Written:')
    for s, n in n_written.items():
        if n > 0: print(f'  {s}: {n}')

    # ── Phase 4: Generate thumbnails ──
    print(f'\nGenerating thumbnails...', flush=True)
    n_plotted = 0
    for mid, info in sorted(all_sources.items(), key=lambda x: -x[1]['snr']):
        target = info['target']
        snr = info['snr']
        ra_s, dec_s = info['ra'], info['dec']
        segments = ['Segment3', 'Segment4'] if target == 'Liller1' else ['Segment2']

        fig, axes = plt.subplots(len(segments), 2,
                                 figsize=(14, 4*len(segments)), squeeze=False)
        has_data = False

        for si, seg in enumerate(segments):
            for ci, ch_name in enumerate(['SW', 'LW']):
                ax = axes[si, ci]
                key = (target, mid, seg, ch_name)
                ss = seg.replace('Segment', 'S')

                if key not in all_results:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                            color='gray', transform=ax.transAxes)
                    ax.set_title(f'{ss} {ch_name}', fontsize=10)
                    continue

                stages = all_results[key]
                best = stages['best']
                t, f, sc = stages[best]
                has_data = True

                clr = '#d62728' if ch_name == 'SW' else '#1f77b4'
                ax.scatter(t, f, s=0.5, c=clr, alpha=0.3, rasterized=True)
                bsz = max(1, len(f) // 100); nb = len(f) // bsz
                if nb > 1:
                    nu = nb * bsz
                    ax.plot(t[:nu].reshape(nb, bsz).mean(1),
                            f[:nu].reshape(nb, bsz).mean(1), clr, lw=2)

                stage_label = {'groupdiff': '', 'sat_corrected': '*',
                               'slope_corrected': 'S', 'sat_slope': '*S'}[best]
                ax.set_title(f'{ss} {ch_name}{stage_label} — {len(f)}pt sc={sc:.5f}',
                             fontsize=10)
                ax.set_ylabel('Norm flux', fontsize=8)
                if si == len(segments) - 1:
                    ax.set_xlabel('Time (hr)', fontsize=8)

        if not has_data:
            plt.close(fig); continue

        fig.suptitle(f'obj{mid:04d} SNR={snr:.1f} (*=sat, S=slope, *S=both)',
                     fontsize=11, y=1.01)
        fig.tight_layout()
        out = os.path.join(OUT_DIR,
                           f'obj{mid:04d}_snr{snr:05.1f}_{ra_s:.5f}_{dec_s:.5f}.png')
        fig.savefig(out, dpi=100, bbox_inches='tight')
        plt.close(fig)
        n_plotted += 1
        if n_plotted % 100 == 0:
            print(f'  {n_plotted} plotted ({time.time()-t0_global:.0f}s)', flush=True)

    print(f'\nDone: {n_plotted} thumbnails in {(time.time()-t0_global)/60:.1f} min')
    print(f'Output: {OUT_DIR}/')
    print('Restart catalog server to pick up changes.')


if __name__ == '__main__':
    main()
