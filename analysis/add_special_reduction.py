#!/usr/bin/env python
"""
Add a special reduction lightcurve to master_variable_catalog.h5.

Special reductions are custom extractions (e.g., PSF wing annulus for
saturated sources) that take precedence over all pipeline stages.

Structure in HDF5:
  special_reduction/{target}/{master_id}/{seg}_{ch}/
    times      — float64 array
    flux_norm  — float64 array
    attrs: method, description, aperture, script

Usage:
    python add_special_reduction.py   (runs built-in definitions)
"""
import numpy as np
import glob
import h5py
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import median_filter
from photutils.aperture import CircularAperture
import warnings
warnings.filterwarnings('ignore')

BASE = '/data/Globulars_Pipeline'
CATALOG = f'{BASE}/catalogs/master_variable_catalog.h5'
TGROUP_HR = 21.47354 / 3600.0
RATIO_THRESH = 0.05
TEMPORAL_WINDOW = 7
CLIP_SIGMA = 3.0


def clip_iqr(f, t, cs=18, iq=2.):
    n = len(f); nc = n // cs
    if nc < 1: return f, t
    m = np.ones(n, dtype=bool)
    for b in range(nc):
        s, e = b*cs, (b+1)*cs; sg = f[s:e]
        q1, q3 = np.percentile(sg, [25, 75]); r = q3 - q1
        m[s:e] = (sg >= q1 - iq*r) & (sg <= q3 + iq*r)
    return f[m], t[m]


def extract_wing_sat_corrected(target, seg, det, ix, iy, r_in, r_out):
    """V8 ratio correction using annular aperture pixels."""
    size = int(r_out) + 1
    y, x = np.mgrid[-size:size+1, -size:size+1]
    r = np.sqrt(x**2 + y**2)
    mask = ((r >= r_in) & (r < r_out)).astype(np.float32)

    allpx = []
    for dy in range(-size, size+1):
        for dx in range(-size, size+1):
            w = mask[dy+size, dx+size]
            if w < 0.01: continue
            ppy, ppx = iy+dy, ix+dx
            if 0 <= ppy < 2048 and 0 <= ppx < 2048:
                allpx.append((ppy, ppx, w))

    uncals = sorted(glob.glob(f'/data/JWST/{target}/{seg}/*{det}*uncal.fits'))
    if not uncals: return None
    fb = None
    for uf in uncals:
        try:
            it = fits.getdata(uf, 'INT_TIMES')
            fb = it['int_start_BJD_TDB'].min(); break
        except: continue
    if fb is None: return None

    pix_ramps = {(py, px): [] for py, px, _ in allpx}
    t0s = []
    for uf in uncals:
        d = fits.getdata(uf, 'SCI'); ni, ng = d.shape[0], d.shape[1]
        try:
            it = fits.getdata(uf, 'INT_TIMES'); tsb = it['int_start_BJD_TDB']
        except: continue
        for i in range(ni):
            t0s.append((float(tsb[i]) - fb) * 24.)
            for (ppy, ppx, _) in allpx:
                pix_ramps[(ppy, ppx)].append(d[i, :, ppy, ppx].astype(np.float64))

    nint = len(t0s)
    if nint == 0: return None
    ng = pix_ramps[allpx[0][:2]][0].shape[0]; max_gd = ng - 1

    pix_info = {}
    for (ppy, ppx, w) in allpx:
        ramps = pix_ramps[(ppy, ppx)]
        all_gds = [r_[1:] - r_[:-1] for r_ in ramps]
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
            ratio = giv[vl2] / g0_clean[vl2]; mr = np.median(ratio)
            if mr < 0.03: break
            poly = np.polyfit(g0_clean[vl2], ratio, 2)
            pred = np.polyval(poly, g0_clean[vl2])
            if np.std(ratio - pred) / mr > RATIO_THRESH: break
            rm[g] = poly; max_g = g
        pix_info[(ppy, ppx)] = {'max_g': max_g, 'rm': rm, 'g0v': g0_clean, 'all_gds': all_gds}

    n_slots = nint * max_gd
    time_grid = np.zeros(n_slots)
    for ii in range(nint):
        for g in range(max_gd):
            time_grid[ii*max_gd+g] = t0s[ii] + (g+0.5) * TGROUP_HR

    pix_contribs = []
    for (ppy, ppx, ap_w) in allpx:
        info = pix_info.get((ppy, ppx))
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

    if not pix_contribs: return None

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
    if vc.sum() < 50: return None
    cf = ws[vc] / wa[vc]; cfn = cf / np.median(cf); tc = time_grid[vc]
    so = np.argsort(tc)
    cfn_c, tc_c = clip_iqr(cfn[so], tc[so])
    cfn_c, tc_c = clip_iqr(cfn_c, tc_c)
    if len(cfn_c) < 20: return None
    return tc_c, cfn_c


def add_special(h5, target, master_id, seg, ch, times, flux_norm,
                method, description, aperture, script):
    """Write one special reduction entry to HDF5."""
    path = f'special_reduction/{target}/{master_id}/{seg}_{ch}'
    if path in h5:
        del h5[path]
    grp = h5.require_group(f'special_reduction/{target}/{master_id}')
    sg = grp.create_group(f'{seg}_{ch}')
    sg.create_dataset('times', data=times.astype(np.float64))
    sg.create_dataset('flux_norm', data=flux_norm.astype(np.float64))
    sg.attrs['method'] = method
    sg.attrs['description'] = description
    sg.attrs['aperture'] = aperture
    sg.attrs['script'] = script
    print(f'  Added: {path} ({len(times)} pts)')


def rapid_burster(h5):
    """Special reduction for the Rapid Burster (obj0293, MXB 1730-335).
    Uses PSF wing annulus r=[3,5]px to avoid saturated core."""
    target = 'Liller1'
    master_id = 293
    ra, dec = 263.352452, -33.38896
    r_in, r_out = 3, 5

    print(f'\nRapid Burster (obj{master_id:04d}):')

    for seg in ['Segment3', 'Segment4']:
        det_sw = 'nrcb4'
        rp = f'{BASE}/refs/{target}_{seg}_{det_sw}_ref.fits'
        wcs = WCS(fits.getheader(rp))
        sky = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        px, py = wcs.world_to_pixel(sky)
        ix, iy = int(round(float(px))), int(round(float(py)))

        print(f'  {seg} SW ({det_sw}): pixel ({ix},{iy})')
        result = extract_wing_sat_corrected(target, seg, det_sw, ix, iy, r_in, r_out)
        if result is not None:
            tc, fc = result
            add_special(h5, target, master_id, seg, 'SW',
                        tc, fc,
                        method='wing_annulus_sat_corrected',
                        description=f'PSF wing extraction r=[{r_in},{r_out}]px '
                                    f'with v8 ratio correction. Avoids saturated core.',
                        aperture=f'annulus_r{r_in}_{r_out}',
                        script='add_special_reduction.py::rapid_burster')
        else:
            print(f'    FAILED')


def main():
    import sys
    # Kill catalog server if it has the file locked
    h5 = h5py.File(CATALOG, 'r+')

    # Run all special reductions
    rapid_burster(h5)

    # Add more special reductions here as needed:
    # other_special_source(h5)

    h5.close()
    print('\nDone. Restart catalog server to pick up changes.')


if __name__ == '__main__':
    main()
