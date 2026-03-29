#!/usr/bin/env python
"""FastAPI server for variable star catalog viewer with Aladin Lite frontend.

Serves mosaic FITS files, source catalog, and lightcurve data from HDF5.
Supports multiple targets (Liller1, Terzan5) via /api/{target}/... routes.
Designed to run behind nginx reverse proxy with auth_basic.

Usage:
    python catalog_server.py                      # localhost:8085
    python catalog_server.py --host 0.0.0.0       # all interfaces
"""
import os
import argparse
import threading
import numpy as np
import h5py
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

MOSAIC_DIR = '/data/Globulars_Pipeline/mosaics'
REFS_DIR = '/data/Globulars_Pipeline/refs'

# Target configurations
TARGETS = {
    'Liller1': {
        'center': '263.337 -33.398',
        'segments': ['Segment3', 'Segment4'],
    },
    'Terzan5': {
        'center': '267.020 -24.779',
        'segments': ['Segment2'],
    },
}

REAL_CATALOG_PATH = '/data/Globulars_Pipeline/catalogs/real_variable_catalog.h5'
MASTER_CATALOG_PATH = '/data/Globulars_Pipeline/catalogs/master_variable_catalog.h5'

app = FastAPI()

# Per-target HDF5 state, protected by lock
_h5_cache = {}  # target -> (h5_file, sources_list, table_map)
_h5_lock = threading.Lock()


def get_h5(target):
    """Load real catalog, flatten all tables for this target into a unified source list."""
    if target not in _h5_cache:
        h5 = h5py.File(REAL_CATALOG_PATH, 'r')
        if target not in h5:
            _h5_cache[target] = (h5, [], {})
            return _h5_cache[target]

        sources = []
        table_map = {}  # global_id -> (table_key, local_idx)
        global_id = 0

        for seg_name in sorted(h5[target].keys()):
            for table_name in sorted(h5[target][seg_name].keys()):
                table_key = f'{target}/{seg_name}/{table_name}'
                tbl_srcs = h5[table_key]['sources'][:]
                for local_idx in range(len(tbl_srcs)):
                    s = tbl_srcs[local_idx]
                    # Derive source_type from table name: ramp_sw -> ramp, zf_lw -> zf
                    mode = table_name.split('_')[0]  # ramp or zf
                    channel = table_name.split('_')[1]  # sw or lw
                    source_type = mode  # 'ramp' or 'zf'
                    det_seg = seg_name  # detection segment

                    sources.append({
                        'source_id': global_id,
                        'ra': float(s['ra']),
                        'dec': float(s['dec']),
                        'detector': s['detector'].decode().strip('\x00'),
                        'source_type': source_type,
                        'channel': channel,
                        'det_segment': det_seg,
                        'table_key': table_key,
                        'table_name': table_name,
                        'snr': float(s['snr']),
                        'period_min': float(s['period_min']),
                        'ls_sig': float(s['ls_sig']),
                        'amplitude': float(s['amplitude']),
                        'local_idx': local_idx,
                    })
                    table_map[global_id] = (table_key, local_idx)
                    global_id += 1

        _h5_cache[target] = (h5, sources, table_map)
    return _h5_cache[target]


# ─── Routes ─────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), 'static', 'index.html'),
        media_type='text/html')


@app.get("/api/targets")
def get_targets():
    """Return available targets with metadata."""
    result = {}
    for name, cfg in TARGETS.items():
        _, sources, _ = get_h5(name)
        n_ramp = sum(1 for s in sources if s['source_type'] == 'ramp')
        n_zf = sum(1 for s in sources if s['source_type'] == 'zf')
        result[name] = {
            'center': cfg['center'],
            'segments': cfg['segments'],
            'n_sources': len(sources),
            'n_ramp': n_ramp,
            'n_zf': n_zf,
        }
    return result


@app.get("/fits/mosaic/{target}/{seg}/{imgtype}")
def serve_mosaic(target: str, seg: str, imgtype: str):
    """Serve pre-built mosaic FITS file."""
    if target not in TARGETS:
        return Response(status_code=404)
    if seg not in TARGETS[target]['segments']:
        return Response(status_code=404)
    VALID_TYPES = ('autocorr', 'i2d', 'lw_i2d', 'lw_autocorr',
                   'zf_median', 'zf_autocorr', 'lw_zf_median', 'lw_zf_autocorr')
    if imgtype not in VALID_TYPES:
        return Response(status_code=404)
    path = f'{MOSAIC_DIR}/{target}_{seg}_{imgtype}_mosaic.fits'
    if not os.path.exists(path):
        return Response(status_code=404, content=f'Mosaic not found: {path}')
    return FileResponse(path, media_type='application/fits',
                        headers={'Cache-Control': 'public, max-age=86400'})


@app.get("/fits/detector/{target}/{seg}/{imgtype}/{det}")
def serve_detector_fits(target: str, seg: str, imgtype: str, det: str):
    """Serve individual detector FITS file (no reprojection, native WCS)."""
    if target not in TARGETS:
        return Response(status_code=404)
    VALID_TYPES = ('autocorr', 'zf_autocorr')
    VALID_DETS = ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong')
    if imgtype not in VALID_TYPES or det not in VALID_DETS:
        return Response(status_code=404)
    path = f'{REFS_DIR}/{target}_{seg}_{det}_{imgtype}.fits'
    if not os.path.exists(path):
        return Response(status_code=404, content=f'Not found: {path}')
    return FileResponse(path, media_type='application/fits',
                        headers={'Cache-Control': 'no-cache'})


@app.get("/api/{target}/sources")
def get_sources_api(target: str):
    """Return all sources as JSON array for a target."""
    if target not in TARGETS:
        return Response(status_code=404)
    _, sources, _ = get_h5(target)
    result = []
    for s in sources:
        entry = {
            'id': s['source_id'],
            'ra': round(s['ra'], 7),
            'dec': round(s['dec'], 7),
            'det': s['detector'],
            'type': s['source_type'],
            'dtype': f'{s["det_segment"]}_{s["table_name"]}',
            'channel': s['channel'],
            'period': round(s['period_min'], 1),
            'ls_sig': round(s['ls_sig'], 1),
            'amp': round(s['amplitude'], 4),
            'autocorr': round(s['snr'], 1),
        }
        result.append(entry)
    return result


def _clip_iqr(flux, times, chunk_size=18, iqr_factor=2.0):
    """IQR clipping in chunks, matching pipeline implementation."""
    n = len(flux)
    n_chunks = n // chunk_size
    if n_chunks < 1:
        return flux, times
    mask = np.ones(n, dtype=bool)
    for b in range(n_chunks):
        s, e = b * chunk_size, (b + 1) * chunk_size
        seg = flux[s:e]
        q1, q3 = np.percentile(seg, [25, 75])
        iqr = q3 - q1
        mask[s:e] = (seg >= q1 - iqr_factor * iqr) & (seg <= q3 + iqr_factor * iqr)
    return flux[mask], times[mask]


def _bin_lc(times, flux, bin_size=9):
    """Bin lightcurve by averaging consecutive points."""
    n = len(flux) // bin_size
    if n < 1:
        return times.tolist(), flux.tolist()
    t_bin = [float(times[i * bin_size:(i + 1) * bin_size].mean()) for i in range(n)]
    f_bin = [float(flux[i * bin_size:(i + 1) * bin_size].mean()) for i in range(n)]
    return t_bin, f_bin


def _is_detected(src_type, det_type, seg, mode):
    """Check if lightcurve is from detection or forced photometry."""
    if src_type == mode:
        if det_type == 'matched':
            return True
        return (det_type == 'seg3_only' and seg == 'Segment3') or \
               (det_type == 'seg4_only' and seg == 'Segment4') or \
               (det_type == 'seg2_only' and seg == 'Segment2')
    return False


@app.get("/api/{target}/lightcurve/{source_id}")
def get_lightcurve(target: str, source_id: int):
    """Return IQR-cleaned and binned lightcurve data for a source."""
    if target not in TARGETS:
        return Response(status_code=404)

    with _h5_lock:
        h5, sources, table_map = get_h5(target)
        if source_id < 0 or source_id >= len(sources):
            return Response(status_code=404)

        src = sources[source_id]
        table_key, local_idx = table_map[source_id]

        lcs = {}
        lc_grp = h5[f'{table_key}/lightcurves']
        time_grp = h5[f'{table_key}/times']

        for lc_seg in lc_grp:
            for lc_mode_ch in lc_grp[lc_seg]:
                for lc_det in lc_grp[lc_seg][lc_mode_ch]:
                    lc_path = f'{lc_seg}/{lc_mode_ch}/{lc_det}'
                    flux = lc_grp[lc_path][local_idx, :].astype(np.float64)
                    times = time_grp[lc_path][:].astype(np.float64)

                    valid = np.isfinite(flux) & (flux != 0)
                    if valid.sum() < 10:
                        continue

                    fv, tv = flux[valid], times[valid]
                    tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
                    # IQR clipping already applied in HDF5 by clean_catalog_lcs.py
                    f_clean, t_clean = fv, tv_hr
                    if len(f_clean) < 4:
                        continue

                    bsz = 9 if 'ramp' in lc_mode_ch else 4
                    t_bin, f_bin = _bin_lc(t_clean, f_clean, bin_size=bsz)

                    med = float(np.nanmedian(f_clean))
                    std = float(np.nanstd(f_clean))
                    is_detection = (lc_seg == src['det_segment'] and
                                    src['detector'] == lc_det and
                                    src['table_name'].split('_')[0] in lc_mode_ch)

                    lc_name = f'{lc_seg}_{lc_mode_ch}_{lc_det}'
                    lcs[lc_name] = {
                        't': [round(float(x), 4) for x in t_clean],
                        'f': [round(float(x), 5) for x in f_clean],
                        'tb': t_bin,
                        'fb': f_bin,
                        'med': round(med, 1),
                        'amp': round(std / abs(med), 4) if med != 0 else 0,
                        'n': len(f_clean),
                        'detected': is_detection,
                    }

    return {
        'id': source_id,
        'ra': round(src['ra'], 7),
        'dec': round(src['dec'], 7),
        'det': src['detector'],
        'type': src['source_type'],
        'dtype': f'{src["det_segment"]}_{src["table_name"]}',
        'channel': src['channel'],
        'period': round(src['period_min'], 1),
        'ls_sig': round(src['ls_sig'], 1),
        'amp': round(src['amplitude'], 4),
        'snr': round(src['snr'], 1),
        'lcs': lcs,
    }


# ─── Real catalog routes ─────────────────────────────────────
REAL_CATALOG = '/data/Globulars_Pipeline/catalogs/real_variable_catalog.h5'
_real_h5 = None
_real_lock = threading.Lock()


def get_real_h5():
    global _real_h5
    if _real_h5 is None and os.path.exists(REAL_CATALOG):
        _real_h5 = h5py.File(REAL_CATALOG, 'r')
    return _real_h5


@app.get("/api/real/tables")
def get_real_tables():
    """Return list of available tables in the real catalog."""
    h5 = get_real_h5()
    if h5 is None:
        return Response(status_code=404)
    tables = []
    for target_name in h5:
        for seg_name in h5[target_name]:
            for table_name in h5[target_name][seg_name]:
                key = f'{target_name}/{seg_name}/{table_name}'
                grp = h5[key]
                n = int(grp.attrs.get('n_sources', len(grp['sources'])))
                tables.append({'key': key, 'n_sources': n})
    return tables


@app.get("/api/real/{target}/{segment}/{table}/sources")
def get_real_sources(target: str, segment: str, table: str):
    """Return sources for a real catalog table."""
    h5 = get_real_h5()
    if h5 is None:
        return Response(status_code=404)
    key = f'{target}/{segment}/{table}'
    if key not in h5:
        return Response(status_code=404, content=f'Table {key} not found')
    srcs = h5[f'{key}/sources'][:]
    result = []
    for s in srcs:
        result.append({
            'id': int(s['source_id']),
            'ra': round(float(s['ra']), 7),
            'dec': round(float(s['dec']), 7),
            'det': s['detector'].decode().strip('\x00'),
            'snr': round(float(s['snr']), 1),
            'period': round(float(s['period_min']), 1),
            'ls_sig': round(float(s['ls_sig']), 1),
            'amp': round(float(s['amplitude']), 4),
        })
    return result


@app.get("/api/real/{target}/{segment}/{table}/lightcurve/{source_id}")
def get_real_lightcurve(target: str, segment: str, table: str, source_id: int):
    """Return all lightcurves (detection + forced) for a real catalog source."""
    h5 = get_real_h5()
    if h5 is None:
        return Response(status_code=404)
    key = f'{target}/{segment}/{table}'
    if key not in h5:
        return Response(status_code=404)

    with _real_lock:
        srcs = h5[f'{key}/sources'][:]
        if source_id < 0 or source_id >= len(srcs):
            return Response(status_code=404)

        src = srcs[source_id]
        det = src['detector'].decode().strip('\x00')

        lcs = {}
        lc_grp = h5[f'{key}/lightcurves']
        time_grp = h5[f'{key}/times']

        # Walk all available lightcurve datasets
        for lc_seg in lc_grp:
            for lc_mode_ch in lc_grp[lc_seg]:
                for lc_det in lc_grp[lc_seg][lc_mode_ch]:
                    lc_path = f'{lc_seg}/{lc_mode_ch}/{lc_det}'
                    flux = lc_grp[lc_path][source_id, :].astype(np.float64)
                    times = time_grp[lc_path][:].astype(np.float64)

                    valid = np.isfinite(flux) & (flux != 0)
                    if valid.sum() < 10:
                        continue

                    fv, tv = flux[valid], times[valid]
                    # Convert to hours from start
                    tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv  # MJD vs hours
                    # IQR clipping already applied in HDF5 by clean_catalog_lcs.py
                    f_clean, t_clean = fv, tv_hr
                    if len(f_clean) < 4:
                        continue

                    bsz = 9 if 'ramp' in lc_mode_ch else 4
                    t_bin, f_bin = _bin_lc(t_clean, f_clean, bin_size=bsz)

                    med = float(np.nanmedian(f_clean))
                    # Is this the detection LC?
                    is_detection = (lc_seg == segment and det == lc_det and
                                    table.split('_')[0] in lc_mode_ch)

                    lc_name = f'{lc_seg}_{lc_mode_ch}_{lc_det}'
                    lcs[lc_name] = {
                        't': [round(float(x), 4) for x in t_clean],
                        'f': [round(float(x), 5) for x in f_clean],
                        'tb': t_bin,
                        'fb': f_bin,
                        'med': round(med, 1),
                        'n': len(f_clean),
                        'detected': is_detection,
                    }

    return {
        'id': source_id,
        'ra': round(float(src['ra']), 7),
        'dec': round(float(src['dec']), 7),
        'det': det,
        'snr': round(float(src['snr']), 1),
        'period': round(float(src['period_min']), 1),
        'ls_sig': round(float(src['ls_sig']), 1),
        'amp': round(float(src['amplitude']), 4),
        'lcs': lcs,
    }


# ─── Master catalog routes ───────────────────────────────────
_master_h5 = None
_master_lock = threading.Lock()
_master_cache = {}  # target -> (sources, n)


def get_master_h5():
    global _master_h5
    if _master_h5 is None and os.path.exists(MASTER_CATALOG_PATH):
        _master_h5 = h5py.File(MASTER_CATALOG_PATH, 'r')
    return _master_h5


@app.get("/api/master/{target}/sources")
def get_master_sources(target: str):
    h5 = get_master_h5()
    if h5 is None or target not in h5:
        return Response(status_code=404)
    srcs = h5[f'{target}/sources'][:]
    return [{
        'id': int(s['master_id']),
        'ra': round(float(s['ra']), 7),
        'dec': round(float(s['dec']), 7),
        'snr': round(float(s['best_snr']), 1),
        'n_det': int(s['n_detections']),
    } for s in srcs]


@app.get("/api/master/{target}/lightcurve/{idx}")
def get_master_lightcurve(target: str, idx: int):
    h5 = get_master_h5()
    if h5 is None or target not in h5:
        return Response(status_code=404)
    with _master_lock:
        srcs = h5[f'{target}/sources'][:]
        if idx < 0 or idx >= len(srcs):
            return Response(status_code=404)
        src = srcs[idx]
        mid = int(src['master_id'])
        lcs = {}

        # Load best_stage table to know which correction to serve per seg/ch
        best_stages = {}  # (seg, ch) -> stage name
        if 'best_stage' in h5 and target in h5['best_stage']:
            bs_tbl = h5[f'best_stage/{target}'][:]
            bs_row = bs_tbl[bs_tbl['master_id'] == mid]
            if len(bs_row) > 0:
                for col in bs_tbl.dtype.names:
                    if col == 'master_id': continue
                    # col is like 'Segment3_SW'
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2:
                        best_stages[(parts[0], parts[1])] = bs_row[0][col].decode()

        # Helper to add a variable-length LC from a correction group
        def _add_corrected_lc(group_name, seg_part, ch_part, lc_label):
            mid_str = str(mid)
            path = f'{group_name}/{target}/{mid_str}/{seg_part}_{ch_part}'
            if path not in h5: return False
            cgrp = h5[path]
            tc = cgrp['times'][:].astype(np.float64)
            fc = cgrp['flux_norm'][:].astype(np.float64)
            if len(tc) < 10: return False
            bsz = 9
            t_bin, f_bin = _bin_lc(tc, fc, bin_size=bsz)
            med = float(np.nanmedian(fc))
            std = float(np.nanstd(fc))
            lc_name = f'{seg_part}_ramp_{"nrcblong" if ch_part == "LW" else lc_label}'
            lcs[lc_name] = {
                't': [round(float(x), 4) for x in tc],
                'f': [round(float(x), 5) for x in fc],
                'tb': t_bin, 'fb': f_bin,
                'med': round(med, 1),
                'amp': round(std / abs(med), 4) if med != 0 else 0,
                'n': len(tc),
                'detected': False,
            }
            return True

        # Serve best ramp LC per seg/ch based on best_stage table
        corrected_keys = set()  # track which seg/mode/det combos are handled
        lc_grp = h5[f'{target}/lightcurves']
        segments = list(lc_grp.keys())

        for seg in segments:
            for ch_name in ['SW', 'LW']:
                stage = best_stages.get((seg, ch_name), 'groupdiff')

                if stage == 'special':
                    if _add_corrected_lc('special_reduction', seg, ch_name, 'special_SW'):
                        dets = ['nrcblong'] if ch_name == 'LW' else ['nrcb1','nrcb2','nrcb3','nrcb4']
                        for d in dets: corrected_keys.add(f'{seg}_ramp_{d}')
                elif stage in ('sat_corrected', 'slope_corrected', 'sat_slope'):
                    if _add_corrected_lc(stage, seg, ch_name, f'{stage}_SW'):
                        dets = ['nrcblong'] if ch_name == 'LW' else ['nrcb1','nrcb2','nrcb3','nrcb4']
                        for d in dets: corrected_keys.add(f'{seg}_ramp_{d}')
                # 'groupdiff' falls through to the raw LC loop below

        # Serve raw (groupdiff + ZF) LCs for anything not covered above
        time_grp = h5[f'{target}/times']
        for seg in lc_grp:
            for mode in lc_grp[seg]:
                for det in lc_grp[seg][mode]:
                    lc_path = f'{seg}/{mode}/{det}'
                    lc_name = f'{seg}_{mode}_{det}'

                    if lc_name in corrected_keys:
                        continue

                    flux = lc_grp[lc_path][idx, :].astype(np.float64)
                    times = time_grp[lc_path][:].astype(np.float64)
                    valid = np.isfinite(flux) & (flux != 0)
                    if valid.sum() < 10:
                        continue
                    fv, tv = flux[valid], times[valid]
                    tv_hr = (tv - tv[0]) * 24.0 if tv[0] > 100 else tv
                    cs = 18 if mode == 'ramp' else 4
                    f_clean, t_clean = _clip_iqr(fv, tv_hr, chunk_size=cs)
                    f_clean, t_clean = _clip_iqr(f_clean, t_clean, chunk_size=cs)
                    if len(f_clean) < 4:
                        continue
                    bsz = 9 if mode == 'ramp' else 4
                    t_bin, f_bin = _bin_lc(t_clean, f_clean, bin_size=bsz)
                    med = float(np.nanmedian(f_clean))
                    std = float(np.nanstd(f_clean))
                    lcs[lc_name] = {
                        't': [round(float(x), 4) for x in t_clean],
                        'f': [round(float(x), 5) for x in f_clean],
                        'tb': t_bin, 'fb': f_bin,
                        'med': round(med, 1),
                        'amp': round(std / abs(med), 4) if med != 0 else 0,
                        'n': len(f_clean),
                        'detected': False,
                    }
    return {
        'id': int(src['master_id']),
        'ra': round(float(src['ra']), 7),
        'dec': round(float(src['dec']), 7),
        'snr': round(float(src['best_snr']), 1),
        'n_det': int(src['n_detections']),
        'lcs': lcs,
    }


# Mount static files last (catch-all)
_static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


def main():
    parser = argparse.ArgumentParser(description='Catalog viewer server')
    parser.add_argument('--port', type=int, default=8085)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()

    # Pre-load all catalogs from real catalog
    if not os.path.exists(REAL_CATALOG_PATH):
        print(f"ERROR: Real catalog not found at {REAL_CATALOG_PATH}")
        return

    for name, cfg in TARGETS.items():
        _, sources, _ = get_h5(name)
        n_ramp = sum(1 for s in sources if s['source_type'] == 'ramp')
        n_zf = sum(1 for s in sources if s['source_type'] == 'zf')
        print(f"{name}: {len(sources)} sources ({n_ramp} ramp, {n_zf} ZF)")

        # Check mosaics
        for seg in cfg['segments']:
            for imgtype in ['autocorr', 'i2d']:
                path = f'{MOSAIC_DIR}/{name}_{seg}_{imgtype}_mosaic.fits'
                if os.path.exists(path):
                    sz = os.path.getsize(path) / 1e6
                    print(f"  {os.path.basename(path)} ({sz:.1f} MB)")
                else:
                    print(f"  WARNING: {path} not found")

    print(f"\nServer: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port,
                log_level='info', access_log=True)


if __name__ == '__main__':
    main()
