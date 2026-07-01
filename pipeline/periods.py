"""Period search: Lomb-Scargle and Box Least Squares.

Ported from run_period_search.py — the search that produced the published
periods and significances (run on the best-stage corrected lightcurves; the
paper's Section on period searching describes this implementation):

* LS over a linear frequency grid (default 5000 points, 20 min - 12 hr);
  significance = (peak - median) / MAD of the periodogram power.
* BLS over a linear period grid (default 1000 periods, same range) with 30
  geometrically-spaced trial durations from 1 min up to 0.95x the minimum
  period; significance = (peak - mean) / std (the standard BLS SDE).
* Sources with fewer than 50 points, and any search failure, return the
  720-minute sentinel ("no period detected") with significance 0.

Historical note: the *extraction-stage* search in the original ramp_pipeline
used a wider grid (1 min - 12 hr) plus a post-hoc high-frequency rejection at
20 min; the final published periods come from this implementation instead.
"""
from __future__ import annotations

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle

SENTINEL_MIN = 720.0   # "no period detected"
MIN_POINTS = 50


def lomb_scargle(t, y, freq_min_cph, freq_max_cph, n_points=5000):
    """Lomb-Scargle over a linear frequency grid (cycles/hr). t in hours.

    Returns dict with best_period_min, ls_significance, freqs, power.
    """
    out = {"freqs": None, "power": None,
           "best_period_min": SENTINEL_MIN, "ls_significance": 0.0}
    if len(t) < MIN_POINTS:
        return out
    freqs = np.linspace(freq_min_cph, freq_max_cph, n_points)
    ls = LombScargle(t, y)
    power = ls.power(freqs)
    out["freqs"], out["power"] = freqs, power
    if np.isfinite(power).any():
        bi = int(np.nanargmax(power))
        best_f = float(freqs[bi])
        if best_f > 0:
            out["best_period_min"] = 60.0 / best_f
        med = np.median(power)
        mad = np.median(np.abs(power - med))
        out["ls_significance"] = float((power[bi] - med) / mad) if mad > 0 else 0.0
    return out


def box_least_squares(t, y, period_min_hr, period_max_hr,
                      duration_min_min=1.0, n_periods=1000, n_durations=30):
    """BLS over a linear period grid. t in hours.

    Durations are geomspaced from duration_min_min up to 0.95x the minimum
    trial period (BLS requires max duration < min period). Significance is the
    BLS Signal Detection Efficiency, (peak - mean) / std.
    """
    out = {"bls_period_min": SENTINEL_MIN, "bls_depth": 0.0, "bls_significance": 0.0}
    if len(t) < MIN_POINTS:
        return out
    min_dur_hr = duration_min_min / 60.0
    max_dur_hr = 0.95 * period_min_hr
    if max_dur_hr <= min_dur_hr:
        return out
    durations = np.geomspace(min_dur_hr, max_dur_hr, n_durations)
    periods = np.linspace(period_min_hr, period_max_hr, n_periods)
    try:
        bls = BoxLeastSquares(t, y)
        result = bls.power(periods, durations)
        bi = int(np.argmax(result.power))
        out["bls_period_min"] = float(result.period[bi]) * 60.0
        out["bls_depth"] = float(result.depth[bi])
        mean_pow = np.mean(result.power)
        std_pow = np.std(result.power)
        out["bls_significance"] = (
            float((result.power[bi] - mean_pow) / std_pow) if std_pow > 0 else 0.0
        )
    except Exception:
        pass
    return out


def search(t_hours, y, cfg, mode="ramp"):
    """Run both searches using the period_search section of the config.

    mode="zf" uses the zeroframe LS grid (period_search.zf) when present.
    """
    ps = cfg["period_search"]
    ls_cfg = ps.get("zf", ps["lomb_scargle"]) if mode == "zf" else ps["lomb_scargle"]
    ls = lomb_scargle(t_hours, y, ls_cfg["freq_min_cph"], ls_cfg["freq_max_cph"],
                      ls_cfg.get("n_points", ps["lomb_scargle"].get("n_points", 5000)))
    b = ps["bls"]
    bls = box_least_squares(
        t_hours, y, b["period_min_hr"], b["period_max_hr"],
        duration_min_min=b.get("duration_min_min", 1.0),
        n_periods=b.get("n_periods", 1000), n_durations=b.get("n_durations", 30),
    )
    return {**ls, **bls}
