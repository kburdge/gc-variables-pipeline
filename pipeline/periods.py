"""Period search: Lomb-Scargle and Box Least Squares.

Peak significance is reported as (peak - median) / MAD of the periodogram power,
matching the paper. Ported from the period-search block of ramp_pipeline.py.
"""
from __future__ import annotations

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle


def compute_significance(power, peak_power):
    """(peak - median) / MAD of the periodogram power values."""
    power = np.asarray(power, dtype=float)
    if not np.isfinite(power).any() or not np.isfinite(peak_power):
        return np.nan
    p = power[np.isfinite(power)]
    median = np.median(p)
    mad = np.median(np.abs(p - median))
    if mad <= 0:
        return np.nan
    return float((peak_power - median) / mad)


def lomb_scargle(t, y, freq_min_cph, freq_max_cph, n_points=5000):
    """Lomb-Scargle over a linear frequency grid (cycles/hr). t in hours.

    Returns dict with best_period_min, ls_significance, freqs, power.
    """
    freqs = np.linspace(freq_min_cph, freq_max_cph, n_points)
    ls = LombScargle(t, y, fit_mean=True, center_data=True)
    power = ls.power(freqs)
    out = {"freqs": freqs, "power": power, "best_period_min": np.nan, "ls_significance": np.nan}
    if np.isfinite(power).any():
        bi = int(np.nanargmax(power))
        best_f = float(freqs[bi])
        out["best_period_min"] = 60.0 / best_f if best_f > 0 else np.nan
        out["ls_significance"] = compute_significance(power, float(power[bi]))
    return out


def box_least_squares(t, y, period_min_hr, period_max_hr, duration_min_min, duration_max_min):
    """BLS over a linear period grid. t in hours. Returns dict with bls_significance."""
    out = {"bls_significance": np.nan, "bls_period_min": np.nan}
    try:
        bls = BoxLeastSquares(t, y)
        periods = np.linspace(period_min_hr, period_max_hr, 1000)
        durations = [duration_min_min / 60.0, duration_max_min / 60.0]
        res = bls.power(periods, duration=durations)
        power = res.power
        if np.isfinite(power).any():
            bi = int(np.nanargmax(power))
            out["bls_significance"] = compute_significance(power, float(power[bi]))
            out["bls_period_min"] = float(periods[bi] * 60.0)
    except Exception:
        pass
    return out


def search(t_hours, y, cfg):
    """Run both searches using the period_search section of the config."""
    ps = cfg["period_search"]
    ls = lomb_scargle(t_hours, y, ps["lomb_scargle"]["freq_min_cph"],
                      ps["lomb_scargle"]["freq_max_cph"], ps["lomb_scargle"].get("n_points", 5000))
    b = ps["bls"]
    bls = box_least_squares(t_hours, y, b["period_min_hr"], b["period_max_hr"],
                            b["duration_min_min"], b["duration_max_min"])
    return {**ls, **bls}
