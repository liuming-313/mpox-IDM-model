"""Objective functions for mpox ABM calibration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from calibration_config import (
    OBSERVED_CUMULATIVE_CASES,
    OBSERVED_DAILY_CASES,
    OBSERVED_LATE_HETEROSEXUAL_MALE_GROWTH,
    SCORING_WEIGHTS,
    TARGETS,
)


COMPOSITION_METRICS = ["hetero", "bi", "homo", "female_known"]


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def case_composition_mae(summary: dict[str, Any], targets: dict[str, float] = TARGETS) -> float:
    """Mean absolute error in percentage points for the primary composition targets."""
    errors = []
    for metric in COMPOSITION_METRICS:
        current = _finite_float(summary.get(metric))
        if math.isnan(current):
            continue
        errors.append(abs(current - float(targets[metric])) * 100.0)
    return float(np.mean(errors)) if errors else math.nan


def _normalized_rmse(sim: Sequence[float] | None, obs: Sequence[float] | None) -> float | None:
    if sim is None or obs is None:
        return None
    n = min(len(sim), len(obs))
    if n == 0:
        return None
    s = np.asarray(sim[:n], dtype=float)
    o = np.asarray(obs[:n], dtype=float)
    mask = np.isfinite(s) & np.isfinite(o)
    if not np.any(mask):
        return None
    denom = max(float(np.mean(o[mask])), 1.0)
    return float(np.sqrt(np.mean((s[mask] - o[mask]) ** 2)) / denom)


def cumulative_case_error(summary: dict[str, Any], observed_cumulative: Sequence[float] | None = OBSERVED_CUMULATIVE_CASES) -> float | None:
    """Optional normalized RMSE against observed cumulative cases."""
    sim_series = summary.get("daily_cumulative_cases") or summary.get("cumulative_cases")
    if sim_series is not None and observed_cumulative is not None:
        return _normalized_rmse(sim_series, observed_cumulative)
    observed_final = summary.get("observed_cumulative_cases")
    total_cases = summary.get("total_cases")
    if observed_final is not None and total_cases is not None:
        obs = max(_finite_float(observed_final), 1.0)
        return abs(_finite_float(total_cases) - obs) / obs
    return None


def peak_timing_error(summary: dict[str, Any], observed_daily: Sequence[float] | None = OBSERVED_DAILY_CASES) -> float | None:
    """Optional absolute difference in peak day, scaled to a 244-day season."""
    sim_daily = summary.get("daily_cases")
    if sim_daily is None or observed_daily is None:
        return None
    n = min(len(sim_daily), len(observed_daily))
    if n == 0:
        return None
    sim_peak = int(np.argmax(np.asarray(sim_daily[:n], dtype=float)))
    obs_peak = int(np.argmax(np.asarray(observed_daily[:n], dtype=float)))
    return abs(sim_peak - obs_peak) / max(n, 1)


def late_heterosexual_male_growth_error(
    summary: dict[str, Any],
    observed_growth: float | None = OBSERVED_LATE_HETEROSEXUAL_MALE_GROWTH,
) -> float | None:
    """Optional error term for late heterosexual-male growth if supplied."""
    if observed_growth is None:
        return None
    current = summary.get("late_heterosexual_male_growth")
    if current is None:
        return None
    denom = max(abs(float(observed_growth)), 1e-9)
    return abs(_finite_float(current) - float(observed_growth)) / denom


def calculate_objective(summary: dict[str, Any], weights: dict[str, float] = SCORING_WEIGHTS) -> dict[str, float | None]:
    comp = case_composition_mae(summary)
    cum = cumulative_case_error(summary)
    peak = peak_timing_error(summary)
    late = late_heterosexual_male_growth_error(summary)

    components = {
        "case_composition_mae": comp,
        "cumulative_case_error": cum,
        "peak_timing_error": peak,
        "late_heterosexual_male_growth_error": late,
    }
    score = 0.0
    used_weight = 0.0
    for name, value in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0.0 or value is None or math.isnan(float(value)):
            continue
        score += weight * float(value)
        used_weight += weight
    components["score"] = score / used_weight if used_weight > 0 else comp
    return components

