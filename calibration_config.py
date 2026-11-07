"""Configuration for reproducible mpox ABM calibration.

Only Table S7 primary calibration parameters are enabled by default. Secondary
parameters are defined for later sensitivity/refinement work but are not used
unless explicitly requested.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_DIR / "main.py"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CALIBRATION_OUTPUT_DIR = OUTPUT_DIR

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
    "homo_male": 0.7796,
    "bi_male": 0.1272,
}

SEARCH_SPACE_PRIMARY = {
    "p_bi_m_to_msm": [0.55, 0.75],
    "party_core_intensity": [0.45, 0.70],
    "social_contact_scale": [2.5, 6.0],
    "social_same_sex_assort": [0.65, 0.85],
    "late_social_ramp": [1.0, 6.0],
}

# Implemented but disabled by default. Do not include these in primary runs.
SEARCH_SPACE_SECONDARY = {
    "mix_mf_anal_frac": [0.08, 0.18],
    "stable_mf_anal_frac": [0.03, 0.08],
    "oneoff_active_frac": [0.005, 0.020],
    "oneoff_p": [0.10, 0.25],
    "party_mix_target_mean": [1.5, 2.5],
}

BASELINE_PARAMS = {
    "p_bi_m_to_msm": 0.66,
    "party_core_intensity": 0.52,
    "social_contact_scale": 5.0,
    "social_same_sex_assort": 0.75,
    "late_social_ramp": 5.5,
}

DEFAULT_RUN_SETTINGS = {
    "n_samples": 200,
    "n_seeds": 3,
    "pop_size": 1_000_000,
    "base_seed": 100000,
    "make_plots": False,
    "diagnostic_plot_seeds": [100000],
}

SCORING_WEIGHTS = {
    "case_composition_mae": 1.0,
    "cumulative_case_error": 0.0,
    "peak_timing_error": 0.0,
    "late_heterosexual_male_growth_error": 0.0,
}

# Optional observed incidence targets. Leave as None unless a clean observed
# series is supplied; objective.py will skip unavailable terms.
OBSERVED_DAILY_CASES = None
OBSERVED_CUMULATIVE_CASES = None
OBSERVED_LATE_HETEROSEXUAL_MALE_GROWTH = None

RESULTS_CSV = OUTPUT_DIR / "calibration_results.csv"
BEST_PARAMS_JSON = OUTPUT_DIR / "best_params.json"
BEST_SUMMARY_CSV = OUTPUT_DIR / "best_summary.csv"
DIAGNOSTIC_PLOT_DIR = OUTPUT_DIR / "calibration_diagnostic_plots"

