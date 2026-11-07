"""Table S8 one-at-a-time sensitivity configuration.

Prepared for future runs only. These scenarios are sensitivity analyses around
the frozen Table S7 calibrated baseline; they are not additional calibration
candidates.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
FREEZE_DIR = PROJECT_DIR / "outputs" / "final_model_freeze"
FROZEN_BASELINE_JSON = FREEZE_DIR / "final_baseline_params.json"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "sensitivity_table_s8"

SEEDS = list(range(100000, 100020))
POP_SIZE = 1_000_000

# Frozen accepted baseline from current main.py. The runner will also read
# final_baseline_params.json and verify these values before launching jobs.
FROZEN_TABLE_S7_BASELINE = {
    "p_bi_m_to_msm": 0.66,
    "party_core_intensity": 0.52,
    "social_contact_scale": 5.0,
    "social_same_sex_assort": 0.75,
    "late_social_ramp": 5.5,
}

# Frozen non-calibration baseline values used by joint sensitivity scenarios.
BASELINE_STABLE_ACTIVATION = {
    "p_stable_sex_general_per_day": 0.10,
    "p_stable_sex_one_mcp_per_day": 0.26,
    "p_stable_sex_both_mcp_per_day": 0.39,
}

BASELINE_PHASE_TESTING = {
    "phase1_growth": {
        "test_pathway_prob": 0.75,
        "test_seek_delay_mean_days": 5.0,
    },
    "phase2_control": {
        "test_pathway_prob": 0.90,
        "test_seek_delay_mean_days": 4.0,
    },
    "phase3_postpeak": {
        "test_pathway_prob": 0.85,
        "test_seek_delay_mean_days": 4.5,
    },
}

# One-at-a-time Table S8 sensitivity values. These do not vary the calibrated
# Table S7 primary parameters.
TABLE_S8_ONE_AT_A_TIME = {
    "party_touch_frac": [0.10, 0.20, 0.25],
    "close_touch": [0.004, 0.0067, 0.010],
    "sexual_risk_scale": [0.25, 0.35, 0.45],
    "stable_activation_mult": [0.8, 1.0, 1.2],
    "testing_prob_mult": [0.9, 1.0, 1.1],
    "testing_delay_shift_days": [-1.0, 0.0, 1.0],
    "monitor_sexual_contact_mult": [0.20, 0.30, 0.50],
    "monitor_social_contact_mult": [0.50, 0.70, 0.90],
    "MONITORING_DAYS": [14, 21],
}
