"""Selected Table S8 sensitivity analyses around the frozen final v3 baseline."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one import PROJECT_DIR, run_model


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_table_s8_selected_sensitivity"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100020))

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
}

FROZEN_BASELINE = {
    "p_bi_m_to_msm": 0.70,
    "party_core_intensity": 0.70,
    "social_contact_scale": 6.0,
    "social_same_sex_assort": 0.75,
    "late_social_ramp": 6.0,
    "mix_mf_anal_frac": 0.13,
    "stable_mf_anal_frac": 0.00,
    "party_mix_homo_spill_frac": 0.01,
}

SCENARIOS = [
    {
        "scenario_id": "s8_001_baseline_frozen",
        "parameter": "baseline",
        "value": "frozen",
        "description": "Frozen final v3 calibrated baseline.",
        "direct_overrides": {},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_002_party_touch_frac_0_10",
        "parameter": "party_touch_frac",
        "value": 0.10,
        "description": "Shared event close-touch fraction set to 0.10.",
        "direct_overrides": {"party_touch_frac": 0.10},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_003_party_touch_frac_0_25",
        "parameter": "party_touch_frac",
        "value": 0.25,
        "description": "Shared event close-touch fraction set to 0.25.",
        "direct_overrides": {"party_touch_frac": 0.25},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_004_close_touch_0_004",
        "parameter": "close_touch",
        "value": 0.004,
        "description": "Per-contact close-touch transmission risk set to 0.004.",
        "direct_overrides": {"close_touch": 0.004},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_005_close_touch_0_010",
        "parameter": "close_touch",
        "value": 0.010,
        "description": "Per-contact close-touch transmission risk set to 0.010.",
        "direct_overrides": {"close_touch": 0.010},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_006_sexual_risk_scale_0_25",
        "parameter": "sexual_risk_scale",
        "value": 0.25,
        "description": "Sexual transmission risk scale set to 0.25.",
        "direct_overrides": {"sexual_risk_scale": 0.25},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_007_sexual_risk_scale_0_45",
        "parameter": "sexual_risk_scale",
        "value": 0.45,
        "description": "Sexual transmission risk scale set to 0.45.",
        "direct_overrides": {"sexual_risk_scale": 0.45},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_008_testing_delay_shift_minus1",
        "parameter": "testing_delay_shift_days",
        "value": -1,
        "description": "Testing delay means shifted by -1 day: phase1=4.0, phase2=3.0, phase3=3.5.",
        "direct_overrides": {},
        "policy_adjustments": {
            "testing_delay_mean_days_by_phase": {
                "phase1_growth": 4.0,
                "phase2_control": 3.0,
                "phase3_postpeak": 3.5,
            },
            "testing_delay_shift_days": -1,
        },
    },
    {
        "scenario_id": "s8_009_testing_delay_shift_plus1",
        "parameter": "testing_delay_shift_days",
        "value": 1,
        "description": "Testing delay means shifted by +1 day: phase1=6.0, phase2=5.0, phase3=5.5.",
        "direct_overrides": {},
        "policy_adjustments": {
            "testing_delay_mean_days_by_phase": {
                "phase1_growth": 6.0,
                "phase2_control": 5.0,
                "phase3_postpeak": 5.5,
            },
            "testing_delay_shift_days": 1,
        },
    },
    {
        "scenario_id": "s8_010_monitor_social_contact_mult_0_50",
        "parameter": "monitor_social_contact_mult",
        "value": 0.50,
        "description": "Monitoring social-contact multiplier set to 0.50.",
        "direct_overrides": {"monitor_social_contact_mult": 0.50},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_011_monitor_social_contact_mult_0_90",
        "parameter": "monitor_social_contact_mult",
        "value": 0.90,
        "description": "Monitoring social-contact multiplier set to 0.90.",
        "direct_overrides": {"monitor_social_contact_mult": 0.90},
        "policy_adjustments": {},
    },
    {
        "scenario_id": "s8_012_monitoring_days_14",
        "parameter": "MONITORING_DAYS",
        "value": 14,
        "description": "Monitoring duration set to 14 days.",
        "direct_overrides": {"MONITORING_DAYS": 14},
        "policy_adjustments": {},
    },
]

RESULT_FIELDS = [
    "timestamp",
    "scenario_id",
    "parameter",
    "value",
    "description",
    "seed",
    "pop_size",
    "run_id",
    "status",
    *FROZEN_BASELINE.keys(),
    "direct_overrides_json",
    "policy_adjustments_json",
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "homo_male",
    "bi_male",
    "MAE_pp",
    "error",
    "stdout_log",
    "stderr_log",
    "diagnostics_copy",
    "elapsed_s",
]

SUMMARY_FIELDS = [
    "scenario_id",
    "parameter",
    "value",
    "description",
    "successful_runs",
    "failed_runs",
    "total_cases_mean",
    "total_cases_sd",
    "hetero_mean",
    "hetero_sd",
    "bi_mean",
    "bi_sd",
    "homo_mean",
    "homo_sd",
    "female_known_mean",
    "female_known_sd",
    "MAE_pp_mean",
    "MAE_pp_sd",
    "delta_total_cases_vs_baseline",
    "delta_hetero_vs_baseline_pp",
    "delta_bi_vs_baseline_pp",
    "delta_homo_vs_baseline_pp",
    "delta_female_known_vs_baseline_pp",
    "delta_MAE_pp_vs_baseline",
    "case_composition_stable",
]


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.fmean(vals) if vals else math.nan


def _sd(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) > 1:
        return statistics.stdev(vals)
    if len(vals) == 1:
        return 0.0
    return math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _mae_pp(run: dict[str, Any]) -> float:
    value = _finite(run.get("objective_case_composition_mae"))
    if math.isfinite(value):
        return value
    errors = []
    for metric, target in TARGETS.items():
        current = _finite(run.get(metric))
        if math.isfinite(current):
            errors.append(abs(current - target) * 100.0)
    return statistics.fmean(errors) if errors else math.nan


def _copy_diagnostics(run_id: str) -> str:
    if not SOURCE_DIAGNOSTICS.exists() or SOURCE_DIAGNOSTICS.stat().st_size == 0:
        return ""
    dst = DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv"
    shutil.copy2(SOURCE_DIAGNOSTICS, dst)
    return str(dst)


def _scenario_overrides(scenario: dict[str, Any]) -> dict[str, float]:
    overrides = dict(FROZEN_BASELINE)
    overrides.update(scenario.get("direct_overrides", {}))
    return overrides


def _completed(results_path: Path) -> set[tuple[str, int]]:
    done = set()
    for row in _read_rows(results_path):
        if row.get("status") == "ok":
            done.add((row.get("scenario_id", ""), int(float(row.get("seed", -1)))))
    return done


def _run_row(scenario: dict[str, Any], seed: int, run: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario_id": scenario["scenario_id"],
        "parameter": scenario["parameter"],
        "value": scenario["value"],
        "description": scenario["description"],
        "seed": seed,
        "pop_size": POP_SIZE,
        "run_id": run.get("run_id", ""),
        "status": run.get("status", "failed"),
        **FROZEN_BASELINE,
        "direct_overrides_json": json.dumps(scenario.get("direct_overrides", {}), sort_keys=True),
        "policy_adjustments_json": json.dumps(scenario.get("policy_adjustments", {}), sort_keys=True),
        "error": run.get("error", ""),
        "stdout_log": run.get("stdout_log", ""),
        "stderr_log": run.get("stderr_log", ""),
        "elapsed_s": run.get("elapsed_s", ""),
    }
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male"]:
        row[metric] = run.get(metric, "")
    row["MAE_pp"] = _mae_pp(run)
    row["diagnostics_copy"] = _copy_diagnostics(str(run["run_id"])) if row["status"] == "ok" else ""
    return row


def _scenario_summary(scenario: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("scenario_id") == scenario["scenario_id"] and row.get("status") == "ok"]
    failed = [row for row in rows if row.get("scenario_id") == scenario["scenario_id"] and row.get("status") != "ok"]
    out: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "parameter": scenario["parameter"],
        "value": scenario["value"],
        "description": scenario["description"],
        "successful_runs": len(ok),
        "failed_runs": len(failed),
    }
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]:
        values = [_finite(row.get(metric)) for row in ok]
        out[f"{metric}_mean"] = _mean(values)
        out[f"{metric}_sd"] = _sd(values)
    out["case_composition_stable"] = (
        out["successful_runs"] == len(SEEDS)
        and out["MAE_pp_mean"] <= 1.5
        and out["hetero_mean"] >= 0.080
        and 0.115 <= out["bi_mean"] <= 0.138
        and out["homo_mean"] <= 0.805
        and 0.028 <= out["female_known_mean"] <= 0.045
    )
    return out


def _write_manifest() -> None:
    path = OUTPUT_DIR / "selected_table_s8_manifest.csv"
    fields = [
        "scenario_id",
        "parameter",
        "value",
        "description",
        "seed_start",
        "seed_end",
        "n_seeds",
        "pop_size",
        "direct_overrides_json",
        "policy_adjustments_json",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scenario in SCENARIOS:
            writer.writerow(
                {
                    "scenario_id": scenario["scenario_id"],
                    "parameter": scenario["parameter"],
                    "value": scenario["value"],
                    "description": scenario["description"],
                    "seed_start": min(SEEDS),
                    "seed_end": max(SEEDS),
                    "n_seeds": len(SEEDS),
                    "pop_size": POP_SIZE,
                    "direct_overrides_json": json.dumps(scenario.get("direct_overrides", {}), sort_keys=True),
                    "policy_adjustments_json": json.dumps(scenario.get("policy_adjustments", {}), sort_keys=True),
                }
            )


def _write_params() -> None:
    payload = {
        "model_version": "v3_final_table_s7_tertiary_calibrated_baseline",
        "frozen_baseline": FROZEN_BASELINE,
        "sensitivity_pop_size": POP_SIZE,
        "sensitivity_seeds": [min(SEEDS), max(SEEDS)],
        "notes": [
            "Selected Table S8 sensitivity analyses only; no recalibration or baseline reselection.",
            "party_hetero_leak, social_male_bias, and party_mix_female_mult are not restored.",
            "population.py is not modified by this workflow.",
        ],
    }
    (OUTPUT_DIR / "frozen_v3_baseline_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary(results_path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(results_path)
    summaries = [_scenario_summary(scenario, rows) for scenario in SCENARIOS]
    baseline = summaries[0]
    for summary in summaries:
        summary["delta_total_cases_vs_baseline"] = summary["total_cases_mean"] - baseline["total_cases_mean"]
        summary["delta_hetero_vs_baseline_pp"] = (summary["hetero_mean"] - baseline["hetero_mean"]) * 100.0
        summary["delta_bi_vs_baseline_pp"] = (summary["bi_mean"] - baseline["bi_mean"]) * 100.0
        summary["delta_homo_vs_baseline_pp"] = (summary["homo_mean"] - baseline["homo_mean"]) * 100.0
        summary["delta_female_known_vs_baseline_pp"] = (summary["female_known_mean"] - baseline["female_known_mean"]) * 100.0
        summary["delta_MAE_pp_vs_baseline"] = summary["MAE_pp_mean"] - baseline["MAE_pp_mean"]

    path = OUTPUT_DIR / "selected_table_s8_sensitivity_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in SUMMARY_FIELDS})
    return summaries


def _write_readme(summaries: list[dict[str, Any]]) -> None:
    stable_count = sum(1 for row in summaries if row.get("case_composition_stable"))
    lines = [
        "# Selected Table S8 sensitivity analyses",
        "",
        "These runs evaluate selected one-at-a-time sensitivity scenarios around the frozen final v3 calibrated baseline.",
        "They are not recalibration runs and do not select a new baseline.",
        "",
        f"Population size: {POP_SIZE:,}",
        f"Seeds: {min(SEEDS)}-{max(SEEDS)}",
        "",
        "MAKE_PLOTS was set to False during batch runs through temporary run-time patching and restored after each run.",
        "",
        "Stability criterion used here: successful 20/20 seeds, mean MAE_pp <= 1.5, heterosexual >= 8.0%, bisexual 11.5%-13.8%, homosexual <= 80.5%, and female_known 2.8%-4.5%.",
        "",
        f"Stable scenarios by this criterion: {stable_count}/{len(summaries)}.",
        "",
        "## Frozen baseline",
        "",
        "```json",
        json.dumps(FROZEN_BASELINE, indent=2),
        "```",
        "",
        "## Scenario summary",
        "",
        "| Scenario | total cases | hetero | bi | homo | female_known | MAE_pp | stable |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['scenario_id']} | "
            f"{row['total_cases_mean']:.2f} | "
            f"{row['hetero_mean']:.4f} | "
            f"{row['bi_mean']:.4f} | "
            f"{row['homo_mean']:.4f} | "
            f"{row['female_known_mean']:.4f} | "
            f"{row['MAE_pp_mean']:.3f} | "
            f"{'yes' if row['case_composition_stable'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "No Table S7 recalibration or parameter expansion was performed.",
        ]
    )
    (OUTPUT_DIR / "selected_table_s8_sensitivity_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    _write_manifest()
    _write_params()
    results_path = OUTPUT_DIR / "selected_table_s8_sensitivity_results.csv"
    done = _completed(results_path)

    for scenario in SCENARIOS:
        for seed in SEEDS:
            if (scenario["scenario_id"], seed) in done:
                print(f"[skip] {scenario['scenario_id']} seed={seed}", flush=True)
                continue
            run_id = f"{scenario['scenario_id']}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            print(f"[run] {scenario['scenario_id']} seed={seed}", flush=True)
            run = run_model(
                _scenario_overrides(scenario),
                seed=seed,
                pop_size=POP_SIZE,
                run_id=run_id,
                make_plots=False,
                log_dir=LOG_DIR,
                copy_plots=False,
                policy_adjustments=scenario.get("policy_adjustments", {}),
            )
            row = _run_row(scenario, seed, run)
            _append_row(results_path, row)
            print(
                f"[done] {scenario['scenario_id']} seed={seed} status={row['status']} "
                f"total_cases={row.get('total_cases')} MAE_pp={row.get('MAE_pp')}",
                flush=True,
            )

    summaries = _write_summary(results_path)
    _write_readme(summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
