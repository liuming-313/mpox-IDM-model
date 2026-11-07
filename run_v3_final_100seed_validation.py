"""Final 100-seed validation for the frozen v3 calibrated baseline."""

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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_final_100seed_validation"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"
POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100100))

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
    "homo_male": 0.7796,
    "bi_male": 0.1272,
}

FINAL_PARAMS = {
    "p_bi_m_to_msm": 0.70,
    "party_core_intensity": 0.70,
    "social_contact_scale": 6.0,
    "social_same_sex_assort": 0.75,
    "late_social_ramp": 6.0,
    "mix_mf_anal_frac": 0.13,
    "stable_mf_anal_frac": 0.00,
    "party_mix_homo_spill_frac": 0.01,
}

RESULT_FIELDS = [
    "timestamp",
    "run_id",
    "seed",
    "pop_size",
    "status",
    *FINAL_PARAMS.keys(),
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "homo_male",
    "bi_male",
    "MAE_pp",
    "diff_hetero_pp",
    "diff_bi_pp",
    "diff_homo_pp",
    "diff_female_known_pp",
    "diff_homo_male_pp",
    "diff_bi_male_pp",
    "error",
    "stdout_log",
    "stderr_log",
    "diagnostics_copy",
    "elapsed_s",
]

SUMMARY_FIELDS = [
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
    "homo_male_mean",
    "homo_male_sd",
    "bi_male_mean",
    "bi_male_sd",
    "MAE_pp_mean",
    "MAE_pp_sd",
    "prop_MAE_le_1_0",
    "prop_MAE_le_1_2",
    "prop_MAE_le_1_5",
    "passes_acceptance",
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


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _append_result(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _copy_diagnostics(run_id: str) -> str:
    if not SOURCE_DIAGNOSTICS.exists() or SOURCE_DIAGNOSTICS.stat().st_size == 0:
        return ""
    dst = DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv"
    shutil.copy2(SOURCE_DIAGNOSTICS, dst)
    return str(dst)


def _mae_pp(row: dict[str, Any]) -> float:
    value = _finite(row.get("objective_case_composition_mae"))
    if math.isfinite(value):
        return value
    errors = []
    for metric in ["hetero", "bi", "homo", "female_known"]:
        current = _finite(row.get(metric))
        if math.isfinite(current):
            errors.append(abs(current - TARGETS[metric]) * 100.0)
    return statistics.fmean(errors) if errors else math.nan


def _result_row(seed: int, run: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run["run_id"],
        "seed": seed,
        "pop_size": POP_SIZE,
        "status": run.get("status", "failed"),
        **FINAL_PARAMS,
        "error": run.get("error", ""),
        "stdout_log": run.get("stdout_log", ""),
        "stderr_log": run.get("stderr_log", ""),
        "elapsed_s": run.get("elapsed_s", ""),
    }
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male"]:
        row[metric] = run.get(metric, "")
    row["MAE_pp"] = _mae_pp(run)
    for metric, target in TARGETS.items():
        current = _finite(row.get(metric))
        row[f"diff_{metric}_pp"] = (current - target) * 100.0 if math.isfinite(current) else ""
    row["diagnostics_copy"] = _copy_diagnostics(str(run["run_id"])) if row["status"] == "ok" else ""
    return row


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    summary: dict[str, Any] = {"successful_runs": len(ok), "failed_runs": len(failed)}
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male", "MAE_pp"]:
        values = [_finite(row.get(metric)) for row in ok]
        summary[f"{metric}_mean"] = _mean(values)
        summary[f"{metric}_sd"] = _sd(values)
    mae_values = [_finite(row.get("MAE_pp")) for row in ok]
    denom = len([v for v in mae_values if math.isfinite(v)])
    for threshold, label in [(1.0, "1_0"), (1.2, "1_2"), (1.5, "1_5")]:
        summary[f"prop_MAE_le_{label}"] = (
            sum(1 for v in mae_values if math.isfinite(v) and v <= threshold) / denom if denom else math.nan
        )
    summary["passes_acceptance"] = (
        len(ok) == len(SEEDS)
        and summary["MAE_pp_mean"] <= 1.2
        and 0.115 <= summary["bi_mean"] <= 0.138
        and 0.028 <= summary["female_known_mean"] <= 0.045
        and summary["hetero_mean"] >= 0.080
        and summary["homo_mean"] <= 0.805
    )
    return summary


def _write_summary(summary: dict[str, Any]) -> None:
    path = OUTPUT_DIR / "final_100seed_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in SUMMARY_FIELDS})


def _write_params() -> None:
    payload = {
        "model_version": "v3_final_table_s7_tertiary_calibrated_baseline",
        "pop_size_for_formal_validation": POP_SIZE,
        "validated_seed_range": "100000-100099",
        "parameters": FINAL_PARAMS,
        "fixed_not_restored": ["party_hetero_leak", "social_male_bias", "party_mix_female_mult"],
        "notes": [
            "Final baseline selected after tertiary 20-seed 10M validation passed acceptance criteria.",
            "Disease natural history, testing, monitoring, policy timing, importations, social_p_eff, close_touch, party_touch_frac, sexual_risk_scale, and age-linked susceptibility were not calibrated here.",
        ],
    }
    (OUTPUT_DIR / "final_v3_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_readme(summary: dict[str, Any]) -> None:
    status = "passes" if summary.get("passes_acceptance") else "does not pass"
    text = f"""# Final v3 100-seed validation

This directory contains the formal 100-seed validation for the frozen v3 calibrated baseline.

## Frozen baseline

```json
{json.dumps(FINAL_PARAMS, indent=2)}
```

Population size: {POP_SIZE:,}

Seeds: 100000-100099

MAKE_PLOTS was set to False during batch validation runs through temporary run-time patching and the canonical `main.py` baseline was restored after each run.

## Summary

- successful runs: {summary['successful_runs']}
- failed runs: {summary['failed_runs']}
- total_cases mean: {summary['total_cases_mean']:.3f}; SD: {summary['total_cases_sd']:.3f}
- heterosexual mean: {summary['hetero_mean']:.6f}; SD: {summary['hetero_sd']:.6f}
- bisexual mean: {summary['bi_mean']:.6f}; SD: {summary['bi_sd']:.6f}
- homosexual mean: {summary['homo_mean']:.6f}; SD: {summary['homo_sd']:.6f}
- female_known mean: {summary['female_known_mean']:.6f}; SD: {summary['female_known_sd']:.6f}
- homo_male mean: {summary['homo_male_mean']:.6f}; SD: {summary['homo_male_sd']:.6f}
- bi_male mean: {summary['bi_male_mean']:.6f}; SD: {summary['bi_male_sd']:.6f}
- MAE_pp mean: {summary['MAE_pp_mean']:.3f}; SD: {summary['MAE_pp_sd']:.3f}
- proportion with MAE_pp <= 1.0: {summary['prop_MAE_le_1_0']:.3f}
- proportion with MAE_pp <= 1.2: {summary['prop_MAE_le_1_2']:.3f}
- proportion with MAE_pp <= 1.5: {summary['prop_MAE_le_1_5']:.3f}

The 100-seed validation {status} the predefined acceptance criteria:
mean MAE_pp <= 1.2, bisexual 11.5%-13.8%, female_known 2.8%-4.5%,
heterosexual >= 8.0%, and homosexual <= 80.5%.

No Table S8 sensitivity simulations were run as part of this validation.
"""
    (OUTPUT_DIR / "final_100seed_README.md").write_text(text, encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    _write_params()
    results_path = OUTPUT_DIR / "final_100seed_results.csv"
    completed = {int(float(row["seed"])) for row in _read_results(results_path) if row.get("status") == "ok"}

    for seed in SEEDS:
        if seed in completed:
            print(f"[skip] seed={seed} already completed", flush=True)
            continue
        run_id = f"v3_final_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        print(f"[run] seed={seed} run_id={run_id}", flush=True)
        run = run_model(
            FINAL_PARAMS,
            seed=seed,
            pop_size=POP_SIZE,
            run_id=run_id,
            make_plots=False,
            log_dir=LOG_DIR,
            copy_plots=False,
        )
        row = _result_row(seed, run)
        _append_result(results_path, row)
        print(
            f"[done] seed={seed} status={row['status']} total_cases={row.get('total_cases')} MAE_pp={row.get('MAE_pp')}",
            flush=True,
        )

    rows = _read_results(results_path)
    summary = _summarize(rows)
    _write_summary(summary)
    _write_readme(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
