"""Prepared runner for Table S8 one-at-a-time sensitivity analyses.

This script is intentionally not run during model freeze. When launched later,
it varies one Table S8 sensitivity parameter or joint scenario at a time,
records every seed, and uses run_one.run_model() so main.py is restored after
each subprocess.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one import run_model
from table_s8_sensitivity_config import (
    BASELINE_PHASE_TESTING,
    BASELINE_STABLE_ACTIVATION,
    FROZEN_BASELINE_JSON,
    FROZEN_TABLE_S7_BASELINE,
    OUTPUT_DIR,
    POP_SIZE,
    SEEDS,
    TABLE_S8_ONE_AT_A_TIME,
)


FIELDS = [
    "timestamp",
    "parameter",
    "value",
    "description",
    "direct_param_overrides",
    "policy_adjustments",
    "seed",
    "pop_size",
    "run_id",
    "status",
    *FROZEN_TABLE_S7_BASELINE.keys(),
    "party_touch_frac",
    "close_touch",
    "sexual_risk_scale",
    "p_stable_sex_general_per_day",
    "p_stable_sex_one_mcp_per_day",
    "p_stable_sex_both_mcp_per_day",
    "monitor_sexual_contact_mult",
    "monitor_social_contact_mult",
    "MONITORING_DAYS",
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "homo_male",
    "bi_male",
    "MAE_pp",
    "score",
    "objective_case_composition_mae",
    "error",
    "stdout_log",
    "stderr_log",
    "elapsed_s",
]

SUMMARY_FIELDS = [
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

BASELINE_SCENARIO_VALUES = {
    "party_touch_frac": 0.20,
    "close_touch": 0.0067,
    "sexual_risk_scale": 0.35,
    "stable_activation_mult": 1.0,
    "testing_prob_mult": 1.0,
    "testing_delay_shift_days": 0.0,
    "monitor_sexual_contact_mult": 0.30,
    "monitor_social_contact_mult": 0.70,
    "MONITORING_DAYS": 21.0,
}

STABILITY_BOUNDS = {
    "hetero": (0.075, 0.115),
    "bi": (0.105, 0.150),
    "homo": (0.740, 0.825),
    "female_known": (0.020, 0.055),
}


def iter_scenarios(only_parameter: str | None = None):
    items = TABLE_S8_ONE_AT_A_TIME.items()
    if only_parameter:
        items = [(only_parameter, TABLE_S8_ONE_AT_A_TIME[only_parameter])]
    for parameter, values in items:
        for value in values:
            yield parameter, float(value)


def build_scenario(parameter: str, value: float, baseline: dict[str, float]) -> tuple[dict[str, float], dict[str, float], str]:
    """Return direct parameter overrides, policy adjustments, and description."""
    direct = dict(baseline)
    policy_adjustments: dict[str, float] = {}

    if parameter == "stable_activation_mult":
        for key, base_value in BASELINE_STABLE_ACTIVATION.items():
            direct[key] = float(base_value) * float(value)
        description = (
            f"Joint multiplier {value:g} applied to p_stable_sex_general_per_day, "
            "p_stable_sex_one_mcp_per_day, and p_stable_sex_both_mcp_per_day."
        )
    elif parameter == "testing_prob_mult":
        policy_adjustments["testing_prob_mult"] = float(value)
        adjusted = {
            phase: min(1.0, vals["test_pathway_prob"] * float(value))
            for phase, vals in BASELINE_PHASE_TESTING.items()
        }
        description = f"Joint multiplier {value:g} applied to phase-specific test_pathway_prob values, capped at 1.0: {adjusted}."
    elif parameter == "testing_delay_shift_days":
        policy_adjustments["testing_delay_shift_days"] = float(value)
        adjusted = {
            phase: max(0.0, vals["test_seek_delay_mean_days"] + float(value))
            for phase, vals in BASELINE_PHASE_TESTING.items()
        }
        policy_adjustments["testing_delay_mean_days_by_phase"] = adjusted
        description = f"Joint shift {value:g} days applied to phase-specific test_seek_delay_mean_days values: {adjusted}."
    else:
        direct[parameter] = float(value)
        description = f"One-at-a-time sensitivity: {parameter} set to {value:g}."

    return direct, policy_adjustments, description


def load_frozen_baseline() -> dict[str, float]:
    if not FROZEN_BASELINE_JSON.exists():
        raise FileNotFoundError(f"Missing frozen baseline JSON: {FROZEN_BASELINE_JSON}")
    data = json.loads(FROZEN_BASELINE_JSON.read_text(encoding="utf-8"))
    params = data["table_s7_primary_parameters"]
    for key, value in FROZEN_TABLE_S7_BASELINE.items():
        if abs(float(params[key]) - float(value)) > 1e-9:
            raise ValueError(f"Frozen baseline mismatch for {key}: {params[key]} != {value}")
    return {key: float(value) for key, value in params.items()}


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def completed(path: Path) -> set[tuple[str, float, int]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {
            (row["parameter"], float(row["value"]), int(row["seed"]))
            for row in csv.DictReader(f)
            if row.get("status") == "ok"
        }


def _float_or_nan(value: Any) -> float:
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
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else math.nan


def _fmt(value: Any, digits: int = 4) -> str:
    val = _float_or_nan(value)
    return "" if not math.isfinite(val) else f"{val:.{digits}f}"


def _scenario_key(parameter: str, value: Any) -> tuple[str, float]:
    return (parameter, round(float(value), 10))


def _case_composition_stable(row: dict[str, Any]) -> str:
    for metric, (lo, hi) in STABILITY_BOUNDS.items():
        value = _float_or_nan(row.get(f"{metric}_mean"))
        if not math.isfinite(value) or value < lo or value > hi:
            return "no"
    mae = _float_or_nan(row.get("MAE_pp_mean"))
    return "yes" if math.isfinite(mae) and mae <= 1.5 else "no"


def write_summary_and_readme(results_path: Path, summary_path: Path, readme_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as f:
        manifest_rows = list(csv.DictReader(f))
    descriptions = {
        _scenario_key(row["parameter"], row["value"]): row.get("description", "")
        for row in manifest_rows
    }

    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    with results_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(_scenario_key(row["parameter"], row["value"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (list(TABLE_S8_ONE_AT_A_TIME).index(item[0]), item[1])):
        parameter, value = key
        scenario_rows = grouped[key]
        ok_rows = [row for row in scenario_rows if row.get("status") == "ok"]
        failed_rows = [row for row in scenario_rows if row.get("status") != "ok"]
        summary = {
            "parameter": parameter,
            "value": value,
            "description": descriptions.get(key, ""),
            "successful_runs": len(ok_rows),
            "failed_runs": len(failed_rows),
        }
        for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]:
            vals = [_float_or_nan(row.get(metric)) for row in ok_rows]
            summary[f"{metric}_mean"] = _mean(vals)
            summary[f"{metric}_sd"] = _sd(vals)
        rows.append(summary)

    baselines = {
        parameter: next(
            (
                row for row in rows
                if row["parameter"] == parameter
                and abs(float(row["value"]) - float(BASELINE_SCENARIO_VALUES[parameter])) < 1e-9
            ),
            None,
        )
        for parameter in BASELINE_SCENARIO_VALUES
    }

    for row in rows:
        base = baselines.get(row["parameter"])
        if base:
            row["delta_total_cases_vs_baseline"] = _float_or_nan(row["total_cases_mean"]) - _float_or_nan(base["total_cases_mean"])
            row["delta_hetero_vs_baseline_pp"] = (_float_or_nan(row["hetero_mean"]) - _float_or_nan(base["hetero_mean"])) * 100.0
            row["delta_bi_vs_baseline_pp"] = (_float_or_nan(row["bi_mean"]) - _float_or_nan(base["bi_mean"])) * 100.0
            row["delta_homo_vs_baseline_pp"] = (_float_or_nan(row["homo_mean"]) - _float_or_nan(base["homo_mean"])) * 100.0
            row["delta_female_known_vs_baseline_pp"] = (_float_or_nan(row["female_known_mean"]) - _float_or_nan(base["female_known_mean"])) * 100.0
            row["delta_MAE_pp_vs_baseline"] = _float_or_nan(row["MAE_pp_mean"]) - _float_or_nan(base["MAE_pp_mean"])
        row["case_composition_stable"] = _case_composition_stable(row)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    stable_count = sum(1 for row in rows if row["case_composition_stable"] == "yes")
    failed_total = sum(int(row["failed_runs"]) for row in rows)
    readme_lines = [
        "# Table S8 one-at-a-time sensitivity results",
        "",
        "These analyses are sensitivity analyses around the frozen Table S7 calibrated baseline. They are not recalibration runs, and no new baseline is selected from these results.",
        "",
        "## Run design",
        "",
        "- Scenarios: 26 one-at-a-time Table S8 sensitivity scenarios",
        "- Seeds per scenario: 20 (`100000-100019`)",
        "- Population size: `1,000,000`",
        "- Frozen Table S7 baseline: `p_bi_m_to_msm=0.66`, `party_core_intensity=0.52`, `social_contact_scale=5.0`, `social_same_sex_assort=0.75`, `late_social_ramp=5.5`",
        "- `main.py` is temporarily patched per subprocess and restored after each run; `population.py` is not modified.",
        "",
        "## Output files",
        "",
        f"- Per-seed results: `{results_path}`",
        f"- Scenario summary: `{summary_path}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Stability summary",
        "",
        f"- Scenarios with stable main case-composition conclusions: {stable_count} / {len(rows)}",
        f"- Failed model runs recorded: {failed_total}",
        "",
        "The `case_composition_stable` flag uses broad manuscript-oriented stability bounds: heterosexual 7.5-11.5%, bisexual 10.5-15.0%, homosexual 74.0-82.5%, female-known 2.0-5.5%, and mean MAE <= 1.5 percentage points.",
        "",
        "## Scenario table",
        "",
        "| parameter | value | successful | failed | total cases mean (SD) | hetero mean (SD) | bi mean (SD) | homo mean (SD) | female-known mean (SD) | MAE mean (SD) | stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        readme_lines.append(
            "| {parameter} | {value:g} | {successful_runs} | {failed_runs} | {tc} ({tc_sd}) | {he} ({he_sd}) | {bi} ({bi_sd}) | {ho} ({ho_sd}) | {fk} ({fk_sd}) | {mae} ({mae_sd}) | {stable} |".format(
                parameter=row["parameter"],
                value=float(row["value"]),
                successful_runs=row["successful_runs"],
                failed_runs=row["failed_runs"],
                tc=_fmt(row["total_cases_mean"], 1),
                tc_sd=_fmt(row["total_cases_sd"], 1),
                he=_fmt(row["hetero_mean"], 4),
                he_sd=_fmt(row["hetero_sd"], 4),
                bi=_fmt(row["bi_mean"], 4),
                bi_sd=_fmt(row["bi_sd"], 4),
                ho=_fmt(row["homo_mean"], 4),
                ho_sd=_fmt(row["homo_sd"], 4),
                fk=_fmt(row["female_known_mean"], 4),
                fk_sd=_fmt(row["female_known_sd"], 4),
                mae=_fmt(row["MAE_pp_mean"], 3),
                mae_sd=_fmt(row["MAE_pp_sd"], 3),
                stable=row["case_composition_stable"],
            )
        )
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prepared Table S8 one-at-a-time sensitivities")
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-seeds", type=int, default=len(SEEDS))
    parser.add_argument("--seed-start", type=int, default=SEEDS[0])
    parser.add_argument("--only-parameter", choices=sorted(TABLE_S8_ONE_AT_A_TIME), default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write the scenario manifest without running models")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_DIR / "logs"
    results_path = OUTPUT_DIR / "table_s8_sensitivity_results.csv"
    summary_path = OUTPUT_DIR / "table_s8_sensitivity_summary.csv"
    readme_path = OUTPUT_DIR / "table_s8_sensitivity_readme.md"
    done = completed(results_path)
    baseline = load_frozen_baseline()
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    manifest_path = OUTPUT_DIR / "table_s8_sensitivity_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "parameter",
                "value",
                "description",
                "seed_start",
                "seed_end",
                "n_seeds",
                "pop_size",
                *FROZEN_TABLE_S7_BASELINE.keys(),
                "direct_param_overrides",
                "policy_adjustments",
            ],
        )
        writer.writeheader()
        for i, (parameter, value) in enumerate(iter_scenarios(args.only_parameter), start=1):
            params, policy_adjustments, description = build_scenario(parameter, value, baseline)
            writer.writerow(
                {
                    "scenario_id": f"table_s8_{i:03d}",
                    "parameter": parameter,
                    "value": value,
                    "description": description,
                    "seed_start": seeds[0],
                    "seed_end": seeds[-1],
                    "n_seeds": len(seeds),
                    "pop_size": args.pop_size,
                    **{k: params[k] for k in FROZEN_TABLE_S7_BASELINE},
                    "direct_param_overrides": json.dumps(
                        {k: v for k, v in params.items() if k not in FROZEN_TABLE_S7_BASELINE},
                        sort_keys=True,
                    ),
                    "policy_adjustments": json.dumps(policy_adjustments, sort_keys=True),
                }
            )

    if args.dry_run:
        print(f"[dry-run] wrote manifest: {manifest_path}", flush=True)
        return 0

    for parameter, value in iter_scenarios(args.only_parameter):
        params, policy_adjustments, description = build_scenario(parameter, value, baseline)
        for seed in seeds:
            key = (parameter, float(value), seed)
            if key in done:
                print(f"[skip] {parameter}={value} seed={seed}", flush=True)
                continue
            run_id = f"table_s8_{parameter}_{value:g}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[run] {run_id}", flush=True)
            result = run_model(
                params,
                seed=seed,
                pop_size=args.pop_size,
                run_id=run_id,
                make_plots=False,
                log_dir=log_dir,
                policy_adjustments=policy_adjustments,
            )
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "parameter": parameter,
                "value": value,
                "description": description,
                "direct_param_overrides": json.dumps(
                    {k: v for k, v in params.items() if k not in FROZEN_TABLE_S7_BASELINE},
                    sort_keys=True,
                ),
                "policy_adjustments": json.dumps(policy_adjustments, sort_keys=True),
                **params,
                **result,
            }
            row["MAE_pp"] = result.get("objective_case_composition_mae", "")
            append_row(results_path, row)
            print(f"[done] {run_id} status={row['status']} score={row.get('score', '')}", flush=True)
    write_summary_and_readme(results_path, summary_path, readme_path, manifest_path)
    print(f"[summary] wrote {summary_path}", flush=True)
    print(f"[summary] wrote {readme_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
