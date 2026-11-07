"""Selected one-at-a-time Table S8 sensitivity around frozen final E.

This runner keeps population.py unchanged. Each run temporarily patches main.py
to the frozen final E baseline, applies exactly one scenario change, runs the
model, and restores main.py afterward.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one import MAIN_PATH, PROJECT_DIR


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_final_E_table_s8_selected_sensitivity"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100020))
OBSERVED_TOTAL_REPORTED_CASES = 2280.0
OBSERVED_KNOWN_ORIENTATION_DENOMINATOR = 2012.0

TARGETS = {
    "hetero": 0.0989065606,
    "male_heterosexual": 0.0621272366,
    "female_known": 0.0367793241,
    "bi": 0.1257455268,
    "homo": 0.7753479125,
    "homo_male": 0.7753479125,
    "bi_male": 0.1257455268,
}

FROZEN_FINAL_E_PARAMS = {
    "p_bi_m_to_msm": 0.70,
    "party_core_intensity": 0.70,
    "mix_mf_anal_frac": 0.13,
    "stable_mf_anal_frac": 0.00,
    "party_mix_homo_spill_frac": 0.01,
    "sexual_anal_receptive": 0.2800,
    "sexual_anal_insertive": 0.0600,
    "sexual_vaginal": 0.0350,
    "sexual_oral": 0.0175,
    "social_contact_scale": 10.0,
    "social_same_sex_assort": 0.90,
    "community_bridge_scale": 1.5,
    "party_touch_frac": 0.20,
    "close_touch": 0.0067,
    "monitor_social_contact_mult": 0.70,
    "MONITORING_DAYS": 21,
}

SCENARIOS = [
    {
        "scenario_id": "baseline_final_E",
        "description": "Frozen final E baseline; no sensitivity change.",
        "params_override": {},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "party_touch_frac_low",
        "description": "party_touch_frac = 0.10.",
        "params_override": {"party_touch_frac": 0.10},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "party_touch_frac_high",
        "description": "party_touch_frac = 0.25.",
        "params_override": {"party_touch_frac": 0.25},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "close_touch_low",
        "description": "close_touch = 0.004.",
        "params_override": {"close_touch": 0.004},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "close_touch_high",
        "description": "close_touch = 0.010.",
        "params_override": {"close_touch": 0.010},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "testing_delay_minus_1_day",
        "description": "Shift all test_seek_delay_mean_days values by -1 day.",
        "params_override": {},
        "testing_delay_shift_days": -1.0,
    },
    {
        "scenario_id": "testing_delay_plus_1_day",
        "description": "Shift all test_seek_delay_mean_days values by +1 day.",
        "params_override": {},
        "testing_delay_shift_days": 1.0,
    },
    {
        "scenario_id": "monitor_social_low",
        "description": "monitor_social_contact_mult = 0.50.",
        "params_override": {"monitor_social_contact_mult": 0.50},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "monitor_social_high",
        "description": "monitor_social_contact_mult = 0.90.",
        "params_override": {"monitor_social_contact_mult": 0.90},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "monitoring_days_14",
        "description": "MONITORING_DAYS = 14.",
        "params_override": {"MONITORING_DAYS": 14},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "community_bridge_low",
        "description": "community_bridge_scale = 1.0.",
        "params_override": {"community_bridge_scale": 1.0},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "community_bridge_high",
        "description": "community_bridge_scale = 2.0.",
        "params_override": {"community_bridge_scale": 2.0},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "sexual_insertive_low",
        "description": "sexual_anal_insertive = 0.0525.",
        "params_override": {"sexual_anal_insertive": 0.0525},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "sexual_insertive_high",
        "description": "sexual_anal_insertive = 0.0700.",
        "params_override": {"sexual_anal_insertive": 0.0700},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "mixed_sex_anal_low",
        "description": "mix_mf_anal_frac = 0.08.",
        "params_override": {"mix_mf_anal_frac": 0.08},
        "testing_delay_shift_days": None,
    },
    {
        "scenario_id": "mixed_sex_anal_high",
        "description": "mix_mf_anal_frac = 0.18.",
        "params_override": {"mix_mf_anal_frac": 0.18},
        "testing_delay_shift_days": None,
    },
]

RESULT_FIELDS = [
    "timestamp",
    "scenario_id",
    "description",
    "seed",
    "pop_size",
    "run_id",
    "status",
    "scenario_params_json",
    "testing_delay_shift_days",
    "total_cases",
    "hetero",
    "implied_male_heterosexual",
    "female_known",
    "bi",
    "homo",
    "homo_male",
    "bi_male",
    "MAE_pp",
    "distance_to_observed_total_cases",
    "distance_to_known_orientation_denominator",
    "error",
    "stdout_log",
    "stderr_log",
    "diagnostics_copy",
    "elapsed_s",
]

SUMMARY_FIELDS = [
    "scenario_id",
    "description",
    "successful_runs",
    "failed_runs",
    "scenario_params_json",
    "testing_delay_shift_days",
    "total_cases_mean",
    "total_cases_sd",
    "hetero_mean",
    "hetero_sd",
    "implied_male_heterosexual_mean",
    "implied_male_heterosexual_sd",
    "female_known_mean",
    "female_known_sd",
    "bi_mean",
    "bi_sd",
    "homo_mean",
    "homo_sd",
    "homo_male_mean",
    "homo_male_sd",
    "bi_male_mean",
    "bi_male_sd",
    "MAE_pp_mean",
    "MAE_pp_sd",
    "prop_MAE_pp_le_1_0",
    "prop_MAE_pp_le_1_2",
    "distance_to_observed_total_cases_2280",
    "distance_to_known_orientation_denominator_2012",
    "delta_total_cases_vs_baseline",
    "delta_hetero_vs_baseline",
    "delta_male_heterosexual_vs_baseline",
    "delta_female_known_vs_baseline",
    "delta_bi_vs_baseline",
    "delta_homo_vs_baseline",
    "delta_MAE_pp_vs_baseline",
    "composition_stable",
]

MANIFEST_FIELDS = [
    "scenario_id",
    "description",
    "scenario_params_json",
    "testing_delay_shift_days",
    "seeds",
    "pop_size",
]


def _float_pattern() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _patch_param(text: str, key: str, value: float) -> str:
    pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({_float_pattern()})'
    text, n = re.subn(pattern, rf"\g<1>{float(value):.8g}", text, count=1, flags=re.MULTILINE)
    if n == 0:
        insert_pattern = r'(^\s*"calibration_targets"\s*:\s*CALIBRATION_TARGETS\s*,)'
        text, n = re.subn(
            insert_pattern,
            rf'    "{key}": {float(value):.8g},\n\g<1>',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    if n != 1:
        raise ValueError(f"Could not safely patch or insert {key}")
    return text


def _patch_testing_delay(text: str, shift_days: float | None) -> str:
    if shift_days is None:
        return text

    def repl_delay(match: re.Match[str]) -> str:
        value = max(0.0, float(match.group(2)) + float(shift_days))
        return f"{match.group(1)}{value:g}"

    text, n = re.subn(
        r"(test_seek_delay_mean_days\s*=\s*)(" + _float_pattern() + r")",
        repl_delay,
        text,
    )
    if n < 1:
        raise ValueError("Could not patch policy test_seek_delay_mean_days values")

    pattern = r'(^\s*"test_seek_delay_mean_days"\s*:\s*)(' + _float_pattern() + r")"
    text, n2 = re.subn(pattern, repl_delay, text, count=1, flags=re.MULTILINE)
    if n2 != 1:
        raise ValueError("Could not patch global test_seek_delay_mean_days")
    return text


def _patch_main_text(original: str, seed: int, scenario: dict[str, Any]) -> str:
    text = original
    patch_values = {**FROZEN_FINAL_E_PARAMS, **scenario["params_override"]}
    for key, value in patch_values.items():
        text = _patch_param(text, key, float(value))

    replacements = [
        (rf"(^\s*base_run_seed\s*=\s*)({_float_pattern()})", rf"\g<1>{int(seed)}", "base_run_seed"),
        (rf"(^\s*POPULATION_SIZE\s*=\s*)(.+)$", rf"\g<1>int({POP_SIZE})", "POPULATION_SIZE"),
        (r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)", r"\g<1>False", "MAKE_PLOTS"),
    ]
    for pattern, replacement, name in replacements:
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely patch {name}")
    return _patch_testing_delay(text, scenario.get("testing_delay_shift_days"))


def _parse_final_state(stdout: str) -> dict[str, Any]:
    match = re.search(r"Final state:\s*(\{.*?\})", stdout, flags=re.DOTALL)
    if not match:
        return {}
    raw = re.sub(r"np\.(?:float64|int64)\((.*?)\)", r"\1", match.group(1))
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def _read_diagnostics(retries: int = 10) -> dict[str, float]:
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            values: dict[str, float] = {}
            with SOURCE_DIAGNOSTICS.open("r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    metric = (row.get("metric") or "").strip()
                    if metric in {"hetero", "female_known", "bi", "homo", "homo_male", "bi_male"}:
                        values[metric] = float(row["current"])
                    elif metric == "mean_absolute_error_pp":
                        values["MAE_pp"] = float(row["diff_percentage_points"])
            missing = sorted({"hetero", "female_known", "bi", "homo", "homo_male", "bi_male"} - set(values))
            if missing or "MAE_pp" not in values:
                raise ValueError(f"Missing diagnostics: {missing}")
            values["implied_male_heterosexual"] = values["hetero"] - values["female_known"]
            return values
        except (PermissionError, FileNotFoundError, ValueError) as exc:
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(f"Could not read diagnostics: {last_exc}")


def _total_cases(summary: dict[str, Any]) -> float | None:
    count_keys = [
        "episode_female_hetero",
        "episode_male_hetero",
        "episode_male_homo",
        "episode_female_homo",
        "episode_male_bi",
        "episode_female_bi",
    ]
    if all(isinstance(summary.get(k), (int, float)) for k in count_keys):
        return float(sum(summary[k] for k in count_keys))
    for key in ["episode_total_cases", "total_episode_cases", "diagnosed_total", "total_cases"]:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _copy_diagnostics(run_id: str) -> str:
    if not SOURCE_DIAGNOSTICS.exists():
        return ""
    dest = DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv"
    dest.write_bytes(SOURCE_DIAGNOSTICS.read_bytes())
    return str(dest)


def _run_one(scenario: dict[str, Any], seed: int) -> dict[str, Any]:
    run_id = f"{scenario['scenario_id']}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    original = MAIN_PATH.read_text(encoding="utf-8")
    started = datetime.now()
    row: dict[str, Any] = {
        "timestamp": started.isoformat(timespec="seconds"),
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "seed": seed,
        "pop_size": POP_SIZE,
        "run_id": run_id,
        "status": "failed",
        "scenario_params_json": json.dumps(scenario["params_override"], sort_keys=True),
        "testing_delay_shift_days": scenario.get("testing_delay_shift_days"),
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "diagnostics_copy": "",
    }
    try:
        MAIN_PATH.write_text(_patch_main_text(original, seed=seed, scenario=scenario), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=PROJECT_DIR, text=True, capture_output=True)
        stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            row["error"] = f"main.py exited with return code {proc.returncode}"
            return row
        final_state = _parse_final_state(proc.stdout)
        diagnostics = _read_diagnostics()
        total_cases = _total_cases(final_state)
        row.update(diagnostics)
        row["total_cases"] = total_cases
        if total_cases is not None:
            row["distance_to_observed_total_cases"] = abs(float(total_cases) - OBSERVED_TOTAL_REPORTED_CASES)
            row["distance_to_known_orientation_denominator"] = abs(float(total_cases) - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR)
        row["diagnostics_copy"] = _copy_diagnostics(run_id)
        row["status"] = "ok"
    except Exception:
        row["error"] = traceback.format_exc(limit=6)
    finally:
        MAIN_PATH.write_text(original, encoding="utf-8")
        row["elapsed_s"] = (datetime.now() - started).total_seconds()
    return row


def _append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _completed(path: Path) -> set[tuple[str, int]]:
    done = set()
    for row in _read_rows(path):
        if row.get("status") == "ok":
            done.add((row.get("scenario_id", ""), int(float(row.get("seed", -1)))))
    return done


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.fmean(clean) if clean else math.nan


def _sd(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return statistics.stdev(clean) if len(clean) > 1 else 0.0 if len(clean) == 1 else math.nan


def _prop(values: list[float], threshold: float) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return sum(v <= threshold for v in clean) / len(clean) if clean else math.nan


def _composition_stable(summary: dict[str, Any]) -> bool:
    return (
        int(summary["successful_runs"]) == len(SEEDS)
        and _finite(summary.get("hetero_mean")) >= 0.080
        and 0.028 <= _finite(summary.get("female_known_mean")) <= 0.045
        and 0.115 <= _finite(summary.get("bi_mean")) <= 0.138
        and _finite(summary.get("homo_mean")) <= 0.805
        and _finite(summary.get("MAE_pp_mean")) <= 1.2
    )


def _summary_for(scenario: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics = [
        "total_cases",
        "hetero",
        "implied_male_heterosexual",
        "female_known",
        "bi",
        "homo",
        "homo_male",
        "bi_male",
        "MAE_pp",
    ]
    selected = [row for row in rows if row.get("scenario_id") == scenario["scenario_id"]]
    ok = [row for row in selected if row.get("status") == "ok"]
    failed = [row for row in selected if row.get("status") != "ok"]
    summary: dict[str, Any] = {
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "successful_runs": len(ok),
        "failed_runs": len(failed),
        "scenario_params_json": json.dumps(scenario["params_override"], sort_keys=True),
        "testing_delay_shift_days": scenario.get("testing_delay_shift_days"),
    }
    for metric in metrics:
        values = [_finite(row.get(metric)) for row in ok]
        summary[f"{metric}_mean"] = _mean(values)
        summary[f"{metric}_sd"] = _sd(values)
    mae_values = [_finite(row.get("MAE_pp")) for row in ok]
    summary["prop_MAE_pp_le_1_0"] = _prop(mae_values, 1.0)
    summary["prop_MAE_pp_le_1_2"] = _prop(mae_values, 1.2)
    total_mean = _finite(summary.get("total_cases_mean"))
    summary["distance_to_observed_total_cases_2280"] = abs(total_mean - OBSERVED_TOTAL_REPORTED_CASES)
    summary["distance_to_known_orientation_denominator_2012"] = abs(total_mean - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR)
    summary["composition_stable"] = _composition_stable(summary)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summaries(results_path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(results_path)
    summaries = [_summary_for(scenario, rows) for scenario in SCENARIOS]
    baseline = next((s for s in summaries if s["scenario_id"] == "baseline_final_E"), None)
    delta_metrics = {
        "total_cases": "delta_total_cases_vs_baseline",
        "hetero": "delta_hetero_vs_baseline",
        "implied_male_heterosexual": "delta_male_heterosexual_vs_baseline",
        "female_known": "delta_female_known_vs_baseline",
        "bi": "delta_bi_vs_baseline",
        "homo": "delta_homo_vs_baseline",
        "MAE_pp": "delta_MAE_pp_vs_baseline",
    }
    if baseline:
        for summary in summaries:
            for metric, out_key in delta_metrics.items():
                summary[out_key] = _finite(summary.get(f"{metric}_mean")) - _finite(baseline.get(f"{metric}_mean"))
    return summaries


def _write_manifest() -> None:
    rows = [
        {
            "scenario_id": scenario["scenario_id"],
            "description": scenario["description"],
            "scenario_params_json": json.dumps(scenario["params_override"], sort_keys=True),
            "testing_delay_shift_days": scenario.get("testing_delay_shift_days"),
            "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
            "pop_size": POP_SIZE,
        }
        for scenario in SCENARIOS
    ]
    _write_csv(OUTPUT_DIR / "final_E_table_s8_manifest.csv", rows, MANIFEST_FIELDS)


def _write_frozen_params() -> None:
    payload = {
        "targets": TARGETS,
        "observed_total_reported_cases": OBSERVED_TOTAL_REPORTED_CASES,
        "observed_known_orientation_denominator": OBSERVED_KNOWN_ORIENTATION_DENOMINATOR,
        "frozen_final_E_params": FROZEN_FINAL_E_PARAMS,
        "importation_schedule_daily_n": [2, 4, 9, 4, 0],
        "seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "pop_size": POP_SIZE,
        "scenarios": SCENARIOS,
    }
    (OUTPUT_DIR / "frozen_final_E_params.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_readme(summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# V3 final E selected Table S8 sensitivity",
        "",
        "One-at-a-time sensitivity scenarios around the frozen final E baseline.",
        "",
        "population.py was not modified. main.py was temporarily patched per run and restored after every run.",
        "",
        f"Population size: {POP_SIZE:,}",
        f"Seeds per scenario: {SEEDS[0]}-{SEEDS[-1]}",
        "",
        "| Scenario | total_cases | hetero | male_hetero | female_known | bi | homo | MAE_pp | stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['scenario_id']} | {s['total_cases_mean']:.2f} ({s['total_cases_sd']:.2f}) | "
            f"{s['hetero_mean']:.4f} ({s['hetero_sd']:.4f}) | "
            f"{s['implied_male_heterosexual_mean']:.4f} ({s['implied_male_heterosexual_sd']:.4f}) | "
            f"{s['female_known_mean']:.4f} ({s['female_known_sd']:.4f}) | "
            f"{s['bi_mean']:.4f} ({s['bi_sd']:.4f}) | "
            f"{s['homo_mean']:.4f} ({s['homo_sd']:.4f}) | "
            f"{s['MAE_pp_mean']:.3f} ({s['MAE_pp_sd']:.3f}) | "
            f"{'yes' if s['composition_stable'] else 'no'} |"
        )
    lines.append("")
    lines.append("No additional sensitivity analyses were run.")
    (OUTPUT_DIR / "final_E_table_s8_sensitivity_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(results_path: Path) -> list[dict[str, Any]]:
    summaries = _summaries(results_path)
    _write_csv(OUTPUT_DIR / "final_E_table_s8_sensitivity_summary.csv", summaries, SUMMARY_FIELDS)
    _write_manifest()
    _write_frozen_params()
    _write_readme(summaries)
    return summaries


def main() -> None:
    _ensure_dirs()
    results_path = OUTPUT_DIR / "final_E_table_s8_sensitivity_results.csv"
    completed = _completed(results_path)
    for scenario in SCENARIOS:
        for seed in SEEDS:
            key = (scenario["scenario_id"], seed)
            if key in completed:
                print(f"[skip] {scenario['scenario_id']} seed={seed}", flush=True)
                continue
            print(f"[run] {scenario['scenario_id']} seed={seed}", flush=True)
            row = _run_one(scenario, seed)
            _append_row(results_path, row)
            print(
                f"[done] {scenario['scenario_id']} seed={seed} status={row['status']} "
                f"total_cases={row.get('total_cases')} hetero={row.get('hetero')} "
                f"male_hetero={row.get('implied_male_heterosexual')} MAE_pp={row.get('MAE_pp')}",
                flush=True,
            )
            _write_outputs(results_path)
    summaries = _write_outputs(results_path)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
