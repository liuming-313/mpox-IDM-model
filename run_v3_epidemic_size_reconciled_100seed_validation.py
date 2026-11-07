"""Final 100-seed validation for epidemic-size reconciled v3 baseline.

Candidate baseline:
  - frozen final v3 calibrated transmission/contact parameters
  - importation_scale = 1.75, implemented as daily_n = 2, 4, 9, 4, 0

The runner temporarily patches main.py for each subprocess run and restores it
afterward. It does not modify population.py.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one import MAIN_PATH, PROJECT_DIR


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_epidemic_size_reconciled_100seed_validation"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100100))
IMPORTATION_SCALE = 1.75
OBSERVED_TOTAL_REPORTED_CASES = 2280.0
OBSERVED_KNOWN_ORIENTATION_DENOMINATOR = 1965.0

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

IMPORTATION_SCHEDULE = [
    {"start": 0, "end": 30, "daily_n": 2},
    {"start": 31, "end": 70, "daily_n": 4},
    {"start": 71, "end": 110, "daily_n": 9},
    {"start": 111, "end": 150, "daily_n": 4},
    {"start": 151, "end": "SIM_DAYS", "daily_n": 0},
]

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
    "homo_male": 0.7796,
    "bi_male": 0.1272,
}

RESULT_FIELDS = [
    "timestamp",
    "seed",
    "pop_size",
    "run_id",
    "status",
    "importation_scale",
    "importation_schedule_json",
    *FROZEN_BASELINE.keys(),
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
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
    "distance_to_observed_total_reported_cases_2280",
    "signed_difference_vs_observed_total_reported_cases_2280",
    "distance_to_known_orientation_denominator_1965",
    "signed_difference_vs_known_orientation_denominator_1965",
    "composition_acceptance_pass",
]


def _float_pattern() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _patch_main_text(original: str, seed: int) -> str:
    text = original
    for key, value in FROZEN_BASELINE.items():
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({_float_pattern()})'
        text, n = re.subn(pattern, rf"\g<1>{float(value):g}", text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely patch frozen parameter {key}")

    replacements = [
        (rf"(^\s*base_run_seed\s*=\s*)({_float_pattern()})", rf"\g<1>{int(seed)}", "base_run_seed"),
        (rf"(^\s*POPULATION_SIZE\s*=\s*)(.+)$", rf"\g<1>int({POP_SIZE})", "POPULATION_SIZE"),
        (r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)", r"\g<1>False", "MAKE_PLOTS"),
    ]
    for pattern, replacement, name in replacements:
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely patch {name}")

    importation_block = "\n".join(
        [
            "    importations = [",
            "        dict(start=0,   end=30,  daily_n=2),",
            "        dict(start=31,  end=70,  daily_n=4),",
            "        dict(start=71,  end=110, daily_n=9),",
            "        dict(start=111, end=150, daily_n=4),",
            "        dict(start=151, end=SIM_DAYS, daily_n=0),",
            "    ]",
        ]
    )
    pattern = (
        r"    importations = \[\n"
        r"        dict\(start=0,\s+end=30,\s+daily_n=\d+\),\n"
        r"        dict\(start=31,\s+end=70,\s+daily_n=\d+\),\n"
        r"        dict\(start=71,\s+end=110,\s+daily_n=\d+\),\n"
        r"        dict\(start=111,\s+end=150,\s+daily_n=\d+\),\n"
        r"        dict\(start=151,\s+end=SIM_DAYS,\s+daily_n=\d+\),\n"
        r"    \]"
    )
    text, n = re.subn(pattern, importation_block, text, count=1)
    if n != 1:
        raise ValueError("Could not safely patch importation schedule")
    return text


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
                    if metric in TARGETS:
                        values[metric] = float(row["current"])
                    elif metric == "mean_absolute_error_pp":
                        values["MAE_pp"] = float(row["diff_percentage_points"])
            missing = sorted(set(TARGETS) - set(values))
            if missing or "MAE_pp" not in values:
                raise ValueError(f"Missing diagnostics: {missing}")
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
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _copy_diagnostics(run_id: str) -> str:
    if not SOURCE_DIAGNOSTICS.exists() or SOURCE_DIAGNOSTICS.stat().st_size == 0:
        return ""
    dst = DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv"
    shutil.copy2(SOURCE_DIAGNOSTICS, dst)
    return str(dst)


def _run_one(seed: int) -> dict[str, Any]:
    run_id = f"v3_epi_size_reconciled_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    original = MAIN_PATH.read_text(encoding="utf-8")
    started = datetime.now()
    row: dict[str, Any] = {
        "timestamp": started.isoformat(timespec="seconds"),
        "seed": seed,
        "pop_size": POP_SIZE,
        "run_id": run_id,
        "status": "failed",
        "importation_scale": IMPORTATION_SCALE,
        "importation_schedule_json": json.dumps(IMPORTATION_SCHEDULE, sort_keys=True),
        **FROZEN_BASELINE,
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "diagnostics_copy": "",
    }
    try:
        MAIN_PATH.write_text(_patch_main_text(original, seed=seed), encoding="utf-8")
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


def _completed(path: Path) -> set[int]:
    done = set()
    for row in _read_rows(path):
        if row.get("status") == "ok":
            done.add(int(float(row.get("seed", -1))))
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


def _composition_pass(summary: dict[str, Any]) -> bool:
    return (
        int(summary["successful_runs"]) == len(SEEDS)
        and _finite(summary.get("MAE_pp_mean")) <= 1.5
        and _finite(summary.get("hetero_mean")) >= 0.080
        and 0.115 <= _finite(summary.get("bi_mean")) <= 0.138
        and _finite(summary.get("homo_mean")) <= 0.805
        and 0.028 <= _finite(summary.get("female_known_mean")) <= 0.045
    )


def _summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    summary: dict[str, Any] = {
        "successful_runs": len(ok),
        "failed_runs": len(failed),
    }
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male", "MAE_pp"]:
        values = [_finite(row.get(metric)) for row in ok]
        summary[f"{metric}_mean"] = _mean(values)
        summary[f"{metric}_sd"] = _sd(values)
    mae_values = [_finite(row.get("MAE_pp")) for row in ok]
    clean_mae = [v for v in mae_values if math.isfinite(v)]
    n = len(clean_mae)
    summary["prop_MAE_le_1_0"] = sum(v <= 1.0 for v in clean_mae) / n if n else math.nan
    summary["prop_MAE_le_1_2"] = sum(v <= 1.2 for v in clean_mae) / n if n else math.nan
    summary["prop_MAE_le_1_5"] = sum(v <= 1.5 for v in clean_mae) / n if n else math.nan
    total_mean = _finite(summary.get("total_cases_mean"))
    summary["distance_to_observed_total_reported_cases_2280"] = abs(total_mean - OBSERVED_TOTAL_REPORTED_CASES)
    summary["signed_difference_vs_observed_total_reported_cases_2280"] = total_mean - OBSERVED_TOTAL_REPORTED_CASES
    summary["distance_to_known_orientation_denominator_1965"] = abs(total_mean - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR)
    summary["signed_difference_vs_known_orientation_denominator_1965"] = total_mean - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR
    summary["composition_acceptance_pass"] = _composition_pass(summary)
    return summary


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in SUMMARY_FIELDS})


def _write_params() -> None:
    payload = {
        "model_version": "v3_epidemic_size_reconciled_candidate_baseline",
        "pop_size_for_formal_validation": POP_SIZE,
        "validated_seed_range": f"{SEEDS[0]}-{SEEDS[-1]}",
        "importation_scale": IMPORTATION_SCALE,
        "importation_schedule": IMPORTATION_SCHEDULE,
        "parameters": FROZEN_BASELINE,
        "fixed_not_restored": [
            "party_hetero_leak",
            "social_male_bias",
            "party_mix_female_mult",
        ],
        "notes": [
            "Transmission, contact, testing, monitoring, disease natural-history, and policy parameters were frozen.",
            "Only the existing importation schedule daily_n values were set to the adopted importation_scale=1.75 schedule.",
        ],
    }
    (OUTPUT_DIR / "final_reconciled_v3_params.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_readme(summary: dict[str, Any]) -> None:
    lines = [
        "# Epidemic-size reconciled v3 100-seed validation",
        "",
        "This directory contains the final 100-seed validation for the candidate epidemic-size reconciled v3 baseline.",
        "",
        "All calibrated transmission/contact parameters and model logic were frozen. The adopted importation scale is 1.75, implemented as `daily_n = 2, 4, 9, 4, 0` over the existing importation time blocks.",
        "",
        f"Population size: {POP_SIZE:,}",
        f"Seeds: {SEEDS[0]}-{SEEDS[-1]}",
        "MAKE_PLOTS was set to False during batch runs and main.py was restored afterward.",
        "",
        "## Summary",
        "",
        f"- successful runs: {summary['successful_runs']}",
        f"- failed runs: {summary['failed_runs']}",
        f"- total_cases mean: {summary['total_cases_mean']:.3f}; SD: {summary['total_cases_sd']:.3f}",
        f"- heterosexual mean: {summary['hetero_mean']:.6f}; SD: {summary['hetero_sd']:.6f}",
        f"- bisexual mean: {summary['bi_mean']:.6f}; SD: {summary['bi_sd']:.6f}",
        f"- homosexual mean: {summary['homo_mean']:.6f}; SD: {summary['homo_sd']:.6f}",
        f"- female_known mean: {summary['female_known_mean']:.6f}; SD: {summary['female_known_sd']:.6f}",
        f"- homo_male mean: {summary['homo_male_mean']:.6f}; SD: {summary['homo_male_sd']:.6f}",
        f"- bi_male mean: {summary['bi_male_mean']:.6f}; SD: {summary['bi_male_sd']:.6f}",
        f"- MAE_pp mean: {summary['MAE_pp_mean']:.3f}; SD: {summary['MAE_pp_sd']:.3f}",
        f"- proportion with MAE_pp <= 1.0: {summary['prop_MAE_le_1_0']:.3f}",
        f"- proportion with MAE_pp <= 1.2: {summary['prop_MAE_le_1_2']:.3f}",
        f"- proportion with MAE_pp <= 1.5: {summary['prop_MAE_le_1_5']:.3f}",
        f"- distance to observed total reported cases 2280: {summary['distance_to_observed_total_reported_cases_2280']:.3f}",
        f"- signed difference vs observed total reported cases 2280: {summary['signed_difference_vs_observed_total_reported_cases_2280']:.3f}",
        f"- distance to known-orientation denominator 1965: {summary['distance_to_known_orientation_denominator_1965']:.3f}",
        f"- signed difference vs known-orientation denominator 1965: {summary['signed_difference_vs_known_orientation_denominator_1965']:.3f}",
        f"- predefined composition acceptance criteria pass: {summary['composition_acceptance_pass']}",
        "",
        "Composition acceptance criteria: successful 100/100 seeds, mean MAE_pp <= 1.5, heterosexual >= 8.0%, bisexual 11.5%-13.8%, homosexual <= 80.5%, and female_known 2.8%-4.5%.",
    ]
    (OUTPUT_DIR / "final_100seed_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    _write_params()
    results_path = OUTPUT_DIR / "final_100seed_results.csv"
    completed = _completed(results_path)

    for seed in SEEDS:
        if seed in completed:
            print(f"[skip] seed={seed}", flush=True)
            continue
        print(f"[run] seed={seed}", flush=True)
        row = _run_one(seed)
        _append_row(results_path, row)
        print(
            f"[done] seed={seed} status={row['status']} "
            f"total_cases={row.get('total_cases')} MAE_pp={row.get('MAE_pp')}",
            flush=True,
        )

    rows = _read_rows(results_path)
    summary = _summarize(rows)
    _write_summary(OUTPUT_DIR / "final_100seed_summary.csv", summary)
    _write_readme(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
