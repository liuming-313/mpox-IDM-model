"""Importation-only epidemic-size reconciliation for frozen final v3 baseline.

This runner keeps the final calibrated transmission/contact parameters fixed and
varies only a multiplier on the existing importation schedule daily_n values.
It temporarily patches main.py for each subprocess run and restores it
afterward.
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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_epidemic_size_reconciliation"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SCREEN_SEEDS = list(range(100000, 100005))
VALIDATION_SEEDS = list(range(100000, 100020))
IMPORTATION_SCALES = [1.0, 1.25, 1.50, 1.75, 2.00, 2.25]
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

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
}

RESULT_FIELDS = [
    "stage",
    "timestamp",
    "scenario_id",
    "importation_scale",
    "importation_schedule_json",
    "seed",
    "pop_size",
    "run_id",
    "status",
    *FROZEN_BASELINE.keys(),
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
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
    "stage",
    "scenario_id",
    "importation_scale",
    "importation_schedule_json",
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
    "composition_criteria_pass",
    "closer_to_observed_2280_than_baseline",
    "closer_to_known_1965_than_baseline",
    "selected_for_20seed_validation",
]


def _float_pattern() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _rounded_daily_n(value: int, scale: float) -> int:
    if value <= 0:
        return 0
    return int(math.floor(float(value) * float(scale) + 0.5))


def _scaled_schedule(scale: float) -> list[dict[str, int]]:
    base = [
        {"start": 0, "end": 30, "daily_n": 1},
        {"start": 31, "end": 70, "daily_n": 2},
        {"start": 71, "end": 110, "daily_n": 5},
        {"start": 111, "end": 150, "daily_n": 2},
        {"start": 151, "end": "SIM_DAYS", "daily_n": 0},
    ]
    return [
        {**row, "daily_n": _rounded_daily_n(int(row["daily_n"]), scale)}
        for row in base
    ]


def _patch_main_text(original: str, seed: int, scale: float) -> str:
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

    schedule = _scaled_schedule(scale)
    importation_block = "\n".join(
        [
            "    importations = [",
            f"        dict(start=0,   end=30,  daily_n={schedule[0]['daily_n']}),",
            f"        dict(start=31,  end=70,  daily_n={schedule[1]['daily_n']}),",
            f"        dict(start=71,  end=110, daily_n={schedule[2]['daily_n']}),",
            f"        dict(start=111, end=150, daily_n={schedule[3]['daily_n']}),",
            f"        dict(start=151, end=SIM_DAYS, daily_n={schedule[4]['daily_n']}),",
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
    path = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            values: dict[str, float] = {}
            with path.open("r", newline="", encoding="utf-8-sig") as f:
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


def _run_one(stage: str, scale: float, seed: int) -> dict[str, Any]:
    scenario_id = f"importation_scale_{scale:.2f}".replace(".", "_")
    run_id = f"{stage}_{scenario_id}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    original = MAIN_PATH.read_text(encoding="utf-8")
    started = datetime.now()
    row: dict[str, Any] = {
        "stage": stage,
        "timestamp": started.isoformat(timespec="seconds"),
        "scenario_id": scenario_id,
        "importation_scale": scale,
        "importation_schedule_json": json.dumps(_scaled_schedule(scale), sort_keys=True),
        "seed": seed,
        "pop_size": POP_SIZE,
        "run_id": run_id,
        "status": "failed",
        **FROZEN_BASELINE,
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "diagnostics_copy": "",
    }
    try:
        MAIN_PATH.write_text(_patch_main_text(original, seed=seed, scale=scale), encoding="utf-8")
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


def _completed(path: Path) -> set[tuple[str, str, int]]:
    out = set()
    for row in _read_rows(path):
        if row.get("status") == "ok":
            out.add((row.get("stage", ""), row.get("scenario_id", ""), int(float(row.get("seed", -1)))))
    return out


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
        _finite(summary.get("MAE_pp_mean")) <= 1.5
        and _finite(summary.get("hetero_mean")) >= 0.080
        and 0.115 <= _finite(summary.get("bi_mean")) <= 0.138
        and _finite(summary.get("homo_mean")) <= 0.805
        and 0.028 <= _finite(summary.get("female_known_mean")) <= 0.045
    )


def _summaries(rows: list[dict[str, str]], stage: str, baseline_total_cases: float | None = None) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for scale in IMPORTATION_SCALES:
        scenario_id = f"importation_scale_{scale:.2f}".replace(".", "_")
        selected = [r for r in rows if r.get("stage") == stage and r.get("scenario_id") == scenario_id]
        ok = [r for r in selected if r.get("status") == "ok"]
        failed = [r for r in selected if r.get("status") != "ok"]
        summary: dict[str, Any] = {
            "stage": stage,
            "scenario_id": scenario_id,
            "importation_scale": scale,
            "importation_schedule_json": json.dumps(_scaled_schedule(scale), sort_keys=True),
            "successful_runs": len(ok),
            "failed_runs": len(failed),
        }
        for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]:
            values = [_finite(row.get(metric)) for row in ok]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_sd"] = _sd(values)
        summary["composition_criteria_pass"] = _composition_pass(summary)
        summaries.append(summary)
    baseline = next((s for s in summaries if abs(float(s["importation_scale"]) - 1.0) < 1e-9), None)
    if baseline or baseline_total_cases is not None:
        base_total = _finite(baseline.get("total_cases_mean")) if baseline else math.nan
        if not math.isfinite(base_total) and baseline_total_cases is not None:
            base_total = float(baseline_total_cases)
        base_dist_2280 = abs(base_total - OBSERVED_TOTAL_REPORTED_CASES)
        base_dist_1965 = abs(base_total - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR)
        for summary in summaries:
            total = _finite(summary.get("total_cases_mean"))
            summary["closer_to_observed_2280_than_baseline"] = abs(total - OBSERVED_TOTAL_REPORTED_CASES) < base_dist_2280
            summary["closer_to_known_1965_than_baseline"] = abs(total - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR) < base_dist_1965
    return summaries


def _select_candidates(screen_summaries: list[dict[str, Any]]) -> list[float]:
    baseline = next(s for s in screen_summaries if abs(float(s["importation_scale"]) - 1.0) < 1e-9)
    base_total = _finite(baseline.get("total_cases_mean"))
    eligible = [
        s for s in screen_summaries
        if float(s["importation_scale"]) != 1.0
        and s["successful_runs"] == len(SCREEN_SEEDS)
        and bool(s.get("composition_criteria_pass"))
        and _finite(s.get("total_cases_mean")) > base_total
        and bool(s.get("closer_to_observed_2280_than_baseline"))
    ]
    eligible.sort(key=lambda s: (abs(_finite(s.get("total_cases_mean")) - OBSERVED_TOTAL_REPORTED_CASES), _finite(s.get("MAE_pp_mean"))))
    return [float(s["importation_scale"]) for s in eligible[:2]]


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_readme(screen: list[dict[str, Any]], validation: list[dict[str, Any]], candidates: list[float]) -> None:
    lines = [
        "# V3 epidemic-size reconciliation",
        "",
        "Read-only audit conclusion before running: `total_cases` is the sum of the six sex-by-orientation episode-date diagnosed case categories from `Population.summary()`, not cumulative infections.",
        f"Observed total reported cases target: approximately {OBSERVED_TOTAL_REPORTED_CASES:.0f}.",
        f"Observed known-orientation denominator for composition: {OBSERVED_KNOWN_ORIENTATION_DENOMINATOR:.0f}.",
        "",
        "All final v3 calibrated transmission/contact parameters were frozen. Only the importation schedule `daily_n` values were multiplied and rounded while preserving the original temporal pattern.",
        "",
        "## Screening summary",
        "",
        "| scale | total cases | hetero | bi | homo | female_known | MAE_pp | composition pass | closer to 2280 | selected |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in screen:
        scale = float(row["importation_scale"])
        selected = scale in candidates
        lines.append(
            f"| {scale:.2f} | {row['total_cases_mean']:.2f} ({row['total_cases_sd']:.2f}) | "
            f"{row['hetero_mean']:.4f} | {row['bi_mean']:.4f} | {row['homo_mean']:.4f} | "
            f"{row['female_known_mean']:.4f} | {row['MAE_pp_mean']:.3f} | "
            f"{'yes' if row['composition_criteria_pass'] else 'no'} | "
            f"{'yes' if row.get('closer_to_observed_2280_than_baseline') else 'no'} | "
            f"{'yes' if selected else 'no'} |"
        )
    if validation:
        lines.extend(
            [
                "",
                "## 20-seed validation summary",
                "",
                "| scale | total cases | hetero | bi | homo | female_known | MAE_pp | composition pass | closer to 2280 |",
                "|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in validation:
            lines.append(
                f"| {float(row['importation_scale']):.2f} | {row['total_cases_mean']:.2f} ({row['total_cases_sd']:.2f}) | "
                f"{row['hetero_mean']:.4f} | {row['bi_mean']:.4f} | {row['homo_mean']:.4f} | "
                f"{row['female_known_mean']:.4f} | {row['MAE_pp_mean']:.3f} | "
                f"{'yes' if row['composition_criteria_pass'] else 'no'} | "
                f"{'yes' if row.get('closer_to_observed_2280_than_baseline') else 'no'} |"
            )
    else:
        lines.extend(["", "No scale was selected for 20-seed validation."])
    lines.append("")
    lines.append("`MAKE_PLOTS` was set to False during runs and `main.py` was restored after each run.")
    (OUTPUT_DIR / "epidemic_size_reconciliation_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    results_path = OUTPUT_DIR / "epidemic_size_reconciliation_results.csv"
    completed = _completed(results_path)

    for scale in IMPORTATION_SCALES:
        scenario_id = f"importation_scale_{scale:.2f}".replace(".", "_")
        for seed in SCREEN_SEEDS:
            if ("screen", scenario_id, seed) in completed:
                print(f"[skip] screen scale={scale:.2f} seed={seed}", flush=True)
                continue
            print(f"[run] screen scale={scale:.2f} seed={seed}", flush=True)
            row = _run_one("screen", scale, seed)
            _append_row(results_path, row)
            print(
                f"[done] screen scale={scale:.2f} seed={seed} status={row['status']} "
                f"total_cases={row.get('total_cases')} MAE_pp={row.get('MAE_pp')}",
                flush=True,
            )

    rows = _read_rows(results_path)
    screen = _summaries(rows, "screen")
    candidates = _select_candidates(screen)
    for row in screen:
        row["selected_for_20seed_validation"] = float(row["importation_scale"]) in candidates
    _write_csv(OUTPUT_DIR / "epidemic_size_reconciliation_screen_summary.csv", screen, SUMMARY_FIELDS)

    for scale in candidates:
        scenario_id = f"importation_scale_{scale:.2f}".replace(".", "_")
        for seed in VALIDATION_SEEDS:
            if ("validation20", scenario_id, seed) in completed:
                print(f"[skip] validation20 scale={scale:.2f} seed={seed}", flush=True)
                continue
            print(f"[run] validation20 scale={scale:.2f} seed={seed}", flush=True)
            row = _run_one("validation20", scale, seed)
            _append_row(results_path, row)
            print(
                f"[done] validation20 scale={scale:.2f} seed={seed} status={row['status']} "
                f"total_cases={row.get('total_cases')} MAE_pp={row.get('MAE_pp')}",
                flush=True,
            )

    rows = _read_rows(results_path)
    screen_baseline = next(row for row in screen if abs(float(row["importation_scale"]) - 1.0) < 1e-9)
    validation = _summaries(rows, "validation20", baseline_total_cases=_finite(screen_baseline["total_cases_mean"])) if candidates else []
    validation = [row for row in validation if float(row["importation_scale"]) in candidates]
    _write_csv(OUTPUT_DIR / "epidemic_size_reconciliation_validation20_summary.csv", validation, SUMMARY_FIELDS)
    _write_readme(screen, validation, candidates)
    print(json.dumps({"screen": screen, "selected_candidates": candidates, "validation20": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
