"""Final 100-seed validation for frozen v3 Candidate E.

This runner keeps population.py unchanged. It temporarily patches only the
allowed main.py parameters for each subprocess run and restores main.py after
every run.
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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_final_E_100seed_validation"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100100))
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

FIXED_PARAMS = {
    "sexual_anal_receptive": 0.2800,
    "sexual_vaginal": 0.0350,
    "sexual_oral": 0.0175,
}

CANDIDATE_E = {
    "candidate_id": "E_final_candidate",
    "description": "Frozen final v3 Candidate E after 20-seed validation.",
    "params": {
        "sexual_anal_insertive": 0.0600,
        "social_contact_scale": 10.0,
        "social_same_sex_assort": 0.90,
        "community_bridge_scale": 1.5,
    },
}

PATCHABLE_PARAMS = [
    "sexual_anal_insertive",
    "social_contact_scale",
    "social_same_sex_assort",
    "community_bridge_scale",
]

RESULT_FIELDS = [
    "timestamp",
    "candidate_id",
    "description",
    "seed",
    "pop_size",
    "run_id",
    "status",
    "sexual_anal_receptive",
    "sexual_anal_insertive",
    "sexual_vaginal",
    "sexual_oral",
    "social_contact_scale",
    "social_same_sex_assort",
    "community_bridge_scale",
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
    "candidate_id",
    "description",
    "successful_runs",
    "failed_runs",
    "sexual_anal_receptive",
    "sexual_anal_insertive",
    "sexual_vaginal",
    "sexual_oral",
    "social_contact_scale",
    "social_same_sex_assort",
    "community_bridge_scale",
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
    "composition_acceptance_pass",
]


def _float_pattern() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _patch_main_text(original: str, seed: int, params: dict[str, float]) -> str:
    text = original
    patch_values = {**FIXED_PARAMS, **params}
    for key, value in patch_values.items():
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({_float_pattern()})'
        text, n = re.subn(pattern, rf"\g<1>{float(value):.6g}", text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely patch {key}")
    replacements = [
        (rf"(^\s*base_run_seed\s*=\s*)({_float_pattern()})", rf"\g<1>{int(seed)}", "base_run_seed"),
        (rf"(^\s*POPULATION_SIZE\s*=\s*)(.+)$", rf"\g<1>int({POP_SIZE})", "POPULATION_SIZE"),
        (r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)", r"\g<1>False", "MAKE_PLOTS"),
    ]
    for pattern, replacement, name in replacements:
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely patch {name}")
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


def _run_one(seed: int) -> dict[str, Any]:
    params = CANDIDATE_E["params"]
    run_id = f"{CANDIDATE_E['candidate_id']}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    original = MAIN_PATH.read_text(encoding="utf-8")
    started = datetime.now()
    row: dict[str, Any] = {
        "timestamp": started.isoformat(timespec="seconds"),
        "candidate_id": CANDIDATE_E["candidate_id"],
        "description": CANDIDATE_E["description"],
        "seed": seed,
        "pop_size": POP_SIZE,
        "run_id": run_id,
        "status": "failed",
        **FIXED_PARAMS,
        **params,
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "diagnostics_copy": "",
    }
    try:
        MAIN_PATH.write_text(_patch_main_text(original, seed=seed, params=params), encoding="utf-8")
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


def _prop(values: list[float], threshold: float) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return sum(v <= threshold for v in clean) / len(clean) if clean else math.nan


def _composition_pass(summary: dict[str, Any]) -> bool:
    return (
        int(summary["successful_runs"]) == len(SEEDS)
        and _finite(summary.get("MAE_pp_mean")) <= 1.2
        and _finite(summary.get("hetero_mean")) >= 0.085
        and _finite(summary.get("implied_male_heterosexual_mean")) >= 0.055
        and 0.028 <= _finite(summary.get("female_known_mean")) <= 0.045
        and 0.115 <= _finite(summary.get("bi_mean")) <= 0.138
        and _finite(summary.get("homo_mean")) <= 0.805
        and 1800 <= _finite(summary.get("total_cases_mean")) <= 2500
    )


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
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
    ok = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    summary: dict[str, Any] = {
        "candidate_id": CANDIDATE_E["candidate_id"],
        "description": CANDIDATE_E["description"],
        "successful_runs": len(ok),
        "failed_runs": len(failed),
        **FIXED_PARAMS,
        **CANDIDATE_E["params"],
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
    summary["composition_acceptance_pass"] = _composition_pass(summary)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_params(summary: dict[str, Any]) -> None:
    payload = {
        "targets": TARGETS,
        "observed_total_reported_cases": OBSERVED_TOTAL_REPORTED_CASES,
        "observed_known_orientation_denominator": OBSERVED_KNOWN_ORIENTATION_DENOMINATOR,
        "fixed_params": FIXED_PARAMS,
        "validation_seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "pop_size": POP_SIZE,
        "candidate": {
            "candidate_id": CANDIDATE_E["candidate_id"],
            "description": CANDIDATE_E["description"],
            "params": {k: summary[k] for k in [*FIXED_PARAMS, *PATCHABLE_PARAMS]},
            "summary": {k: summary[k] for k in SUMMARY_FIELDS if k in summary and k not in [*FIXED_PARAMS, *PATCHABLE_PARAMS]},
        },
    }
    (OUTPUT_DIR / "final_E_params.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_readme(summary: dict[str, Any]) -> None:
    lines = [
        "# V3 final E 100-seed validation",
        "",
        "Final frozen Candidate E after 20-seed validation.",
        "",
        "population.py was not modified. main.py was temporarily patched per run and restored after every run.",
        "",
        "Frozen: importation schedule daily_n=2/4/9/4/0, p_bi_m_to_msm=0.70, party_core_intensity=0.70, mix_mf_anal_frac=0.13, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.01, disease natural history, testing, monitoring, vaccination/susceptibility, policy timing, party_touch_frac, social_p_eff, close_touch, and excluded female/leak/bias knobs.",
        "",
        f"Population size: {POP_SIZE:,}",
        f"Seeds: {SEEDS[0]}-{SEEDS[-1]}",
        "",
        "| total_cases | hetero | male_hetero | female_known | bi | homo | homo_male | bi_male | MAE_pp | MAE<=1.0 | MAE<=1.2 | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| {summary['total_cases_mean']:.2f} ({summary['total_cases_sd']:.2f}) | "
            f"{summary['hetero_mean']:.4f} ({summary['hetero_sd']:.4f}) | "
            f"{summary['implied_male_heterosexual_mean']:.4f} ({summary['implied_male_heterosexual_sd']:.4f}) | "
            f"{summary['female_known_mean']:.4f} ({summary['female_known_sd']:.4f}) | "
            f"{summary['bi_mean']:.4f} ({summary['bi_sd']:.4f}) | "
            f"{summary['homo_mean']:.4f} ({summary['homo_sd']:.4f}) | "
            f"{summary['homo_male_mean']:.4f} ({summary['homo_male_sd']:.4f}) | "
            f"{summary['bi_male_mean']:.4f} ({summary['bi_male_sd']:.4f}) | "
            f"{summary['MAE_pp_mean']:.3f} ({summary['MAE_pp_sd']:.3f}) | "
            f"{summary['prop_MAE_pp_le_1_0']:.2f} | {summary['prop_MAE_pp_le_1_2']:.2f} | "
            f"{'yes' if summary['composition_acceptance_pass'] else 'no'} |"
        ),
        "",
        "No additional sensitivity analyses were run.",
        "",
    ]
    (OUTPUT_DIR / "final_E_100seed_README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(results_path: Path) -> dict[str, Any]:
    summary = _summary(_read_rows(results_path))
    _write_csv(OUTPUT_DIR / "final_E_100seed_summary.csv", [summary], SUMMARY_FIELDS)
    _write_params(summary)
    _write_readme(summary)
    return summary


def main() -> None:
    _ensure_dirs()
    results_path = OUTPUT_DIR / "final_E_100seed_results.csv"
    completed = _completed(results_path)
    for seed in SEEDS:
        if seed in completed:
            print(f"[skip] {CANDIDATE_E['candidate_id']} seed={seed}", flush=True)
            continue
        print(f"[run] {CANDIDATE_E['candidate_id']} seed={seed}", flush=True)
        row = _run_one(seed)
        _append_row(results_path, row)
        print(
            f"[done] {CANDIDATE_E['candidate_id']} seed={seed} status={row['status']} "
            f"total_cases={row.get('total_cases')} hetero={row.get('hetero')} "
            f"male_hetero={row.get('implied_male_heterosexual')} MAE_pp={row.get('MAE_pp')}",
            flush=True,
        )
        _write_outputs(results_path)
    summary = _write_outputs(results_path)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
