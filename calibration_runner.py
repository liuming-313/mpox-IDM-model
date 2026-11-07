"""Coordinate-search runner for mpox/STI case-composition calibration.

This script edits only the approved calibration parameters in main.py, runs the
model as a subprocess, and records every attempted parameter set.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_DIR / "main.py"
BACKUP_PATH = PROJECT_DIR / "main_before_calibration.py"
SEARCH_DIR = PROJECT_DIR / "outputs" / "calibration_search"
LOG_DIR = SEARCH_DIR / "logs"
DIAGNOSTICS_DIR = SEARCH_DIR / "diagnostics"
HISTORY_PATH = SEARCH_DIR / "history.csv"
BEST_PARAMS_PATH = SEARCH_DIR / "best_params.json"
DIAGNOSTICS_CANDIDATES = [
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv",
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "figures" / "calibration_diagnostics.csv",
]


ALLOWED_PARAMS = [
    "party_mix_female_mult",
    "p_bi_m_to_msm",
    "p_bi_f_to_wsw",
    "unhoused_hetero_male_boost",
    "social_target_male_share",
    "social_contact_underascertain_mult",
    "unhoused_contact_mult_max",
    "party_mix_homo_spill_frac",
    "mix_mf_anal_frac",
    "party_mix_attend_prob",
]

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
    "homo_male": 0.7796,
    "bi_male": 0.1272,
}

TOLERANCES = {
    "hetero": (0.087, 0.101),
    "bi": (0.117, 0.137),
    "homo": (0.768, 0.792),
    "female_known": (0.028, 0.041),
}

CANDIDATES = {
    "party_mix_female_mult": [0.18, 0.20, 0.22],
    "p_bi_m_to_msm": [0.64, 0.655, 0.665, 0.68],
    "p_bi_f_to_wsw": [0.70, 0.75, 0.80],
    "unhoused_hetero_male_boost": [0.57, 0.59, 0.61],
    "social_target_male_share": [0.91, 0.92, 0.93],
    "social_contact_underascertain_mult": [3.8, 4.2, 4.6],
    "unhoused_contact_mult_max": [4.5, 5.0, 5.5],
    "party_mix_homo_spill_frac": [0.02, 0.03, 0.04],
    "mix_mf_anal_frac": [0.11, 0.13, 0.15],
    "party_mix_attend_prob": [0.50, 0.55, 0.60],
}

VALIDATION_SEEDS = [100000, 100001, 100002]


@dataclass
class RunResult:
    run_id: str
    status: str
    params: dict[str, float]
    changed: dict[str, float]
    seed: int | None
    metrics: dict[str, float]
    mae_pp: float | None
    total_cases: float | None
    stdout_log: Path
    stderr_log: Path
    error: str = ""
    validation: bool = False


def _float_re() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def ensure_dirs() -> None:
    SEARCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_backup() -> None:
    if not BACKUP_PATH.exists():
        shutil.copy2(MAIN_PATH, BACKUP_PATH)


def make_run_id(prefix: str, seed: int | None, label: str | None = None) -> str:
    """Create a unique, readable run id with seed and wall-clock timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = f"seed{seed}" if seed is not None else "seedNA"
    label_part = f"_{label}" if label else ""
    base = f"{prefix}_{seed_part}_{timestamp}{label_part}"
    candidate = base
    i = 1
    existing_ids = {row.get("run_id", "") for row in load_results_from_history()}
    while (
        candidate in existing_ids
        or (LOG_DIR / f"{candidate}.stdout.log").exists()
        or (LOG_DIR / f"{candidate}.stderr.log").exists()
        or (DIAGNOSTICS_DIR / f"{candidate}_calibration_diagnostics.csv").exists()
    ):
        i += 1
        candidate = f"{base}_{i}"
    return candidate


def read_main() -> str:
    return MAIN_PATH.read_text(encoding="utf-8")


def write_main(text: str) -> None:
    MAIN_PATH.write_text(text, encoding="utf-8")


def extract_params(text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for key in ALLOWED_PARAMS:
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({_float_re()})'
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Could not find allowed parameter in main.py: {key}")
        params[key] = float(match.group(2))
    return params


def extract_seed(text: str) -> int | None:
    match = re.search(rf"(^\s*base_run_seed\s*=\s*)({_float_re()})", text, re.MULTILINE)
    return int(float(match.group(2))) if match else None


def update_main(params: dict[str, float], seed: int | None = None) -> None:
    text = read_main()
    for key, value in params.items():
        if key not in ALLOWED_PARAMS:
            raise ValueError(f"Refusing to edit non-whitelisted parameter: {key}")
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({_float_re()})'
        text, n = re.subn(pattern, rf"\g<1>{value:g}", text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely update parameter: {key}")
    if seed is not None:
        text, n = re.subn(
            rf"(^\s*base_run_seed\s*=\s*)({_float_re()})",
            rf"\g<1>{int(seed)}",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if n != 1:
            raise ValueError("Could not safely update base_run_seed")
    write_main(text)


def param_hash(params: dict[str, float], seed: int | None, validation: bool) -> str:
    payload = {"params": {k: params[k] for k in sorted(params)}, "seed": seed, "validation": validation}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def load_history_keys() -> set[str]:
    if not HISTORY_PATH.exists():
        return set()
    keys: set[str] = set()
    with HISTORY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("param_hash"):
                keys.add(row["param_hash"])
    return keys


def append_history(result: RunResult, p_hash: str, elapsed_s: float) -> None:
    ensure_dirs()
    fieldnames = [
        "timestamp",
        "run_id",
        "param_hash",
        "status",
        "validation",
        "seed",
        "mae_pp",
        "total_cases",
        "error",
        "stdout_log",
        "stderr_log",
        *ALLOWED_PARAMS,
        *[f"{m}_current" for m in TARGETS],
        *[f"{m}_diff_pp" for m in TARGETS],
        "elapsed_s",
    ]
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": result.run_id,
            "param_hash": p_hash,
            "status": result.status,
            "validation": result.validation,
            "seed": result.seed if result.seed is not None else "",
            "mae_pp": result.mae_pp if result.mae_pp is not None else "",
            "total_cases": result.total_cases if result.total_cases is not None else "",
            "error": result.error,
            "stdout_log": str(result.stdout_log),
            "stderr_log": str(result.stderr_log),
            "elapsed_s": f"{elapsed_s:.1f}",
        }
        row.update(result.params)
        for metric in TARGETS:
            cur = result.metrics.get(metric)
            row[f"{metric}_current"] = "" if cur is None else cur
            row[f"{metric}_diff_pp"] = "" if cur is None else 100.0 * (cur - TARGETS[metric])
        writer.writerow(row)


def parse_diagnostics(run_id: str | None = None) -> tuple[dict[str, float], float]:
    diag_path = next((p for p in DIAGNOSTICS_CANDIDATES if p.exists() and p.stat().st_size > 0), None)
    if diag_path is None:
        raise FileNotFoundError("No non-empty calibration_diagnostics.csv found")
    if run_id is not None:
        shutil.copy2(diag_path, DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv")
    metrics: dict[str, float] = {}
    mae_pp: float | None = None
    with diag_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            metric = (row.get("metric") or "").strip()
            if metric in TARGETS:
                value = float(row["current"])
                if not math.isfinite(value):
                    raise ValueError(f"NaN/inf diagnostic value for {metric}")
                metrics[metric] = value
            elif metric == "mean_absolute_error_pp":
                mae_pp = float(row["diff_percentage_points"])
    missing = sorted(set(TARGETS) - set(metrics))
    if missing:
        raise ValueError(f"Missing diagnostics metrics: {missing}")
    if mae_pp is None or not math.isfinite(mae_pp):
        raise ValueError("Missing or invalid mean_absolute_error_pp")
    return metrics, mae_pp


def parse_total_cases(stdout: str) -> float | None:
    match = re.search(r"Final state:\s*(\{.*?\})", stdout, flags=re.DOTALL)
    if not match:
        return None
    raw = re.sub(r"np\.(?:float64|int64)\((.*?)\)", r"\1", match.group(1))
    try:
        summary = ast.literal_eval(raw)
    except Exception:
        return None
    keys = [
        "episode_total_cases",
        "total_episode_cases",
        "diagnosed_total",
        "total_diagnosed",
        "diagnosed_cases",
        "cum_diagnosed",
        "total_cases",
    ]
    for key in keys:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    episode_count_keys = [
        "episode_female_hetero",
        "episode_male_hetero",
        "episode_male_homo",
        "episode_female_homo",
        "episode_male_bi",
        "episode_female_bi",
    ]
    if all(isinstance(summary.get(key), (int, float)) for key in episode_count_keys):
        return float(sum(summary[key] for key in episode_count_keys))
    return None


def run_main(run_id: str, params: dict[str, float], changed: dict[str, float], seed: int | None, python_exe: str, validation: bool) -> RunResult:
    ensure_dirs()
    if (
        (LOG_DIR / f"{run_id}.stdout.log").exists()
        or (LOG_DIR / f"{run_id}.stderr.log").exists()
        or (DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv").exists()
    ):
        run_id = make_run_id(run_id, seed)
    update_main(params, seed=seed)
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    t0 = datetime.now()
    proc = subprocess.run(
        [python_exe, str(MAIN_PATH)],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )
    elapsed_s = (datetime.now() - t0).total_seconds()
    stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
    status = "ok" if proc.returncode == 0 else "failed"
    metrics: dict[str, float] = {}
    mae_pp: float | None = None
    total_cases = parse_total_cases(proc.stdout)
    error = ""
    if status == "ok":
        try:
            metrics, mae_pp = parse_diagnostics(run_id)
        except Exception as exc:
            status = "failed"
            error = str(exc)
    else:
        error = f"main.py exited with return code {proc.returncode}"
    result = RunResult(
        run_id=run_id,
        status=status,
        params=dict(params),
        changed=dict(changed),
        seed=seed,
        metrics=metrics,
        mae_pp=mae_pp,
        total_cases=total_cases,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        error=error,
        validation=validation,
    )
    append_history(result, param_hash(params, seed, validation), elapsed_s)
    return result


def metric_report(result: RunResult) -> str:
    lines = [
        f"\nrun_id={result.run_id} status={result.status} seed={result.seed}",
        f"changed={result.changed if result.changed else '(baseline)'}",
    ]
    if result.status != "ok":
        lines.append(f"error={result.error}")
        return "\n".join(lines)
    lines.append(f"{'metric':<14} {'target':>9} {'current':>9} {'diff_pp':>9}")
    for metric, target in TARGETS.items():
        cur = result.metrics[metric]
        lines.append(f"{metric:<14} {target:>8.2%} {cur:>8.2%} {100.0 * (cur - target):>+8.2f}")
    total = "n/a" if result.total_cases is None else f"{result.total_cases:.0f}"
    lines.append(f"MAE={result.mae_pp:.3f} pp  total_cases={total}  stop={primary_stop(result)}")
    return "\n".join(lines)


def primary_stop(result: RunResult) -> bool:
    if result.status != "ok" or result.mae_pp is None or result.mae_pp > 0.80:
        return False
    for metric, (lo, hi) in TOLERANCES.items():
        value = result.metrics.get(metric)
        if value is None or not (lo <= value <= hi):
            return False
    if result.total_cases is not None and not (2500 <= result.total_cases <= 3300):
        return False
    return True


def secondary_good(result: RunResult) -> bool:
    return result.status == "ok" and result.mae_pp is not None and result.mae_pp <= 1.0


def important_failed_metric_improved(candidate: RunResult, best: RunResult | None) -> bool:
    if candidate.status != "ok" or best is None or best.status != "ok":
        return False
    failed = []
    for metric, (lo, hi) in TOLERANCES.items():
        cur = best.metrics[metric]
        if cur < lo or cur > hi:
            failed.append(metric)
    if not failed:
        return False
    improved = False
    for metric in failed:
        before = abs(100.0 * (best.metrics[metric] - TARGETS[metric]))
        after = abs(100.0 * (candidate.metrics[metric] - TARGETS[metric]))
        improved = improved or after < before
    female_ok = candidate.metrics.get("female_known", 9.0) <= 0.041
    return improved and female_ok


def should_accept(candidate: RunResult, best: RunResult | None) -> bool:
    if candidate.status != "ok":
        return False
    if best is None or best.status != "ok" or best.mae_pp is None:
        return True
    if candidate.mae_pp is not None and candidate.mae_pp < best.mae_pp:
        return True
    if candidate.mae_pp is not None and candidate.mae_pp <= best.mae_pp + 0.20:
        return important_failed_metric_improved(candidate, best)
    return False


def save_best(result: RunResult, validation: list[RunResult] | None = None) -> None:
    payload: dict[str, Any] = {
        "run_id": result.run_id,
        "seed": result.seed,
        "mae_pp": result.mae_pp,
        "total_cases": result.total_cases,
        "params": result.params,
        "metrics": result.metrics,
        "diff_pp": {m: 100.0 * (result.metrics[m] - TARGETS[m]) for m in result.metrics},
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if validation:
        ok = [r for r in validation if r.status == "ok"]
        payload["validation"] = [
            {"run_id": r.run_id, "seed": r.seed, "mae_pp": r.mae_pp, "metrics": r.metrics, "total_cases": r.total_cases}
            for r in validation
        ]
        if ok:
            payload["validation_mean"] = {
                "mae_pp": sum(float(r.mae_pp) for r in ok if r.mae_pp is not None) / len(ok),
                "metrics": {m: sum(r.metrics[m] for r in ok) / len(ok) for m in TARGETS},
            }
    BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def candidate_order(current: dict[str, float]) -> list[tuple[dict[str, float], dict[str, float]]]:
    plan: list[tuple[dict[str, float], dict[str, float]]] = [({}, dict(current))]
    seen = {json.dumps(current, sort_keys=True)}
    priority = [
        "social_contact_underascertain_mult",
        "unhoused_contact_mult_max",
        "unhoused_hetero_male_boost",
        "social_target_male_share",
        "p_bi_m_to_msm",
        "p_bi_f_to_wsw",
        "party_mix_female_mult",
        "party_mix_homo_spill_frac",
        "mix_mf_anal_frac",
        "party_mix_attend_prob",
    ]
    for key in priority:
        for value in CANDIDATES[key]:
            params = dict(current)
            params[key] = value
            marker = json.dumps(params, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                plan.append(({key: value}, params))

    pair_tests = [
        {"p_bi_m_to_msm": 0.64, "p_bi_f_to_wsw": 0.80},
        {"p_bi_m_to_msm": 0.655, "party_mix_female_mult": 0.18},
        {"unhoused_hetero_male_boost": 0.61, "social_target_male_share": 0.93},
        {"social_contact_underascertain_mult": 4.6, "unhoused_contact_mult_max": 5.5},
        {"party_mix_homo_spill_frac": 0.04, "party_mix_attend_prob": 0.60},
    ]
    for changes in pair_tests:
        params = dict(current)
        params.update(changes)
        marker = json.dumps(params, sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            plan.append((changes, params))
    return plan


def load_results_from_history() -> list[dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def next_run_index() -> int:
    max_idx = -1
    for row in load_results_from_history():
        match = re.fullmatch(r"run_(\d+)", row.get("run_id", ""))
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


def print_summary_table() -> None:
    rows = [r for r in load_results_from_history() if r.get("status") == "ok" and r.get("mae_pp")]
    rows.sort(key=lambda r: float(r["mae_pp"]))
    print("\nFinal summary table sorted by MAE")
    print(f"{'run_id':<16} {'seed':>8} {'MAE':>8} {'hetero':>8} {'bi':>8} {'homo':>8} {'female':>8}")
    for row in rows[:20]:
        print(
            f"{row['run_id']:<16} {row.get('seed',''):>8} {float(row['mae_pp']):>8.3f} "
            f"{float(row['hetero_current']):>8.3%} {float(row['bi_current']):>8.3%} "
            f"{float(row['homo_current']):>8.3%} {float(row['female_known_current']):>8.3%}"
        )


def validate_best(best: RunResult, args: argparse.Namespace, used_runs: int, history_keys: set[str]) -> list[RunResult]:
    results: list[RunResult] = []
    if not primary_stop(best):
        return results
    print("\nPrimary stopping criteria reached; validating seeds 100000, 100001, 100002.")
    for seed in VALIDATION_SEEDS:
        if used_runs >= args.max_runs:
            print("Validation stopped because --max-runs was reached.")
            break
        run_id = make_run_id("validation", seed)
        p_hash = param_hash(best.params, seed, True)
        if args.resume and p_hash in history_keys:
            print(f"Skipping validation seed {seed}; already in history.csv")
            continue
        result = run_main(run_id, best.params, {"base_run_seed": seed}, seed, args.python, validation=True)
        history_keys.add(p_hash)
        used_runs += 1
        results.append(result)
        print(metric_report(result))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Coordinate-search calibration runner for main.py")
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    ensure_dirs()
    ensure_backup()
    original_text = read_main()
    original_seed = extract_seed(original_text)
    current = extract_params(original_text)
    plan = candidate_order(current)

    if args.dry_run:
        print(f"Current seed: {original_seed}")
        print("Dry run: planned parameter sets")
        for i, (changed, params) in enumerate(plan[: args.max_runs]):
            print(f"run_{i:03d}: changed={changed if changed else '(baseline)'} params={params}")
        return 0

    history_keys = load_history_keys() if args.resume else set()
    best: RunResult | None = None
    completed = 0
    accepted_params = dict(current)
    validation_results: list[RunResult] = []
    run_start = next_run_index()

    try:
        for i, (changed, params) in enumerate(plan):
            if completed >= args.max_runs:
                break
            if best is not None and changed:
                params = dict(accepted_params)
                params.update(changed)
            run_id = make_run_id("run", original_seed, f"{run_start + i:03d}")
            p_hash = param_hash(params, original_seed, False)
            if args.resume and p_hash in history_keys:
                print(f"Skipping {run_id}; parameter set already in history.csv")
                continue
            result = run_main(run_id, params, changed, original_seed, args.python, validation=False)
            history_keys.add(p_hash)
            completed += 1
            print(metric_report(result))

            if should_accept(result, best):
                best = result
                accepted_params = dict(result.params)
                save_best(result)
                print(f"Accepted {run_id} as current best.")

            if best is not None and primary_stop(best):
                validation_results = validate_best(best, args, completed, history_keys)
                if validation_results:
                    save_best(best, validation_results)
                break

            if best is not None and secondary_good(best):
                print("Secondary stopping criterion reached: MAE <= 1.0 pp. Holding best set unless a later planned run clearly improves it.")

        if best is not None:
            update_main(best.params, seed=original_seed)
            save_best(best, validation_results or None)
        print_summary_table()
        return 0
    finally:
        if best is None:
            write_main(original_text)


if __name__ == "__main__":
    raise SystemExit(main())
