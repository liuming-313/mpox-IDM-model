"""Final 100-seed validation for adopted simplified baseline combo_manual_02.

This script performs validation only. It does not tune parameters and does not
modify population.py. During each subprocess run it temporarily sets the fixed
combo_manual_02 parameters, base_run_seed, and MAKE_PLOTS=False; when finished
it restores main.py to combo_manual_02 with base_run_seed=100000 and
MAKE_PLOTS=True.
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


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_DIR / "main.py"
OUT_DIR = PROJECT_DIR / "outputs" / "simplified_calibration" / "final_100seed_combo_manual_02"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_PATH = OUT_DIR / "validation_100seed_summary.csv"
README_PATH = OUT_DIR / "validation_100seed_readme.md"
PARAMS_PATH = OUT_DIR / "validation_100seed_params.json"

FIXED_PARAMS = {
    "p_bi_m_to_msm": 0.66,
    "party_core_intensity": 0.52,
    "party_hetero_leak": 0.026,
    "social_contact_scale": 5.0,
    "social_male_bias": 0.92,
    "late_social_ramp": 5.5,
}

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
}

METRIC_COLUMNS = [
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "MAE_pp",
]

DIAGNOSTICS_CANDIDATES = [
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv",
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "figures" / "calibration_diagnostics.csv",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def fnum() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def update_main_text(text: str, seed: int, make_plots: bool) -> str:
    for key, value in FIXED_PARAMS.items():
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({fnum()})'
        text, n = re.subn(pattern, rf"\g<1>{value:g}", text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely set {key}")
    text, n_seed = re.subn(
        rf"(^\s*base_run_seed\s*=\s*)({fnum()})",
        rf"\g<1>{int(seed)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_seed != 1:
        raise ValueError("Could not safely set base_run_seed")
    text, n_plots = re.subn(
        r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)",
        rf"\g<1>{str(bool(make_plots))}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_plots != 1:
        raise ValueError("Could not safely set MAKE_PLOTS")
    return text


def restore_main(original_text: str) -> None:
    MAIN_PATH.write_text(update_main_text(original_text, seed=100000, make_plots=True), encoding="utf-8")


def parse_total_cases(stdout: str) -> float | None:
    match = re.search(r"Final state:\s*(\{.*?\})", stdout, flags=re.DOTALL)
    if not match:
        return None
    raw = re.sub(r"np\.(?:float64|int64)\((.*?)\)", r"\1", match.group(1))
    try:
        summary = ast.literal_eval(raw)
    except Exception:
        return None
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
    return None


def parse_diagnostics() -> dict[str, float]:
    diag = next((p for p in DIAGNOSTICS_CANDIDATES if p.exists() and p.stat().st_size > 0), None)
    if diag is None:
        raise FileNotFoundError("No non-empty calibration_diagnostics.csv found")
    values: dict[str, float] = {}
    last_exc: Exception | None = None
    for _ in range(10):
        try:
            with diag.open("r", newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    metric = (row.get("metric") or "").strip()
                    if metric in TARGETS:
                        values[metric] = float(row["current"])
            break
        except PermissionError as exc:
            last_exc = exc
            time.sleep(1.0)
    else:
        if last_exc is not None:
            raise last_exc
    missing = sorted(set(TARGETS) - set(values))
    if missing:
        raise ValueError(f"Missing diagnostics metrics: {missing}")
    values["MAE_pp"] = statistics.mean(abs(values[m] - TARGETS[m]) for m in TARGETS) * 100.0
    return values


def fieldnames() -> list[str]:
    return [
        "timestamp",
        "seed",
        "run_id",
        "status",
        *METRIC_COLUMNS,
        "error",
        "stdout_log",
        "stderr_log",
    ]


def ensure_summary_header() -> None:
    if SUMMARY_PATH.exists() and SUMMARY_PATH.stat().st_size > 0:
        return
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writeheader()


def append_summary(row: dict[str, Any]) -> None:
    ensure_summary_header()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writerow({k: row.get(k, "") for k in fieldnames()})


def completed_ok_seeds() -> set[int]:
    if not SUMMARY_PATH.exists():
        return set()
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return {int(r["seed"]) for r in csv.DictReader(f) if r.get("status") == "ok" and r.get("seed")}


def run_seed(seed: int, original_text: str) -> dict[str, Any]:
    run_id = f"combo_manual_02_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "run_id": run_id,
        "status": "failed",
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = ""
    try:
        MAIN_PATH.write_text(update_main_text(original_text, seed=seed, make_plots=False), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=PROJECT_DIR, text=True, capture_output=True)
        stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            row["error"] = f"main.py exited with return code {proc.returncode}"
            return row
        diagnostics = parse_diagnostics()
        total_cases = parse_total_cases(proc.stdout)
        if total_cases is None or not math.isfinite(total_cases):
            raise ValueError("Could not parse total_cases")
        row.update(
            {
                "total_cases": total_cases,
                "hetero": diagnostics["hetero"],
                "bi": diagnostics["bi"],
                "homo": diagnostics["homo"],
                "female_known": diagnostics["female_known"],
                "MAE_pp": diagnostics["MAE_pp"],
                "status": "ok",
            }
        )
    except Exception:
        row["error"] = traceback.format_exc(limit=5)
    return row


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compact_summary_to_latest_seed_rows() -> None:
    rows = load_rows()
    latest: dict[int, dict[str, str]] = {}
    for row in rows:
        if not row.get("seed"):
            continue
        latest[int(row["seed"])] = row
    ordered = [latest[seed] for seed in sorted(latest)]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames())
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k, "") for k in fieldnames()})


def build_readme() -> dict[str, Any]:
    rows = load_rows()
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    failed_rows = [r for r in rows if r.get("status") != "ok"]
    stats: dict[str, dict[str, float]] = {}
    for metric in METRIC_COLUMNS:
        vals = [float(r[metric]) for r in ok_rows if r.get(metric) != ""]
        stats[metric] = {
            "mean": statistics.mean(vals) if vals else float("nan"),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals) if vals else float("nan"),
            "p2_5": percentile(vals, 2.5),
            "p97_5": percentile(vals, 97.5),
        }
    mae_vals = [float(r["MAE_pp"]) for r in ok_rows if r.get("MAE_pp") != ""]
    prop_mae_1 = sum(v <= 1.0 for v in mae_vals) / len(mae_vals) if mae_vals else float("nan")
    prop_mae_1_5 = sum(v <= 1.5 for v in mae_vals) / len(mae_vals) if mae_vals else float("nan")
    accepted = (
        stats["MAE_pp"]["mean"] <= 1.2
        and 0.115 <= stats["bi"]["mean"] <= 0.138
        and 0.028 <= stats["female_known"]["mean"] <= 0.045
        and stats["hetero"]["mean"] >= 0.080
    )
    payload = {
        "fixed_params": FIXED_PARAMS,
        "successful_runs": len(ok_rows),
        "failed_runs": len(failed_rows),
        "stats": stats,
        "prop_mae_le_1_0": prop_mae_1,
        "prop_mae_le_1_5": prop_mae_1_5,
        "accepted": accepted,
        "failed_seed_details": [{"seed": r.get("seed"), "error": r.get("error")} for r in failed_rows],
    }
    PARAMS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Final 100-seed validation: combo_manual_02",
        "",
        f"Successful runs: {len(ok_rows)}",
        f"Failed runs: {len(failed_rows)}",
        "",
        "## Fixed parameters",
        "",
        "```json",
        json.dumps(FIXED_PARAMS, indent=2, sort_keys=True),
        "```",
        "",
        "## Summary",
        "",
        "| metric | mean | SD | median | 2.5th pct | 97.5th pct |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRIC_COLUMNS:
        s = stats[metric]
        if metric == "MAE_pp":
            lines.append(f"| {metric} | {s['mean']:.3f} pp | {s['sd']:.3f} pp | {s['median']:.3f} pp | {s['p2_5']:.3f} pp | {s['p97_5']:.3f} pp |")
        elif metric == "total_cases":
            lines.append(f"| {metric} | {s['mean']:.1f} | {s['sd']:.1f} | {s['median']:.1f} | {s['p2_5']:.1f} | {s['p97_5']:.1f} |")
        else:
            lines.append(f"| {metric} | {100*s['mean']:.3f}% | {100*s['sd']:.3f}% | {100*s['median']:.3f}% | {100*s['p2_5']:.3f}% | {100*s['p97_5']:.3f}% |")
    lines.extend(
        [
            "",
            "## MAE proportions",
            "",
            f"- Proportion MAE <= 1.0 pp: {prop_mae_1:.3f}",
            f"- Proportion MAE <= 1.5 pp: {prop_mae_1_5:.3f}",
            "",
            "## Acceptance decision",
            "",
            "Accepted under the requested rules." if accepted else "Not accepted under the requested rules.",
        ]
    )
    if failed_rows:
        lines.extend(["", "## Failed seeds", ""])
        for r in failed_rows:
            lines.append(f"- Seed {r.get('seed')}: {r.get('error')}")
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    ensure_dirs()
    ensure_summary_header()
    original_text = MAIN_PATH.read_text(encoding="utf-8")
    try:
        done = completed_ok_seeds()
        for seed in range(100000, 100100):
            if seed in done:
                print(f"[skip] seed={seed} already completed", flush=True)
                continue
            print(f"[run] seed={seed}", flush=True)
            row = run_seed(seed, original_text)
            append_summary(row)
            print(f"[done] seed={seed} status={row['status']} MAE={row.get('MAE_pp', '')}", flush=True)
        compact_summary_to_latest_seed_rows()
        build_readme()
        return 0
    finally:
        restore_main(original_text)


if __name__ == "__main__":
    raise SystemExit(main())
