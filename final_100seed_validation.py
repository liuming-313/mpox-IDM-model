"""Run final 100-seed validation for the fixed calibrated baseline.

This script does not tune parameters. It only changes base_run_seed and
MAKE_PLOTS in main.py while each validation subprocess runs, then restores the
original main.py text before exiting.
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
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_DIR / "main.py"
OUT_DIR = PROJECT_DIR / "outputs" / "final_100seed_validation"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_PATH = OUT_DIR / "validation_100seed_summary.csv"
README_PATH = OUT_DIR / "validation_100seed_readme.md"
DIAGNOSTICS_CANDIDATES = [
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv",
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "figures" / "calibration_diagnostics.csv",
]

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
    "homo_male": 0.7796,
    "bi_male": 0.1272,
}

METRIC_COLUMNS = [
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "homo_male",
    "bi_male",
    "MAE_pp",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def update_main_for_seed(text: str, seed: int) -> str:
    text, n_seed = re.subn(
        r"(^\s*base_run_seed\s*=\s*)([-+]?\d+)",
        rf"\g<1>{seed}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_seed != 1:
        raise ValueError("Could not safely update base_run_seed in main.py")
    text, n_plots = re.subn(
        r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)",
        r"\g<1>False",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_plots != 1:
        raise ValueError("Could not safely update MAKE_PLOTS in main.py")
    return text


def parse_total_cases(stdout: str) -> float | None:
    match = re.search(r"Final state:\s*(\{.*?\})", stdout, flags=re.DOTALL)
    if not match:
        return None
    raw = re.sub(r"np\.(?:float64|int64)\((.*?)\)", r"\1", match.group(1))
    try:
        summary = ast.literal_eval(raw)
    except Exception:
        return None
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
    for key in ["episode_total_cases", "total_episode_cases", "diagnosed_total", "total_cases"]:
        value = summary.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def parse_diagnostics() -> tuple[dict[str, float], float]:
    diag_path = next((p for p in DIAGNOSTICS_CANDIDATES if p.exists() and p.stat().st_size > 0), None)
    if diag_path is None:
        raise FileNotFoundError("No non-empty calibration_diagnostics.csv found")
    metrics: dict[str, float] = {}
    mae_pp: float | None = None
    with diag_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            metric = (row.get("metric") or "").strip()
            if metric in TARGETS:
                value = float(row["current"])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite diagnostic value for {metric}")
                metrics[metric] = value
            elif metric == "mean_absolute_error_pp":
                mae_pp = float(row["diff_percentage_points"])
    missing = sorted(set(TARGETS) - set(metrics))
    if missing:
        raise ValueError(f"Missing diagnostics metrics: {missing}")
    if mae_pp is None or not math.isfinite(mae_pp):
        raise ValueError("Missing or invalid MAE")
    return metrics, mae_pp


def write_summary_header_if_needed() -> None:
    if SUMMARY_PATH.exists() and SUMMARY_PATH.stat().st_size > 0:
        return
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "seed",
                "status",
                *METRIC_COLUMNS,
                "error",
                "stdout_log",
                "stderr_log",
            ],
        )
        writer.writeheader()


def append_row(row: dict[str, Any]) -> None:
    write_summary_header_if_needed()
    with SUMMARY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "seed",
                "status",
                *METRIC_COLUMNS,
                "error",
                "stdout_log",
                "stderr_log",
            ],
        )
        writer.writerow(row)


def completed_seeds() -> set[int]:
    if not SUMMARY_PATH.exists():
        return set()
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return {int(row["seed"]) for row in csv.DictReader(f) if row.get("seed") and row.get("status") == "ok"}


def run_seed(seed: int, original_text: str, python_exe: str) -> dict[str, Any]:
    run_id = f"seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    MAIN_PATH.write_text(update_main_for_seed(original_text, seed), encoding="utf-8")
    proc = subprocess.run(
        [python_exe, str(MAIN_PATH)],
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )
    stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "status": "ok" if proc.returncode == 0 else "failed",
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    for col in METRIC_COLUMNS:
        row[col] = ""
    if proc.returncode != 0:
        row["error"] = f"main.py exited with return code {proc.returncode}"
        return row
    try:
        metrics, mae_pp = parse_diagnostics()
        total_cases = parse_total_cases(proc.stdout)
        if total_cases is None or not math.isfinite(total_cases):
            raise ValueError("Could not parse total episode cases from stdout")
        row.update(
            {
                "total_cases": total_cases,
                "hetero": metrics["hetero"],
                "bi": metrics["bi"],
                "homo": metrics["homo"],
                "female_known": metrics["female_known"],
                "homo_male": metrics["homo_male"],
                "bi_male": metrics["bi_male"],
                "MAE_pp": mae_pp,
            }
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
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
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def build_readme() -> None:
    rows: list[dict[str, str]]
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]

    stats: dict[str, dict[str, float]] = {}
    for metric in METRIC_COLUMNS:
        vals = [float(row[metric]) for row in ok_rows if row.get(metric) != ""]
        stats[metric] = {
            "mean": statistics.mean(vals) if vals else float("nan"),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median": statistics.median(vals) if vals else float("nan"),
            "p2_5": percentile(vals, 2.5),
            "p97_5": percentile(vals, 97.5),
        }

    mae_vals = [float(row["MAE_pp"]) for row in ok_rows if row.get("MAE_pp") != ""]
    prop_mae_1 = sum(v <= 1.0 for v in mae_vals) / len(mae_vals) if mae_vals else float("nan")
    prop_mae_1_5 = sum(v <= 1.5 for v in mae_vals) / len(mae_vals) if mae_vals else float("nan")
    mean_mae = stats["MAE_pp"]["mean"]

    lines = [
        "# Final 100-seed validation",
        "",
        f"Completed seeds: {len(ok_rows)} successful, {len(failed_rows)} failed.",
        f"Seed range requested: 100000 to 100099.",
        "",
        "## Summary statistics",
        "",
        "| metric | mean | SD | median | 2.5th percentile | 97.5th percentile |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in METRIC_COLUMNS:
        s = stats[metric]
        if metric == "MAE_pp":
            lines.append(
                f"| {metric} | {s['mean']:.3f} pp | {s['sd']:.3f} pp | {s['median']:.3f} pp | {s['p2_5']:.3f} pp | {s['p97_5']:.3f} pp |"
            )
        elif metric == "total_cases":
            lines.append(
                f"| {metric} | {s['mean']:.1f} | {s['sd']:.1f} | {s['median']:.1f} | {s['p2_5']:.1f} | {s['p97_5']:.1f} |"
            )
        else:
            lines.append(
                f"| {metric} | {100*s['mean']:.3f}% | {100*s['sd']:.3f}% | {100*s['median']:.3f}% | {100*s['p2_5']:.3f}% | {100*s['p97_5']:.3f}% |"
            )
    lines.extend(
        [
            "",
            "## Calibration pass proportions",
            "",
            f"- Proportion of successful seeds with MAE <= 1.0 pp: {prop_mae_1:.3f}",
            f"- Proportion of successful seeds with MAE <= 1.5 pp: {prop_mae_1_5:.3f}",
            "",
            "## Interpretation",
            "",
        ]
    )
    if mean_mae <= 1.5:
        lines.append(
            "The 100-seed mean remains close to the observed case-composition targets. "
            "The mean MAE is not clearly worse than 1.5 pp, so the fixed calibrated baseline is retained."
        )
    else:
        lines.append(
            "The 100-seed mean MAE is clearly worse than 1.5 pp. This suggests the fixed baseline should be reviewed, but no new baseline was selected in this validation script."
        )
    if failed_rows:
        lines.extend(["", "## Failed seeds", ""])
        for row in failed_rows:
            lines.append(f"- Seed {row['seed']}: {row['error']}")
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    original_text = MAIN_PATH.read_text(encoding="utf-8")
    try:
        done = completed_seeds()
        for seed in range(100000, 100100):
            if seed in done:
                print(f"[skip] seed={seed} already completed", flush=True)
                continue
            print(f"[run] seed={seed}", flush=True)
            try:
                row = run_seed(seed, original_text, sys.executable)
            except Exception:
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "seed": seed,
                    "status": "failed",
                    "error": traceback.format_exc(limit=5),
                    "stdout_log": "",
                    "stderr_log": "",
                }
                for col in METRIC_COLUMNS:
                    row[col] = ""
            append_row(row)
            print(f"[done] seed={seed} status={row['status']} MAE={row.get('MAE_pp', '')}", flush=True)
        build_readme()
        return 0
    finally:
        MAIN_PATH.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
