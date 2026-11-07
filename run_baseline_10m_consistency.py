"""Run frozen 10M baseline consistency checks for Table S8 gating.

This is validation only, not recalibration. It uses run_one.run_model(), which
temporarily patches runtime controls for each subprocess and restores main.py
after each run.
"""

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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "sensitivity_table_s8" / "baseline_10m_consistency"
RESULTS_CSV = OUTPUT_DIR / "baseline_10m_consistency_results.csv"
SUMMARY_CSV = OUTPUT_DIR / "baseline_10m_consistency_summary.csv"
README_MD = OUTPUT_DIR / "baseline_10m_consistency_readme.md"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
DIAGNOSTICS_SOURCE = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"
FINAL_BASELINE_JSON = PROJECT_DIR / "outputs" / "final_model_freeze" / "final_baseline_params.json"
MAIN_PATH = PROJECT_DIR / "main.py"

SEEDS = list(range(100000, 100020))
POP_SIZE = 10_000_000
FROZEN_TABLE_S7_BASELINE = {
    "p_bi_m_to_msm": 0.66,
    "party_core_intensity": 0.52,
    "social_contact_scale": 5.0,
    "social_same_sex_assort": 0.75,
    "late_social_ramp": 5.5,
}
REFERENCE_100_SEED = {
    "total_cases": 2322.3,
    "hetero": 0.08286,
    "bi": 0.12246,
    "homo": 0.79468,
    "female_known": 0.03555,
    "MAE_pp": 0.967,
}
FIELDS = [
    "timestamp",
    "seed",
    "run_id",
    "status",
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


def _fmt(value: Any, digits: int = 3) -> str:
    val = _float_or_nan(value)
    return "NA" if not math.isfinite(val) else f"{val:.{digits}f}"


def _load_completed() -> set[int]:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        return {
            int(row["seed"])
            for row in csv.DictReader(f)
            if row.get("status") == "ok" and row.get("seed")
        }


def _append_row(row: dict[str, Any]) -> None:
    exists = RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def _load_rows() -> list[dict[str, str]]:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _verify_frozen_baseline() -> None:
    data = json.loads(FINAL_BASELINE_JSON.read_text(encoding="utf-8"))
    params = data["table_s7_primary_parameters"]
    for key, value in FROZEN_TABLE_S7_BASELINE.items():
        if abs(float(params[key]) - float(value)) > 1e-9:
            raise ValueError(f"Frozen Table S7 baseline mismatch for {key}: {params[key]} != {value}")


def _write_summary(main_restored: bool) -> dict[str, Any]:
    rows = _load_rows()
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pop_size": POP_SIZE,
        "seed_start": SEEDS[0],
        "seed_end": SEEDS[-1],
        "successful_runs": len(ok_rows),
        "failed_runs": len(failed_rows),
        "main_restored_after_runs": main_restored,
    }
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]:
        vals = [_float_or_nan(row.get(metric)) for row in ok_rows]
        mean = _mean(vals)
        sd = _sd(vals)
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_sd"] = sd
        summary[f"{metric}_reference_100seed"] = REFERENCE_100_SEED[metric]
        summary[f"{metric}_delta_vs_reference"] = mean - REFERENCE_100_SEED[metric]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    readme = [
        "# 10M frozen-baseline consistency check",
        "",
        "This is a baseline consistency check only. It is not recalibration and does not select a new baseline.",
        "",
        "## Run design",
        "",
        f"- Population size: `{POP_SIZE:,}`",
        f"- Seeds: `{SEEDS[0]}-{SEEDS[-1]}`",
        "- MAKE_PLOTS: temporarily set to `False` inside each subprocess; restored afterward",
        "- Frozen Table S7 primary baseline unchanged",
        "",
        "## Results",
        "",
        f"- Successful runs: {summary['successful_runs']}",
        f"- Failed runs: {summary['failed_runs']}",
        f"- main.py restored after runs: {main_restored}",
        "",
        "| metric | 20-seed mean | 20-seed SD | previous 100-seed mean | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]:
        digits = 1 if metric == "total_cases" else 4 if metric != "MAE_pp" else 3
        readme.append(
            f"| {metric} | {_fmt(summary[f'{metric}_mean'], digits)} | "
            f"{_fmt(summary[f'{metric}_sd'], digits)} | "
            f"{_fmt(summary[f'{metric}_reference_100seed'], digits)} | "
            f"{_fmt(summary[f'{metric}_delta_vs_reference'], digits)} |"
        )
    if failed_rows:
        readme.extend(["", "## Failed seeds", ""])
        for row in failed_rows:
            readme.append(f"- seed `{row.get('seed')}`: {row.get('error', '')}")
    README_MD.write_text("\n".join(readme) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    _verify_frozen_baseline()
    original_main = MAIN_PATH.read_text(encoding="utf-8")
    completed = _load_completed()

    for seed in SEEDS:
        if seed in completed:
            print(f"[skip] seed={seed}", flush=True)
            continue
        run_id = f"baseline_10m_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"[run] {run_id}", flush=True)
        result = run_model(
            FROZEN_TABLE_S7_BASELINE,
            seed=seed,
            pop_size=POP_SIZE,
            run_id=run_id,
            make_plots=False,
            log_dir=LOG_DIR,
        )
        diagnostics_copy = ""
        if result.get("status") == "ok" and DIAGNOSTICS_SOURCE.exists():
            dst = DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv"
            shutil.copy2(DIAGNOSTICS_SOURCE, dst)
            diagnostics_copy = str(dst)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "MAE_pp": result.get("objective_case_composition_mae", ""),
            "diagnostics_copy": diagnostics_copy,
            **result,
        }
        _append_row(row)
        print(f"[done] {run_id} status={row['status']} MAE_pp={row.get('MAE_pp', '')}", flush=True)

    main_restored = MAIN_PATH.read_text(encoding="utf-8") == original_main
    summary = _write_summary(main_restored)
    print(f"[summary] wrote {SUMMARY_CSV}", flush=True)
    print(f"[summary] wrote {README_MD}", flush=True)
    print(
        f"[result] ok={summary['successful_runs']} failed={summary['failed_runs']} "
        f"MAE_pp_mean={summary['MAE_pp_mean']:.3f}",
        flush=True,
    )
    return 0 if main_restored else 2


if __name__ == "__main__":
    raise SystemExit(main())
