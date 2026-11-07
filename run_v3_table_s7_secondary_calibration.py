"""Secondary local calibration for the v3 Table S7 baseline.

Only mix_mf_anal_frac and stable_mf_anal_frac are varied. The five primary
v3 Table S7 parameters are held fixed at the best Stage 4 primary candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from run_one import PROJECT_DIR, run_model


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_table_s7_secondary_calibration"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"
MAIN_PATH = PROJECT_DIR / "main.py"

POP_SIZE = 10_000_000
BASE_SEED = 100000
TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
    "homo_male": 0.7796,
    "bi_male": 0.1272,
}
FIXED_PRIMARY = {
    "p_bi_m_to_msm": 0.70,
    "party_core_intensity": 0.70,
    "social_contact_scale": 6.0,
    "social_same_sex_assort": 0.65,
    "late_social_ramp": 6.0,
}
SECONDARY_BASELINE = {
    "mix_mf_anal_frac": 0.13,
    "stable_mf_anal_frac": 0.05,
}
SECONDARY_VALUES = {
    "mix_mf_anal_frac": [0.08, 0.10, 0.13, 0.16, 0.18],
    "stable_mf_anal_frac": [0.03, 0.05, 0.08],
}
PARAM_FIELDS = [*FIXED_PRIMARY.keys(), *SECONDARY_BASELINE.keys()]
RESULT_FIELDS = [
    "timestamp",
    "stage",
    "candidate_id",
    "candidate_label",
    "seed",
    "pop_size",
    "run_id",
    "status",
    *PARAM_FIELDS,
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
    "stage",
    "candidate_id",
    "candidate_label",
    *PARAM_FIELDS,
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
    "diff_hetero_pp_mean",
    "diff_bi_pp_mean",
    "diff_homo_pp_mean",
    "diff_female_known_pp_mean",
    "diff_homo_male_pp_mean",
    "diff_bi_male_pp_mean",
    "passes_acceptance",
]


def _f(value: Any) -> float:
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


def _params_key(params: dict[str, float]) -> str:
    return json.dumps({key: round(float(params[key]), 10) for key in PARAM_FIELDS}, sort_keys=True)


def _candidate_id(stage: str, index: int) -> str:
    return f"{stage}_{index:03d}"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _results_path(stage: str) -> Path:
    return OUTPUT_DIR / f"{stage}_results.csv"


def _summary_path(stage: str) -> Path:
    return OUTPUT_DIR / f"{stage}_summary.csv"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _completed(stage: str) -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    for row in _read_rows(_results_path(stage)):
        if row.get("status") != "ok":
            continue
        params = {key: _f(row.get(key)) for key in PARAM_FIELDS}
        done.add((_params_key(params), int(float(row["seed"])), row.get("candidate_label", "")))
    return done


def _params(mix_mf_anal_frac: float, stable_mf_anal_frac: float) -> dict[str, float]:
    return {
        **FIXED_PRIMARY,
        "mix_mf_anal_frac": float(mix_mf_anal_frac),
        "stable_mf_anal_frac": float(stable_mf_anal_frac),
    }


def screen_candidates() -> list[tuple[str, dict[str, float]]]:
    candidates: list[tuple[str, dict[str, float]]] = []
    for value in SECONDARY_VALUES["mix_mf_anal_frac"]:
        candidates.append((f"mix_mf_anal_frac_{value:g}", _params(value, SECONDARY_BASELINE["stable_mf_anal_frac"])))
    for value in SECONDARY_VALUES["stable_mf_anal_frac"]:
        candidates.append((f"stable_mf_anal_frac_{value:g}", _params(SECONDARY_BASELINE["mix_mf_anal_frac"], value)))
    return candidates


def combined_candidates() -> list[tuple[str, dict[str, float]]]:
    # Six local combinations selected from Stage 1: keep the current secondary
    # baseline as a comparator, retain the stable-frac signals already tested,
    # and add female-lowering mix values crossed with the lower stable value.
    # This is deliberately not a broad 5x3 grid.
    pairs = [
        (0.13, 0.05),
        (0.13, 0.03),
        (0.13, 0.08),
        (0.16, 0.03),
        (0.18, 0.03),
        (0.16, 0.08),
    ]
    return [(f"mix_{mix:g}_stable_{stable:g}", _params(mix, stable)) for mix, stable in pairs]


def _copy_diagnostics(run_id: str) -> str:
    if not SOURCE_DIAGNOSTICS.exists():
        return ""
    dst = DIAGNOSTICS_DIR / f"{run_id}_calibration_diagnostics.csv"
    shutil.copy2(SOURCE_DIAGNOSTICS, dst)
    return str(dst)


def _decorate_result(stage: str, cid: str, label: str, params: dict[str, float], result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "candidate_id": cid,
        "candidate_label": label,
        **params,
        **result,
    }
    mae_values = []
    for metric, target in TARGETS.items():
        current = _f(result.get(metric))
        diff = (current - target) * 100.0 if math.isfinite(current) else math.nan
        row[f"diff_{metric}_pp"] = diff
        if metric in {"hetero", "bi", "homo", "female_known"} and math.isfinite(current):
            mae_values.append(abs(diff))
    row["MAE_pp"] = _mean(mae_values)
    if result.get("status") == "ok":
        row["diagnostics_copy"] = _copy_diagnostics(str(result.get("run_id", cid)))
    return row


def run_candidates(stage: str, candidates: list[tuple[str, dict[str, float]]], seeds: list[int]) -> None:
    _ensure_dirs()
    done = _completed(stage)
    for index, (label, params) in enumerate(candidates, start=1):
        cid = _candidate_id(stage, index)
        key = _params_key(params)
        for seed in seeds:
            if (key, seed, label) in done:
                print(f"[skip] {stage} {label} seed={seed}", flush=True)
                continue
            run_id = f"v3_secondary_{stage}_{label}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[run] {stage} {label} seed={seed}", flush=True)
            result = run_model(
                params,
                seed=seed,
                pop_size=POP_SIZE,
                run_id=run_id,
                make_plots=False,
                log_dir=LOG_DIR / stage,
            )
            row = _decorate_result(stage, cid, label, params, result)
            _append_row(_results_path(stage), row)
            print(f"[done] {stage} {label} seed={seed} status={row['status']} MAE_pp={row.get('MAE_pp', '')}", flush=True)
    write_stage_summary(stage)
    write_stage_readme(stage, candidates, seeds)


def _passes_acceptance(row: dict[str, Any]) -> str:
    mae = _f(row.get("MAE_pp_mean"))
    hetero = _f(row.get("hetero_mean"))
    bi = _f(row.get("bi_mean"))
    homo = _f(row.get("homo_mean"))
    female = _f(row.get("female_known_mean"))
    ok = (
        math.isfinite(mae)
        and mae <= 1.2
        and 0.115 <= bi <= 0.138
        and 0.028 <= female <= 0.045
        and hetero >= 0.080
        and homo <= 0.805
    )
    return "yes" if ok else "no"


def write_stage_summary(stage: str) -> list[dict[str, Any]]:
    rows = _read_rows(_results_path(stage))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["candidate_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for cid, vals in groups.items():
        first = vals[0]
        ok_rows = [row for row in vals if row.get("status") == "ok"]
        summary: dict[str, Any] = {
            "stage": stage,
            "candidate_id": cid,
            "candidate_label": first.get("candidate_label", ""),
            **{key: _f(first.get(key)) for key in PARAM_FIELDS},
            "successful_runs": len(ok_rows),
            "failed_runs": len(vals) - len(ok_rows),
        }
        for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male", "MAE_pp"]:
            values = [_f(row.get(metric)) for row in ok_rows]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_sd"] = _sd(values)
        for metric in TARGETS:
            summary[f"diff_{metric}_pp_mean"] = (_f(summary.get(f"{metric}_mean")) - TARGETS[metric]) * 100.0
        summary["passes_acceptance"] = _passes_acceptance(summary)
        summaries.append(summary)
    summaries.sort(key=lambda row: (_f(row.get("MAE_pp_mean")), max(0.0, _f(row.get("female_known_mean")) - 0.045)))
    with _summary_path(stage).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return summaries


def write_stage_readme(stage: str, candidates: list[tuple[str, dict[str, float]]], seeds: list[int]) -> None:
    summaries = write_stage_summary(stage)
    lines = [
        f"# v3 Table S7 secondary calibration - {stage}",
        "",
        "This stage keeps the current v3 model structure and holds the five primary Table S7 parameters fixed at the best Stage 4 primary candidate.",
        "",
        f"- Population size: `{POP_SIZE:,}`",
        f"- Seeds: `{seeds[0]}-{seeds[-1]}`" if len(seeds) > 1 else f"- Seed: `{seeds[0]}`",
        "- Varying only: `mix_mf_anal_frac`, `stable_mf_anal_frac`",
        "- MAKE_PLOTS: temporarily `False` during subprocess runs; restored after each run",
        "- `population.py`: not modified",
        "",
        f"- Candidate sets: {len(candidates)}",
        f"- Result file: `{_results_path(stage)}`",
        f"- Summary file: `{_summary_path(stage)}`",
        "",
        "| rank | candidate | MAE_pp mean | total cases | hetero | bi | homo | female_known | passes acceptance |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(summaries[:10], start=1):
        lines.append(
            "| {rank} | {label} | {mae:.3f} | {cases:.1f} | {hetero:.4f} | {bi:.4f} | {homo:.4f} | {female:.4f} | {passes} |".format(
                rank=i,
                label=row["candidate_label"],
                mae=_f(row.get("MAE_pp_mean")),
                cases=_f(row.get("total_cases_mean")),
                hetero=_f(row.get("hetero_mean")),
                bi=_f(row.get("bi_mean")),
                homo=_f(row.get("homo_mean")),
                female=_f(row.get("female_known_mean")),
                passes=row.get("passes_acceptance", ""),
            )
        )
    (OUTPUT_DIR / f"{stage}_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_best_candidates(source_stage: str, n: int) -> list[tuple[str, dict[str, float]]]:
    summaries = write_stage_summary(source_stage)
    selected = []
    for row in summaries[:n]:
        label = f"from_{source_stage}_{row['candidate_label']}"
        params = {key: _f(row.get(key)) for key in PARAM_FIELDS}
        selected.append((label, params))
    return selected


def stage_candidates(stage: str) -> tuple[list[tuple[str, dict[str, float]]], list[int]]:
    if stage == "stage1_secondary_screen":
        return screen_candidates(), [BASE_SEED]
    if stage == "stage2_secondary_combined":
        return combined_candidates(), [BASE_SEED]
    if stage == "stage3_secondary_short_validation":
        return select_best_candidates("stage2_secondary_combined", 3), list(range(100000, 100005))
    if stage == "stage4_secondary_extended_validation":
        return select_best_candidates("stage3_secondary_short_validation", 1), list(range(100000, 100020))
    raise ValueError(f"Unknown stage: {stage}")


def write_overall_readme() -> None:
    lines = [
        "# v3 Table S7 secondary calibration workflow",
        "",
        "Secondary local calibration after the five-parameter v3 primary search failed because female_known was high.",
        "",
        "## Fixed Primary Candidate",
        "",
        json.dumps(FIXED_PRIMARY, indent=2),
        "",
        "## Secondary Parameters Only",
        "",
        "- `mix_mf_anal_frac`: `0.08-0.18`",
        "- `stable_mf_anal_frac`: `0.03-0.08`",
        "",
        "No `party_hetero_leak`, `social_male_bias`, old `combo_manual_02` structure, Table S8 sensitivity parameters, policy/testing/monitoring parameters, disease natural history, `party_touch_frac`, `sexual_risk_scale`, `close_touch`, `social_p_eff`, or age-linked susceptibility are tuned here.",
        "",
        "## Stage Files",
        "",
        "- `stage1_secondary_screen_results.csv`, `stage1_secondary_screen_summary.csv`, `stage1_secondary_screen_README.md`",
        "- `stage2_secondary_combined_results.csv`, `stage2_secondary_combined_summary.csv`, `stage2_secondary_combined_README.md`",
        "- `stage3_secondary_short_validation_results.csv`, `stage3_secondary_short_validation_summary.csv`, `stage3_secondary_short_validation_README.md`",
        "- `stage4_secondary_extended_validation_results.csv`, `stage4_secondary_extended_validation_summary.csv`, `stage4_secondary_extended_validation_README.md`",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v3 Table S7 secondary calibration")
    parser.add_argument(
        "--stage",
        choices=[
            "stage1_secondary_screen",
            "stage2_secondary_combined",
            "stage3_secondary_short_validation",
            "stage4_secondary_extended_validation",
            "all",
        ],
        required=True,
    )
    args = parser.parse_args()
    _ensure_dirs()
    write_overall_readme()
    original_main = MAIN_PATH.read_text(encoding="utf-8")
    stages = (
        [
            "stage1_secondary_screen",
            "stage2_secondary_combined",
            "stage3_secondary_short_validation",
            "stage4_secondary_extended_validation",
        ]
        if args.stage == "all"
        else [args.stage]
    )
    try:
        for stage in stages:
            candidates, seeds = stage_candidates(stage)
            run_candidates(stage, candidates, seeds)
    finally:
        if MAIN_PATH.read_text(encoding="utf-8") != original_main:
            MAIN_PATH.write_text(original_main, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
