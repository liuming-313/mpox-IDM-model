"""Formal v3 Table S7 calibration workflow at 10M population scale.

This workflow keeps the current v3 model structure:
five Table S7 primary calibration parameters with social_same_sex_assort.
It does not restore party_hetero_leak, social_male_bias, or the old
combo_manual_02 six-parameter structure.
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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_table_s7_calibration"
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
PRIMARY_PARAMS = {
    "p_bi_m_to_msm": (0.55, 0.75),
    "party_core_intensity": (0.45, 0.70),
    "social_contact_scale": (2.5, 6.0),
    "social_same_sex_assort": (0.65, 0.85),
    "late_social_ramp": (1.0, 6.0),
}
BASELINE = {
    "p_bi_m_to_msm": 0.66,
    "party_core_intensity": 0.52,
    "social_contact_scale": 5.0,
    "social_same_sex_assort": 0.75,
    "late_social_ramp": 5.5,
}

RESULT_FIELDS = [
    "timestamp",
    "stage",
    "candidate_id",
    "candidate_label",
    "seed",
    "pop_size",
    "run_id",
    "status",
    *PRIMARY_PARAMS.keys(),
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
    *PRIMARY_PARAMS.keys(),
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
    return json.dumps({k: round(float(params[k]), 10) for k in PRIMARY_PARAMS}, sort_keys=True)


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


def _completed(stage: str) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    for row in _read_rows(_results_path(stage)):
        if row.get("status") != "ok":
            continue
        params = {k: _f(row.get(k)) for k in PRIMARY_PARAMS}
        done.add((_params_key(params), int(float(row["seed"]))))
    return done


def _within_bounds(params: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in params.items():
        lo, hi = PRIMARY_PARAMS[key]
        out[key] = min(float(hi), max(float(lo), float(value)))
    return out


def stage0_candidates() -> list[tuple[str, dict[str, float]]]:
    return [("current_v3_baseline", dict(BASELINE))]


def stage1_candidates() -> list[tuple[str, dict[str, float]]]:
    values = {
        "p_bi_m_to_msm": [0.55, 0.60, 0.66, 0.70, 0.75],
        "party_core_intensity": [0.45, 0.52, 0.58, 0.64, 0.70],
        "social_contact_scale": [4.5, 5.0, 5.5, 6.0],
        "social_same_sex_assort": [0.65, 0.70, 0.75, 0.80, 0.85],
        "late_social_ramp": [4.5, 5.0, 5.5, 6.0],
    }
    candidates: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for param, param_values in values.items():
        for value in param_values:
            params = dict(BASELINE)
            params[param] = value
            key = _params_key(params)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((f"{param}_{value:g}", params))
    return candidates


def manual_stage2_candidates() -> list[tuple[str, dict[str, float]]]:
    raw = [
        ("balanced_social_high_party", dict(p_bi_m_to_msm=0.60, party_core_intensity=0.64, social_contact_scale=6.0, social_same_sex_assort=0.70, late_social_ramp=6.0)),
        ("low_bi_route_high_social", dict(p_bi_m_to_msm=0.55, party_core_intensity=0.58, social_contact_scale=6.0, social_same_sex_assort=0.70, late_social_ramp=6.0)),
        ("mid_bi_high_party_low_assort", dict(p_bi_m_to_msm=0.60, party_core_intensity=0.70, social_contact_scale=5.5, social_same_sex_assort=0.65, late_social_ramp=6.0)),
        ("baseline_bi_high_social", dict(p_bi_m_to_msm=0.66, party_core_intensity=0.64, social_contact_scale=6.0, social_same_sex_assort=0.70, late_social_ramp=6.0)),
        ("low_assort_high_party", dict(p_bi_m_to_msm=0.66, party_core_intensity=0.70, social_contact_scale=6.0, social_same_sex_assort=0.65, late_social_ramp=5.5)),
        ("more_msm_route_more_party", dict(p_bi_m_to_msm=0.70, party_core_intensity=0.70, social_contact_scale=6.0, social_same_sex_assort=0.65, late_social_ramp=6.0)),
        ("moderate_all_up", dict(p_bi_m_to_msm=0.60, party_core_intensity=0.58, social_contact_scale=5.5, social_same_sex_assort=0.70, late_social_ramp=6.0)),
        ("cases_up_assort_baseline", dict(p_bi_m_to_msm=0.60, party_core_intensity=0.70, social_contact_scale=5.0, social_same_sex_assort=0.75, late_social_ramp=6.0)),
        ("social_up_low_assort_party_base", dict(p_bi_m_to_msm=0.60, party_core_intensity=0.52, social_contact_scale=6.0, social_same_sex_assort=0.65, late_social_ramp=6.0)),
        ("low_bi_low_assort_max_core", dict(p_bi_m_to_msm=0.55, party_core_intensity=0.70, social_contact_scale=6.0, social_same_sex_assort=0.65, late_social_ramp=6.0)),
        ("bi_target_push", dict(p_bi_m_to_msm=0.75, party_core_intensity=0.64, social_contact_scale=5.5, social_same_sex_assort=0.70, late_social_ramp=6.0)),
        ("hetero_push_no_max_core", dict(p_bi_m_to_msm=0.55, party_core_intensity=0.52, social_contact_scale=6.0, social_same_sex_assort=0.65, late_social_ramp=6.0)),
    ]
    return [(label, _within_bounds(params)) for label, params in raw]


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
            if (key, seed) in done:
                print(f"[skip] {stage} {label} seed={seed}", flush=True)
                continue
            run_id = f"v3_{stage}_{label}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            **{key: _f(first.get(key)) for key in PRIMARY_PARAMS},
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
    summaries.sort(key=lambda row: (_f(row.get("MAE_pp_mean")), -_f(row.get("hetero_mean"))))
    with _summary_path(stage).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return summaries


def write_stage_readme(stage: str, candidates: list[tuple[str, dict[str, float]]], seeds: list[int]) -> None:
    summaries = write_stage_summary(stage)
    lines = [
        f"# v3 Table S7 calibration - {stage}",
        "",
        "This stage uses the current v3 model structure with five Table S7 primary calibration parameters. It is not Table S8 sensitivity analysis and does not use 1M simulations for parameter selection.",
        "",
        f"- Population size: `{POP_SIZE:,}`",
        f"- Seeds: `{seeds[0]}-{seeds[-1]}`" if len(seeds) > 1 else f"- Seed: `{seeds[0]}`",
        "- MAKE_PLOTS: temporarily `False` during subprocess runs; restored after each run",
        "- `population.py`: not modified",
        "",
        "## Candidate Count",
        "",
        f"- Candidate sets: {len(candidates)}",
        f"- Result file: `{_results_path(stage)}`",
        f"- Summary file: `{_summary_path(stage)}`",
        "",
        "## Top Candidates By MAE",
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
        params = {key: _f(row.get(key)) for key in PRIMARY_PARAMS}
        selected.append((label, params))
    return selected


def stage_candidates(stage: str) -> tuple[list[tuple[str, dict[str, float]]], list[int]]:
    if stage == "stage0_baseline":
        return stage0_candidates(), list(range(100000, 100005))
    if stage == "stage1_primary_screen":
        return stage1_candidates(), [BASE_SEED]
    if stage == "stage2_combined":
        return manual_stage2_candidates(), [BASE_SEED]
    if stage == "stage3_short_validation":
        return select_best_candidates("stage2_combined", 3), list(range(100000, 100005))
    if stage == "stage4_extended_validation":
        return select_best_candidates("stage3_short_validation", 1), list(range(100000, 100020))
    raise ValueError(f"Unknown stage: {stage}")


def write_overall_readme() -> None:
    lines = [
        "# v3 Table S7 calibration workflow",
        "",
        "Formal calibration output for the current v3 model structure documented in the Supplementary Appendix.",
        "",
        "## Fixed Decisions",
        "",
        "- Do not restore `party_hetero_leak`.",
        "- Do not restore `social_male_bias`.",
        "- Do not use old `combo_manual_02` six-parameter structure.",
        "- Primary calibration uses only five Table S7 parameters: `p_bi_m_to_msm`, `party_core_intensity`, `social_contact_scale`, `social_same_sex_assort`, and `late_social_ramp`.",
        "- Formal population scale is `10,000,000`; 1M runs are not interpreted as calibration evidence.",
        "- Table S8 sensitivity is deferred until a new v3 baseline passes Stage 4.",
        "",
        "## Targets",
        "",
        json.dumps(TARGETS, indent=2),
        "",
        "## Stage Files",
        "",
    ]
    for stage in ["stage0_baseline", "stage1_primary_screen", "stage2_combined", "stage3_short_validation", "stage4_extended_validation"]:
        lines.append(f"- `{stage}_results.csv`, `{stage}_summary.csv`, `{stage}_README.md`")
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run formal v3 Table S7 calibration stages")
    parser.add_argument(
        "--stage",
        choices=["stage0_baseline", "stage1_primary_screen", "stage2_combined", "stage3_short_validation", "stage4_extended_validation", "all"],
        required=True,
    )
    args = parser.parse_args()
    _ensure_dirs()
    write_overall_readme()
    original_main = MAIN_PATH.read_text(encoding="utf-8")
    stages = (
        ["stage0_baseline", "stage1_primary_screen", "stage2_combined", "stage3_short_validation", "stage4_extended_validation"]
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
