"""Targeted tertiary v3 calibration focused on reducing female_known.

This runner keeps the current v3 model structure. It does not restore
party_hetero_leak, social_male_bias, party_mix_female_mult, or the old
combo_manual_02 structure.
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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_table_s7_tertiary_calibration"
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
STARTING_PARAMS = {
    "p_bi_m_to_msm": 0.70,
    "party_core_intensity": 0.70,
    "social_contact_scale": 6.0,
    "social_same_sex_assort": 0.65,
    "late_social_ramp": 6.0,
    "mix_mf_anal_frac": 0.18,
    "stable_mf_anal_frac": 0.03,
    "party_mix_homo_spill_frac": 0.03,
}
TERTIARY_VALUES = {
    "social_same_sex_assort": [0.70, 0.75, 0.80, 0.85],
    "mix_mf_anal_frac": [0.08, 0.10, 0.13, 0.16, 0.18],
    "stable_mf_anal_frac": [0.00, 0.01, 0.03],
    "party_mix_homo_spill_frac": [0.00, 0.01, 0.03],
}
OPTIONAL_VALUES = {
    "p_bi_f_to_wsw": [0.75, 0.85, 0.90],
}
PARAM_FIELDS = [
    "p_bi_m_to_msm",
    "party_core_intensity",
    "social_contact_scale",
    "social_same_sex_assort",
    "late_social_ramp",
    "mix_mf_anal_frac",
    "stable_mf_anal_frac",
    "party_mix_homo_spill_frac",
]
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
    "selection_score",
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


def _params(**overrides: float) -> dict[str, float]:
    out = dict(STARTING_PARAMS)
    out.update({key: float(value) for key, value in overrides.items()})
    return out


def screen_candidates() -> list[tuple[str, dict[str, float]]]:
    """Focused screen: OAT plus targeted female_known-reduction combinations."""
    candidates: list[tuple[str, dict[str, float]]] = [("start_secondary_best", _params())]
    for value in TERTIARY_VALUES["social_same_sex_assort"]:
        candidates.append((f"social_same_sex_assort_{value:g}", _params(social_same_sex_assort=value)))
    for value in [0.08, 0.10, 0.13, 0.16]:
        candidates.append((f"mix_mf_anal_frac_{value:g}", _params(mix_mf_anal_frac=value)))
    for value in [0.00, 0.01]:
        candidates.append((f"stable_mf_anal_frac_{value:g}", _params(stable_mf_anal_frac=value)))
    for value in [0.00, 0.01]:
        candidates.append((f"party_mix_homo_spill_frac_{value:g}", _params(party_mix_homo_spill_frac=value)))

    targeted = [
        ("social70_mix18_stable00_spill03", _params(social_same_sex_assort=0.70, mix_mf_anal_frac=0.18, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.03)),
        ("social75_mix18_stable00_spill03", _params(social_same_sex_assort=0.75, mix_mf_anal_frac=0.18, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.03)),
        ("social70_mix18_stable01_spill03", _params(social_same_sex_assort=0.70, mix_mf_anal_frac=0.18, stable_mf_anal_frac=0.01, party_mix_homo_spill_frac=0.03)),
        ("social70_mix16_stable00_spill03", _params(social_same_sex_assort=0.70, mix_mf_anal_frac=0.16, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.03)),
        ("social70_mix16_stable01_spill03", _params(social_same_sex_assort=0.70, mix_mf_anal_frac=0.16, stable_mf_anal_frac=0.01, party_mix_homo_spill_frac=0.03)),
        ("social70_mix18_stable00_spill01", _params(social_same_sex_assort=0.70, mix_mf_anal_frac=0.18, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.01)),
        ("social75_mix16_stable00_spill01", _params(social_same_sex_assort=0.75, mix_mf_anal_frac=0.16, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.01)),
        ("social75_mix13_stable00_spill01", _params(social_same_sex_assort=0.75, mix_mf_anal_frac=0.13, stable_mf_anal_frac=0.00, party_mix_homo_spill_frac=0.01)),
    ]
    candidates.extend(targeted)

    unique: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for label, params in candidates:
        key = _params_key(params)
        if key in seen:
            continue
        seen.add(key)
        unique.append((label, params))
    return unique


def _selection_score(row: dict[str, Any]) -> float:
    mae = _f(row.get("MAE_pp_mean"))
    hetero = _f(row.get("hetero_mean"))
    bi = _f(row.get("bi_mean"))
    homo = _f(row.get("homo_mean"))
    female = _f(row.get("female_known_mean"))
    if not all(math.isfinite(v) for v in [mae, hetero, bi, homo, female]):
        return 1e9
    penalty = 0.0
    penalty += max(0.0, 0.080 - hetero) * 200.0
    penalty += max(0.0, 0.115 - bi) * 200.0
    penalty += max(0.0, bi - 0.138) * 200.0
    penalty += max(0.0, homo - 0.805) * 200.0
    penalty += max(0.0, female - 0.045) * 220.0
    penalty += max(0.0, 0.028 - female) * 220.0
    return mae + penalty


def select_screen_candidates(n: int = 8) -> list[tuple[str, dict[str, float]]]:
    summaries = write_stage_summary("stage1_tertiary_screen")
    for row in summaries:
        row["selection_score"] = _selection_score(row)
    summaries.sort(key=lambda row: (_f(row.get("selection_score")), _f(row.get("MAE_pp_mean"))))
    selected = []
    with (OUTPUT_DIR / "stage1_selected_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summaries[:n]:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
            label = f"from_screen_{row['candidate_label']}"
            selected.append((label, {key: _f(row[key]) for key in PARAM_FIELDS}))
    return selected


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
            run_id = f"v3_tertiary_{stage}_{label}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        and 0.028 <= female <= 0.045
        and 0.115 <= bi <= 0.138
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
        summary["selection_score"] = _selection_score(summary)
        summary["passes_acceptance"] = _passes_acceptance(summary)
        summaries.append(summary)
    summaries.sort(key=lambda row: (_f(row.get("selection_score")), _f(row.get("MAE_pp_mean"))))
    with _summary_path(stage).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    return summaries


def write_stage_readme(stage: str, candidates: list[tuple[str, dict[str, float]]], seeds: list[int]) -> None:
    summaries = write_stage_summary(stage)
    lines = [
        f"# v3 Table S7 tertiary calibration - {stage}",
        "",
        "This stage keeps the current v3 model structure and focuses on reducing female_known while preserving observed cumulative case composition.",
        "",
        f"- Population size: `{POP_SIZE:,}`",
        f"- Seeds: `{seeds[0]}-{seeds[-1]}`" if len(seeds) > 1 else f"- Seed: `{seeds[0]}`",
        "- MAKE_PLOTS: temporarily `False` during subprocess runs; restored after each run",
        "- `population.py`: not modified",
        "",
        f"- Candidate sets: {len(candidates)}",
        f"- Result file: `{_results_path(stage)}`",
        f"- Summary file: `{_summary_path(stage)}`",
        "",
        "| rank | candidate | MAE_pp mean | female_known | hetero | bi | homo | total cases | passes |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(summaries[:10], start=1):
        lines.append(
            "| {rank} | {label} | {mae:.3f} | {female:.4f} | {hetero:.4f} | {bi:.4f} | {homo:.4f} | {cases:.1f} | {passes} |".format(
                rank=i,
                label=row["candidate_label"],
                mae=_f(row.get("MAE_pp_mean")),
                female=_f(row.get("female_known_mean")),
                hetero=_f(row.get("hetero_mean")),
                bi=_f(row.get("bi_mean")),
                homo=_f(row.get("homo_mean")),
                cases=_f(row.get("total_cases_mean")),
                passes=row.get("passes_acceptance", ""),
            )
        )
    (OUTPUT_DIR / f"{stage}_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_best_candidates(source_stage: str, n: int) -> list[tuple[str, dict[str, float]]]:
    summaries = write_stage_summary(source_stage)
    selected = []
    for row in summaries[:n]:
        label = f"from_{source_stage}_{row['candidate_label']}"
        selected.append((label, {key: _f(row[key]) for key in PARAM_FIELDS}))
    return selected


def stage_candidates(stage: str) -> tuple[list[tuple[str, dict[str, float]]], list[int]]:
    if stage == "stage1_tertiary_screen":
        return screen_candidates(), [BASE_SEED]
    if stage == "stage2_tertiary_short_validation":
        return select_best_candidates("stage1_tertiary_screen", 3), list(range(100000, 100005))
    if stage == "stage3_tertiary_extended_validation":
        return select_best_candidates("stage2_tertiary_short_validation", 1), list(range(100000, 100020))
    raise ValueError(f"Unknown stage: {stage}")


def write_overall_readme() -> None:
    lines = [
        "# v3 Table S7 tertiary calibration workflow",
        "",
        "Targeted local calibration after secondary Stage 4 failed because female_known remained high.",
        "",
        "## Starting Parameters",
        "",
        json.dumps(STARTING_PARAMS, indent=2),
        "",
        "## Tertiary Search Parameters",
        "",
        json.dumps(TERTIARY_VALUES, indent=2),
        "",
        "Optional `p_bi_f_to_wsw` values are documented but not run unless needed after the first tertiary screen.",
        "",
        "No `party_hetero_leak`, `social_male_bias`, `party_mix_female_mult`, old `combo_manual_02` structure, Table S8 sensitivity parameters, policy/testing/monitoring parameters, disease natural history, `party_touch_frac`, `sexual_risk_scale`, `close_touch`, `social_p_eff`, or age-linked susceptibility are tuned here.",
        "",
        "## Stage Files",
        "",
        "- `stage1_tertiary_screen_results.csv`, `stage1_tertiary_screen_summary.csv`, `stage1_tertiary_screen_README.md`",
        "- `stage1_selected_candidates.csv`",
        "- `stage2_tertiary_short_validation_results.csv`, `stage2_tertiary_short_validation_summary.csv`, `stage2_tertiary_short_validation_README.md`",
        "- `stage3_tertiary_extended_validation_results.csv`, `stage3_tertiary_extended_validation_summary.csv`, `stage3_tertiary_extended_validation_README.md`",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v3 tertiary local calibration")
    parser.add_argument(
        "--stage",
        choices=[
            "stage1_tertiary_screen",
            "stage2_tertiary_short_validation",
            "stage3_tertiary_extended_validation",
            "all",
        ],
        required=True,
    )
    args = parser.parse_args()
    _ensure_dirs()
    write_overall_readme()
    original_main = MAIN_PATH.read_text(encoding="utf-8")
    stages = (
        ["stage1_tertiary_screen", "stage2_tertiary_short_validation", "stage3_tertiary_extended_validation"]
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
