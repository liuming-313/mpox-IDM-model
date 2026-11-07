"""Staged calibration for the six simplified mpox/STI parameters.

The runner changes only the six simplified calibration parameters, base_run_seed,
and MAKE_PLOTS while subprocesses run. It does not edit population.py or model
structure.
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
OUT_DIR = PROJECT_DIR / "outputs" / "simplified_calibration"
LOG_DIR = OUT_DIR / "logs"
HISTORY_PATH = OUT_DIR / "history.csv"

TARGETS = {
    "hetero": 0.0931,
    "bi": 0.1272,
    "homo": 0.7796,
    "female_known": 0.0336,
}

TUNABLES = [
    "p_bi_m_to_msm",
    "party_core_intensity",
    "party_hetero_leak",
    "social_contact_scale",
    "social_male_bias",
    "late_social_ramp",
]

BASELINE = {
    "p_bi_m_to_msm": 0.66,
    "party_core_intensity": 0.55,
    "party_hetero_leak": 0.026,
    "social_contact_scale": 4.20,
    "social_male_bias": 0.92,
    "late_social_ramp": 5.00,
}

BOUNDS = {
    "p_bi_m_to_msm": (0.55, 0.75),
    "party_core_intensity": (0.45, 0.70),
    "party_hetero_leak": (0.015, 0.040),
    "social_contact_scale": (2.5, 6.0),
    "social_male_bias": (0.75, 0.95),
    "late_social_ramp": (1.0, 6.0),
}

SINGLE_GRID = {
    "p_bi_m_to_msm": [0.62, 0.66, 0.70],
    "party_core_intensity": [0.52, 0.55, 0.58],
    "party_hetero_leak": [0.022, 0.026, 0.030],
    "social_contact_scale": [3.8, 4.2, 4.6, 5.0],
    "social_male_bias": [0.90, 0.92, 0.94],
    "late_social_ramp": [4.0, 5.0, 5.5, 6.0],
}

DIAGNOSTICS_CANDIDATES = [
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv",
    PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "figures" / "calibration_diagnostics.csv",
]

FIELDNAMES = [
    "timestamp",
    "stage",
    "candidate",
    "run_id",
    "seed",
    "status",
    *TUNABLES,
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "MAE_pp",
    "error",
    "stdout_log",
    "stderr_log",
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def fnum() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def update_main_text(original: str, params: dict[str, float], seed: int, make_plots: bool = False) -> str:
    text = original
    for key, value in params.items():
        if key not in TUNABLES:
            raise ValueError(f"Refusing to update non-tunable parameter: {key}")
        lo, hi = BOUNDS[key]
        if not (lo <= float(value) <= hi):
            raise ValueError(f"{key}={value} outside allowed range {lo}-{hi}")
        pattern = rf'(^\s*"{re.escape(key)}"\s*:\s*)({fnum()})'
        text, n = re.subn(pattern, rf"\g<1>{float(value):g}", text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f"Could not safely update {key}")
    text, n_seed = re.subn(
        rf"(^\s*base_run_seed\s*=\s*)({fnum()})",
        rf"\g<1>{int(seed)}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_seed != 1:
        raise ValueError("Could not safely update base_run_seed")
    text, n_plot = re.subn(
        r"(^\s*MAKE_PLOTS\s*=\s*)(True|False)",
        rf"\g<1>{str(bool(make_plots))}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_plot != 1:
        raise ValueError("Could not safely update MAKE_PLOTS")
    return text


def write_main(params: dict[str, float], seed: int = 100000, make_plots: bool = True) -> None:
    original = MAIN_PATH.read_text(encoding="utf-8")
    MAIN_PATH.write_text(update_main_text(original, params, seed, make_plots), encoding="utf-8")


def run_id(stage: str, candidate: str, seed: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_")
    return f"{stage}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"


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
        raise FileNotFoundError("No calibration_diagnostics.csv found")
    values: dict[str, float] = {}
    with diag.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            metric = (row.get("metric") or "").strip()
            if metric in TARGETS:
                values[metric] = float(row["current"])
    missing = sorted(set(TARGETS) - set(values))
    if missing:
        raise ValueError(f"Missing diagnostics metrics: {missing}")
    values["MAE_pp"] = statistics.mean(abs(values[m] - TARGETS[m]) for m in TARGETS) * 100.0
    return values


def append_history(row: dict[str, Any]) -> None:
    exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_history() -> list[dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def history_match(stage: str, candidate: str, seed: int, params: dict[str, float]) -> dict[str, str] | None:
    for row in reversed(load_history()):
        if row.get("stage") != stage or row.get("candidate") != candidate or row.get("seed") != str(seed):
            continue
        if row.get("status") != "ok":
            continue
        if all(abs(float(row[k]) - float(params[k])) < 1e-9 for k in TUNABLES):
            return row
    return None


def run_once(stage: str, candidate: str, seed: int, params: dict[str, float], original_text: str) -> dict[str, Any]:
    cached = history_match(stage, candidate, seed, params)
    if cached:
        print(f"[skip] {stage} {candidate} seed={seed}", flush=True)
        return dict(cached)
    rid = run_id(stage, candidate, seed)
    stdout_log = LOG_DIR / f"{rid}.stdout.log"
    stderr_log = LOG_DIR / f"{rid}.stderr.log"
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "candidate": candidate,
        "run_id": rid,
        "seed": seed,
        "status": "failed",
        **params,
        "total_cases": "",
        "hetero": "",
        "bi": "",
        "homo": "",
        "female_known": "",
        "MAE_pp": "",
        "error": "",
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    try:
        MAIN_PATH.write_text(update_main_text(original_text, params, seed, make_plots=False), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(MAIN_PATH)], cwd=PROJECT_DIR, text=True, capture_output=True)
        stdout_log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        stderr_log.write_text(proc.stderr, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            row["error"] = f"main.py exited with return code {proc.returncode}"
        else:
            metrics = parse_diagnostics()
            total_cases = parse_total_cases(proc.stdout)
            if total_cases is None:
                raise ValueError("Could not parse total_cases")
            row.update({m: metrics[m] for m in TARGETS})
            row["MAE_pp"] = metrics["MAE_pp"]
            row["total_cases"] = total_cases
            row["status"] = "ok"
    except Exception:
        row["error"] = traceback.format_exc(limit=5)
    append_history(row)
    print(f"[done] {stage} {candidate} seed={seed} status={row['status']} MAE={row['MAE_pp']}", flush=True)
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def ok(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok"


def hard_ok(row: dict[str, Any]) -> bool:
    if not ok(row):
        return False
    return (
        0.115 <= float(row["bi"]) <= 0.138
        and 0.028 <= float(row["female_known"]) <= 0.045
        and float(row["hetero"]) >= 0.080
        and float(row["homo"]) <= 0.805
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    good = [r for r in rows if ok(r)]
    metrics = ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        vals = [float(r[metric]) for r in good]
        out[metric] = {
            "mean": statistics.mean(vals) if vals else float("nan"),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def stage2_candidates(single_rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, float]]]:
    good_single = [r for r in single_rows if hard_ok(r)]
    good_single.sort(key=lambda r: (float(r["MAE_pp"]), -float(r["hetero"])))

    candidates: list[tuple[str, dict[str, float]]] = []
    manual = [
        {"party_core_intensity": 0.52, "social_contact_scale": 4.6, "late_social_ramp": 5.5},
        {"party_core_intensity": 0.52, "social_contact_scale": 5.0, "late_social_ramp": 5.5},
        {"party_core_intensity": 0.52, "social_male_bias": 0.94, "late_social_ramp": 5.5},
        {"party_core_intensity": 0.55, "social_male_bias": 0.94, "late_social_ramp": 5.5},
        {"party_core_intensity": 0.52, "social_contact_scale": 4.6, "social_male_bias": 0.94},
        {"party_core_intensity": 0.58, "social_male_bias": 0.94, "party_hetero_leak": 0.022},
        {"social_contact_scale": 4.6, "social_male_bias": 0.94, "late_social_ramp": 5.5},
        {"party_core_intensity": 0.52, "party_hetero_leak": 0.022, "social_contact_scale": 4.6},
    ]
    for i, changes in enumerate(manual, start=1):
        p = dict(BASELINE)
        p.update(changes)
        candidates.append((f"combo_manual_{i:02d}", p))

    for i, row in enumerate(good_single[:7], start=1):
        changed = {k: float(row[k]) for k in TUNABLES if abs(float(row[k]) - BASELINE[k]) > 1e-9}
        if changed:
            p = dict(BASELINE)
            p.update(changed)
            candidates.append((f"combo_singlebest_{i:02d}", p))

    dedup: list[tuple[str, dict[str, float]]] = []
    seen: set[str] = set()
    for name, params in candidates:
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((name, params))
    return dedup[:15]


def select_best3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Keep total cases in the same rough scale as the validated baseline rather
    # than imposing an artificial hard 2200 lower bound.
    eligible = [r for r in rows if hard_ok(r) and 1500 <= float(r["total_cases"]) <= 3300]
    eligible.sort(key=lambda r: (float(r["MAE_pp"]), -float(r["hetero"])))
    return eligible[:3]


def validation_decision(baseline_rows: list[dict[str, Any]], candidate_groups: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, float], dict[str, Any]]:
    base_sum = summarize(baseline_rows)
    best_name = "baseline"
    best_params = dict(BASELINE)
    best_info: dict[str, Any] = {"baseline_summary": base_sum, "candidate_summaries": {}}
    base_mae = base_sum["MAE_pp"]["mean"]
    base_female_sd = base_sum["female_known"]["sd"]
    base_bi_sd = base_sum["bi"]["sd"]
    for name, rows in candidate_groups.items():
        s = summarize(rows)
        best_info["candidate_summaries"][name] = s
        if not all(ok(r) for r in rows):
            continue
        if s["female_known"]["mean"] > base_sum["female_known"]["mean"] + 0.002:
            continue
        if s["bi"]["sd"] > base_bi_sd + 0.005 or s["female_known"]["sd"] > base_female_sd + 0.005:
            continue
        if s["MAE_pp"]["mean"] < base_mae - 0.05:
            first = rows[0]
            best_name = name
            best_params = {k: float(first[k]) for k in TUNABLES}
            base_mae = s["MAE_pp"]["mean"]
    return best_name, best_params, best_info


def write_final_outputs(chosen: str, params: dict[str, float], validation_rows: list[dict[str, Any]], info: dict[str, Any], failed: list[dict[str, Any]]) -> None:
    summary = summarize(validation_rows)
    payload = {
        "chosen": chosen,
        "params": params,
        "summary": summary,
        "failed_runs": failed,
        "selection_info": info,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT_DIR / "final_simplified_params.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    final_rows = []
    for metric, values in summary.items():
        final_rows.append({"metric": metric, "mean": values["mean"], "sd": values["sd"]})
    with (OUT_DIR / "final_validation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "sd"])
        writer.writeheader()
        writer.writerows(final_rows)

    lines = [
        "# Simplified calibration final validation",
        "",
        f"Final decision: {'retained current simplified baseline' if chosen == 'baseline' else 'adopted ' + chosen}.",
        "",
        "## Final parameters",
        "",
        "```json",
        json.dumps(params, indent=2, sort_keys=True),
        "```",
        "",
        "## Final validation summary",
        "",
        "| metric | mean | SD |",
        "|---|---:|---:|",
    ]
    for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "MAE_pp"]:
        vals = summary[metric]
        if metric in {"hetero", "bi", "homo", "female_known"}:
            lines.append(f"| {metric} | {100*vals['mean']:.3f}% | {100*vals['sd']:.3f}% |")
        elif metric == "MAE_pp":
            lines.append(f"| {metric} | {vals['mean']:.3f} pp | {vals['sd']:.3f} pp |")
        else:
            lines.append(f"| {metric} | {vals['mean']:.1f} | {vals['sd']:.1f} |")
    lines.extend(
        [
            "",
            f"Number of validation seeds in final comparison: {len([r for r in validation_rows if ok(r)])}.",
            f"Failed runs: {len(failed)}.",
            "",
            "## Selection rationale",
            "",
        ]
    )
    if chosen == "baseline":
        lines.append("No candidate clearly improved mean MAE without risking bisexual or female-known stability, so the current simplified baseline was retained.")
    else:
        lines.append("The selected candidate improved validation performance enough to justify replacing the current simplified baseline.")
    (OUT_DIR / "final_validation_readme.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ensure_dirs()
    original_text = MAIN_PATH.read_text(encoding="utf-8")
    failed: list[dict[str, Any]] = []
    try:
        # Stage 0
        baseline_rows = [run_once("stage0_baseline", "baseline", seed, BASELINE, original_text) for seed in range(100000, 100005)]
        write_rows(OUT_DIR / "baseline_5seed.csv", baseline_rows)
        failed.extend([r for r in baseline_rows if not ok(r)])

        # Stage 1
        single_rows: list[dict[str, Any]] = []
        for param, values in SINGLE_GRID.items():
            for value in values:
                p = dict(BASELINE)
                p[param] = value
                single_rows.append(run_once("stage1_single", f"{param}_{value:g}", 100000, p, original_text))
        write_rows(OUT_DIR / "screening_single_param.csv", single_rows)
        failed.extend([r for r in single_rows if not ok(r)])

        # Stage 2
        combo_rows: list[dict[str, Any]] = []
        for name, params in stage2_candidates(single_rows):
            combo_rows.append(run_once("stage2_combined", name, 100000, params, original_text))
        write_rows(OUT_DIR / "screening_combined.csv", combo_rows)
        failed.extend([r for r in combo_rows if not ok(r)])
        best3 = select_best3(combo_rows)

        # Stage 3
        validation_rows: list[dict[str, Any]] = []
        validation_rows.extend([run_once("stage3_validation", "baseline", seed, BASELINE, original_text) for seed in range(100000, 100005)])
        candidate_groups: dict[str, list[dict[str, Any]]] = {}
        for row in best3:
            name = str(row["candidate"])
            params = {k: float(row[k]) for k in TUNABLES}
            rows = [run_once("stage3_validation", name, seed, params, original_text) for seed in range(100000, 100005)]
            candidate_groups[name] = rows
            validation_rows.extend(rows)
        write_rows(OUT_DIR / "validation_5seed.csv", validation_rows)
        failed.extend([r for r in validation_rows if not ok(r)])

        chosen, chosen_params, info = validation_decision(
            [r for r in validation_rows if r["candidate"] == "baseline"],
            candidate_groups,
        )

        final_validation_rows = [r for r in validation_rows if r["candidate"] == ("baseline" if chosen == "baseline" else chosen)]

        # Stage 4
        if chosen != "baseline":
            stage4_rows: list[dict[str, Any]] = []
            for seed in range(100000, 100020):
                stage4_rows.append(run_once("stage4_validation20", "baseline", seed, BASELINE, original_text))
                stage4_rows.append(run_once("stage4_validation20", chosen, seed, chosen_params, original_text))
            write_rows(OUT_DIR / "validation_20seed.csv", stage4_rows)
            failed.extend([r for r in stage4_rows if not ok(r)])
            base20 = [r for r in stage4_rows if r["candidate"] == "baseline"]
            cand20 = [r for r in stage4_rows if r["candidate"] == chosen]
            base20_sum = summarize(base20)
            cand20_sum = summarize(cand20)
            improves_mae = cand20_sum["MAE_pp"]["mean"] <= base20_sum["MAE_pp"]["mean"] - 0.15
            improves_hetero = cand20_sum["hetero"]["mean"] >= base20_sum["hetero"]["mean"] + 0.002 and cand20_sum["bi"]["mean"] >= 0.117 and cand20_sum["female_known"]["mean"] <= 0.045
            if improves_mae or improves_hetero:
                final_validation_rows = cand20
                info["stage4_baseline_summary"] = base20_sum
                info["stage4_candidate_summary"] = cand20_sum
            else:
                chosen = "baseline"
                chosen_params = dict(BASELINE)
                final_validation_rows = base20
                info["stage4_baseline_summary"] = base20_sum
                info["stage4_candidate_summary"] = cand20_sum

        write_final_outputs(chosen, chosen_params, final_validation_rows, info, failed)
        write_main(chosen_params, seed=100000, make_plots=True)
        return 0
    finally:
        # If an exception occurs before final write, restore original text.
        if not (OUT_DIR / "final_simplified_params.json").exists():
            MAIN_PATH.write_text(original_text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
