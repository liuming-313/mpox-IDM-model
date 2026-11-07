"""Small screening search over direct sexual transmission-risk parameters.

This runner keeps the current v3 epidemic-size reconciled baseline fixed:
community_bridge_scale=1.0, importation schedule daily_n=2,4,9,4,0, policy,
testing/monitoring, disease natural history, and nonsexual/contact parameters.

It temporarily patches only the four direct sexual transmission-risk parameters,
base_run_seed, POPULATION_SIZE, and MAKE_PLOTS in main.py for each subprocess
run, then restores main.py afterward.
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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_direct_transmission_risk_search"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100005))
OBSERVED_TOTAL_REPORTED_CASES = 2280.0
OBSERVED_KNOWN_ORIENTATION_DENOMINATOR = 2012.0

TARGETS = {
    "hetero": (125 + 74) / 2012,
    "bi": 253 / 2012,
    "homo": 1560 / 2012,
    "female_known": 74 / 2012,
    "homo_male": 1560 / 2012,
    "bi_male": 253 / 2012,
}

BASELINE_DIRECT_RISKS = {
    "sexual_anal_receptive": 0.2800,
    "sexual_anal_insertive": 0.0525,
    "sexual_vaginal": 0.0350,
    "sexual_oral": 0.0175,
}

CANDIDATES = [
    {
        "candidate_id": "A_baseline_direct_risks",
        "description": "Baseline direct risks.",
        "params": {
            "sexual_anal_receptive": 0.2800,
            "sexual_anal_insertive": 0.0525,
            "sexual_vaginal": 0.0350,
            "sexual_oral": 0.0175,
        },
    },
    {
        "candidate_id": "B_lower_vaginal",
        "description": "Lower mixed-sex vaginal risk.",
        "params": {
            "sexual_anal_receptive": 0.2800,
            "sexual_anal_insertive": 0.0525,
            "sexual_vaginal": 0.0250,
            "sexual_oral": 0.0175,
        },
    },
    {
        "candidate_id": "C_lower_vaginal_higher_insertive",
        "description": "Lower vaginal plus slightly higher insertive anal.",
        "params": {
            "sexual_anal_receptive": 0.2800,
            "sexual_anal_insertive": 0.0600,
            "sexual_vaginal": 0.0250,
            "sexual_oral": 0.0175,
        },
    },
    {
        "candidate_id": "D_preserve_msm_lower_vaginal_oral",
        "description": "Preserve MSM-core, lower vaginal and oral.",
        "params": {
            "sexual_anal_receptive": 0.2800,
            "sexual_anal_insertive": 0.0525,
            "sexual_vaginal": 0.0200,
            "sexual_oral": 0.0125,
        },
    },
    {
        "candidate_id": "E_lower_receptive_lower_vaginal",
        "description": "Slightly lower receptive anal and lower vaginal.",
        "params": {
            "sexual_anal_receptive": 0.2600,
            "sexual_anal_insertive": 0.0525,
            "sexual_vaginal": 0.0250,
            "sexual_oral": 0.0175,
        },
    },
    {
        "candidate_id": "F_higher_receptive_lower_vaginal",
        "description": "Slightly higher receptive anal and lower vaginal.",
        "params": {
            "sexual_anal_receptive": 0.3000,
            "sexual_anal_insertive": 0.0525,
            "sexual_vaginal": 0.0250,
            "sexual_oral": 0.0175,
        },
    },
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
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
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
    "distance_to_observed_total_cases_2280",
    "distance_to_known_orientation_denominator_2012",
    "composition_acceptance_pass",
    "female_known_delta_vs_baseline",
    "female_known_improved_vs_baseline",
    "total_cases_delta_vs_baseline",
    "risk_change_l1_vs_baseline",
    "selected_for_20seed_validation",
]


def _float_pattern() -> str:
    return r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _patch_main_text(original: str, seed: int, params: dict[str, float]) -> str:
    text = original
    for key, value in params.items():
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


def _run_one(candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    params = candidate["params"]
    run_id = f"{candidate['candidate_id']}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    stdout_log = LOG_DIR / f"{run_id}.stdout.log"
    stderr_log = LOG_DIR / f"{run_id}.stderr.log"
    original = MAIN_PATH.read_text(encoding="utf-8")
    started = datetime.now()
    row: dict[str, Any] = {
        "timestamp": started.isoformat(timespec="seconds"),
        "candidate_id": candidate["candidate_id"],
        "description": candidate["description"],
        "seed": seed,
        "pop_size": POP_SIZE,
        "run_id": run_id,
        "status": "failed",
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


def _completed(path: Path) -> set[tuple[str, int]]:
    done = set()
    for row in _read_rows(path):
        if row.get("status") == "ok":
            done.add((row.get("candidate_id", ""), int(float(row.get("seed", -1)))))
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


def _composition_pass(summary: dict[str, Any]) -> bool:
    return (
        int(summary["successful_runs"]) == len(SEEDS)
        and _finite(summary.get("MAE_pp_mean")) <= 1.5
        and _finite(summary.get("hetero_mean")) >= 0.080
        and 0.115 <= _finite(summary.get("bi_mean")) <= 0.138
        and _finite(summary.get("homo_mean")) <= 0.805
        and 0.028 <= _finite(summary.get("female_known_mean")) <= 0.045
    )


def _risk_change_l1(params: dict[str, float]) -> float:
    return sum(abs(float(params[k]) - float(BASELINE_DIRECT_RISKS[k])) for k in BASELINE_DIRECT_RISKS)


def _summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        cid = candidate["candidate_id"]
        selected = [row for row in rows if row.get("candidate_id") == cid]
        ok = [row for row in selected if row.get("status") == "ok"]
        failed = [row for row in selected if row.get("status") != "ok"]
        summary: dict[str, Any] = {
            "candidate_id": cid,
            "description": candidate["description"],
            "successful_runs": len(ok),
            "failed_runs": len(failed),
            **candidate["params"],
            "risk_change_l1_vs_baseline": _risk_change_l1(candidate["params"]),
            "selected_for_20seed_validation": False,
        }
        for metric in ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male", "MAE_pp"]:
            values = [_finite(row.get(metric)) for row in ok]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_sd"] = _sd(values)
        total_mean = _finite(summary.get("total_cases_mean"))
        summary["distance_to_observed_total_cases_2280"] = abs(total_mean - OBSERVED_TOTAL_REPORTED_CASES)
        summary["distance_to_known_orientation_denominator_2012"] = abs(total_mean - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR)
        summary["composition_acceptance_pass"] = _composition_pass(summary)
        summaries.append(summary)

    baseline = next((s for s in summaries if s["candidate_id"] == "A_baseline_direct_risks"), None)
    if baseline:
        base_female = _finite(baseline["female_known_mean"])
        base_total_dist = _finite(baseline["distance_to_observed_total_cases_2280"])
        for summary in summaries:
            female = _finite(summary["female_known_mean"])
            summary["female_known_delta_vs_baseline"] = female - base_female
            summary["female_known_improved_vs_baseline"] = abs(female - TARGETS["female_known"]) < abs(base_female - TARGETS["female_known"])
            summary["total_cases_delta_vs_baseline"] = _finite(summary["total_cases_mean"]) - _finite(baseline["total_cases_mean"])
            # Selection guard: do not trade away epidemic-size fit badly.
            summary["_keeps_total_reasonable"] = _finite(summary["distance_to_observed_total_cases_2280"]) <= (base_total_dist + 250.0)

    eligible = [
        s for s in summaries
        if s["candidate_id"] != "A_baseline_direct_risks"
        and s.get("composition_acceptance_pass")
        and s.get("female_known_improved_vs_baseline")
        and s.get("_keeps_total_reasonable")
    ]
    eligible.sort(
        key=lambda s: (
            abs(_finite(s["female_known_mean"]) - TARGETS["female_known"]),
            _finite(s["MAE_pp_mean"]),
            _finite(s["distance_to_observed_total_cases_2280"]),
            _finite(s["risk_change_l1_vs_baseline"]),
        )
    )
    for summary in eligible[:2]:
        summary["selected_for_20seed_validation"] = True
    for summary in summaries:
        summary.pop("_keeps_total_reasonable", None)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_candidate_params(summaries: list[dict[str, Any]]) -> None:
    selected = [s for s in summaries if s.get("selected_for_20seed_validation")]
    payload = {
        "targets": TARGETS,
        "observed_total_reported_cases": OBSERVED_TOTAL_REPORTED_CASES,
        "observed_known_orientation_denominator": OBSERVED_KNOWN_ORIENTATION_DENOMINATOR,
        "screening_seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "pop_size": POP_SIZE,
        "selected_for_20seed_validation": [
            {
                "candidate_id": s["candidate_id"],
                "description": s["description"],
                "params": {k: s[k] for k in BASELINE_DIRECT_RISKS},
                "summary": {k: s[k] for k in SUMMARY_FIELDS if k in s and k not in BASELINE_DIRECT_RISKS},
            }
            for s in selected
        ],
    }
    (OUTPUT_DIR / "candidate_params.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_readme(summaries: list[dict[str, Any]]) -> None:
    selected = [s for s in summaries if s.get("selected_for_20seed_validation")]
    lines = [
        "# V3 direct transmission-risk search",
        "",
        "Small mechanism-focused screening over direct per-act sexual transmission-risk parameters.",
        "",
        "Frozen during this search: community_bridge_scale=1.0, importation schedule daily_n=2/4/9/4/0, policy timing, testing, monitoring, disease natural history, social_p_eff, close_touch, party_touch_frac, and all direct female-ratio knobs.",
        "",
        f"Population size: {POP_SIZE:,}",
        f"Seeds per candidate: {SEEDS[0]}-{SEEDS[-1]}",
        "",
        "## Screening summary",
        "",
        "| Candidate | total_cases | hetero | bi | homo | female_known | homo_male | bi_male | MAE_pp | composition pass | female improved | selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['candidate_id']} | {s['total_cases_mean']:.2f} ({s['total_cases_sd']:.2f}) | "
            f"{s['hetero_mean']:.4f} | {s['bi_mean']:.4f} | {s['homo_mean']:.4f} | "
            f"{s['female_known_mean']:.4f} | {s['homo_male_mean']:.4f} | {s['bi_male_mean']:.4f} | "
            f"{s['MAE_pp_mean']:.3f} | {'yes' if s['composition_acceptance_pass'] else 'no'} | "
            f"{'yes' if s.get('female_known_improved_vs_baseline') else 'no'} | "
            f"{'yes' if s.get('selected_for_20seed_validation') else 'no'} |"
        )
    lines.extend(["", "## Recommendation", ""])
    if selected:
        ids = ", ".join(s["candidate_id"] for s in selected)
        lines.append(f"Selected for possible 20-seed validation: {ids}.")
    else:
        lines.append("No candidate was selected for 20-seed validation under the predefined guards.")
    lines.append("")
    lines.append("Do not run final 100-seed validation until a 20-seed validation passes.")
    (OUTPUT_DIR / "direct_transmission_risk_search_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    results_path = OUTPUT_DIR / "direct_transmission_risk_search_results.csv"
    completed = _completed(results_path)
    for candidate in CANDIDATES:
        for seed in SEEDS:
            key = (candidate["candidate_id"], seed)
            if key in completed:
                print(f"[skip] {candidate['candidate_id']} seed={seed}", flush=True)
                continue
            print(f"[run] {candidate['candidate_id']} seed={seed}", flush=True)
            row = _run_one(candidate, seed)
            _append_row(results_path, row)
            print(
                f"[done] {candidate['candidate_id']} seed={seed} status={row['status']} "
                f"total_cases={row.get('total_cases')} MAE_pp={row.get('MAE_pp')}",
                flush=True,
            )

    summaries = _summaries(_read_rows(results_path))
    _write_csv(OUTPUT_DIR / "direct_transmission_risk_search_summary.csv", summaries, SUMMARY_FIELDS)
    _write_candidate_params(summaries)
    _write_readme(summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
