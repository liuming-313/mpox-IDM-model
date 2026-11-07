"""Focused D/F-combination social recovery search for v3.

This runner keeps population.py unchanged. For each run it temporarily patches
only the allowed main.py parameters:

  - sexual_anal_insertive
  - social_contact_scale
  - social_same_sex_assort
  - community_bridge_scale
  - base_run_seed, POPULATION_SIZE, MAKE_PLOTS

Fixed direct risks during this search:
  - sexual_anal_receptive = 0.2800
  - sexual_vaginal = 0.0350
  - sexual_oral = 0.0175

main.py is restored after every subprocess run.
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


OUTPUT_DIR = PROJECT_DIR / "outputs" / "v3_df_combined_social_recovery_search"
LOG_DIR = OUTPUT_DIR / "logs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
SOURCE_DIAGNOSTICS = PROJECT_DIR / "outputs" / "10M_homo_male_mcp" / "calibration_diagnostics.csv"

POP_SIZE = 10_000_000
SEEDS = list(range(100000, 100005))
OBSERVED_TOTAL_REPORTED_CASES = 2280.0
OBSERVED_KNOWN_ORIENTATION_DENOMINATOR = 2012.0

TARGETS = {
    "hetero": (125 + 74) / 2012,
    "male_heterosexual": 125 / 2012,
    "female_known": 74 / 2012,
    "bi": 253 / 2012,
    "homo": 1560 / 2012,
    "homo_male": 1560 / 2012,
    "bi_male": 253 / 2012,
}

PREVIOUS_D = {
    "candidate_id": "previous_D_higher_insertive_social_bridge",
    "total_cases_mean": 2302.4,
    "hetero_mean": 0.07227325473244746,
    "implied_male_heterosexual_mean": 0.03825979837039798,
    "female_known_mean": 0.03401345636204948,
    "bi_mean": 0.1263210561566806,
    "homo_mean": 0.801405689110872,
    "MAE_pp_mean": 1.5602371970329174,
}

PREVIOUS_F = {
    "candidate_id": "previous_F_fixed_risks_max_social_pathway",
    "total_cases_mean": 1902.8,
    "hetero_mean": 0.08232918183389275,
    "implied_male_heterosexual_mean": 0.05290939481851884,
    "female_known_mean": 0.029419787015373906,
    "bi_mean": 0.11369934940843762,
    "homo_mean": 0.8039714687576696,
    "MAE_pp_mean": 1.8227931335373835,
}

PATCHABLE_PARAMS = [
    "sexual_anal_insertive",
    "social_contact_scale",
    "social_same_sex_assort",
    "community_bridge_scale",
]

FIXED_PARAMS = {
    "sexual_anal_receptive": 0.2800,
    "sexual_vaginal": 0.0350,
    "sexual_oral": 0.0175,
}

CANDIDATES = [
    {
        "candidate_id": "A_df_midpoint",
        "description": "D/F midpoint.",
        "params": {
            "sexual_anal_insertive": 0.0600,
            "social_contact_scale": 9.0,
            "social_same_sex_assort": 0.88,
            "community_bridge_scale": 1.2,
        },
    },
    {
        "candidate_id": "B_stronger_male_social_bridge",
        "description": "Stronger male social bridge.",
        "params": {
            "sexual_anal_insertive": 0.0600,
            "social_contact_scale": 10.0,
            "social_same_sex_assort": 0.90,
            "community_bridge_scale": 1.2,
        },
    },
    {
        "candidate_id": "C_stronger_community_recovery",
        "description": "Stronger community recovery.",
        "params": {
            "sexual_anal_insertive": 0.0600,
            "social_contact_scale": 9.0,
            "social_same_sex_assort": 0.90,
            "community_bridge_scale": 1.5,
        },
    },
    {
        "candidate_id": "D_total_preservation_plus_recovery",
        "description": "D-like total preservation plus recovery.",
        "params": {
            "sexual_anal_insertive": 0.0600,
            "social_contact_scale": 8.0,
            "social_same_sex_assort": 0.85,
            "community_bridge_scale": 1.5,
        },
    },
    {
        "candidate_id": "E_f_like_male_hetero_plus_insertive",
        "description": "F-like male-hetero improvement plus insertive risk.",
        "params": {
            "sexual_anal_insertive": 0.0600,
            "social_contact_scale": 10.0,
            "social_same_sex_assort": 0.90,
            "community_bridge_scale": 1.5,
        },
    },
    {
        "candidate_id": "F_upper_social_moderate_recovery",
        "description": "Upper social bridge but moderate recovery.",
        "params": {
            "sexual_anal_insertive": 0.0600,
            "social_contact_scale": 11.0,
            "social_same_sex_assort": 0.92,
            "community_bridge_scale": 1.2,
        },
    },
    {
        "candidate_id": "G_higher_insertive_moderate_bridge",
        "description": "Higher insertive risk with moderate bridge.",
        "params": {
            "sexual_anal_insertive": 0.0650,
            "social_contact_scale": 9.0,
            "social_same_sex_assort": 0.88,
            "community_bridge_scale": 1.2,
        },
    },
    {
        "candidate_id": "H_higher_insertive_stronger_bridge",
        "description": "Higher insertive risk with stronger bridge.",
        "params": {
            "sexual_anal_insertive": 0.0650,
            "social_contact_scale": 10.0,
            "social_same_sex_assort": 0.90,
            "community_bridge_scale": 1.5,
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
    "distance_to_observed_total_cases_2280",
    "distance_to_known_orientation_denominator_2012",
    "composition_acceptance_pass",
    "hetero_improved_vs_previous_D",
    "male_heterosexual_improved_vs_previous_D",
    "total_cases_improved_vs_previous_D",
    "hetero_improved_vs_previous_F",
    "male_heterosexual_improved_vs_previous_F",
    "total_cases_improved_vs_previous_F",
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


def _dist(value: float, target: float) -> float:
    return abs(value - target)


def _summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
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
            **FIXED_PARAMS,
            **candidate["params"],
            "selected_for_20seed_validation": False,
        }
        for metric in metrics:
            values = [_finite(row.get(metric)) for row in ok]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_sd"] = _sd(values)
        total_mean = _finite(summary.get("total_cases_mean"))
        summary["distance_to_observed_total_cases_2280"] = abs(total_mean - OBSERVED_TOTAL_REPORTED_CASES)
        summary["distance_to_known_orientation_denominator_2012"] = abs(total_mean - OBSERVED_KNOWN_ORIENTATION_DENOMINATOR)
        summary["composition_acceptance_pass"] = _composition_pass(summary)

        summary["hetero_improved_vs_previous_D"] = (
            _dist(_finite(summary["hetero_mean"]), TARGETS["hetero"])
            < _dist(PREVIOUS_D["hetero_mean"], TARGETS["hetero"])
        )
        summary["male_heterosexual_improved_vs_previous_D"] = (
            _dist(_finite(summary["implied_male_heterosexual_mean"]), TARGETS["male_heterosexual"])
            < _dist(PREVIOUS_D["implied_male_heterosexual_mean"], TARGETS["male_heterosexual"])
        )
        summary["total_cases_improved_vs_previous_D"] = (
            _dist(total_mean, OBSERVED_TOTAL_REPORTED_CASES)
            < _dist(PREVIOUS_D["total_cases_mean"], OBSERVED_TOTAL_REPORTED_CASES)
        )
        summary["hetero_improved_vs_previous_F"] = (
            _dist(_finite(summary["hetero_mean"]), TARGETS["hetero"])
            < _dist(PREVIOUS_F["hetero_mean"], TARGETS["hetero"])
        )
        summary["male_heterosexual_improved_vs_previous_F"] = (
            _dist(_finite(summary["implied_male_heterosexual_mean"]), TARGETS["male_heterosexual"])
            < _dist(PREVIOUS_F["implied_male_heterosexual_mean"], TARGETS["male_heterosexual"])
        )
        summary["total_cases_improved_vs_previous_F"] = (
            _dist(total_mean, OBSERVED_TOTAL_REPORTED_CASES)
            < _dist(PREVIOUS_F["total_cases_mean"], OBSERVED_TOTAL_REPORTED_CASES)
        )
        summaries.append(summary)

    eligible = [
        s for s in summaries
        if s.get("composition_acceptance_pass")
        and _finite(s["distance_to_observed_total_cases_2280"]) <= 480.0
        and _finite(s["implied_male_heterosexual_mean"]) >= PREVIOUS_F["implied_male_heterosexual_mean"]
        and _finite(s["hetero_mean"]) >= PREVIOUS_F["hetero_mean"]
    ]
    eligible.sort(
        key=lambda s: (
            _dist(_finite(s["hetero_mean"]), TARGETS["hetero"]),
            _dist(_finite(s["implied_male_heterosexual_mean"]), TARGETS["male_heterosexual"]),
            _dist(_finite(s["female_known_mean"]), TARGETS["female_known"]),
            _finite(s["MAE_pp_mean"]),
            _finite(s["distance_to_observed_total_cases_2280"]),
        )
    )
    for summary in eligible[:2]:
        summary["selected_for_20seed_validation"] = True
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
        "previous_D": PREVIOUS_D,
        "previous_F": PREVIOUS_F,
        "fixed_params": FIXED_PARAMS,
        "screening_seeds": f"{SEEDS[0]}-{SEEDS[-1]}",
        "pop_size": POP_SIZE,
        "selected_for_20seed_validation": [
            {
                "candidate_id": s["candidate_id"],
                "description": s["description"],
                "params": {k: s[k] for k in [*FIXED_PARAMS, *PATCHABLE_PARAMS]},
                "summary": {k: s[k] for k in SUMMARY_FIELDS if k in s and k not in [*FIXED_PARAMS, *PATCHABLE_PARAMS]},
            }
            for s in selected
        ],
    }
    (OUTPUT_DIR / "candidate_params.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_readme(summaries: list[dict[str, Any]]) -> None:
    selected = [s for s in summaries if s.get("selected_for_20seed_validation")]
    lines = [
        "# V3 D/F combined social recovery search",
        "",
        "Focused screen combining D-like insertive risk with F-like social bridging and community_bridge_scale.",
        "",
        "population.py was not modified. community_bridge_scale was already implemented as a social/community contact-rate recovery multiplier.",
        "",
        "Frozen: importation schedule daily_n=2/4/9/4/0, disease natural history, testing, monitoring, vaccination/susceptibility, policy timing, party_touch_frac, social_p_eff, close_touch, direct female-ratio knobs, party_hetero_leak, social_male_bias, party_mix_female_mult, and late_social_ramp.",
        "",
        f"Population size: {POP_SIZE:,}",
        f"Seeds per candidate: {SEEDS[0]}-{SEEDS[-1]}",
        "",
        "## Screening summary",
        "",
        "| Candidate | total_cases | hetero | male_hetero | female_known | bi | homo | MAE_pp | pass | selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['candidate_id']} | {s['total_cases_mean']:.2f} ({s['total_cases_sd']:.2f}) | "
            f"{s['hetero_mean']:.4f} | {s['implied_male_heterosexual_mean']:.4f} | "
            f"{s['female_known_mean']:.4f} | {s['bi_mean']:.4f} | {s['homo_mean']:.4f} | "
            f"{s['MAE_pp_mean']:.3f} | {'yes' if s['composition_acceptance_pass'] else 'no'} | "
            f"{'yes' if s.get('selected_for_20seed_validation') else 'no'} |"
        )
    lines.extend(["", "## Recommendation", ""])
    if selected:
        ids = ", ".join(s["candidate_id"] for s in selected)
        lines.append(f"Selected for possible 20-seed validation: {ids}.")
    else:
        lines.append("No candidate was selected for 20-seed validation under the predefined guards.")
    lines.append("")
    lines.append("Do not run final 100-seed validation.")
    (OUTPUT_DIR / "df_combined_search_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    results_path = OUTPUT_DIR / "df_combined_search_results.csv"
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
                f"total_cases={row.get('total_cases')} hetero={row.get('hetero')} "
                f"male_hetero={row.get('implied_male_heterosexual')} MAE_pp={row.get('MAE_pp')}",
                flush=True,
            )

    summaries = _summaries(_read_rows(results_path))
    _write_csv(OUTPUT_DIR / "df_combined_search_summary.csv", summaries, SUMMARY_FIELDS)
    _write_candidate_params(summaries)
    _write_readme(summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
