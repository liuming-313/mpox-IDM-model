"""CLI calibration workflow for the mpox agent-based model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from calibration_config import (
    BASELINE_PARAMS,
    BEST_PARAMS_JSON,
    BEST_SUMMARY_CSV,
    CALIBRATION_OUTPUT_DIR,
    DEFAULT_RUN_SETTINGS,
    DIAGNOSTIC_PLOT_DIR,
    RESULTS_CSV,
    SEARCH_SPACE_PRIMARY,
    SEARCH_SPACE_SECONDARY,
)
from run_one import run_model


OUTPUT_ROOT = CALIBRATION_OUTPUT_DIR
RESULTS_PATH = RESULTS_CSV
BEST_PARAMS_PATH = BEST_PARAMS_JSON
BEST_SUMMARY_PATH = BEST_SUMMARY_CSV
PLOT_DIR = DIAGNOSTIC_PLOT_DIR

RESULT_FIELDS = [
    "timestamp",
    "stage",
    "candidate_id",
    "run_id",
    "seed",
    "pop_size",
    "status",
    *sorted(set(SEARCH_SPACE_PRIMARY) | set(SEARCH_SPACE_SECONDARY)),
    "total_cases",
    "hetero",
    "bi",
    "homo",
    "female_known",
    "homo_male",
    "bi_male",
    "objective_case_composition_mae",
    "objective_cumulative_case_error",
    "objective_peak_timing_error",
    "objective_late_heterosexual_male_growth_error",
    "score",
    "error",
    "elapsed_s",
    "stdout_log",
    "stderr_log",
    "diagnostic_plot_dir",
]


def configure_output_dir(output_dir: str | None) -> None:
    global OUTPUT_ROOT, RESULTS_PATH, BEST_PARAMS_PATH, BEST_SUMMARY_PATH, PLOT_DIR
    if not output_dir:
        return
    root = Path(output_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    OUTPUT_ROOT = root
    RESULTS_PATH = root / "calibration_results.csv"
    BEST_PARAMS_PATH = root / "best_params.json"
    BEST_SUMMARY_PATH = root / "best_summary.csv"
    PLOT_DIR = root / "diagnostic_plots"


def ensure_outputs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "calibration_logs").mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(path: Path | None = None) -> list[dict[str, str]]:
    path = path or RESULTS_PATH
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_results(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    path = path or RESULTS_PATH
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def candidate_key(params: dict[str, float]) -> str:
    return json.dumps({k: round(float(v), 10) for k, v in sorted(params.items())}, sort_keys=True)


def existing_completed_keys(stage: str, pop_size: int) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in load_results():
        if row.get("stage") != stage or row.get("status") != "ok":
            continue
        if int(float(row.get("pop_size", 0))) != int(pop_size):
            continue
        params = {k: float(row[k]) for k in SEARCH_SPACE_PRIMARY if row.get(k) not in {"", None}}
        for k in SEARCH_SPACE_SECONDARY:
            if row.get(k) not in {"", None}:
                params[k] = float(row[k])
        keys.add((candidate_key(params), int(row["seed"])))
    return keys


def sample_uniform(space: dict[str, list[float]], rng: random.Random) -> dict[str, float]:
    return {key: rng.uniform(float(bounds[0]), float(bounds[1])) for key, bounds in space.items()}


def primary_candidates(n_samples: int, rng: random.Random) -> list[dict[str, float]]:
    candidates = [dict(BASELINE_PARAMS)]
    while len(candidates) < n_samples:
        candidates.append(sample_uniform(SEARCH_SPACE_PRIMARY, rng))
    return candidates[:n_samples]


def mutate_around(params: dict[str, float], space: dict[str, list[float]], rng: random.Random, scale: float = 0.15) -> dict[str, float]:
    out = dict(params)
    for key, bounds in space.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        width = (hi - lo) * scale
        out[key] = min(hi, max(lo, rng.gauss(float(params[key]), width)))
    return out


def aggregate_candidate_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("score") in {"", None}:
            continue
        params = {k: float(row[k]) for k in SEARCH_SPACE_PRIMARY if row.get(k) not in {"", None}}
        for k in SEARCH_SPACE_SECONDARY:
            if row.get(k) not in {"", None}:
                params[k] = float(row[k])
        groups.setdefault(candidate_key(params), []).append(row)
    aggs = []
    for key, vals in groups.items():
        scores = [float(v["score"]) for v in vals]
        maes = [float(v["objective_case_composition_mae"]) for v in vals]
        params = json.loads(key)
        aggs.append(
            {
                "params": params,
                "n": len(vals),
                "mean_score": statistics.mean(scores),
                "sd_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "mean_mae": statistics.mean(maes),
                "sd_mae": statistics.stdev(maes) if len(maes) > 1 else 0.0,
            }
        )
    aggs.sort(key=lambda x: (x["mean_score"], x["mean_mae"]))
    return aggs


def write_best_outputs(best: dict[str, Any], rows: list[dict[str, str]]) -> None:
    params = best["params"]
    matching = []
    key = candidate_key(params)
    for row in rows:
        if row.get("status") != "ok":
            continue
        row_params = {k: float(row[k]) for k in SEARCH_SPACE_PRIMARY if row.get(k) not in {"", None}}
        for k in SEARCH_SPACE_SECONDARY:
            if row.get(k) not in {"", None}:
                row_params[k] = float(row[k])
        if candidate_key(row_params) == key:
            matching.append(row)

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "summary": best,
    }
    BEST_PARAMS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    metric_fields = ["total_cases", "hetero", "bi", "homo", "female_known", "homo_male", "bi_male", "objective_case_composition_mae", "score"]
    with BEST_SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "sd", "n"])
        writer.writeheader()
        for metric in metric_fields:
            vals = [float(r[metric]) for r in matching if r.get(metric) not in {"", None}]
            if vals:
                writer.writerow(
                    {
                        "metric": metric,
                        "mean": statistics.mean(vals),
                        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals),
                    }
                )


def run_candidates(stage: str, candidates: list[dict[str, float]], seeds: list[int], pop_size: int, plot_best: bool = False) -> list[dict[str, Any]]:
    ensure_outputs()
    done = existing_completed_keys(stage, pop_size)
    new_rows: list[dict[str, Any]] = []
    log_dir = OUTPUT_ROOT / "calibration_logs"
    for idx, params in enumerate(candidates):
        cid = f"{stage}_{idx:05d}"
        key = candidate_key(params)
        for seed in seeds:
            if (key, seed) in done:
                print(f"[skip] {cid} seed={seed}", flush=True)
                continue
            rid = f"{cid}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[run] {cid} seed={seed}", flush=True)
            result = run_model(params, seed=seed, pop_size=pop_size, run_id=rid, make_plots=False, log_dir=log_dir)
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": stage,
                "candidate_id": cid,
                **params,
                **result,
            }
            new_rows.append(row)
            append_results([row])
            print(f"[done] {cid} seed={seed} status={row['status']} score={row.get('score', '')}", flush=True)
    if plot_best:
        rows = load_results()
        aggs = aggregate_candidate_rows(rows)
        if aggs:
            best_params = aggs[0]["params"]
            rid = f"diagnostic_best_seed{seeds[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"[plot] best diagnostic run {rid}", flush=True)
            plot_result = run_model(
                best_params,
                seed=seeds[0],
                pop_size=pop_size,
                run_id=rid,
                make_plots=True,
                log_dir=log_dir,
                copy_plots=True,
                diagnostic_plot_dir=PLOT_DIR,
            )
            append_results([
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "stage": f"{stage}_diagnostic_plot",
                    "candidate_id": "best",
                    **best_params,
                    **plot_result,
                }
            ])
    return new_rows


def select_refine_centers(min_seeds: int = 1, top_k: int = 5) -> list[dict[str, float]]:
    aggs = [a for a in aggregate_candidate_rows(load_results()) if a["n"] >= min_seeds]
    return [a["params"] for a in aggs[:top_k]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproducible mpox ABM calibration workflow")
    parser.add_argument("--stage", choices=["primary", "refine"], default="primary")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_RUN_SETTINGS["n_samples"])
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_RUN_SETTINGS["n_seeds"])
    parser.add_argument("--pop-size", type=int, default=DEFAULT_RUN_SETTINGS["pop_size"])
    parser.add_argument("--seed-start", type=int, default=DEFAULT_RUN_SETTINGS["base_seed"])
    parser.add_argument("--random-seed", type=int, default=12345)
    parser.add_argument("--include-secondary", action="store_true", help="Enable secondary parameters in refine stage only")
    parser.add_argument("--validate-best", action="store_true")
    parser.add_argument("--diagnostic-plots", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Directory for calibration_results.csv, best files, logs, and plots")
    args = parser.parse_args()

    configure_output_dir(args.output_dir)
    rng = random.Random(args.random_seed)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    ensure_outputs()

    if args.validate_best:
        rows = load_results()
        aggs = aggregate_candidate_rows(rows)
        if BEST_PARAMS_PATH.exists():
            params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))["params"]
        elif aggs:
            params = aggs[0]["params"]
        else:
            params = dict(BASELINE_PARAMS)
        run_candidates("validate_best", [params], seeds, args.pop_size, plot_best=True)
    elif args.stage == "primary":
        candidates = primary_candidates(args.n_samples, rng)
        run_candidates("primary", candidates, seeds, args.pop_size, plot_best=args.diagnostic_plots)
    else:
        centers = select_refine_centers(min_seeds=max(1, min(args.n_seeds, 3)), top_k=5)
        if not centers:
            centers = [dict(BASELINE_PARAMS)]
        space = dict(SEARCH_SPACE_PRIMARY)
        if args.include_secondary:
            space.update(SEARCH_SPACE_SECONDARY)
            centers = [{**c, **{k: statistics.mean(v) for k, v in SEARCH_SPACE_SECONDARY.items()}} for c in centers]
        candidates = []
        while len(candidates) < args.n_samples:
            center = centers[len(candidates) % len(centers)]
            candidates.append(mutate_around(center, space, rng, scale=0.10))
        run_candidates("refine", candidates, seeds, args.pop_size, plot_best=args.diagnostic_plots)

    all_rows = load_results()
    aggs = aggregate_candidate_rows(all_rows)
    if aggs:
        write_best_outputs(aggs[0], all_rows)
        print("[best]", json.dumps(aggs[0], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
