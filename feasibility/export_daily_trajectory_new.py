"""Export daily trajectory for the CURRENT frozen baseline (post social-ramp removal).

Unlike the older v3_final_E_daily_trajectory_validation runner (which text-patched
main.py with a stale FINAL_E_PARAMS dict), this script:
  1. imports main.py's `params` dict directly, so it always reflects the current
     frozen parameter values (bi=1.0, mix direction-split, social ramp removed, etc.);
  2. runs N seeds by constructing Population objects directly;
  3. exports the same per-day trajectory fields used by the manuscript figures.

This is a validation/export runner, NOT recalibration. population.py is not modified.

Usage:
  python feasibility/export_daily_trajectory_new.py --seeds 300,311,322 --pop-size 10000000
  python feasibility/export_daily_trajectory_new.py --seeds 300-399 --pop-size 10000000 --outdir outputs/v4_daily_trajectory
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import main as _main  # noqa: E402  (imports params, CALIBRATION_TARGETS, Population via main)
from population import Population  # noqa: E402

START_DATE = date(2022, 5, 1)

DAILY_FIELDS = [
    "seed", "day", "date",
    "daily_detected", "cumulative_detected",
    "daily_episode_cases", "cumulative_episode_cases",
    "daily_episode_male", "daily_episode_female",
    "cumulative_episode_male", "cumulative_episode_female",
    "daily_episode_hetero", "daily_episode_bi", "daily_episode_homo",
    "cumulative_episode_hetero", "cumulative_episode_bi", "cumulative_episode_homo",
    "daily_episode_female_hetero", "daily_episode_male_hetero", "daily_episode_male_homo",
    "daily_episode_female_homo", "daily_episode_male_bi", "daily_episode_female_bi",
    "cumulative_episode_female_hetero", "cumulative_episode_male_hetero", "cumulative_episode_male_homo",
    "cumulative_episode_female_homo", "cumulative_episode_male_bi", "cumulative_episode_female_bi",
    "cum_ratio_hetero", "cum_ratio_male_heterosexual", "cum_ratio_female_known",
    "cum_ratio_bi", "cum_ratio_homo", "cum_ratio_homo_male", "cum_ratio_bi_male",
]
AGGREGATE_METRICS = [f for f in DAILY_FIELDS if f not in {"seed", "day", "date"}]


def _parse_seed_spec(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s, e = part.split("-", 1)
            seeds.extend(range(int(s), int(e) + 1))
        else:
            seeds.append(int(part))
    return sorted(dict.fromkeys(seeds))


def _safe_div(num, den):
    return float(num) / float(den) if float(den) > 0 else 0.0


def run_seed(seed: int, pop_size: int, sim_days: int) -> dict:
    """Run one seed and return the per-day trajectory rows + summary."""
    params = dict(_main.params)  # copy the frozen param dict
    params["n_days"] = sim_days + 1
    params["verbose"] = False

    policy = [
        dict(name="phase1_growth", start=0, end=91,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95, social_mult=0.90,
             test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
             test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
        dict(name="phase2_control", start=92, end=170,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
        dict(name="phase3_postpeak", start=171, end=sim_days,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]

    pop = Population(n=pop_size, params=params, seed=seed, cache_label=f"traj_new_{seed}",
                     policy=policy, importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(_main.INITIAL_INFECTIONS, mask=mask)
    pop.run_simulation(sim_days)

    T = int(min(len(getattr(pop, "daily_new_diagnoses", [])),
                len(getattr(pop, "daily_episode_cases", [])),
                getattr(pop, "n_days", sim_days + 1)))
    dx = np.asarray(pop.daily_new_diagnoses[:T], dtype=int)
    ep = np.asarray(pop.daily_episode_cases[:T], dtype=int)
    sex = np.asarray(pop.daily_episode_cases_by_sex[:T, :], dtype=int)
    ori = np.asarray(pop.daily_episode_cases_by_ori[:T, :], dtype=int)
    sexcat = np.asarray(pop.daily_episode_cases_by_sexcat[:T, :], dtype=int)
    cdx = np.cumsum(dx)
    cep = np.cumsum(ep)
    csex = np.cumsum(sex, axis=0)
    cori = np.cumsum(ori, axis=0)
    csexcat = np.cumsum(sexcat, axis=0)

    rows = []
    for d in range(T):
        known = max(float(cep[d]), 0.0)
        female_known = float(csexcat[d, 0] + csexcat[d, 3] + csexcat[d, 5])
        rows.append({
            "seed": int(seed),
            "day": int(d),
            "date": (START_DATE + timedelta(days=int(d))).isoformat(),
            "daily_detected": int(dx[d]),
            "cumulative_detected": int(cdx[d]),
            "daily_episode_cases": int(ep[d]),
            "cumulative_episode_cases": int(cep[d]),
            "daily_episode_male": int(sex[d, 0]),
            "daily_episode_female": int(sex[d, 1]),
            "cumulative_episode_male": int(csex[d, 0]),
            "cumulative_episode_female": int(csex[d, 1]),
            "daily_episode_hetero": int(ori[d, 0]),
            "daily_episode_bi": int(ori[d, 1]),
            "daily_episode_homo": int(ori[d, 2]),
            "cumulative_episode_hetero": int(cori[d, 0]),
            "cumulative_episode_bi": int(cori[d, 1]),
            "cumulative_episode_homo": int(cori[d, 2]),
            "daily_episode_female_hetero": int(sexcat[d, 0]),
            "daily_episode_male_hetero": int(sexcat[d, 1]),
            "daily_episode_male_homo": int(sexcat[d, 2]),
            "daily_episode_female_homo": int(sexcat[d, 3]),
            "daily_episode_male_bi": int(sexcat[d, 4]),
            "daily_episode_female_bi": int(sexcat[d, 5]),
            "cumulative_episode_female_hetero": int(csexcat[d, 0]),
            "cumulative_episode_male_hetero": int(csexcat[d, 1]),
            "cumulative_episode_male_homo": int(csexcat[d, 2]),
            "cumulative_episode_female_homo": int(csexcat[d, 3]),
            "cumulative_episode_male_bi": int(csexcat[d, 4]),
            "cumulative_episode_female_bi": int(csexcat[d, 5]),
            "cum_ratio_hetero": _safe_div(cori[d, 0], known),
            "cum_ratio_male_heterosexual": _safe_div(csexcat[d, 1], known),
            "cum_ratio_female_known": _safe_div(female_known, known),
            "cum_ratio_bi": _safe_div(cori[d, 1], known),
            "cum_ratio_homo": _safe_div(cori[d, 2], known),
            "cum_ratio_homo_male": _safe_div(csexcat[d, 2], known),
            "cum_ratio_bi_male": _safe_div(csexcat[d, 4], known),
        })
    s = pop.summary()
    return rows, s


def _write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _summarize_daily(long_path: Path, out_path: Path):
    with long_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_day = {}
    for r in rows:
        by_day.setdefault(int(r["day"]), []).append(r)
    fields = ["day", "date", "n_seeds"]
    for m in AGGREGATE_METRICS:
        fields += [f"{m}_mean", f"{m}_sd", f"{m}_p025", f"{m}_p975"]
    out_rows = []
    for day in sorted(by_day):
        grp = by_day[day]
        out = {"day": day, "date": grp[0]["date"], "n_seeds": len(grp)}
        for m in AGGREGATE_METRICS:
            vals = sorted(float(r[m]) for r in grp)
            mean = statistics.fmean(vals) if vals else math.nan
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0

            def pct(p):
                if not vals:
                    return math.nan
                k = (len(vals) - 1) * p / 100.0
                lo, hi = math.floor(k), math.ceil(k)
                return vals[int(k)] if lo == hi else vals[lo] * (hi - k) + vals[hi] * (k - lo)
            out[f"{m}_mean"] = mean
            out[f"{m}_sd"] = sd
            out[f"{m}_p025"] = pct(2.5)
            out[f"{m}_p975"] = pct(97.5)
        out_rows.append(out)
    _write_csv(out_path, fields, out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="300-329")
    ap.add_argument("--pop-size", type=int, default=10_000_000)
    ap.add_argument("--sim-days", type=int, default=244)
    ap.add_argument("--outdir", default="outputs/v4_daily_trajectory")
    args = ap.parse_args()

    seeds = _parse_seed_spec(args.seeds)
    outdir = PROJECT_DIR / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    per_seed = outdir / "per_seed_daily"
    per_seed.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for i, seed in enumerate(seeds):
        t0 = time.time()
        daily_path = per_seed / f"seed{seed}_daily_trajectory.csv"
        if daily_path.exists() and daily_path.stat().st_size > 0:
            print(f"[skip] seed={seed} (exists)", flush=True)
            continue
        try:
            rows, s = run_seed(seed, args.pop_size, args.sim_days)
            _write_csv(daily_path, DAILY_FIELDS, rows)
            tot = rows[-1]["cumulative_episode_cases"]
            run_rows.append({
                "seed": seed, "status": "ok",
                "episode_total_cases": tot,
                "episode_homo_ratio": s.get("episode_homo_ratio", ""),
                "episode_bi_ratio": s.get("episode_bi_ratio", ""),
                "episode_hetero_ratio": s.get("episode_hetero_ratio", ""),
                "episode_female_ratio": s.get("episode_female_ratio", ""),
            })
            print(f"[ok] seed={seed} total={tot} homo={s.get('episode_homo_ratio')} "
                  f"bi={s.get('episode_bi_ratio')} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc(limit=6)
            run_rows.append({"seed": seed, "status": "failed", "episode_total_cases": ""})
            print(f"[fail] seed={seed}: {e}", flush=True)

    # rebuild long file
    all_rows = []
    for seed in seeds:
        p = per_seed / f"seed{seed}_daily_trajectory.csv"
        if p.exists() and p.stat().st_size > 0:
            with p.open(newline="", encoding="utf-8-sig") as f:
                all_rows.extend(list(csv.DictReader(f)))
    if not all_rows:
        raise RuntimeError("No successful per-seed trajectory files")
    long_path = outdir / "daily_trajectory_long.csv"
    _write_csv(long_path, DAILY_FIELDS, all_rows)
    _summarize_daily(long_path, outdir / "daily_trajectory_summary_by_day.csv")
    _write_csv(outdir / "run_summary.csv",
               ["seed", "status", "episode_total_cases", "episode_homo_ratio",
                "episode_bi_ratio", "episode_hetero_ratio", "episode_female_ratio"],
               run_rows)
    print(f"Done. {len(all_rows)} rows, {len(run_rows)} seeds -> {outdir}")


if __name__ == "__main__":
    main()
