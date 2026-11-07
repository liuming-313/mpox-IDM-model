"""Multi-seed validation of f in {7.0, 7.5, 8.0}.

Goal: determine which f gives median total ~ 2280 AND median peak ~ 92, and
whether the critical-band bistability (total jump between f=7.0 and f=7.5)
persists across seeds.

Each f runs N_SEEDS seeds (300, 311, 322, ...). Results written incrementally
to CSV so rows are never lost. Day-level logging suppressed.
"""
import os, sys, io, time, csv, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import params, SIM_DAYS, POPULATION_SIZE, INITIAL_INFECTIONS
from population import Population, Case_data


def make_policy():
    return [
        dict(name='phase1_growth', start=0, end=91,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95, social_mult=0.90,
             test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
             test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
        dict(name='phase2_control', start=92, end=170,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
        dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]


def run_one(seed, factor):
    oneoff_mean = 0.5 * factor
    oneoff_p = 1.0 / (1.0 + oneoff_mean)
    top_mean = 1.0 * factor

    p = dict(params)
    p["oneoff_p"] = oneoff_p
    p["party_ht_top_mean"] = top_mean
    p["sexual_anal_receptive"] = 0.28

    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"contact_{factor:.2f}_{seed}", policy=make_policy(),
                     importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.seed_infection(INITIAL_INFECTIONS, mask=mask)
        pop.run_simulation(SIM_DAYS)

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    c_rmse = float(np.sqrt(np.mean((np.cumsum(sim) - np.cumsum(obs)) ** 2)))

    s = pop.summary() if hasattr(pop, 'summary') else {}
    total = float(sim.sum())
    peak = int(np.argmax(sim))
    homo = s.get('episode_homo_ratio', float('nan'))
    hetero = s.get('episode_hetero_ratio', float('nan'))
    female = s.get('episode_female_known_ratio', float('nan'))

    row = dict(seed=seed, factor=factor, oneoff_mean=oneoff_mean, top_mean=top_mean,
               total=total, peak=peak, cRMSE=c_rmse, homo=homo, hetero=hetero,
               female=female, seconds=round(time.time() - t0))
    print(f"RESULT f={factor:5.2f} seed={seed} total={total:6.0f} peak={peak:3d} "
          f"cRMSE={c_rmse:5.0f} homo={homo:.3f} hetero={hetero:.3f} "
          f"female={female:.3f}", flush=True)
    return row


if __name__ == "__main__":
    factors = [7.0, 7.5, 8.0]
    n_seeds = 8
    seeds = [300 + i * 11 for i in range(n_seeds)]

    out_csv = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs",
        "contact_volume", "multi_seed_f.csv"))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    fieldnames = ["seed", "factor", "oneoff_mean", "top_mean", "total",
                  "peak", "cRMSE", "homo", "hetero", "female", "seconds"]
    for f in factors:
        for s in seeds:
            rows.append(run_one(s, f))
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    print(f"\nsaved to {out_csv}", flush=True)

    # summary per factor
    print("\n=== per-factor summary (median / mean±std over seeds) ===", flush=True)
    for f in factors:
        fr = [r for r in rows if r["factor"] == f]
        tot = sorted(r["total"] for r in fr)
        med = tot[len(tot) // 2]
        mean = np.mean(tot)
        std = np.std(tot)
        pk = sorted(r["peak"] for r in fr)
        med_pk = pk[len(pk) // 2]
        print(f"f={f:5.2f}: total median={med:6.0f} mean={mean:6.0f}±{std:4.0f} "
              f"peak median={med_pk}  n={len(fr)}", flush=True)
    print("target: total=2280, peak=92", flush=True)
