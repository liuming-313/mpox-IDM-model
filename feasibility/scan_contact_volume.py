"""Scan contact-volume level to find where total returns to ~2280.

Context (2026-11-01 audit): oneoff + sex_party are the DOMINANT transmission
layers (63.5% + 50.7% of total in the pre-cut config), NOT stable. Cutting
both partner-count multipliers to the literature lower bound (oneoff_p=0.667
-> 0.5 partners/day; top_mean=1.0) collapsed total 5938 -> 20: the outbreak
cannot establish.

Correct rebuild = restore oneoff/sex_party contact volume to a level that is
BOTH literature-defensible AND lets the outbreak establish, with per-act risk
held at the literature-anchored 0.28.

This script freezes per-act=0.28 and jointly scans the two partner-count
multipliers upward from the (too low) lower bound toward the (too high)
pre-cut level, to locate the band where total ~ 2280.

Scenarios (joint, multiplicative factor f applied to BOTH multipliers'
partner-count means):
  oneoff_p       -> mean partners/day = 0.5 * f   (p = 1/(1+mean))
  party_ht_top_mean -> top E[k] = 1.0 * f          (scaled single-night draw)

One seed (300) per scenario. Reports total / peak / cRMSE / composition.
"""
import os, sys, time
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
    # oneoff_p -> NegBin(1, p) mean = (1-p)/p. Set mean = 0.5 * factor.
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
    print(f"f={factor:5.2f}  oneoff_mean={oneoff_mean:5.2f}  top_mean={top_mean:5.2f}  "
          f"total={total:6.0f}  peak={peak:3d}  cRMSE={c_rmse:5.0f}  "
          f"homo={homo:.3f}  hetero={hetero:.3f}  female={female:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return dict(factor=factor, oneoff_mean=oneoff_mean, top_mean=top_mean,
                total=total, peak=peak, cRMSE=c_rmse, homo=homo, hetero=hetero,
                female=female)


if __name__ == "__main__":
    seed = 300
    # Fine sweep around the f~7.2 interpolation point (f=6 -> 1737, f=8 -> 2645)
    # to locate the exact total ~ 2280 (peak ~ 92).
    factors = [6.5, 7.0, 7.5, 8.0]
    print(f"per-act frozen at 0.28; scaling BOTH oneoff mean (0.5*f) and "
          f"top_mean (1.0*f) jointly by factor f", flush=True)
    results = []
    for f in factors:
        results.append(run_one(seed, f))
    print("\n=== summary (seed 300, per-act=0.28) ===", flush=True)
    print("f | oneoff_mean | top_mean | total | peak | cRMSE | homo | hetero | female", flush=True)
    for r in results:
        print(f"{r['factor']:5.2f} | {r['oneoff_mean']:5.2f} | {r['top_mean']:5.2f} | "
              f"{r['total']:6.0f} | {r['peak']:3d} | {r['cRMSE']:5.0f} | "
              f"{r['homo']:.3f} | {r['hetero']:.3f} | {r['female']:.3f}", flush=True)
    print("target: total=2280, peak=92", flush=True)
