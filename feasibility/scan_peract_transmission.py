"""Scan per-act transmission risk under literature-anchored contact volumes.

Context (2026-10-31 / 11-01): anchoring BOTH partner-count multipliers to
literature (oneoff_p=0.667 -> 0.5 partners/day; party_ht_top_mean=1.0 -> top
E[k]=1.0 single-night) collapsed total from 5785 -> 20: the outbreak can no
longer establish. The correct rebuild is therefore a joint (contact-volume,
transmission-risk) calibration, not contact-volume alone.

This script freezes the literature contact volumes and scans the dominant
per-act risk `sexual_anal_receptive` (currently 0.28) upward across the
literature range (Dutch fading beta_s 0.25-0.40; CDC Spicknall high 0.60-0.90)
to find where total returns toward the 2280 target.

One seed (300) per scenario; reports total / peak day / cRMSE / composition.
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


def run_one(seed, anal_rec):
    p = dict(params)
    p["sexual_anal_receptive"] = anal_rec
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"peract_{anal_rec}_{seed}", policy=make_policy(),
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
    csim = np.cumsum(sim)
    cobs = np.cumsum(obs)
    c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))

    s = pop.summary() if hasattr(pop, 'summary') else {}
    total = float(sim.sum())
    peak = int(np.argmax(sim))
    homo = s.get('episode_homo_ratio', float('nan'))
    hetero = s.get('episode_hetero_ratio', float('nan'))
    female = s.get('episode_female_known_ratio', float('nan'))
    print(f"anal_rec={anal_rec:.2f}  total={total:.0f}  peak={peak}  "
          f"cRMSE={c_rmse:.0f}  homo={homo:.3f}  hetero={hetero:.3f}  "
          f"female={female:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(anal_rec=anal_rec, total=total, peak=peak, cRMSE=c_rmse,
                homo=homo, hetero=hetero, female=female)


if __name__ == "__main__":
    seed = 300
    scans = [0.28, 0.35, 0.40, 0.50, 0.60]
    print(f"contact volumes frozen: oneoff_p={params['oneoff_p']} "
          f"(mean={1/params['oneoff_p']-1:.3f}/day), "
          f"party_ht_top_mean={params['party_ht_top_mean']}, "
          f"social_contact_scale={params['social_contact_scale']}", flush=True)
    results = []
    for ar in scans:
        results.append(run_one(seed, ar))
    print("\n=== summary (seed 300, literature contact volumes) ===", flush=True)
    print("anal_rec | total | peak | cRMSE | homo | hetero | female", flush=True)
    for r in results:
        print(f"{r['anal_rec']:.2f} | {r['total']:.0f} | {r['peak']} | "
              f"{r['cRMSE']:.0f} | {r['homo']:.3f} | {r['hetero']:.3f} | "
              f"{r['female']:.3f}", flush=True)
    print("target: total=2280, peak=92", flush=True)
