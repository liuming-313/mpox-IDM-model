"""Sweep the number of seeds placed in the high-activity core (K of 20).

Root cause of large variance (std~578): the 20 seeds are drawn uniformly from
the whole MSM core, so the number that land in the high-activity core
(`sex_party_high_active`) is random, making "ignition" highly stochastic.

Here we FIX K of the 20 seeds to sit in the high-activity core and draw the
remaining 20-K from the bulk MSM core. K is the controllable "ignition
certainty" knob. We sweep K = 0/2/4/6/8/10 and report mean/median/std of total
episodes + composition, to find the K that minimizes variance without
overshooting the 2280 target.

Uses long_tail2 policy (party_mult 0.95/0.70/0.70) as the baseline shape.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = 10_000_000
SIM_DAYS = 244

def base_params():
    return {
        "oneoff_active_frac": 0.01, "oneoff_p": 0.15, "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10, "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
        "sexual_anal_receptive": 0.28, "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035, "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7, "party_core_intensity": 0.7,
        "social_contact_scale": 10, "social_same_sex_assort": 0.9,
        "party_heavy_tail": True, "party_ht_frac": 0.05,
        "party_ht_top_mean": 20.0, "party_ht_top_clip": 30,
        "n_days": SIM_DAYS + 1, "verbose": False,
    }

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

def seed_with_ht(pop, n_total, n_ht, seed, rng):
    """Seed n_total infections, of which exactly n_ht sit in the high-activity core."""
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    msm_mask = (g == 0) & (o == 'homo') & mcp
    ht_mask = (pop.sex_party_activity_level == 5) & msm_mask
    bulk_mask = msm_mask & ~ht_mask

    ht_uids = np.where(ht_mask)[0]
    bulk_uids = np.where(bulk_mask)[0]

    n_ht = min(n_ht, ht_uids.size)
    n_bulk = n_total - n_ht

    chosen_ht = rng.choice(ht_uids, size=n_ht, replace=False)
    chosen_bulk = rng.choice(bulk_uids, size=n_bulk, replace=False)
    idx = np.concatenate([chosen_ht, chosen_bulk])

    pop.susceptible[idx] = False
    pop.exposed[idx] = True
    pop.date_exposed[idx] = pop.day
    pop.incubation_period[idx] = np.clip(rng.lognormal(2.108, 0.254, size=n_total), 3, 17)
    pop.symptom_onset_delay[idx] = rng.uniform(1, 4, size=n_total)
    pop.infectious_period[idx] = np.clip(rng.gamma(9, 2.33, size=n_total), 14, 28)
    rash = pop.date_exposed[idx] + pop.incubation_period[idx]
    pop.infectious_start_day[idx] = np.maximum(
        pop.date_exposed[idx], (rash - pop.symptom_onset_delay[idx]).astype(int))

def run_one(seed, n_ht, cache_label):
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label=cache_label, policy=make_policy(), importations=[])
    rng = np.random.default_rng(seed * 1000 + 7)
    seed_with_ht(pop, 20, n_ht, seed, rng)
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    s = pop.summary()
    return sim.sum(), s

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])
    args = ap.parse_args()

    print(f"{'K_ht':>5} {'mean':>6} {'median':>6} {'std':>5} {'min':>5} {'max':>6} "
          f"{'homo':>6} {'hetero':>7} {'female':>7}")
    print("-" * 70)
    for K in args.ks:
        totals, homo, hetero, female = [], [], [], []
        t0 = time.time()
        for i in range(args.reps):
            seed = 200 + K * 100 + i * 7
            tot, s = run_one(seed, K, f"ht_mix_{K}_{seed}")
            totals.append(tot)
            homo.append(s.get("episode_homo_ratio", 0.0))
            hetero.append(s.get("episode_hetero_ratio", 0.0))
            female.append(s.get("episode_female_ratio", 0.0))
        print(f"{K:>5} {np.mean(totals):>6.0f} {np.median(totals):>6.0f} "
              f"{np.std(totals):>5.0f} {np.min(totals):>5.0f} {np.max(totals):>6.0f} "
              f"{np.mean(homo):>6.1%} {np.mean(hetero):>7.1%} {np.mean(female):>7.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)
