"""Dump the current model's daily episode curve to compare against the real
2022 mpox epidemic curve (rise → peak → decline shape).

Runs ONE 10M simulation with the current main.py baseline (heavy-tail ON,
top_mean=20, 20 seeds) and prints the daily episode series plus shape
diagnostics: peak day, peak height, cumulative by phase.
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
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.42, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
        dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.47, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]

if __name__ == "__main__":
    seed = 42
    t0 = time.time()
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label="curve_dump", policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    ep = np.asarray(pop.daily_episode_cases, dtype=float)

    print(f"runtime {time.time()-t0:.0f}s")
    print(f"total episode = {int(ep.sum())}")
    peak_day = int(np.argmax(ep))
    print(f"peak day = {peak_day}, peak height = {int(ep.max())}")

    # cumulative share by 30-day windows (roughly months)
    print("\ncumulative share by 30-day windows:")
    csum = np.cumsum(ep)
    tot = csum[-1]
    for d in range(30, SIM_DAYS + 1, 30):
        print(f"  by day {d}: {100*csum[d-1]/tot:.1f}%")

    # daily series, aggregated every 7 days for readability
    print("\nweekly episode counts (day: count):")
    for d in range(0, SIM_DAYS, 7):
        print(f"  {d:3d}-{d+6:3d}: {int(ep[d:d+7].sum())}")
