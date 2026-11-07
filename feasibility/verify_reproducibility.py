"""Verify seed-to-seed reproducibility after fixing seed_infection to use self._rng.

Runs the SAME seed twice and checks that (a) the chosen 20 seed individuals are
identical, and (b) the full daily-case trajectory is bit-identical. No global
np.random.seed() is called, to prove the result depends only on the Population's
own self._rng (default_rng(seed)) and not on ambient global RNG state.

Also runs with a DIFFERENT construction path (different cache_label) to confirm
the result is stable across cache hits/misses.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

N = 500_000
SEED = 42
SIM_DAYS = 60

def make_params():
    return {
        "oneoff_active_frac": 0.01, "oneoff_p": 0.15, "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10, "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
        "sexual_anal_receptive": 0.28, "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035, "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7, "party_core_intensity": 0.7,
        "social_contact_scale": 10, "social_same_sex_assort": 0.9,
        "community_bridge_scale": 1.5,
        "party_heavy_tail": True, "party_ht_frac": 0.05,
        "party_ht_top_mean": 18.0, "party_ht_top_clip": 30,
        "n_days": SIM_DAYS + 1, "verbose": False,
    }

def make_policy():
    return [
        dict(name='p1', start=0, end=30, contact_mult=1.0, oneoff_mult=1.0, party_mult=0.95,
             social_mult=0.90, test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
             test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
        dict(name='p2', start=31, end=SIM_DAYS, contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70,
             social_mult=1.05, test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
    ]

def run(seed, cache_label):
    pop = Population(n=N, params=make_params(), seed=seed, cache_label=cache_label,
                     policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    # NOTE: no np.random.seed() here — rely on self._rng only.
    pop.seed_infection(20, mask=mask)
    # capture chosen seeds: the newly non-susceptible individuals
    seed_ids = np.where(~pop.susceptible)[0]
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    return seed_ids, sim

# 1) same cache_label, two runs -> must be identical
s1, sim1 = run(SEED, "repro_test_a")
s2, sim2 = run(SEED, "repro_test_a")
print("=== same cache_label, two runs ===")
print("seed ids identical:", np.array_equal(s1, s2), "| n_seeds =", s1.size)
print("trajectory identical:", np.array_equal(sim1, sim2), "| total =", sim1.sum(), sim2.sum())

# 2) different cache_label (forces fresh cache build path) -> must STILL be identical
s3, sim3 = run(SEED, "repro_test_b_alt")
print("\n=== different cache_label (different construction path) ===")
print("seed ids identical:", np.array_equal(s1, s3))
print("trajectory identical:", np.array_equal(sim1, sim3), "| total =", sim1.sum(), sim3.sum())

print("\nRESULT:", "PASS" if (np.array_equal(sim1, sim2) and np.array_equal(sim1, sim3)) else "FAIL")
