"""Audit per-layer infection composition for the PRE-cut vs CURRENT configs.

Answers: which layers actually drive `total`? This is the benchmark needed
before deciding how to rebuild transmission toward the 2280 target.

Config A (pre-cut, "the calibration that hit 2280"):
  - oneoff_p         = 0.15   (NegBin mean 5.67 partners/day)
  - party_ht_top_mean = 18.0  (top E[k] single-night 18)
  - social_contact_scale = 10.0, social_same_sex_assort = 0.9
  - community_bridge_scale = 1.5
Config B (current, "all anchors reverted"):
  - whatever main.py currently holds (oneoff_p=0.667, top_mean=1.0, scale=1.0, ...)

Runs ONE seed (300) each, prints total + per-layer infection counts + shares.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population, Case_data
import main


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
        dict(name='phase3_postpeak', start=171, end=main.SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]


def run_one(params, seed, label):
    t0 = time.time()
    pop = Population(n=main.POPULATION_SIZE, params=params, seed=seed,
                     cache_label=f"layer_audit_{label}_{seed}", policy=make_policy(),
                     importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(main.INITIAL_INFECTIONS, mask=mask)
    pop.run_simulation(main.SIM_DAYS)

    sim = np.asarray(pop.daily_episode_cases[:main.SIM_DAYS], dtype=float)
    obs = np.asarray(Case_data, dtype=float)[:main.SIM_DAYS]
    c_rmse = float(np.sqrt(np.mean((np.cumsum(sim) - np.cumsum(obs)) ** 2)))

    counts = pop.layer_infection_counts
    total = float(sim.sum())
    print(f"\n=== {label} (seed {seed})  runtime {time.time()-t0:.0f}s ===", flush=True)
    print(f"total = {total:.0f}   peak={int(np.argmax(sim))}   cRMSE={c_rmse:.0f}", flush=True)
    for layer, c in counts.items():
        share = 100.0 * c / total if total > 0 else 0.0
        print(f"  {layer:16s} {c:8d}  ({share:5.1f}%)", flush=True)
    return total


if __name__ == "__main__":
    seed = 300

    # Config A: pre-cut params (overlay on a copy of main.params)
    pre = dict(main.params)
    pre.update({
        "oneoff_p": 0.15,
        "party_ht_top_mean": 18.0,
        "social_contact_scale": 10.0,
        "social_same_sex_assort": 0.9,
        "community_bridge_scale": 1.5,
    })
    print("=== Config A (pre-cut) params: oneoff_p=0.15, top_mean=18, scale=10, assort=0.9, bridge=1.5 ===", flush=True)
    run_one(pre, seed, "A_precut")

    # Config B: current main.py
    print("\n=== Config B (current) params ===", flush=True)
    print(f"  oneoff_p={main.params['oneoff_p']}, top_mean={main.params['party_ht_top_mean']}, "
          f"scale={main.params['social_contact_scale']}, assort={main.params['social_same_sex_assort']}, "
          f"bridge={main.params['community_bridge_scale']}", flush=True)
    run_one(main.params, seed, "B_current")
