"""Single-seed verification after reverting the unjustified social parameters.

Reverts (done in main.py):
  - community_bridge_scale 1.5 -> 1.0 (disabled; was the 6464 culprit)
  - social_contact_scale  10   -> 1.0 (literature anchor)
  - social_same_sex_assort 0.9 -> 0.80 (literature anchor)

Runs ONE seed (300) with the full main.py param set to quantify how far total
drops once the out-of-range knobs are removed. Outputs total / peak / cRMSE /
composition so we can size the gap to refill via literature-anchored sexual
transmission rates.
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


if __name__ == "__main__":
    seed = 300
    print(f"params check: social_contact_scale={params['social_contact_scale']}, "
          f"assort={params['social_same_sex_assort']}, "
          f"bridge={params['community_bridge_scale']}, "
          f"top_mean={params['party_ht_top_mean']}", flush=True)

    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=params, seed=seed,
                     cache_label=f"revert_check_{seed}", policy=make_policy(), importations=[])
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
    print(f"runtime {time.time()-t0:.0f}s", flush=True)
    print(f"total = {sim.sum():.0f}  (target 2280; before revert 6464)", flush=True)
    print(f"peak day = {int(np.argmax(sim))}  (obs 92)", flush=True)
    print(f"cRMSE = {c_rmse:.0f}", flush=True)
    for k in ('episode_homo_ratio', 'episode_hetero_ratio', 'episode_female_known_ratio',
              'cum_inf_homo_ratio', 'cum_inf_hetero_ratio'):
        if k in s:
            print(f"  {k} = {s[k]:.4f}", flush=True)
