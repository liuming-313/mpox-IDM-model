"""Single-seed verification after re-anchoring the two partner-count multipliers.

This round (the real "contact-volume" fixes, see reference/性网络边生成机制审查.md):
  - oneoff_p        0.15  -> 0.667   (NegBin(1,p) mean 5.67 -> 0.5 partners/day,
                                       was ~20x the CDC top-stratum 0.286/day)
  - party_ht_top_mean 18.0 -> 1.0    (top E[k] single-night draw 18 -> 1.0,
                                       fixing the ~18x 3-week-partner inflation;
                                       6-level E[k] now [0.12..1.0])

Runs ONE seed (300) with the full main.py param set to quantify how far total
drops from the previous 5785 after these two contact-volume fixes.
Outputs total / peak / cRMSE / composition.
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
    print(f"params check: oneoff_p={params['oneoff_p']} "
          f"(mean={1/params['oneoff_p']-1:.3f}/day), "
          f"party_ht_top_mean={params['party_ht_top_mean']}, "
          f"social_contact_scale={params['social_contact_scale']}", flush=True)

    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=params, seed=seed,
                     cache_label=f"sexual_anchor_{seed}", policy=make_policy(), importations=[])
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
    print(f"total = {sim.sum():.0f}  (target 2280; prev revert 5785)", flush=True)
    print(f"peak day = {int(np.argmax(sim))}  (obs 92)", flush=True)
    print(f"cRMSE = {c_rmse:.0f}", flush=True)
    for k in ('episode_homo_ratio', 'episode_hetero_ratio', 'episode_female_known_ratio',
              'cum_inf_homo_ratio', 'cum_inf_hetero_ratio'):
        if k in s:
            print(f"  {k} = {s[k]:.4f}", flush=True)
