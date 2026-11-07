"""Fine-tune phase1 internal shape to remove the residual "rise ~2 weeks too fast".

The long_tail2 config (phase1 party_mult=0.95 uniform over days 0-91) still rises
slightly too fast vs observed (cumulative curve leads by ~2 weeks). The earlier
"slow_rise" test lowered party_mult over ALL 91 days and collapsed the epidemic,
so instead we split phase1 into sub-phases that damp the EARLY days only, then
ramp back up toward the peak.

Sub-phase designs (party_mult by day range), keeping phase2/3 = long_tail2 (0.70/0.70):
  A. fine_early:  0-30 -> 0.70, 31-60 -> 0.85, 61-91 -> 0.95
  B. fine_early2: 0-20 -> 0.60, 21-60 -> 0.85, 61-91 -> 0.95
  C. fine_early3: 0-30 -> 0.75, 31-91 -> 0.95
  D. late_damp:   0-60 -> 0.95, 61-91 -> 0.80   (damp just before peak)

Reports cRMSE (vs observed cumulative), total, peak day, and 25/50/75/90% days.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population, Case_data

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

def make_policy(phase1_segments):
    # phase1_segments: list of (start, end, party_mult)
    phases = []
    for (s, e, pm) in phase1_segments:
        phases.append(dict(name=f'phase1_{s}_{e}', start=s, end=e,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=pm, social_mult=0.90,
             test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
             test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2))
    phases.append(dict(name='phase2_control', start=92, end=170,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2))
    phases.append(dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2))
    return phases

# name -> phase1 segments
CONFIGS = {
    "baseline":      [(0, 91, 0.95)],
    "fine_early":    [(0, 30, 0.70), (31, 60, 0.85), (61, 91, 0.95)],
    "fine_early2":   [(0, 20, 0.60), (21, 60, 0.85), (61, 91, 0.95)],
    "fine_early3":   [(0, 30, 0.75), (31, 91, 0.95)],
    "late_damp":     [(0, 60, 0.95), (61, 91, 0.80)],
}

def run_one(seed, segs, cache_label):
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label=cache_label, policy=make_policy(segs), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    return np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)

if __name__ == "__main__":
    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    obs_total = obs.sum()
    cobs = np.cumsum(obs)
    obs_peak = int(np.argmax(obs))
    obs_pct = {}
    for pct in [25, 50, 75, 90]:
        obs_pct[pct] = int(np.searchsorted(cobs, pct/100*obs_total))

    print(f"observed: total={int(obs_total)}, peak={obs_peak}, "
          f"pct(25/50/75/90)={obs_pct[25]}/{obs_pct[50]}/{obs_pct[75]}/{obs_pct[90]}")
    print(f"{'config':<14} {'total':>6} {'peak':>4} {'d25':>4} {'d50':>4} {'d75':>4} {'d90':>4} {'cRMSE':>8}")
    print("-" * 70)

    for name, segs in CONFIGS.items():
        t0 = time.time()
        sim = run_one(42, segs, f"finetune_{name}")
        tot = sim.sum()
        pk = int(np.argmax(sim))
        csim = np.cumsum(sim)
        s_pct = {}
        for pct in [25, 50, 75, 90]:
            s_pct[pct] = int(np.searchsorted(csim, pct/100*tot)) if tot > 0 else -1
        c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
        print(f"{name:<14} {tot:>6.0f} {pk:>4} {s_pct[25]:>4} {s_pct[50]:>4} {s_pct[75]:>4} {s_pct[90]:>4} {c_rmse:>8.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
