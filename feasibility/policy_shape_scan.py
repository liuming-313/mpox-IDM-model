"""Policy time-shape sensitivity scan: match the observed Case_data curve.

The current model rises too fast (50% cumulative at day 89 vs observed day 103)
and declines too steeply (90% at day 115 vs observed day 143). This scans a
small grid of policy phase timings + party_mult strengths to find a time shape
closer to the observed curve, holding transmission params fixed.

Each config runs ONE 10M sim (seed 42) and reports:
  - peak day (observed 92), total episodes (observed 2280)
  - day to reach 25/50/75/90% of final cumulative (observed 87/103/121/143)
  - daily & cumulative RMSE vs Case_data
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

# policy configs: (name, phase1_end, phase2_end, p1_party, p2_party, p3_party)
# baseline p1=0.95 p2=0.42 p3=0.47, phase1 0-91, phase2 92-170, phase3 171+
CONFIGS = [
    ("baseline",      91, 170, 0.95, 0.42, 0.47),
    # slow the rise (lower p1), keep decline similar
    ("slow_rise",     91, 170, 0.75, 0.42, 0.47),
    ("slow_rise2",    91, 170, 0.60, 0.42, 0.47),
    # lengthen the tail (higher p2/p3)
    ("long_tail",     91, 170, 0.95, 0.60, 0.60),
    ("long_tail2",    91, 170, 0.95, 0.70, 0.70),
    # both: slower rise + longer tail
    ("slow_long",     91, 170, 0.70, 0.60, 0.60),
    # earlier phase2 boundary (pull peak earlier) + longer tail
    ("earlier_ctrl",  85, 165, 0.95, 0.55, 0.55),
    # slower rise + earlier control + long tail
    ("combo",         85, 165, 0.70, 0.55, 0.60),
]

def make_policy(p1_end, p2_end, p1_party, p2_party, p3_party):
    return [
        dict(name='phase1_growth', start=0, end=p1_end,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=p1_party, social_mult=0.90,
             test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
             test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
        dict(name='phase2_control', start=p1_end+1, end=p2_end,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=p2_party, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
        dict(name='phase3_postpeak', start=p2_end+1, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=p3_party, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]

def run_one(seed, cfg, cache_label):
    name, p1e, p2e, p1p, p2p, p3p = cfg
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label=cache_label, policy=make_policy(p1e, p2e, p1p, p2p, p3p),
                     importations=[])
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
    obs_peak = int(np.argmax(obs))
    obs_total = obs.sum()
    cobs = np.cumsum(obs)
    # observed day to reach each % of final
    obs_pct_day = {}
    for pct in [25, 50, 75, 90]:
        obs_pct_day[pct] = int(np.searchsorted(cobs, pct/100*obs_total))

    print(f"observed: total={int(obs_total)}, peak_day={obs_peak}, "
          f"pct_day(25/50/75/90)={obs_pct_day[25]}/{obs_pct_day[50]}/{obs_pct_day[75]}/{obs_pct_day[90]}")
    print(f"{'config':<14} {'total':>6} {'peak':>4} {'d25':>4} {'d50':>4} {'d75':>4} {'d90':>4} "
          f"{'dRMSE':>7} {'cRMSE':>8} {'homo':>5} {'hetero':>6} {'female':>6}")
    print("-" * 95)

    for cfg in CONFIGS:
        name = cfg[0]
        t0 = time.time()
        sim = run_one(42, cfg, f"policyscan_{name}")
        s_total = sim.sum()
        s_peak = int(np.argmax(sim))
        csim = np.cumsum(sim)
        s_pct_day = {}
        for pct in [25, 50, 75, 90]:
            s_pct_day[pct] = int(np.searchsorted(csim, pct/100*s_total)) if s_total > 0 else -1
        d_rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
        c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
        print(f"{name:<14} {s_total:>6.0f} {s_peak:>4} {s_pct_day[25]:>4} {s_pct_day[50]:>4} "
              f"{s_pct_day[75]:>4} {s_pct_day[90]:>4} {d_rmse:>7.3f} {c_rmse:>8.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
