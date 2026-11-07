"""Quantitative fit of the current model's daily episode curve vs observed LAC
Case_data. Reports peak date error, daily RMSE, cumulative RMSE, and early-phase
cumulative shares — the same metrics used in the previous importation-timing work
(CODEX_PROJECT_HANDOFF.md).

Runs ONE 10M simulation (one-shot 20 seeds, current baseline) and compares
against population.Case_data over the first 245 slots.
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
    obs = np.asarray(Case_data, dtype=float)
    T = min(SIM_DAYS, obs.size)
    obs = obs[:T]

    seed = 42
    t0 = time.time()
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label="curve_fit", policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:T], dtype=float)
    print(f"runtime {time.time()-t0:.0f}s")

    obs_peak = int(np.argmax(obs))
    sim_peak = int(np.argmax(sim))
    print(f"observed: total={int(obs.sum())}, peak_day={obs_peak} (height {int(obs.max())})")
    print(f"simulated: total={int(sim.sum())}, peak_day={sim_peak} (height {int(sim.max())})")
    print(f"peak date error: {sim_peak - obs_peak:+d} days")

    # daily RMSE over full window and early window (day 18..91)
    full_rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    early = slice(18, 92)
    early_rmse = float(np.sqrt(np.mean((sim[early] - obs[early]) ** 2)))
    print(f"\ndaily RMSE (full 0..{T-1}): {full_rmse:.4f}")
    print(f"daily RMSE (day18..91, early rise): {early_rmse:.4f}")

    # cumulative RMSE
    cobs = np.cumsum(obs)
    csim = np.cumsum(sim)
    cum_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
    print(f"cumulative RMSE: {cum_rmse:.4f}")

    # cumulative case checkpoints
    print("\ncumulative cases (obs vs sim) at key days:")
    for d in [30, 60, 91, 120, 150, 180]:
        print(f"  day {d:3d}: obs={cobs[d-1]:6.0f}  sim={csim[d-1]:6.0f}  diff={csim[d-1]-cobs[d-1]:+7.0f}")

    # early cumulative share (how fast does it reach 25%/50% of final)
    print("\nshare of final cumulative reached (obs vs sim):")
    for pct in [5, 10, 25, 50, 75, 90]:
        do = int(np.searchsorted(cobs, pct/100*cobs[-1]))
        ds = int(np.searchsorted(csim, pct/100*csim[-1]))
        print(f"  {pct:2d}%: obs day={do:3d}  sim day={ds:3d}  (sim-obs {ds-do:+d})")
