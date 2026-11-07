"""
10M verification of the A-base scenario (heavy-tail + 20 MSM seeds, no importations).

At 2M, A_base_20s gave episode ~450 (target ~456 scaled), homo ~72.9%. This
script confirms at full 10M population whether episode lands near the real
target 2280 and whether composition stays healthy, since population scaling is
nonlinear (2M -> 10M is not a clean 5x).

Reports: episode cases, cum_inf, homo/bi/hetero/female shares, per-layer
contribution, and daily incidence / cumulative fit figures if available.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = 10_000_000
SIM_DAYS = 244
SEED = 42
N_SEEDS = 20

OUT_DIR = "outputs/10M_A_base_verify"


def base_params():
    return {
        "oneoff_active_frac": 0.01,
        "oneoff_p": 0.15,
        "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10,
        "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,   # A-base: no orientation bias
        "sexual_anal_receptive": 0.28,
        "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035,
        "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7,
        "party_core_intensity": 0.7,
        "social_contact_scale": 10,
        "social_same_sex_assort": 0.9,
        "n_days": SIM_DAYS + 1,
        "verbose": False,
    }


def make_policy():
    return [
        dict(name='phase1_growth', start=0, end=91,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95, social_mult=0.90,
             test_mult=0.70, trace_mult=0.80,
             test_pathway_prob=0.75, test_seek_delay_mean_days=5.0,
             test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
        dict(name='phase2_control', start=92, end=170,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.42, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20,
             test_pathway_prob=0.90, test_seek_delay_mean_days=4.0,
             test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
        dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.47, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00,
             test_pathway_prob=0.85, test_seek_delay_mean_days=4.5,
             test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]


def main():
    params = base_params()
    pop = Population(n=POP_SIZE, params=params, seed=SEED,
                     cache_label="verify_A_base_10M", policy=make_policy(),
                     importations=[])

    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(SEED)
    pop._rng = np.random.default_rng(SEED)
    pop.seed_infection(N_SEEDS, mask=mask)

    t0 = time.time()
    pop.run_simulation(SIM_DAYS)
    dt = time.time() - t0

    summ = pop.summary()
    cum_inf = int(summ.get("cum_inf_homo", 0) + summ.get("cum_inf_bi", 0) + summ.get("cum_inf_hetero", 0))
    episode = int(np.sum(pop.daily_episode_cases)) if hasattr(pop, "daily_episode_cases") else 0
    layers = dict(getattr(pop, "layer_infection_counts", {}))

    print("\n===== 10M A-BASE VERIFICATION =====")
    print(f"POP={POP_SIZE}, DAYS={SIM_DAYS}, NSEEDS={N_SEEDS}, heavy-tail ON, bias OFF")
    print(f"runtime = {dt:.0f}s\n")
    print(f"cum_inf          = {cum_inf}")
    print(f"episode cases    = {episode}   (target ~2280)")
    print(f"homo             = {summ.get('episode_homo_ratio', summ.get('cum_inf_homo_ratio', 0))*100:.2f}%   (target 77.53%)")
    print(f"bi               = {summ.get('episode_bi_ratio', summ.get('cum_inf_bi_ratio', 0))*100:.2f}%   (target 12.57%)")
    print(f"hetero           = {summ.get('episode_hetero_ratio', summ.get('cum_inf_hetero_ratio', 0))*100:.2f}%   (target 9.89%)")
    print(f"female_known     = {summ.get('episode_female_known_ratio', summ.get('cum_inf_female_known_ratio', 0))*100:.2f}%   (target 3.68%)")
    print("\nPer-layer infection counts:")
    for k in sorted(layers, key=lambda x: -layers[x]):
        print(f"  {k:>16}: {layers[k]}")

    # summary of full dict for reference
    print("\nFull summary keys with scalar values:")
    for k, v in sorted(summ.items()):
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
