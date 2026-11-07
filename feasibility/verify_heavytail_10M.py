"""
10M verification of sex_party heavy-tail (MSM-core-restricted activity stratum).

A = heavy-tail OFF (old behaviour: every attendee NB(r=1), E[k]~2.7)
B = heavy-tail ON  (5% of MSM core flagged high-activity, E[k]=12, clip 30)

Goal: confirm B lifts episode toward the 2280 target at full 10M scale while
keeping composition healthy (hetero ~9.9%, female ~3.7%).
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


def base_params():
    return {
        "oneoff_active_frac": 0.01,
        "oneoff_p": 0.15,
        "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10,
        "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
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


def run_one(label, cache_label, party_heavy_tail):
    params = base_params()
    params["party_heavy_tail"] = party_heavy_tail
    params["party_ht_frac"] = 0.05
    params["party_ht_top_mean"] = 12.0
    params["party_ht_top_clip"] = 30

    pop = Population(n=POP_SIZE, params=params, seed=SEED,
                     cache_label=cache_label, policy=make_policy(),
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

    homo = summ.get('episode_homo_ratio', summ.get('cum_inf_homo_ratio', 0)) * 100
    bi = summ.get('episode_bi_ratio', summ.get('cum_inf_bi_ratio', 0)) * 100
    hetero = summ.get('episode_hetero_ratio', summ.get('cum_inf_hetero_ratio', 0)) * 100
    female = summ.get('episode_female_known_ratio', summ.get('cum_inf_female_known_ratio', 0)) * 100

    print(f"\n===== 10M {label} =====")
    print(f"party_heavy_tail={party_heavy_tail}, runtime={dt:.0f}s")
    print(f"episode cases = {episode}   (target ~2280)")
    print(f"cum_inf       = {cum_inf}")
    print(f"homo={homo:.2f}%  bi={bi:.2f}%  hetero={hetero:.2f}%  female={female:.2f}%")
    print(f"  (target: homo 77.5%, bi 12.6%, hetero 9.9%, female 3.7%)")
    print("per-layer (top 6):")
    for k in sorted(layers, key=lambda x: -layers[x])[:6]:
        print(f"  {k:>16}: {layers[k]}")
    return dict(episode=episode, cum_inf=cum_inf, homo=homo, bi=bi, hetero=hetero,
                female=female, layers=layers)


def main():
    print(f"POP={POP_SIZE}, DAYS={SIM_DAYS}, NSEEDS={N_SEEDS}, SEED={SEED}")
    a = run_one("A: heavy-tail OFF", "verify10M_ht_off", False)
    b = run_one("B: heavy-tail ON", "verify10M_ht_on", True)

    print("\n===== COMPARISON =====")
    print(f"{'metric':<14} {'A(off)':>12} {'B(on)':>12} {'ratio':>8}")
    for m in ['episode', 'cum_inf']:
        print(f"{m:<14} {a[m]:>12} {b[m]:>12} {b[m]/max(a[m],1):>8.2f}x")
    for m in ['homo', 'bi', 'hetero', 'female']:
        print(f"{m:<14} {a[m]:>12.2f} {b[m]:>12.2f}")


if __name__ == "__main__":
    main()
