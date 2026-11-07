"""
Multi-seed stability test for sex_party heavy-tail (MSM-core activity stratum).

Runs the 10M population with party_heavy_tail=True across multiple seeds and
reports the episode distribution (mean/median/std/min/max) and composition, so
we can judge whether the heavy-tail lifts the *median* episode toward the 2280
target rather than relying on a single seed.

The heavy-tail knobs are exposed so a small sweep over `party_ht_top_mean` can be
run in one go.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = 10_000_000
SIM_DAYS = 244
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


def run_one(seed, ht_top_mean):
    params = base_params()
    params["party_heavy_tail"] = True
    params["party_ht_frac"] = 0.05
    params["party_ht_top_mean"] = ht_top_mean
    params["party_ht_top_clip"] = 30

    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=f"stab_ht_{ht_top_mean:.0f}", policy=make_policy(),
                     importations=[])

    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(N_SEEDS, mask=mask)

    pop.run_simulation(SIM_DAYS)

    summ = pop.summary()
    cum_inf = int(summ.get("cum_inf_homo", 0) + summ.get("cum_inf_bi", 0) + summ.get("cum_inf_hetero", 0))
    episode = int(np.sum(pop.daily_episode_cases)) if hasattr(pop, "daily_episode_cases") else 0
    homo = summ.get('episode_homo_ratio', summ.get('cum_inf_homo_ratio', 0)) * 100
    bi = summ.get('episode_bi_ratio', summ.get('cum_inf_bi_ratio', 0)) * 100
    hetero = summ.get('episode_hetero_ratio', summ.get('cum_inf_hetero_ratio', 0)) * 100
    female = summ.get('episode_female_known_ratio', summ.get('cum_inf_female_known_ratio', 0)) * 100

    return dict(seed=seed, episode=episode, cum_inf=cum_inf, homo=homo, bi=bi,
                hetero=hetero, female=female)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds to run")
    ap.add_argument("--top-mean", type=float, nargs="+", default=[12.0],
                    help="party_ht_top_mean value(s) to sweep")
    args = ap.parse_args()

    print(f"POP={POP_SIZE}, DAYS={SIM_DAYS}, NSEEDS={N_SEEDS}, seeds={args.seeds}, top_mean={args.top_mean}")

    for ht in args.top_mean:
        rows = []
        for s in range(args.seeds):
            t0 = time.time()
            r = run_one(s, ht)
            r["t"] = time.time() - t0
            rows.append(r)
            print(f"[top_mean={ht:.0f}] seed={s}: episode={r['episode']}, cum_inf={r['cum_inf']}, "
                  f"homo={r['homo']:.1f}%, hetero={r['hetero']:.2f}%, female={r['female']:.2f}% "
                  f"({r['t']:.0f}s)")

        ep = np.array([r['episode'] for r in rows])
        cu = np.array([r['cum_inf'] for r in rows])
        def stat(a): return f"mean={a.mean():.0f} median={np.median(a):.0f} std={a.std():.0f} min={a.min()} max={a.max()}"
        print(f"\n===== top_mean={ht:.0f} (n={len(rows)}) =====")
        print(f"episode : {stat(ep)}   (target ~2280)")
        print(f"cum_inf : {stat(cu)}")
        for k in ['homo', 'bi', 'hetero', 'female']:
            vals = np.array([r[k] for r in rows])
            print(f"{k:<8}: mean={vals.mean():.2f}% std={vals.std():.2f}%")

        print("individual runs:")
        for r in rows:
            print(f"  seed={r['seed']}: ep={r['episode']} homo={r['homo']:.1f}% bi={r['bi']:.1f}% "
                  f"hetero={r['hetero']:.2f}% female={r['female']:.2f}%")


if __name__ == "__main__":
    main()
