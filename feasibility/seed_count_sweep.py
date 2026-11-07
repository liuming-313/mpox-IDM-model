"""
Seed-count sensitivity sweep: how many initial MSM infections are needed?

Sweeps INITIAL_INFECTIONS over {5, 10, 20, 40} at 10M, heavy-tail ON
(party_ht_top_mean=20), reporting for each count:
  - episode mean/median/std/min/max
  - extinction rate (fraction of reps with episode near zero)
  - composition (homo/bi/hetero/female)

This directly answers whether 20 seeds is justified vs the literature's 2-5:
if small counts have high extinction probability, that is the quantitative
argument for using 20 (no importations in this model).

Note: the literature's 2-5 seeds are in models that either are deterministic
SEIR (no stochastic extinction) or carry a continuous importation stream; this
model has NEITHER, so it needs enough seeds to reliably establish a chain.
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = 10_000_000
SIM_DAYS = 244


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
        "party_heavy_tail": True,
        "party_ht_frac": 0.05,
        "party_ht_top_mean": 20.0,
        "party_ht_top_clip": 30,
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


def run_one(seed, n_seeds):
    params = base_params()
    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=f"seedcount_{n_seeds}", policy=make_policy(),
                     importations=[])

    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(n_seeds, mask=mask)

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4, help="reps per seed count")
    ap.add_argument("--counts", type=int, nargs="+", default=[5, 10, 20, 40],
                    help="seed counts to sweep")
    args = ap.parse_args()

    print(f"POP={POP_SIZE}, DAYS={SIM_DAYS}, reps={args.reps}, counts={args.counts}, heavy-tail ON (top_mean=20)")

    summary_rows = []
    for nc in args.counts:
        rows = []
        for s in range(args.reps):
            t0 = time.time()
            r = run_one(s, nc)
            r["t"] = time.time() - t0
            rows.append(r)
            print(f"[n_seeds={nc}] rep={s}: episode={r['episode']}, homo={r['homo']:.1f}%, "
                  f"hetero={r['hetero']:.2f}%, female={r['female']:.2f}% ({r['t']:.0f}s)")

        ep = np.array([r['episode'] for r in rows])
        extinct = int(np.sum(ep <= nc))  # episode barely above seed count => chain died
        print(f"\n===== n_seeds={nc} (n={len(rows)}) =====")
        print(f"episode: mean={ep.mean():.0f} median={np.median(ep):.0f} std={ep.std():.0f} "
              f"min={ep.min()} max={ep.max()}")
        print(f"extinction (episode<=seeds): {extinct}/{len(rows)}")
        for k in ['homo', 'bi', 'hetero', 'female']:
            vals = np.array([r[k] for r in rows])
            print(f"{k:<8}: mean={vals.mean():.2f}% std={vals.std():.2f}%")
        print("individual:", ", ".join(f"{r['episode']}" for r in rows))
        summary_rows.append(dict(n_seeds=nc, ep_mean=ep.mean(), ep_median=np.median(ep),
                                 ep_std=ep.std(), extinct=extinct, n=len(rows)))

    print("\n===== SUMMARY =====")
    print(f"{'n_seeds':>8} {'ep_mean':>8} {'ep_median':>9} {'ep_std':>7} {'extinct':>8}")
    for r in summary_rows:
        print(f"{r['n_seeds']:>8} {r['ep_mean']:>8.0f} {r['ep_median']:>9.0f} "
              f"{r['ep_std']:>7.0f} {r['extinct']}/{r['n']}")


if __name__ == "__main__":
    main()
