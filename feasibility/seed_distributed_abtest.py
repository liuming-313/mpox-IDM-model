"""A/B test: one-shot seeding vs distributed seeding of the 20 index cases.

Compares, at 10M with heavy-tail ON (top_mean=20):
  A) "one_shot"   -- current behavior: seed_infection(20) all on day 0.
  B) "distributed" -- 20 cases spread over day 5..24 via the importations
     mechanism (aligned with the real 2022 timeline: first case ~May 6,
     4 MSM cases May 16, then ongoing introductions over the following weeks).

Goal: see whether distributing the 20 index cases over the early weeks
(1) pulls the peak earlier / closer to the observed Aug-1 peak and (2) makes
the early rise less sluggish, WITHOUT changing total scale or composition.

Both arms use the SAME random seeds so the comparison is controlled.
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = 10_000_000
SIM_DAYS = 244
N_SEEDS = 20

# Distributed schedule: total must equal N_SEEDS (20).
#   day 5-14 : 1/day -> 10
#   day 15-24: 1/day -> 10
DIST_IMPORTATIONS = [
    dict(start=5,  end=14, daily_n=1),
    dict(start=15, end=24, daily_n=1),
]


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


def run_one(seed, mode, cache_label):
    if mode == "distributed":
        importations = DIST_IMPORTATIONS
    else:
        importations = []

    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label=cache_label, policy=make_policy(),
                     importations=importations)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)

    if mode == "one_shot":
        pop.seed_infection(N_SEEDS, mask=mask)
    # distributed mode: importations already passed to constructor; _apply_importations
    # will seed them over the scheduled days during run_simulation.

    pop.run_simulation(SIM_DAYS)
    ep = np.asarray(pop.daily_episode_cases, dtype=float)

    summ = pop.summary()
    cum_inf = int(summ.get("cum_inf_homo", 0) + summ.get("cum_inf_bi", 0) + summ.get("cum_inf_hetero", 0))
    episode = int(ep.sum())
    homo = summ.get('episode_homo_ratio', summ.get('cum_inf_homo_ratio', 0)) * 100
    bi = summ.get('episode_bi_ratio', summ.get('cum_inf_bi_ratio', 0)) * 100
    hetero = summ.get('episode_hetero_ratio', summ.get('cum_inf_hetero_ratio', 0)) * 100
    female = summ.get('episode_female_known_ratio', summ.get('cum_inf_female_known_ratio', 0)) * 100

    peak_day = int(np.argmax(ep)) if ep.sum() > 0 else -1
    # cumulative share at key days
    csum = np.cumsum(ep)
    tot = csum[-1] if csum[-1] > 0 else 1
    share30 = 100 * csum[29] / tot
    share60 = 100 * csum[59] / tot
    share90 = 100 * csum[89] / tot

    return dict(seed=seed, episode=episode, cum_inf=cum_inf, homo=homo, bi=bi,
                hetero=hetero, female=female, peak_day=peak_day,
                share30=share30, share60=share60, share90=share90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    # verify distributed total == N_SEEDS
    tot = sum((b['end'] - b['start'] + 1) * b['daily_n'] for b in DIST_IMPORTATIONS)
    assert tot == N_SEEDS, f"distributed total {tot} != {N_SEEDS}"
    print(f"POP={POP_SIZE}, DAYS={SIM_DAYS}, N_SEEDS={N_SEEDS}, reps={args.reps}, "
          f"heavy-tail ON (top_mean=20), distributed total={tot}")

    for mode, cache in [("one_shot", "seedmode_oneshot"),
                        ("distributed", "seedmode_dist")]:
        rows = []
        for s in range(args.reps):
            t0 = time.time()
            r = run_one(s, mode, cache)
            r["t"] = time.time() - t0
            rows.append(r)
            print(f"[{mode}] rep={s}: episode={r['episode']}, peak_day={r['peak_day']}, "
                  f"homo={r['homo']:.1f}%, hetero={r['hetero']:.2f}%, female={r['female']:.2f}% "
                  f"({r['t']:.0f}s)")

        ep = np.array([r['episode'] for r in rows])
        pk = np.array([r['peak_day'] for r in rows], dtype=float)
        s30 = np.array([r['share30'] for r in rows])
        s60 = np.array([r['share60'] for r in rows])
        print(f"\n===== mode={mode} (n={len(rows)}) =====")
        print(f"episode: mean={ep.mean():.0f} median={np.median(ep):.0f} std={ep.std():.0f} "
              f"min={ep.min()} max={ep.max()}")
        print(f"peak_day: mean={pk.mean():.1f} median={np.median(pk):.1f} "
              f"(observed peak ~ day 92)")
        print(f"cum share: day30={s30.mean():.1f}% day60={s60.mean():.1f}% "
              f"day90={np.array([r['share90'] for r in rows]).mean():.1f}%")
        for k in ['homo', 'bi', 'hetero', 'female']:
            vals = np.array([r[k] for r in rows])
            print(f"{k:<8}: mean={vals.mean():.2f}% std={vals.std():.2f}%")
        print("individual episode:", ", ".join(f"{r['episode']}" for r in rows))


if __name__ == "__main__":
    main()
