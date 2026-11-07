"""
Seed-targeting A/B test: where should the initial infections be seeded?

Compares two seeding strategies for the SAME seed count (20), heavy-tail ON
(party_ht_top_mean=20), 10M:
  A) "random MSM core"  -- current behavior: 20 seeds drawn uniformly from all
     MSM core (gay men, MCP-flagged). This is the model's current default.
  B) "high-activity MSM" -- 20 seeds drawn from the sex_party high-activity
     sub-stratum (sex_party_high_active, ~5% of MSM core). Aligned with the
     literature (Netherlands fading model seeds into "very high" activity;
     US DC model inputs group 5-6 daily).

The question: does biasing the seed toward the high-activity sub-stratum
(a) raise the episode median (stronger establishment), and (b) change the
extinction risk, WITHOUT degrading composition? This tells us whether the
current "uniform MSM core" seeding is justified or should be re-targeted.

Note: seed_infection draws `n` individuals from `mask`; to target the
high-activity sub-stratum we pass a mask restricted to it (and if the
sub-stratum is smaller than n we fall back to the full MSM core).
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


def run_one(seed, targeting, cache_label):
    params = base_params()
    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=cache_label, policy=make_policy(),
                     importations=[])

    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    msm_core = (g == 0) & (o == 'homo') & mcp

    if targeting == "high_active":
        # top activity stratum = level 5 in the 6-level riskg gradient
        lvl = getattr(pop, "sex_party_activity_level", None)
        ha = (lvl == 5) if lvl is not None else None
        mask = msm_core & ha if ha is not None else msm_core
        # fallback to full MSM core if the sub-stratum is too small
        if mask.sum() < N_SEEDS:
            mask = msm_core
    else:  # "random" (uniform MSM core)
        mask = msm_core

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5, help="reps per strategy")
    args = ap.parse_args()

    print(f"POP={POP_SIZE}, DAYS={SIM_DAYS}, N_SEEDS={N_SEEDS}, reps={args.reps}, "
          f"heavy-tail ON (top_mean=20)")

    for targeting, cache in [("random", "seedtarget_random"),
                             ("high_active", "seedtarget_high")]:
        rows = []
        for s in range(args.reps):
            t0 = time.time()
            r = run_one(s, targeting, cache)
            r["t"] = time.time() - t0
            rows.append(r)
            print(f"[{targeting}] rep={s}: episode={r['episode']}, homo={r['homo']:.1f}%, "
                  f"hetero={r['hetero']:.2f}%, female={r['female']:.2f}% ({r['t']:.0f}s)")

        ep = np.array([r['episode'] for r in rows])
        extinct = int(np.sum(ep <= N_SEEDS))
        print(f"\n===== targeting={targeting} (n={len(rows)}) =====")
        print(f"episode: mean={ep.mean():.0f} median={np.median(ep):.0f} std={ep.std():.0f} "
              f"min={ep.min()} max={ep.max()}")
        print(f"extinction (episode<=seeds): {extinct}/{len(rows)}")
        for k in ['homo', 'bi', 'hetero', 'female']:
            vals = np.array([r[k] for r in rows])
            print(f"{k:<8}: mean={vals.mean():.2f}% std={vals.std():.2f}%")
        print("individual:", ", ".join(f"{r['episode']}" for r in rows))


if __name__ == "__main__":
    main()
