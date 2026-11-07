"""Diagnose seed-level early-fadeout under top_mean=18 (6-level CDC riskg).

Hypothesis: 20 index cases are drawn UNIFORMLY from the MSM core
(g==0 & homo & mcp). 95% of that pool sits in low CDC riskg strata
(3-week partner count 0.8-1.5), so some seeds draw index cases that are
almost all low-activity -> the outbreak stochastically fades out early.

This script (silent) reruns 10 seeds and records, per seed:
  - total episodes
  - cum cases by day 30 (early-fadeout proxy)
  - the CDC riskg stratum (sex_party_activity_level 0..5) of each of the
    20 index cases
  - the sex_party partner-degree (E[k] / top anchor) implied by those strata

Then prints a compact table + a fadeout-vs-index-activity diagnostic.
"""
import os, sys, io, contextlib, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population, Case_data, CDC_RISKG_PARTNERS_3W

POP_SIZE = 10_000_000
SIM_DAYS = 244
SEEDS = [300 + i * 11 for i in range(10)]

def base_params():
    return {
        "oneoff_active_frac": 0.01, "oneoff_p": 0.15, "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10, "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
        "sexual_anal_receptive": 0.28, "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035, "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7, "party_core_intensity": 0.7,
        "social_contact_scale": 10, "social_same_sex_assort": 0.9,
        "community_bridge_scale": 1.5,
        "party_heavy_tail": True, "party_ht_frac": 0.05,
        "party_ht_top_mean": 18.0, "party_ht_top_clip": 30,
        "n_days": SIM_DAYS + 1, "verbose": False,
    }

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

def run_one(seed, cache_label):
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label=cache_label, policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask

    # snapshot eligible uids BEFORE seeding so we can later recover who got picked
    elig = np.where(mask)[0]

    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)

    # silence the day-level debug prints inside population.py
    with contextlib.redirect_stdout(io.StringIO()):
        pop.seed_infection(20, mask=mask)
        pop.run_simulation(SIM_DAYS)

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)

    # recover the 20 index-case uids = exposed on day 0 (date_exposed == 0)
    idx_uids = np.where(pop.date_exposed == 0)[0]
    # stratum of each index case (-1 if not in MSM sex_party core)
    strata = pop.sex_party_activity_level[idx_uids] if idx_uids.size else np.array([], dtype=np.int8)

    cum30 = sim[:30].sum()
    total = sim.sum()
    return total, cum30, strata, sim

if __name__ == "__main__":
    print("=== seed-level early-fadeout diagnostic (top_mean=18) ===")
    print(f"{'seed':>5} {'total':>6} {'cum30':>6} {'cum30/total':>12}  index-strata(counts 0..5)")
    rows = []
    for seed in SEEDS:
        t0 = time.time()
        total, cum30, strata, sim = run_one(seed, f"diag_{seed}")
        # count strata
        cnt = np.bincount(strata[strata >= 0], minlength=6)
        mean_lvl = strata[strata >= 0].mean() if (strata >= 0).any() else float('nan')
        frac_top = (strata == 5).mean() if strata.size else 0.0
        rows.append(dict(seed=seed, total=total, cum30=cum30, strata=strata,
                         mean_lvl=mean_lvl, frac_top=frac_top, cnt=cnt))
        print(f"{seed:>5} {total:>6.0f} {cum30:>6.0f} {cum30/total:>12.1%}  "
              f"{cnt.tolist()}  mean_lvl={mean_lvl:.2f} top_frac={frac_top:.0%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    totals = [r['total'] for r in rows]
    print("-" * 90)
    print(f"TOTAL: mean={np.mean(totals):.0f} median={np.median(totals):.0f} std={np.std(totals):.0f}")

    # correlation: is early growth (cum30) predictive of final total?
    c30 = [r['cum30'] for r in rows]
    if len(totals) > 2:
        corr = np.corrcoef(totals, c30)[0, 1]
        print(f"corr(total, cum30) = {corr:.3f}")

    # compare index-case activity between low-total and high-total seeds
    th = np.median(totals)
    lo = [r for r in rows if r['total'] < th]
    hi = [r for r in rows if r['total'] >= th]
    lo_lvl = np.mean([r['mean_lvl'] for r in lo])
    hi_lvl = np.mean([r['mean_lvl'] for r in hi])
    lo_top = np.mean([r['frac_top'] for r in lo])
    hi_top = np.mean([r['frac_top'] for r in hi])
    print(f"\nLOW-total seeds  (n={len(lo)}): mean index activity-level={lo_lvl:.2f}, top(5)-frac={lo_top:.1%}")
    print(f"HIGH-total seeds (n={len(hi)}): mean index activity-level={hi_lvl:.2f}, top(5)-frac={hi_top:.1%}")
    print(f"\n(CDC riskg 3-week partner counts by stratum: {CDC_RISKG_PARTNERS_3W.tolist()})")
