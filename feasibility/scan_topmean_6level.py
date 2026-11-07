"""Scan party_ht_top_mean under the 6-level CDC riskg stratification.

After upgrading sex_party partner-degree stratification from a binary
high-activity flag to the full 6-level CDC riskg gradient, the weighted mean
partner count rose (3.565 -> 3.878), overshooting total cases to ~2724 median
(target 2280). This scan re-anchors party_ht_top_mean so the 6-level gradient
preserves the calibrated ~2280 median while keeping the CDC-sourced gradient.

Candidates: top_mean in {14, 16, 18}; 5 seeds each (300, 311, 322, 333, 344).
"""
import os, sys, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population, Case_data

POP_SIZE = 10_000_000
SIM_DAYS = 244
SEEDS = [300, 311, 322, 333, 344]

def base_params(top_mean):
    return {
        "oneoff_active_frac": 0.01, "oneoff_p": 0.15, "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10, "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
        "sexual_anal_receptive": 0.28, "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035, "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7, "party_core_intensity": 0.7,
        "social_contact_scale": 10, "social_same_sex_assort": 0.9,
        "party_heavy_tail": True, "party_ht_frac": 0.05,
        "party_ht_top_mean": float(top_mean), "party_ht_top_clip": 30,
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

def run_one(seed, top_mean, cache_label):
    pop = Population(n=POP_SIZE, params=base_params(top_mean), seed=seed,
                     cache_label=cache_label, policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    return sim, pop.summary()

if __name__ == "__main__":
    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    obs_total = obs.sum()
    cobs = np.cumsum(obs)
    obs_peak = int(np.argmax(obs))

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "topmean_scan")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "topmean_scan_6level.csv")

    print(f"=== top_mean scan (6-level gradient) | observed total={int(obs_total)} peak={obs_peak} ===")
    print(f"{'top':>5} {'seed':>5} {'total':>6} {'peak':>4} {'cRMSE':>8} {'homo':>6} {'hetero':>7} {'female':>7}")

    rows = []
    for top_mean in [14.0, 16.0, 18.0]:
        for seed in SEEDS:
            t0 = time.time()
            sim, s = run_one(seed, top_mean, f"tm{int(top_mean)}_{seed}")
            tot = sim.sum()
            pk = int(np.argmax(sim))
            csim = np.cumsum(sim)
            c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
            h = s.get("episode_homo_ratio", 0.0)
            ht = s.get("episode_hetero_ratio", 0.0)
            f = s.get("episode_female_ratio", 0.0)
            rows.append(dict(top_mean=top_mean, seed=seed, total=tot, peak=pk,
                             crmse=c_rmse, homo=h, hetero=ht, female=f))
            print(f"{top_mean:>5.0f} {seed:>5} {tot:>6.0f} {pk:>4} {c_rmse:>8.1f} {h:>6.1%} {ht:>7.1%} {f:>7.1%} ({time.time()-t0:.0f}s)", flush=True)

    # aggregate per top_mean
    print("-" * 72)
    for top_mean in [14.0, 16.0, 18.0]:
        sub = [r for r in rows if r["top_mean"] == top_mean]
        tot = [r["total"] for r in sub]
        print(f"top_mean={top_mean:.0f}: total mean={np.mean(tot):.0f} median={np.median(tot):.0f} "
              f"peak mean={np.mean([r['peak'] for r in sub]):.1f} cRMSE={np.mean([r['crmse'] for r in sub]):.1f} "
              f"homo={np.mean([r['homo'] for r in sub]):.2%} hetero={np.mean([r['hetero'] for r in sub]):.2%} "
              f"female={np.mean([r['female'] for r in sub]):.2%}")

    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["top_mean", "seed", "total", "peak",
                                           "crmse", "homo", "hetero", "female"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved -> {csv_path}")
