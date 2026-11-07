"""Verify long_tail2 (best policy from shape scan) for composition + stability.

long_tail2: phase1 party_mult=0.95, phase2/phase3 party_mult=0.70/0.70.
This was the best curve fit (cRMSE 193 vs baseline 296, total 2274 ~ 2280 target),
but the scan did NOT report composition. Higher party_mult over a longer tail may
widen hetero/female spill, so we check composition + run multiple seeds.
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

def make_policy_longtail2():
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
                     cache_label=cache_label, policy=make_policy_longtail2(),
                     importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    s = pop.summary()
    return sim, s

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    obs_total = obs.sum()
    cobs = np.cumsum(obs)

    print("=== long_tail2 verification (party_mult 0.95/0.70/0.70) ===")
    print(f"observed: total={int(obs_total)}, peak=92")
    print(f"{'seed':>5} {'total':>6} {'peak':>4} {'cRMSE':>8} "
          f"{'homo':>6} {'hetero':>7} {'female':>7}")

    totals, peaks, crmse, homo, hetero, female = [], [], [], [], [], []
    for i in range(args.seeds):
        seed = 100 + i * 7
        t0 = time.time()
        sim, s = run_one(seed, f"longtail2_verify_{seed}")
        s_total = sim.sum()
        s_peak = int(np.argmax(sim))
        csim = np.cumsum(sim)
        c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
        h = s.get("episode_homo_ratio", 0.0)
        ht = s.get("episode_hetero_ratio", 0.0)
        f = s.get("episode_female_ratio", 0.0)
        totals.append(s_total); peaks.append(s_peak); crmse.append(c_rmse)
        homo.append(h); hetero.append(ht); female.append(f)
        print(f"{seed:>5} {s_total:>6.0f} {s_peak:>4} {c_rmse:>8.1f} "
              f"{h:>6.1%} {ht:>7.1%} {f:>7.1%} ({time.time()-t0:.0f}s)", flush=True)

    print("-" * 70)
    print(f"total:  mean={np.mean(totals):.0f} median={np.median(totals):.0f} std={np.std(totals):.0f}")
    print(f"peak:   mean={np.mean(peaks):.1f} median={np.median(peaks):.0f}")
    print(f"cRMSE:  mean={np.mean(crmse):.1f}")
    print(f"homo:   mean={np.mean(homo):.2%} std={np.std(homo):.2%}")
    print(f"hetero: mean={np.mean(hetero):.2%} std={np.std(hetero):.2%}")
    print(f"female: mean={np.mean(female):.2%} std={np.std(female):.2%}")
