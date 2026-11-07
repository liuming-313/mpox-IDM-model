"""Final baseline verification: lock the chosen configuration and run N seeds.

Final configuration (all decisions recorded in reference/ docs):
  - seed = 20 MSM index cases, drawn UNIFORMLY from the MSM core (K=0 in the
    high-activity-mix sweep). Uniform (not high-activity-targeted) was chosen:
    targeting overshoots 2280 (+56%) and degrades composition.
  - heavy-tail ON: party_ht_frac=0.05, party_ht_top_mean=20, clip=30
  - policy = long_tail2: phase1 party_mult=0.95, phase2/phase3 party_mult=0.70/0.70
    (longer, more gradual tail -> best curve fit, cRMSE 193 vs 296 baseline)

Reports mean/median/std of total episodes, composition, peak day, cRMSE across
seeds so the paper can cite the median (2280 is a median target).
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
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    return sim, pop.summary()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    obs_total = obs.sum()
    cobs = np.cumsum(obs)
    obs_peak = int(np.argmax(obs))

    print("=== FINAL BASELINE (20 uniform MSM seeds, heavy-tail, long_tail2) ===")
    print(f"observed: total={int(obs_total)}, peak_day={obs_peak}")
    print(f"{'seed':>5} {'total':>6} {'peak':>4} {'cRMSE':>8} {'homo':>6} {'hetero':>7} {'female':>7}")

    totals, peaks, crmse, homo, hetero, female = [], [], [], [], [], []
    for i in range(args.seeds):
        seed = 300 + i * 11
        t0 = time.time()
        sim, s = run_one(seed, f"finalbl_{seed}")
        tot = sim.sum()
        pk = int(np.argmax(sim))
        csim = np.cumsum(sim)
        c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
        h = s.get("episode_homo_ratio", 0.0)
        ht = s.get("episode_hetero_ratio", 0.0)
        f = s.get("episode_female_ratio", 0.0)
        totals.append(tot); peaks.append(pk); crmse.append(c_rmse)
        homo.append(h); hetero.append(ht); female.append(f)
        print(f"{seed:>5} {tot:>6.0f} {pk:>4} {c_rmse:>8.1f} {h:>6.1%} {ht:>7.1%} {f:>7.1%} ({time.time()-t0:.0f}s)", flush=True)

    print("-" * 72)
    print(f"TOTAL : mean={np.mean(totals):.0f} median={np.median(totals):.0f} std={np.std(totals):.0f}  [target 2280]")
    print(f"PEAK  : mean={np.mean(peaks):.1f} median={np.median(peaks):.0f}  [observed 92]")
    print(f"cRMSE : mean={np.mean(crmse):.1f}")
    print(f"HOMO  : mean={np.mean(homo):.2%} std={np.std(homo):.2%}")
    print(f"HETERO: mean={np.mean(hetero):.2%} std={np.std(hetero):.2%}")
    print(f"FEMALE: mean={np.mean(female):.2%} std={np.std(female):.2%}")
