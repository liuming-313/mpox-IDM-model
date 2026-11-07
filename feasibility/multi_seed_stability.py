"""
Multi-seed stability experiment.

Single-seed A/B tests and sweeps are noisy. This script runs a small set of
candidate scenarios across MANY random seeds and reports mean +/- std (and
extinction count) for the two calibration targets:
  - size (episode cases, target ~2280 at 10M; here at 2M we read the ratio to a
    reference scenario)
  - composition (homo / bi / hetero / female shares)

Candidate scenarios (all heavy-tail on, no 742 importations):
  A  baseline HT:        af=0.01 p=0.15, n_seeds=20
  B  more seeds:         af=0.01 p=0.15, n_seeds=40
  C  bias + mild oneoff: af=0.02 p=0.10, n_seeds=20, orientation bias ON
  D  bias + seeds:       af=0.02 p=0.10, n_seeds=40, orientation bias ON

Run at 2M population (fast) so we can afford many seeds. Population scaling is
nonlinear, so the *relative* ordering of scenarios is the signal here; a final
10M confirmation is a separate step.
"""
import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = int(os.environ.get("MS_POP", 2_000_000))
SIM_DAYS = int(os.environ.get("MS_DAYS", 244))
REPS = int(os.environ.get("MS_REPS", 10))
SEED_BASE = int(os.environ.get("MS_SEED_BASE", 100000))

# sexcat: 0=FH, 1=MH, 2=MSM, 3=WSW, 4=MB, 5=FB
ORI_BIAS = {0: 0.15, 1: 0.20, 2: 3.00, 3: 0.30, 4: 2.00, 5: 0.40}


def base_params():
    return {
        "oneoff_active_frac": 0.01,
        "oneoff_p": 0.15,
        "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10,
        "oneoff_activity_stratified": True,
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


def run_once(af, op, n_seeds, use_bias, seed):
    params = base_params()
    params["oneoff_active_frac"] = af
    params["oneoff_p"] = op
    params["oneoff_orientation_bias"] = dict(ORI_BIAS) if use_bias else None

    label = f"ms_{af}_{op}_{n_seeds}_{'b' if use_bias else 'n'}_{seed}"
    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=label, policy=make_policy(), importations=[])

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
    layers = dict(getattr(pop, "layer_infection_counts", {}))

    return {
        "cum_inf": cum_inf,
        "episode": episode,
        "homo": float(summ.get("episode_homo_ratio", summ.get("cum_inf_homo_ratio", 0))),
        "bi": float(summ.get("episode_bi_ratio", summ.get("cum_inf_bi_ratio", 0))),
        "hetero": float(summ.get("episode_hetero_ratio", summ.get("cum_inf_hetero_ratio", 0))),
        "female": float(summ.get("episode_female_known_ratio", summ.get("cum_inf_female_known_ratio", 0))),
        "oneoff_contrib": int(layers.get("oneoff", 0)),
        "sex_party_contrib": int(layers.get("sex_party", 0)),
    }


def summarize(name, rows):
    def stat(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(vals.mean()), float(vals.std())

    ep_mean, ep_std = stat("episode")
    ci_mean, ci_std = stat("cum_inf")
    ext = sum(1 for r in rows if r["cum_inf"] <= 2 * r["n_seeds"])
    return {
        "name": name,
        "ep_mean": round(ep_mean, 1), "ep_std": round(ep_std, 1),
        "ci_mean": round(ci_mean, 1), "ci_std": round(ci_std, 1),
        "extinct": ext,
        "homo": round(stat("homo")[0] * 100, 1), "homo_std": round(stat("homo")[1] * 100, 1),
        "bi": round(stat("bi")[0] * 100, 1),
        "hetero": round(stat("hetero")[0] * 100, 1),
        "female": round(stat("female")[0] * 100, 1),
        "oneoff_contrib": round(stat("oneoff_contrib")[0], 1),
        "sex_party_contrib": round(stat("sex_party_contrib")[0], 1),
    }


def main():
    scenarios = [
        ("A_base_20s",  0.01, 0.15, 20, False),
        ("B_base_40s",  0.01, 0.15, 40, False),
        ("C_bias_20s",  0.02, 0.10, 20, True),
        ("D_bias_40s",  0.02, 0.10, 40, True),
    ]

    out_csv = "outputs/feasibility/multi_seed_stability.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = ["scenario", "seed", "af", "op", "n_seeds", "use_bias",
                  "cum_inf", "episode", "homo", "bi", "hetero", "female",
                  "oneoff_contrib", "sex_party_contrib"]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        print(f"Multi-seed stability: {len(scenarios)} scenarios x {REPS} reps, "
              f"POP={POP_SIZE}, DAYS={SIM_DAYS}")
        print(f"Output -> {out_csv}\n")

        agg = {}
        for name, af, op, ns, bias in scenarios:
            rows = []
            for rep in range(REPS):
                seed = SEED_BASE + rep * 100003
                t0 = time.time()
                try:
                    r = run_once(af, op, ns, bias, seed)
                except Exception as e:
                    print(f"[FAIL] {name} rep{rep}: {type(e).__name__}: {e}")
                    continue
                dt = time.time() - t0
                r.update({"scenario": name, "seed": seed, "af": af, "op": op,
                          "n_seeds": ns, "use_bias": bias})
                w.writerow(r)
                f.flush()
                rows.append(r)
                print(f"  {name} rep{rep}/{REPS}: ep={r['episode']} "
                      f"homo={r['homo']*100:.1f}% het={r['hetero']*100:.1f}% "
                      f"({dt:.0f}s)")
            agg[name] = summarize(name, rows)

    # summary table
    print("\n=== SUMMARY (mean +/- std over reps) ===")
    print(f"{'scenario':>14} {'episode':>16} {'cum_inf':>16} {'extinct':>8} "
          f"{'homo%':>8} {'bi%':>7} {'het%':>7} {'fem%':>7} {'oneoff':>7} {'sp':>6}")
    for name, _, _, _, _ in scenarios:
        s = agg[name]
        print(f"{s['name']:>14} {s['ep_mean']:>7.0f}±{s['ep_std']:>5.0f} "
              f"{s['ci_mean']:>7.0f}±{s['ci_std']:>5.0f} {s['extinct']:>4}/{REPS} "
              f"{s['homo']:>7.1f} {s['bi']:>6.1f} {s['hetero']:>6.1f} {s['female']:>6.1f} "
              f"{s['oneoff_contrib']:>6.0f} {s['sex_party_contrib']:>5.0f}")

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
