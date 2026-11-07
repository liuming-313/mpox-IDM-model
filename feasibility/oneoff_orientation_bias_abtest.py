"""
A/B test: orientation-stratified one-off participation.

Answers the key question from the overnight sweep: when one-off activity is
raised (to close the size gap toward 2280), the hetero/female mix explodes
because the one-off active pool is ~72% heterosexual by headcount.

Fix: bias one-off participation toward MSM / male-bi strata (CDC's one-time
network is effectively MSM-only). This test compares, on identical population
+ seed + policy + one-off activity settings:

  A) oneoff_orientation_bias = None        (uniform participation, old behavior)
  B) oneoff_orientation_bias = {...}       (MSM-biased participation)

across a few high one-off-activity settings where the old behavior collapses.
"""
import os
import sys
import time
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population

POP_SIZE = int(os.environ.get("OB_POP", 2_000_000))
SIM_DAYS = int(os.environ.get("OB_DAYS", 244))
BASE_SEED = int(os.environ.get("OB_SEED", 42))
N_SEEDS = int(os.environ.get("OB_NSEEDS", 20))

# sexcat: 0=FH, 1=MH, 2=MSM, 3=WSW, 4=MB, 5=FB
ORI_BIAS = {
    0: 0.15, 1: 0.20, 2: 3.00, 3: 0.30, 4: 2.00, 5: 0.40,
}


def load_params():
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


POLICY = [
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


def run(af, op, use_bias):
    params = load_params()
    params["oneoff_active_frac"] = af
    params["oneoff_p"] = op
    if use_bias:
        params["oneoff_orientation_bias"] = dict(ORI_BIAS)
    else:
        params["oneoff_orientation_bias"] = None
    label = f"ob_{'bias' if use_bias else 'nobias'}_{af}_{op}"

    pop = Population(n=POP_SIZE, params=params, seed=BASE_SEED,
                     cache_label=label, policy=POLICY, importations=[])

    g = pop.genders
    o = pop.orientations
    mcp = pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(BASE_SEED)
    pop._rng = np.random.default_rng(BASE_SEED)
    pop.seed_infection(N_SEEDS, mask=mask)

    t0 = time.time()
    pop.run_simulation(SIM_DAYS)
    dt = time.time() - t0

    summ = pop.summary()
    cum_inf = int(summ.get("cum_inf_total",
                   summ.get("cum_inf_homo", 0) + summ.get("cum_inf_bi", 0) + summ.get("cum_inf_hetero", 0)))

    return {
        "af": af, "op": op, "use_bias": use_bias,
        "cum_inf": cum_inf,
        "homo": float(summ.get("episode_homo_ratio", summ.get("cum_inf_homo_ratio", 0))),
        "bi": float(summ.get("episode_bi_ratio", summ.get("cum_inf_bi_ratio", 0))),
        "hetero": float(summ.get("episode_hetero_ratio", summ.get("cum_inf_hetero_ratio", 0))),
        "female": float(summ.get("episode_female_known_ratio", summ.get("cum_inf_female_known_ratio", 0))),
        "oneoff_contrib": int(getattr(pop, "layer_infection_counts", {}).get("oneoff", 0)),
        "dt": dt,
    }


def main():
    # high one-off settings where the old behavior collapsed in the overnight sweep
    combos = [
        (0.02, 0.10),   # moderate
        (0.04, 0.10),   # collapsed (hetero 58.5%)
        (0.04, 0.06),   # collapsed (hetero 68.5%)
    ]
    out_csv = "outputs/feasibility/oneoff_orientation_bias_abtest.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ["af", "op", "use_bias", "cum_inf", "homo", "bi", "hetero",
                  "female", "oneoff_contrib", "dt"]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"A/B: orientation bias. POP={POP_SIZE}, SEED={BASE_SEED}, NSEEDS={N_SEEDS}")
        for af, op in combos:
            for use_bias in (False, True):
                r = run(af, op, use_bias)
                writer.writerow(r)
                f.flush()
                tag = "bias  " if use_bias else "nobias"
                print(f"[{tag}] af={af} op={op} -> cum_inf={r['cum_inf']:>6} "
                      f"homo={r['homo']*100:5.1f}% bi={r['bi']*100:5.1f}% "
                      f"hetero={r['hetero']*100:5.1f}% female={r['female']*100:5.1f}% "
                      f"oneoff={r['oneoff_contrib']} ({r['dt']:.0f}s)")

    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
