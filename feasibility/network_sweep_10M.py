"""
Overnight 10M network-structure sweep (heavy-tail on).

Runs a small, carefully chosen grid directly at 10M so the nonlinear
population scaling is captured exactly. Each combo ~8 min; 12 combos ~96 min.

Grid rationale (all with oneoff_activity_stratified=True):
  oneoff_active_frac: 0.01 (baseline anchor), 0.02, 0.04
  oneoff_p:          0.15 (baseline), 0.10, 0.06  (lower p -> more partners/active)
  party_core_intensity: 0.7 (baseline)  [kept fixed to isolate one-off effect]

A few combos also vary party_core_intensity to probe whether the sex_party
engine (the current dominant layer) benefits from a stronger core.

Writes each row to CSV as it completes; a summary table is printed at the end.
"""
import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = 10_000_000
SIM_DAYS = 244
BASE_SEED = 42
N_SEEDS = 20

OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "feasibility", "network_sweep_10M.csv")


def base_params():
    return {
        "oneoff_activity_stratified": True,
        "oneoff_mf_anal_frac": 0.10,
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


def run_one(oneoff_active_frac, oneoff_p, party_core_intensity, name):
    params = base_params()
    params["oneoff_active_frac"] = oneoff_active_frac
    params["oneoff_p"] = oneoff_p
    params["party_core_intensity"] = party_core_intensity

    seed = BASE_SEED
    label = f"sw10M_{name}"

    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=label, policy=make_policy(), importations=[])

    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(N_SEEDS, mask=mask)

    t0 = time.time()
    pop.run_simulation(SIM_DAYS)
    dt = time.time() - t0

    summ = pop.summary()
    cum_inf = int(summ.get("cum_inf_homo", 0) + summ.get("cum_inf_bi", 0) + summ.get("cum_inf_hetero", 0))
    episode = int(np.sum(pop.daily_episode_cases)) if hasattr(pop, "daily_episode_cases") else 0
    layers = dict(getattr(pop, "layer_infection_counts", {}))

    return {
        "name": name,
        "oneoff_active_frac": oneoff_active_frac,
        "oneoff_p": oneoff_p,
        "party_core_intensity": party_core_intensity,
        "cum_inf": cum_inf,
        "episode": episode,
        "homo": float(summ.get("episode_homo_ratio", summ.get("cum_inf_homo_ratio", 0))),
        "bi": float(summ.get("episode_bi_ratio", summ.get("cum_inf_bi_ratio", 0))),
        "hetero": float(summ.get("episode_hetero_ratio", summ.get("cum_inf_hetero_ratio", 0))),
        "female": float(summ.get("episode_female_known_ratio", summ.get("cum_inf_female_known_ratio", 0))),
        "oneoff_contrib": int(layers.get("oneoff", 0)),
        "sex_party_contrib": int(layers.get("sex_party", 0)),
        "stable_contrib": int(layers.get("stable", 0)),
        "social_contrib": int(layers.get("social", 0)),
        "dt_s": round(dt, 1),
    }


def main():
    # (oneoff_active_frac, oneoff_p, party_core_intensity, name)
    grid = [
        (0.01, 0.15, 0.7, "af01_p15_pc07"),   # baseline + heavy-tail
        (0.02, 0.15, 0.7, "af02_p15_pc07"),
        (0.04, 0.15, 0.7, "af04_p15_pc07"),
        (0.01, 0.10, 0.7, "af01_p10_pc07"),
        (0.02, 0.10, 0.7, "af02_p10_pc07"),
        (0.04, 0.10, 0.7, "af04_p10_pc07"),
        (0.01, 0.06, 0.7, "af01_p06_pc07"),
        (0.02, 0.06, 0.7, "af02_p06_pc07"),
        (0.04, 0.06, 0.7, "af04_p06_pc07"),
        # probe sex_party core intensity (party_core_intensity 0.7 -> 0.9)
        (0.02, 0.10, 0.9, "af02_p10_pc09"),
        (0.04, 0.06, 0.9, "af04_p06_pc09"),
        # mild anchor with very high partner count
        (0.06, 0.04, 0.7, "af06_p04_pc07"),
    ]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["name", "oneoff_active_frac", "oneoff_p", "party_core_intensity",
                  "cum_inf", "episode", "homo", "bi", "hetero", "female",
                  "oneoff_contrib", "sex_party_contrib", "stable_contrib",
                  "social_contrib", "dt_s"]

    write_header = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    print(f"10M overnight sweep: {len(grid)} combos, POP=10M, DAYS=244, NSEEDS=20")
    print(f"Output -> {OUT_CSV}\n")

    n_done = 0
    t_start = time.time()
    for af, p, pc, name in grid:
        try:
            row = run_one(af, p, pc, name)
            writer.writerow(row)
            f.flush()
            n_done += 1
            print(f"[{n_done}/{len(grid)}] {name}: cum_inf={row['cum_inf']} "
                  f"episode={row['episode']} homo={row['homo']:.3f} "
                  f"hetero={row['hetero']:.3f} female={row['female']:.3f} "
                  f"oneoff={row['oneoff_contrib']} sp={row['sex_party_contrib']} "
                  f"({row['dt_s']}s)")
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    f.close()
    print(f"\n10M sweep done in {(time.time()-t_start)/60:.1f} min. Rows={n_done}")


if __name__ == "__main__":
    main()
