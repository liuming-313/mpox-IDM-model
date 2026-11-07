"""
Overnight network-structure optimization sweep.

Goal: after adding heavy-tail activity stratification to the one-off layer,
systematically explore network-structure parameters to raise endogenous
amplification toward the ~2280-case target (at 10M) while preserving the
MSM-dominant case mix (homo ~77.5%, hetero ~9.9%, female ~3.7%).

Explored dimensions (orthogonal, on top of oneoff_activity_stratified=True):
  1. oneoff_active_frac  — anchor for total one-off contact volume (higher -> more)
  2. oneoff_p           — NegBin(1,p) partner count per active day (lower -> more partners)
  3. oneoff_age_gap     — age-compatibility window (wider -> more eligible partners)

Each combination runs at a reduced population + horizon so the full grid
finishes overnight; every row is appended to CSV as it completes so partial
results survive a crash. A final summary table is printed at the end.

Run controls (env vars):
  SW_POP      population size (default 2_000_000)
  SW_DAYS     horizon days (default 180)
  SW_SEED     base RNG seed (default 42)
  SW_NSEEDS   number of homo_male_mcp index cases (default 20)
  SW_REPS     reps per combination (default 1; set 3 for variance)
"""
import os
import sys
import csv
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population

POP_SIZE = int(os.environ.get("SW_POP", 2_000_000))
SIM_DAYS = int(os.environ.get("SW_DAYS", 180))
BASE_SEED = int(os.environ.get("SW_SEED", 42))
N_SEEDS = int(os.environ.get("SW_NSEEDS", 20))
REPS = int(os.environ.get("SW_REPS", 1))

OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "feasibility", "network_sweep.csv")


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


def run_one(oneoff_active_frac, oneoff_p, age_gap, rep):
    params = base_params()
    params["oneoff_active_frac"] = oneoff_active_frac
    params["oneoff_p"] = oneoff_p
    params["oneoff_age_gap"] = age_gap

    seed = BASE_SEED + rep * 100003
    label = f"sw_{oneoff_active_frac}_{oneoff_p}_{age_gap}_{rep}"

    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=label, policy=POLICY, importations=[])

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
    # episode-date case total (the quantity calibrated against the observed 2280)
    if hasattr(pop, "daily_episode_cases"):
        episode = int(np.sum(pop.daily_episode_cases))
    else:
        episode = 0
    layers = dict(getattr(pop, "layer_infection_counts", {}))

    row = {
        "pop": POP_SIZE,
        "days": SIM_DAYS,
        "seed": seed,
        "n_seeds": N_SEEDS,
        "rep": rep,
        "oneoff_active_frac": oneoff_active_frac,
        "oneoff_p": oneoff_p,
        "age_gap": age_gap,
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
    return row


def main():
    # Grid: on top of heavy-tail stratification.
    # Keep it small enough to finish overnight at 10M (each ~8 min).
    # Env overrides let you run a fast 2M pre-screen or a full 10M grid.
    active_fracs = [float(x) for x in os.environ.get("SW_AFRACS", "0.01,0.02,0.04").split(",")]
    oneoff_ps = [float(x) for x in os.environ.get("SW_PS", "0.15,0.10,0.06").split(",")]
    age_gaps = [int(x) for x in os.environ.get("SW_GAPS", "80").split(",")]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["pop", "days", "seed", "n_seeds", "rep",
                  "oneoff_active_frac", "oneoff_p", "age_gap",
                  "cum_inf", "episode", "homo", "bi", "hetero", "female",
                  "oneoff_contrib", "sex_party_contrib", "stable_contrib",
                  "social_contrib", "dt_s"]

    write_header = not os.path.exists(OUT_CSV)
    f = open(OUT_CSV, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    combos = [(af, p, ag) for af in active_fracs for p in oneoff_ps for ag in age_gaps]
    print(f"Sweep: {len(combos)} combos x {REPS} reps, POP={POP_SIZE}, DAYS={SIM_DAYS}")
    print(f"Output -> {OUT_CSV}\n")

    n_done = 0
    t_start = time.time()
    for af, p, ag in combos:
        for rep in range(REPS):
            try:
                row = run_one(af, p, ag, rep)
                writer.writerow(row)
                f.flush()
                n_done += 1
                print(f"[{n_done}/{len(combos)*REPS}] af={af} p={p} gap={ag} rep={rep} "
                      f"-> cum_inf={row['cum_inf']} episode={row['episode']} "
                      f"homo={row['homo']:.3f} hetero={row['hetero']:.3f} "
                      f"female={row['female']:.3f} oneoff={row['oneoff_contrib']} "
                      f"({row['dt_s']}s)")
            except Exception as e:
                print(f"[FAIL] af={af} p={p} gap={ag} rep={rep}: {type(e).__name__}: {e}")

    f.close()
    print(f"\nSweep done in {(time.time()-t_start)/60:.1f} min. Rows={n_done}")


if __name__ == "__main__":
    main()
