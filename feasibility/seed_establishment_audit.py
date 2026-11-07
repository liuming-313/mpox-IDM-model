"""
Seed-establishment audit (corrected, minimal, read-only w.r.t. main.py/population.py).

Scientific question:
    If we REMOVE the 742 scheduled external importations and instead seed a
    small number of high-activity MSM (homosexual male, MCP) cases, does the
    sex_party engine (the dominant transmission layer, ~66.5% of infections)
    self-sustain the outbreak?

Design (deliberately minimal — NO monkeypatching of one-off or any layer):
    - Keep the model EXACTLY as-is (params/policy/disease/network structure).
    - Remove importations (empty schedule).
    - Seed N high-activity MSM cases via pop.seed_infection(..., mask=homo_male_mcp).
    - Vary N in {5, 10, 20, 50, 100} across several stochastic seeds.
    - Report: extinction probability, final episode cases, layer attribution,
      and orientation composition.

This isolates the true question: is the network's endogenous amplification
sufficient, or does the current fit depend on the 742 external imports?

Corrected bugs vs. the earlier audit_endogenous_amplification.py:
    1. Do NOT `import main` (main.py executes a full 10M run loop on import).
       Instead parse params via exec of the module head (no run loop).
    2. Seed via pop.seed_infection (full disease-timing init), not _expose_batch.
    3. No monkeypatch of _update_oneoff_layer (removes noise/duplicate-edge bug).
    4. Set np.random.seed before seed_infection (it uses global RNG).
"""

import os
import sys
import csv
import time
import copy

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")

from population import Population


def load_main_params():
    """Extract params dict from main.py WITHOUT executing its run loop."""
    main_path = os.path.join(REPO, "main.py")
    src = open(main_path, "r", encoding="utf-8").read()
    cut = src.index("SCENARIOS = [")
    head = src[:cut]
    ns = {}
    exec(compile(head, main_path, "exec"), ns)
    return ns["params"]


# Full 3-phase policy, reproduced verbatim from main.py's SCENARIOS loop.
POLICY = [
    dict(name='phase1_growth', start=0, end=91,
         contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95,
         social_mult=0.90, test_mult=0.70, trace_mult=0.80,
         test_pathway_prob=0.75, test_seek_delay_mean_days=5.0,
         test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
    dict(name='phase2_control', start=92, end=170,
         contact_mult=0.65, oneoff_mult=0.65, party_mult=0.42,
         social_mult=1.05, test_mult=1.50, trace_mult=1.20,
         test_pathway_prob=0.90, test_seek_delay_mean_days=4.0,
         test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
    dict(name='phase3_postpeak', start=171, end=244,
         contact_mult=0.70, oneoff_mult=0.85, party_mult=0.47,
         social_mult=1.20, test_mult=1.20, trace_mult=1.00,
         test_pathway_prob=0.85, test_seek_delay_mean_days=4.5,
         test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
]


def homo_male_mcp_mask(pop):
    g = pop.genders
    o = pop.orientations
    mcp = pop.mcp_flags
    return (g == 0) & (o == 'homo') & mcp


def run_one(pop_size, days, seed, n_seeds, params, label):
    params = copy.deepcopy(params)
    params["n_days"] = days + 1

    pop = Population(n=pop_size, params=params, seed=seed,
                     cache_label=label,
                     policy=POLICY,
                     importations=[])   # <-- NO external importations

    # seed a small number of high-activity MSM
    mask = homo_male_mcp_mask(pop)
    elig = np.where(mask)[0]
    n = int(min(n_seeds, elig.size))
    np.random.seed(seed)          # seed_infection uses global np.random
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(n, mask=mask)

    t0 = time.time()
    pop.run_simulation(days)
    dt = time.time() - t0

    summ = pop.summary()
    # episode-date cases (the calibration-facing case count)
    ep = int(np.sum(pop.daily_episode_cases)) if hasattr(pop, "daily_episode_cases") else 0
    # cumulative infections
    cum = int(np.sum(pop.infections_by_day["total"]))
    # layer attribution
    layer_counts = dict(pop.layer_infection_counts)

    # orientation composition of episode cases
    ori = pop.daily_episode_cases_by_ori
    ot = np.sum(ori, axis=0).astype(float)
    tot_ori = float(ot.sum())
    het = float(ot[0] / tot_ori) if tot_ori else 0.0
    bi = float(ot[1] / tot_ori) if tot_ori else 0.0
    hom = float(ot[2] / tot_ori) if tot_ori else 0.0

    # sex composition (female fraction)
    sex = pop.daily_episode_cases_by_sex
    st = np.sum(sex, axis=0).astype(float)
    female = float(st[1] / st.sum()) if st.sum() else 0.0

    return {
        "seed": seed, "n_seeds": n, "episode": ep, "cum_inf": cum,
        "hetero": het, "bi": bi, "homo": hom, "female": female,
        "layer_counts": layer_counts, "runtime_s": dt,
    }


def main():
    POP = int(os.environ.get("SEED_POP", "2000000"))
    DAYS = 244
    BASE_SEED = int(os.environ.get("SEED_BASE", "100000"))
    REPS = int(os.environ.get("SEED_REPS", "5"))
    _sc = os.environ.get("SEED_COUNTS", "")
    seed_counts = ([int(x) for x in _sc.split(",") if x.strip().isdigit()]
                   if _sc.strip() else [5, 10, 20, 50, 100])

    params = load_main_params()
    label = f"seedaudit_{POP // 1_000_000}M"

    print(f"=== seed-establishment audit: pop={POP}, days={DAYS}, reps={REPS} ===")
    print(f"seed counts: {seed_counts}")
    print(f"params version: {params.get('simplified_parameter_version')}")

    rows = []
    for n_seeds in seed_counts:
        print(f"\n--- N_seeds = {n_seeds} ---")
        ep_list, cum_list, ext = [], [], 0
        for r in range(REPS):
            seed = BASE_SEED + r
            res = run_one(POP, DAYS, seed, n_seeds, params, label)
            ep_list.append(res["episode"])
            cum_list.append(res["cum_inf"])
            extinct = res["cum_inf"] <= (2 * n_seeds)
            if extinct:
                ext += 1
            print(f"  seed={seed}: episode={res['episode']}, cum_inf={res['cum_inf']}, "
                  f"homo={res['homo']:.3f} bi={res['bi']:.3f} hetero={res['hetero']:.3f} "
                  f"female={res['female']:.3f} {'EXTINCT' if extinct else 'SUSTAINED'} "
                  f"({res['runtime_s']:.0f}s)")
            rows.append(res)

        print(f"  -> episode mean={np.mean(ep_list):.0f} (range {min(ep_list)}-{max(ep_list)}), "
              f"cum_inf mean={np.mean(cum_list):.0f}, extinction={ext}/{REPS}")

    # save CSV (layer counts flattened into columns)
    os.makedirs(os.path.join(REPO, "outputs", "feasibility"), exist_ok=True)
    out = os.path.join(REPO, "outputs", "feasibility", "seed_establishment_audit.csv")
    layers = ['stable', 'household_close', 'social', 'oneoff', 'sex_party', 'sex_party_touch']
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["n_seeds", "seed", "episode", "cum_inf", "hetero", "bi", "homo",
                    "female", "runtime_s"] + layers)
        for r in rows:
            lc = r["layer_counts"]
            w.writerow([r["n_seeds"], r["seed"], r["episode"], r["cum_inf"],
                        f"{r['hetero']:.4f}", f"{r['bi']:.4f}", f"{r['homo']:.4f}",
                        f"{r['female']:.4f}", f"{r['runtime_s']:.0f}"]
                       + [lc.get(l, 0) for l in layers])
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
