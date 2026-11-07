"""One-way sensitivity analysis (Table S8) for structural-assumption parameters.

Aligns with the FINAL locked main.py configuration (oneoff_p=0.222,
party_ht_top_mean=7.0, etc.) and varies one structural-assumption parameter at
a time across its declared sensitivity range, reporting total / peak / homo /
hetero / female for each level.

Parameters varied (from Supplementary S1.3, P1–P2):
  - party_touch_frac        [0.10, 0.20, 0.25]
  - oneoff_mf_anal_frac     [0.05, 0.10, 0.20]
  - monitor_sexual_contact_mult [0.15, 0.30, 0.60]
  - monitor_social_contact_mult [0.50, 0.70, 0.90]
  - bi_same_sex_frac        [0.05, 0.10, 0.20]
  - social_same_sex_assort  [0.65, 0.80, 0.85]

Default: 1 seed per level (300). Output CSV: outputs/sensitivity/table_s8.csv
"""
import os, sys, io, time, csv, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population, Case_data

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = open(os.path.join(_root, "main.py"), encoding="utf-8").read()

SIM_DAYS = 244
POPULATION_SIZE = 10_000_000
INITIAL_INFECTIONS = 20

_CT = {'target_homo': 1560/2012, 'target_bi': 253/2012,
       'target_hetero': (125+74)/2012, 'target_female_known': 74/2012,
       'target_homo_male': 1560/2012, 'target_bi_male': 253/2012}
_ps = _src.find("params = {") + len("params = ")
_pe = _src.find("params[", _ps)
BASE_PARAMS = eval(_src[_ps:_pe].rstrip(),
                   {"__builtins__": {}, "SIM_DAYS": SIM_DAYS, "CALIBRATION_TARGETS": _CT}, {})
_pls = _src.find("policy = [")
_ple = _src.find("]\n", _pls) + 1
POLICY = eval(_src[_pls + len("policy = "):_ple],
              {"__builtins__": {}, "dict": dict, "SIM_DAYS": SIM_DAYS}, {})

# (parameter name, [levels])
SWEEPS = [
    ("party_touch_frac", [0.10, 0.20, 0.25]),
    ("oneoff_mf_anal_frac", [0.05, 0.10, 0.20]),
    ("monitor_sexual_contact_mult", [0.15, 0.30, 0.60]),
    ("monitor_social_contact_mult", [0.50, 0.70, 0.90]),
    ("bi_same_sex_frac", [0.05, 0.10, 0.20]),
    ("social_same_sex_assort", [0.65, 0.80, 0.85]),
]


def run_one(param_name, value, seed):
    p = dict(BASE_PARAMS)
    if param_name in p:
        p[param_name] = value
    else:
        # some params live only inside nested dicts; fall back to no-op flag
        pass

    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"sens_{param_name}_{value}_{seed}",
                     policy=POLICY, importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.seed_infection(INITIAL_INFECTIONS, mask=mask)
        pop.run_simulation(SIM_DAYS)

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    s = pop.summary() if hasattr(pop, 'summary') else {}
    row = dict(
        parameter=param_name, value=value, seed=seed,
        total=float(sim.sum()), peak=int(np.argmax(sim)),
        homo=s.get('episode_homo_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        seconds=round(time.time() - t0),
    )
    print(f"{param_name}={value} -> total={row['total']:6.0f} peak={row['peak']:3d} "
          f"homo={row['homo']:.3f} hetero={row['hetero']:.3f} female={row['female']:.3f}",
          flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "table_s8.csv")

    fieldnames = ["parameter", "value", "seed", "total", "peak",
                  "homo", "hetero", "female", "seconds"]
    rows = []

    # baseline row (no override)
    rows.append(run_one("_baseline_", None, seed))

    for param_name, levels in SWEEPS:
        for v in levels:
            rows.append(run_one(param_name, v, seed))
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    print(f"\nsaved to {out_csv}", flush=True)
