"""Diagnostic sweep of sexual-orientation routing parameters.

Target: fix bi overshoot (18% vs 12.6%) and hetero undershoot (4.8% vs 9.9%),
female overshoot (5.3% vs 3.7%).

Mechanism: bi/hetero composition is governed by (a) how bisexual men are routed
into the MSM core vs MIX pool (`p_bi_m_to_msm`), and (b) the size / MF-contact
intensity of the MIX party pool (`party_mix_target_mean`, `mix_mf_anal_frac`).

This sweep varies one parameter at a time across a plausible range and reports
total / peak / homo / bi / hetero / female so we can see the direction and
magnitude of each lever before committing to a recalibration.

Outputs: outputs/sensitivity/orientation_sweep.csv
"""
import os, sys, io, time, csv, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population

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


# (parameter, [levels]) — baseline value included where relevant
SWEEPS = [
    ("p_bi_m_to_msm", [0.45, 0.55, 0.65]),
    ("party_mix_target_mean", [1.5, 2.0, 2.5, 3.0]),
    ("mix_mf_anal_frac", [0.10, 0.13, 0.18, 0.22]),
    ("party_msm_attend_prob", [0.30, 0.40, 0.50]),
]


def run_one(param_name, value, seed):
    p = dict(BASE_PARAMS)
    p[param_name] = value
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"ori_{param_name}_{value}_{seed}",
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
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        seconds=round(time.time() - t0),
    )
    print(f"{param_name}={value} -> total={row['total']:6.0f} peak={row['peak']:3d} "
          f"homo={row['homo']:.3f} bi={row['bi']:.3f} hetero={row['hetero']:.3f} "
          f"female={row['female']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "orientation_sweep.csv")

    fieldnames = ["parameter", "value", "seed", "total", "peak",
                  "homo", "bi", "hetero", "female", "seconds"]
    rows = []

    # baseline (no override)
    rows.append(run_one("_baseline_", None, seed))

    for param_name, levels in SWEEPS:
        for v in levels:
            rows.append(run_one(param_name, v, seed))
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    print(f"\nsaved to {out_csv}", flush=True)
