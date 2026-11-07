"""Focused sweep of oneoff_orientation_bias — the real lever for bi/hetero split.

Diagnosis: `p_bi_m_to_msm` (sex_party routing) barely moves bi (17.8-19.4%).
The dominant pathway for bi-male infection is the one-off (one-time partner)
layer, where `oneoff_orientation_bias[4]=2.0` gives male-bi 10x the one-off
participation weight of male-hetero (0.20). This sweep varies that weight
(and the male-hetero weight) to find the bi/hetero/female rebalance.

sexcat: 0=FH, 1=MH, 2=MSM, 3=WSW, 4=MB, 5=FB.

Outputs: outputs/sensitivity/oneoff_bias_sweep.csv
"""
import os, sys, io, time, csv, contextlib, copy
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


def _set_bias(params, mb, mh):
    """Override oneoff_orientation_bias male-bi (4) and male-hetero (1) weights."""
    bias = dict(params.get("oneoff_orientation_bias", {}))
    bias[4] = float(mb)
    bias[1] = float(mh)
    params["oneoff_orientation_bias"] = bias
    return params


# (label, male_bi_weight, male_hetero_weight)
SWEEPS = [
    ("baseline",      2.00, 0.20),
    ("bi1.5",         1.50, 0.20),
    ("bi1.0",         1.00, 0.20),
    ("bi0.5",         0.50, 0.20),
    ("bi0.3",         0.30, 0.20),
    ("bi0.5_het0.4",  0.50, 0.40),
    ("bi0.5_het0.6",  0.50, 0.60),
    ("bi0.3_het0.6",  0.30, 0.60),
]


_SWEEP_MAP = {s[0]: (s[1], s[2]) for s in SWEEPS}


def run_one(label, seed):
    p = dict(BASE_PARAMS)
    if label == "baseline":
        pass
    else:
        mb, mh = _SWEEP_MAP[label]
        p = _set_bias(p, mb, mh)
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"oneoffbias_{label}_{seed}",
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
        label=label, seed=seed,
        total=float(sim.sum()), peak=int(np.argmax(sim)),
        homo=s.get('episode_homo_ratio', float('nan')),
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        seconds=round(time.time() - t0),
    )
    print(f"{label:16s} -> total={row['total']:6.0f} peak={row['peak']:3d} "
          f"homo={row['homo']:.3f} bi={row['bi']:.3f} hetero={row['hetero']:.3f} "
          f"female={row['female']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "oneoff_bias_sweep.csv")

    fieldnames = ["label", "seed", "total", "peak", "homo", "bi", "hetero",
                  "female", "seconds"]
    rows = []
    for label, mb, mh in SWEEPS:
        rows.append(run_one(label, seed))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== TARGETS: bi=12.6% hetero=9.9% female=3.7% homo=77.5% ===", flush=True)
    print(f"saved to {out_csv}", flush=True)
