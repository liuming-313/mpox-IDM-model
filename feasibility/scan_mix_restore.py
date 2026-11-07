"""Round-2 sweep: fix oneoff bi weight at 1.0 (hits bi target), then lift MIX pool
to restore hetero bridging toward 9.9% without collapsing total.

Round-1 finding: oneoff_orientation_bias[4] 2.0->1.0 nails bi=12.4% but drags
hetero to 2.8% (bi men are the main hetero bridge in the one-off layer). Raising
male-hetero weight in oneoff collapses total (low-transmission vaginal edges
crowd out MSM). So hetero must be restored through the dedicated sex_party MIX
pool, which does not compete with the MSM pool.

Sweep MIX knobs (with bi weight fixed at 1.0):
  - party_mix_target_mean  (2.0 -> 4.0/6.0/8.0): mean partners per MIX attendee
  - mix_mf_anal_frac       (0.13 -> 0.20/0.30): MF anal upgrade fraction
  - party_mix_attend_prob  (base -> 0.60): MIX attendance (currently 3/7)

Outputs: outputs/sensitivity/mix_restore_sweep.csv
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


def _set_bi1(params):
    bias = dict(params.get("oneoff_orientation_bias", {}))
    bias[4] = 1.0
    params["oneoff_orientation_bias"] = bias
    return params


def _override(params, name, value):
    params[name] = value
    return params


# (label, overrides) — bi weight fixed at 1.0 in all
SWEEPS = [
    ("bi1_base",    {}),                              # bi=1.0, MIX unchanged
    ("bi1_mix4",    {"party_mix_target_mean": 4.0}),
    ("bi1_mix6",    {"party_mix_target_mean": 6.0}),
    ("bi1_mix8",    {"party_mix_target_mean": 8.0}),
    ("bi1_anal20",  {"mix_mf_anal_frac": 0.20}),
    ("bi1_anal30",  {"mix_mf_anal_frac": 0.30}),
    ("bi1_att60",   {"party_mix_attend_prob": 0.60}),
    ("bi1_mix6_anal30", {"party_mix_target_mean": 6.0, "mix_mf_anal_frac": 0.30}),
    ("bi1_mix8_anal30", {"party_mix_target_mean": 8.0, "mix_mf_anal_frac": 0.30}),
]


def run_one(label, overrides, seed):
    p = dict(BASE_PARAMS)
    p = _set_bi1(p)
    for k, v in overrides.items():
        p = _override(p, k, v)
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"mixrestore_{label}_{seed}",
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
    print(f"{label:18s} -> total={row['total']:6.0f} peak={row['peak']:3d} "
          f"homo={row['homo']:.3f} bi={row['bi']:.3f} hetero={row['hetero']:.3f} "
          f"female={row['female']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "mix_restore_sweep.csv")

    fieldnames = ["label", "seed", "total", "peak", "homo", "bi", "hetero",
                  "female", "seconds"]
    rows = []
    for label, overrides in SWEEPS:
        rows.append(run_one(label, overrides, seed))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== TARGETS: bi=12.6% hetero=9.9% female=3.7% homo=77.5% total=2280 ===", flush=True)
    print(f"saved to {out_csv}", flush=True)
