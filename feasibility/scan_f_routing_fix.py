"""Sweep f factor to pull total back to ~2280 while keeping the routing fix.

The 2026-11-04 routing fix (oneoff_bias[4]=1.0, mix_mean=8.0, anal=0.30) aligns
homo/bi/hetero composition but total is ~17% high (2663 vs 2280). f scales the
one-off mean (0.5*f) and party heavy-tail top mean (1.0*f) jointly, which mainly
compresses the MSM core (homo) — the bulk of total — without directly touching
the MIX pool. This sweep finds the f that restores total~2280 while confirming
bi/hetero stay aligned.

Outputs: outputs/sensitivity/f_sweep_routing_fix.csv
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


def run_one(seed, factor):
    oneoff_mean = 0.5 * factor
    oneoff_p = 1.0 / (1.0 + oneoff_mean)
    top_mean = 1.0 * factor

    p = dict(BASE_PARAMS)
    p["oneoff_p"] = oneoff_p
    p["party_ht_top_mean"] = top_mean

    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"fsweepfix_{factor:.2f}_{seed}",
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
    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    c_rmse = float(np.sqrt(np.mean((np.cumsum(sim) - np.cumsum(obs)) ** 2)))
    s = pop.summary() if hasattr(pop, 'summary') else {}
    known = float(sim.sum())
    fh = s.get('episode_female_hetero', float('nan'))
    fh_ratio = float(fh / known) if known > 0 else float('nan')

    row = dict(
        factor=factor, seed=seed,
        total=float(sim.sum()), peak=int(np.argmax(sim)),
        cRMSE=c_rmse,
        homo=s.get('episode_homo_ratio', float('nan')),
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        female_hetero=fh_ratio,
        seconds=round(time.time() - t0),
    )
    print(f"f={factor:5.2f} total={row['total']:6.0f} peak={row['peak']:3d} "
          f"homo={row['homo']:.3f} bi={row['bi']:.3f} hetero={row['hetero']:.3f} "
          f"female={row['female']:.3f} fhet={fh_ratio:.3f}", flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    factors = [5.5, 6.0, 6.5, 7.0]
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "f_sweep_routing_fix.csv")

    fieldnames = ["factor", "seed", "total", "peak", "cRMSE", "homo", "bi",
                  "hetero", "female", "female_hetero", "seconds"]
    rows = []
    for f in factors:
        rows.append(run_one(seed, f))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== TARGETS: total=2280 peak=92 homo=77.5% bi=12.6% hetero=9.9% female=3.7% ===")
    print(f"saved to {out_csv}")
