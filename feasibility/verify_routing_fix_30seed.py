"""30-seed verification of the new bi/hetero routing calibration.

Applies the 2026-11-04 fix:
  - oneoff_orientation_bias[4] 2.0 -> 1.0  (bi root-cause fix)
  - party_mix_target_mean 2.0 -> 8.0       (MIX hetero bridge)
  - mix_mf_anal_frac 0.13 -> 0.30          (MF anal upgrade)

These are already written into main.py; this script reads them via eval (same
safe extraction as final_baseline_30seed.py) and runs 30 seeds, reporting
total / peak / homo / bi / hetero / female (+ female_hetero for the cleaner
target alignment) to confirm composition stability.

Output: outputs/sensitivity/routing_fix_30seed.csv
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
params = eval(_src[_ps:_pe].rstrip(),
              {"__builtins__": {}, "SIM_DAYS": SIM_DAYS, "CALIBRATION_TARGETS": _CT}, {})
_pls = _src.find("policy = [")
_ple = _src.find("]\n", _pls) + 1
policy = eval(_src[_pls + len("policy = "):_ple],
              {"__builtins__": {}, "dict": dict, "SIM_DAYS": SIM_DAYS}, {})


def run_one(seed):
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=dict(params), seed=seed,
                     cache_label=f"routefix_{seed}", policy=policy,
                     importations=[])
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

    # female_hetero (cleaner target = 74/2012 = 3.68%)
    fh = s.get('episode_female_hetero', float('nan'))
    known = float(sim.sum())
    fh_ratio = float(fh / known) if known > 0 else float('nan')

    row = dict(
        seed=seed,
        total=float(sim.sum()),
        peak=int(np.argmax(sim)),
        peak_height=float(sim.max()),
        cRMSE=c_rmse,
        homo=s.get('episode_homo_ratio', float('nan')),
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        female_hetero=fh_ratio,
        seconds=round(time.time() - t0),
    )
    print(f"seed={seed} total={row['total']:6.0f} peak={row['peak']:3d} "
          f"homo={row['homo']:.3f} bi={row['bi']:.3f} hetero={row['hetero']:.3f} "
          f"female={row['female']:.3f} fhet={fh_ratio:.3f}", flush=True)
    return row


def pct(vals):
    a = np.asarray(vals, dtype=float)
    return dict(median=float(np.median(a)), mean=float(np.mean(a)),
                q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)))


if __name__ == "__main__":
    seeds = [300 + i * 11 for i in range(30)]
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "routing_fix_30seed.csv")

    fieldnames = ["seed", "total", "peak", "peak_height", "cRMSE", "homo", "bi",
                  "hetero", "female", "female_hetero", "seconds"]

    print("=== ROUTING FIX 30-SEED VERIFICATION ===", flush=True)
    print("params: oneoff_bias[4]=1.0, mix_target_mean=8.0, mix_mf_anal_frac=0.30", flush=True)
    print("targets: total=2280 peak=92 homo=77.5% bi=12.6% hetero=9.9% female=3.7%", flush=True)

    rows = []
    for seed in seeds:
        rows.append(run_one(seed))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print("-" * 70, flush=True)
    for name in ["total", "peak", "peak_height", "cRMSE", "homo", "bi",
                 "hetero", "female", "female_hetero"]:
        st = pct([r[name] for r in rows if r[name] == r[name]])
        print(f"{name:>14}: median={st['median']:8.2f}  mean={st['mean']:8.3f}  "
              f"IQR=[{st['q25']:.3f}, {st['q75']:.3f}]", flush=True)

    print(f"\nsaved to {out_csv}", flush=True)
