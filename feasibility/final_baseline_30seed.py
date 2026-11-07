"""Formal final baseline: run N seeds with the locked main.py configuration.

This is the authoritative baseline for the Results table. It imports the
final calibrated params/policy directly from main.py (so there is no risk of
parameter drift) and runs N seeds, reporting median/IQR of:
  - total episodes (target 2280)
  - peak day (observed 92)
  - cRMSE vs observed cumulative curve
  - composition (homo / bi / hetero / female)

Output: outputs/final_baseline/final_baseline_Nseeds.csv
"""
import os, sys, io, time, csv, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population, Case_data

# --- 安全提取 main.py 的 params/policy/常量（避免 import main 触发顶层模拟） ---
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
                     cache_label=f"finalbl_{seed}", policy=policy,
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

    row = dict(
        seed=seed,
        total=float(sim.sum()),
        peak=int(np.argmax(sim)),
        peak_height=float(sim.max()),
        first_case=int(np.where(sim > 0)[0][0]) if (sim > 0).any() else -1,
        last_case=int(np.where(sim > 0)[0][-1]) if (sim > 0).any() else -1,
        cRMSE=c_rmse,
        homo=s.get('episode_homo_ratio', float('nan')),
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        seconds=round(time.time() - t0),
    )
    print(f"seed={seed} total={row['total']:6.0f} peak={row['peak']:3d} "
          f"cRMSE={c_rmse:6.1f} homo={row['homo']:.3f} hetero={row['hetero']:.3f} "
          f"female={row['female']:.3f}", flush=True)
    return row


def pct(vals):
    a = np.asarray(vals, dtype=float)
    return dict(median=float(np.median(a)), mean=float(np.mean(a)),
                std=float(np.std(a)),
                q25=float(np.percentile(a, 25)), q75=float(np.percentile(a, 75)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()

    seeds = [300 + i * 11 for i in range(args.seeds)]

    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "final_baseline"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"final_baseline_{args.seeds}seeds.csv")

    fieldnames = ["seed", "total", "peak", "peak_height", "first_case",
                  "last_case", "cRMSE", "homo", "bi", "hetero", "female", "seconds"]

    print(f"=== FINAL BASELINE ({args.seeds} seeds, main.py locked config) ===", flush=True)
    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    print(f"observed: total={int(obs.sum())}, peak_day={int(np.argmax(obs))}, "
          f"peak_height={obs.max():.0f}", flush=True)

    rows = []
    for seed in seeds:
        rows.append(run_one(seed))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print("-" * 70, flush=True)
    for name in ["total", "peak", "peak_height", "cRMSE", "homo", "bi",
                 "hetero", "female"]:
        st = pct([r[name] for r in rows if r[name] == r[name]])
        print(f"{name:>11}: median={st['median']:8.1f}  mean={st['mean']:8.1f}±{st['std']:6.1f}  "
              f"IQR=[{st['q25']:.1f}, {st['q75']:.1f}]", flush=True)

    print(f"\nsummary: total median={pct([r['total'] for r in rows])['median']:.0f} "
          f"(target 2280)", flush=True)
    print(f"saved to {out_csv}", flush=True)
