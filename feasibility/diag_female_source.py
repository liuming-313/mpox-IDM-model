"""Diagnose the source-layer decomposition of female / male-hetero / bi cases.

Uses the txlog (successful-transmission log) to attribute each new infection to
the layer it came from, split by the *destination's* sexcat (0=FH, 1=MH, 2=MSM,
3=WSW, 4=MB, 5=FB). This answers: "where do female cases actually get infected?"
across ALL layers (stable / oneoff / sex_party / household_close / social), not
just the sex_party MIX pool.

layer_map (from population.py _fast_transmit):
  1=stable, 2=oneoff, 3=sex_party, 4=household_close, 5=sex_party_touch, 6=social

Outputs: outputs/sensitivity/female_source_diag.csv  (rows = layer x dest-sexcat)
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

LAYER_NAMES = {1: "stable", 2: "oneoff", 3: "sex_party", 4: "household_close",
               5: "sex_party_touch", 6: "social"}
SEXCAT_NAMES = {0: "FH", 1: "MH", 2: "MSM", 3: "WSW", 4: "MB", 5: "FB"}


def run_diag(seed):
    p = dict(BASE_PARAMS)
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"diagf_{seed}",
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

    n = int(pop.txlog_i)
    layer = pop.txlog_layer[:n]
    dst = pop.txlog_dst[:n]
    dst_sexcat = pop.sexcat[dst]  # destination's sexcat

    # cross-tab: layer x dest_sexcat
    table = {}
    for lc in sorted(np.unique(layer)):
        for sc in sorted(np.unique(dst_sexcat)):
            k = int(((layer == lc) & (dst_sexcat == sc)).sum())
            if k > 0:
                table[(int(lc), int(sc))] = k

    rows = []
    for (lc, sc), k in sorted(table.items()):
        rows.append({
            "layer": LAYER_NAMES.get(lc, str(lc)),
            "dest_sexcat": SEXCAT_NAMES.get(sc, str(sc)),
            "n": k,
        })
    print(f"[seed {seed}] total logged tx = {n}", flush=True)
    for r in rows:
        print(f"  {r['layer']:>16} -> {r['dest_sexcat']:>4} : {r['n']}", flush=True)
    return rows, n


if __name__ == "__main__":
    seed = 300
    rows, n = run_diag(seed)
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "female_source_diag.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer", "dest_sexcat", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved to {out_csv}", flush=True)
