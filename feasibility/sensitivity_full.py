"""One-way sensitivity analysis (full parameter set) for the mpox model.

Extends the earlier S8 (6 structural-assumption parameters) to the complete
set of uncertain parameters, organized into four categories:

  A. Natural history  (incubation, infectious period)
  B. Per-act sexual transmission (receptive/insertive anal, vaginal, oral)
  C. Nonsexual contact (social per-contact effective probability)
  D. Structural assumptions (the 6 already in S8)

Each parameter is varied one-at-a-time across 3 levels (low / baseline / high)
using the literature-derived sensitivity ranges. Baseline levels reproduce the
locked main.py configuration. 1 seed per level (seed=300), consistent with the
existing S8 precision — the goal is direction and magnitude, not exact medians.

Outputs: outputs/sensitivity/table_s8_full.csv
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

# ---- Sensitivity sweeps -----------------------------------------------------
# Each entry: (category, parameter_name, [low, baseline, high], level_labels)
# baseline value is read from BASE_PARAMS at runtime and asserted below.

SWEEPS = [
    # A. Natural history
    ("natural_history", "incubation_lognorm_mu", [2.020, 2.108, 2.196]),
    ("natural_history", "incubation_lognorm_sigma", [0.200, 0.254, 0.300]),
    ("natural_history", "infectious_period_gamma_scale", [1.56, 2.33, 3.00]),
    # B. Per-act sexual transmission (bounds from transmission_risk_parameter_bounds)
    ("per_act", "sexual_anal_receptive", [0.25, 0.28, 0.32]),
    ("per_act", "sexual_anal_insertive", [0.035, 0.06, 0.07]),
    ("per_act", "sexual_vaginal", [0.015, 0.035, 0.04]),
    ("per_act", "sexual_oral", [0.005, 0.0175, 0.025]),
    # C. Nonsexual contact
    ("nonsexual", "social_p_eff", [0.003, 0.0063, 0.010]),
    # D. Structural assumptions
    ("structural", "party_touch_frac", [0.10, 0.20, 0.25]),
    ("structural", "oneoff_mf_anal_frac", [0.05, 0.10, 0.20]),
    ("structural", "monitor_sexual_contact_mult", [0.15, 0.30, 0.60]),
    ("structural", "bi_same_sex_frac", [0.05, 0.10, 0.20]),
    ("structural", "social_same_sex_assort", [0.65, 0.80, 0.85]),
    # E. Final-calibration routing/bridging knobs (added 2026-11-04)
    ("structural", "mix_mf_anal_frac", [0.05, 0.15, 0.30]),
    ("structural", "mix_fm_anal_frac", [0.00, 0.05, 0.15]),
    ("structural", "party_mix_target_mean", [4.0, 8.0, 12.0]),
    # Nested dict param: oneoff_orientation_bias[4] (bisexual-male weight),
    # the root cause of the bi-overestimate (2.0 -> 1.0). Swept via a special
    # handler in run_one that mutates the [4] key only.
    ("structural", "oneoff_orientation_bias_bi", [0.5, 1.0, 2.0]),
]

# All parameters are re-run under the FINAL parameter set. The earlier S8
# (6-parameter) run used stale pre-recalibration parameters and is superseded.
SKIP_ALREADY_DONE = set()


def run_one(category, param_name, value, seed):
    p = dict(BASE_PARAMS)
    if param_name == "oneoff_orientation_bias_bi":
        # Nested dict: mutate only the bisexual-male weight (key 4).
        p["oneoff_orientation_bias"] = dict(p["oneoff_orientation_bias"])
        p["oneoff_orientation_bias"][4] = value
    else:
        p[param_name] = value

    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"sfull_{param_name}_{value}_{seed}",
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
        category=category, parameter=param_name, value=value, seed=seed,
        total=float(sim.sum()), peak=int(np.argmax(sim)),
        homo=s.get('episode_homo_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        seconds=round(time.time() - t0),
    )
    print(f"[{category}] {param_name}={value} -> total={row['total']:6.0f} "
          f"peak={row['peak']:3d} homo={row['homo']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "table_s8_final.csv")

    fieldnames = ["category", "parameter", "value", "seed", "total", "peak",
                  "homo", "hetero", "female", "seconds"]
    rows = []

    # baseline row
    rows.append(run_one("baseline", "_baseline_", None, seed))

    for category, param_name, levels in SWEEPS:
        if param_name in SKIP_ALREADY_DONE:
            print(f"[skip] {param_name} (already in S8)", flush=True)
            continue
        assert param_name in BASE_PARAMS or param_name == "oneoff_orientation_bias_bi", \
            f"{param_name} missing in params"
        for v in levels:
            rows.append(run_one(category, param_name, v, seed))
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)

    print(f"\nsaved to {out_csv}", flush=True)
