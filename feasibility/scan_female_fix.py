"""Round-3 sweep: female structural fix via directional MF anal split + bug fix.

Background (2026-11-04):
  - homo/bi/hetero now aligned (bi=1.0 + mix_mean=8.0), but female = 8.2% vs
    target 3.7% (male hetero 4.1% vs target 6.2%).
  - Layer-source diagnosis (diag_female_source.py): 71.6% of female cases come
    from the sex_party MIX pool, where female-as-receptive (male->female) anal
    transmission uses beta_rec=0.28 while male-as-insertive uses beta_ins=0.06.
  - Fix applied to population.py:
      1) is_fm direction bug corrected (female->male now beta_ins, not beta_rec).
      2) mix_mf_anal_frac split into directional mix_mf_anal_frac (M->F, female
         receptive) and mix_fm_anal_frac (F->M, female insertive).
  - Literature: Thornhill Lancet 2022 (136 women) -> cis-women show vulvovaginal
    (not anorectal) lesions, so female receptive-anal should be LOW.

Sweep (with bi=1.0 + mix_mean=8.0 fixed; new directional anal params):
  - mix_mf_anal_frac: 0.15 / 0.10 / 0.05  (female receptive anal)
  - mix_fm_anal_frac: 0.05 / 0.10 / 0.15  (female insertive)

Outputs: outputs/sensitivity/female_fix_sweep.csv
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


# (label, overrides) — bi=1.0 + mix_mean=8.0 fixed in all
SWEEPS = [
    ("mf15_fm05",  {"mix_mf_anal_frac": 0.15, "mix_fm_anal_frac": 0.05}),  # new default
    ("mf10_fm05",  {"mix_mf_anal_frac": 0.10, "mix_fm_anal_frac": 0.05}),
    ("mf05_fm05",  {"mix_mf_anal_frac": 0.05, "mix_fm_anal_frac": 0.05}),
    ("mf15_fm10",  {"mix_mf_anal_frac": 0.15, "mix_fm_anal_frac": 0.10}),
    ("mf10_fm10",  {"mix_mf_anal_frac": 0.10, "mix_fm_anal_frac": 0.10}),
    ("mf15_fm15",  {"mix_mf_anal_frac": 0.15, "mix_fm_anal_frac": 0.15}),
]


def run_one(label, overrides, seed):
    p = dict(BASE_PARAMS)
    p = _set_bi1(p)
    for k, v in overrides.items():
        p[k] = v
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"femalefix_{label}_{seed}",
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
    # female_hetero / male_hetero are int counts; compute ratio over known total.
    known = 0.0
    if hasattr(pop, 'daily_episode_cases_by_sexcat'):
        known = float(np.sum(np.sum(pop.daily_episode_cases_by_sexcat, axis=0)))
    fh = float(s.get('episode_female_hetero', float('nan')))
    mh = float(s.get('episode_male_hetero', float('nan')))
    row = dict(
        label=label, seed=seed,
        total=float(sim.sum()), peak=int(np.argmax(sim)),
        homo=s.get('episode_homo_ratio', float('nan')),
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
        female_hetero=(fh / known) if known > 0 else float('nan'),
        male_hetero=(mh / known) if known > 0 else float('nan'),
        seconds=round(time.time() - t0),
    )
    print(f"{label:14s} -> total={row['total']:6.0f} homo={row['homo']:.3f} bi={row['bi']:.3f} "
          f"hetero={row['hetero']:.3f} female={row['female']:.3f} "
          f"fhet={row['female_hetero']:.3f} mhet={row['male_hetero']:.3f}", flush=True)
    return row


if __name__ == "__main__":
    seed = 300
    out_dir = os.path.abspath(os.path.join(_root, "outputs", "sensitivity"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "female_fix_sweep.csv")

    fieldnames = ["label", "seed", "total", "peak", "homo", "bi", "hetero",
                  "female", "female_hetero", "male_hetero", "seconds"]
    rows = []
    for label, overrides in SWEEPS:
        rows.append(run_one(label, overrides, seed))
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    print(f"\n=== TARGETS: bi=12.6% hetero=9.9% female=3.7% homo=77.5% total=2280 ===", flush=True)
    print(f"saved to {out_csv}", flush=True)
