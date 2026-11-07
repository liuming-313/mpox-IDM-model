"""A/B scan: seed-case introduction schedule vs peak timing.

Tests whether replacing the current "20 index cases injected on day 0" with
the literature-standard "small batch + sustained low-level importation"
(small exposed batch at start + daily importations through the surge) shifts
the epidemic peak later, toward the observed LAC peak (day ~92).

Each scenario is a different introduction schedule; all else (params, seed=300,
10M population, 244 days) matches the frozen baseline. 1 seed per scenario for
a fast direction/magnitude read.

Scenarios (day 0 = 2022-05-01; LAC first case 05-27 = day 26):
  A  baseline:         20 cases injected day 0 (current)
  B  small batch:      4 cases day 0, then no importation (UK-style 4 MSM)
  C  batch + sustained:4 cases day 0 + 1/day importation day 20..90
  D  batch + taper:    4 cases day 0 + 2/day day 20..60 + 1/day day 61..120
  E  batch + peak-fed: 4 cases day 0 + importation peaking mid-surge (align
     phylodynamic finding: introductions peak at outbreak height)
"""
import os, sys, io, time, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
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

POLICY = [
    dict(name="phase1_growth", start=0, end=91,
         contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95, social_mult=0.90,
         test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
         test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
    dict(name="phase2_control", start=92, end=170,
         contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70, social_mult=1.05,
         test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
         test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
    dict(name="phase3_postpeak", start=171, end=SIM_DAYS,
         contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70, social_mult=1.20,
         test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
         test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
]

SCENARIOS = {
    "A_baseline_20_day0": (20, []),
    "B_batch4_only": (4, []),
    "C_batch4_sustain": (4, [dict(start=20, end=90, daily_n=1)]),
    "D_batch4_taper": (4, [dict(start=20, end=60, daily_n=2),
                           dict(start=61, end=120, daily_n=1)]),
    "E_batch4_peakfed": (4, [dict(start=40, end=55, daily_n=1),
                             dict(start=56, end=90, daily_n=3),
                             dict(start=91, end=130, daily_n=1)]),
    "F_batch4_sustain_2pd": (4, [dict(start=20, end=90, daily_n=2)]),
}


def run(name, n_seed, importations):
    p = dict(BASE_PARAMS)
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=300,
                     cache_label=f"ab_{name}", policy=POLICY, importations=importations)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(300)
    pop._rng = np.random.default_rng(300)
    pop.seed_infection(n_seed, mask=mask)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    peak = int(np.argmax(sim))
    total = float(sim.sum())
    print(f"{name:<22} n_seed={n_seed:>2} total={total:6.0f} peak={peak:3d} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return peak, total


if __name__ == "__main__":
    print("目标: total~2280, peak~92 (观测 LAC)", flush=True)
    for name, (n_seed, imp) in SCENARIOS.items():
        run(name, n_seed, imp)
