"""Scan delayed epidemic onset (align seed introduction to LAC timeline).

Root-cause diagnosis (final): the simulated epidemic STARTS too early, not
grows too fast. Observed LAC: first case day 18 (May 19), but sustained
exponential growth only begins ~day 45-50 (mid-June); cumulative at day 40 is
only 0.6%, day 50 only 1.4%. The model injects 20 MSM on day 0 and starts
amplifying immediately, so the whole curve is shifted ~18-40 days early,
producing peak day 74 instead of observed 92.

This scan replaces the day-0 `seed_infection(20)` with an importation schedule
that delays the effective onset. Each scenario = 1 seed (300), 10M pop, 244 d.

Scenarios (delay the sustained-growth onset):
  A  baseline:          20 cases day 0 (current)
  G  delayed onset @40: 2/day day 38..52 then 4/day day 53..90
  H  delayed onset @45: 2/day day 43..57 then 4/day day 58..90
  I  delayed onset @50: 2/day day 48..62 then 4/day day 63..90
  J  delayed @40 strong:4/day day 40..90
  K  delayed @45 strong:4/day day 45..90
"""
import os, sys, io, time, contextlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
SIM_DAYS = 244
POPULATION_SIZE = 10_000_000

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


def _load_obs():
    src = open(os.path.join(ROOT, "population.py"), encoding="utf-8").read()
    start = src.find("Case_data = ") + len("Case_data = ")
    end = src.find("probability_log", start)
    return np.array(eval(src[start:end].rstrip().rstrip(",").strip()))


OBS = _load_obs()
OBSTOT = OBS.sum()
OBS_PEAK = int(np.argmax(OBS))


def run(name, n_seed_day0, importations):
    p = dict(BASE_PARAMS)
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    t0w = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=300,
                     cache_label=f"do_{name}", policy=POLICY, importations=importations)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(300)
    pop._rng = np.random.default_rng(300)
    if n_seed_day0 > 0:
        pop.seed_infection(n_seed_day0, mask=mask)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    peak = int(np.argmax(sim))
    total = float(sim.sum())
    cum_s = np.cumsum(sim); cum_o = np.cumsum(OBS)
    mae_cum = float(np.mean(np.abs(cum_s[:120]/total - cum_o[:120]/OBSTOT)))

    print(f"{name:<26} total={total:6.0f} peak={peak:3d} "
          f"| obs_peak={OBS_PEAK} | cumMAE0-120={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0w:.0f}s)", flush=True)
    return peak, total, mae_cum


if __name__ == "__main__":
    print(f"观测: total={OBSTOT:.0f}, peak={OBS_PEAK}。cumMAE 越小越好，peak 越接近 92 越好", flush=True)
    SCENARIOS = {
        "A_baseline_20_day0": (20, []),
        "G_delay@40": (0, [dict(start=38, end=52, daily_n=2),
                            dict(start=53, end=90, daily_n=4)]),
        "H_delay@45": (0, [dict(start=43, end=57, daily_n=2),
                            dict(start=58, end=90, daily_n=4)]),
        "I_delay@50": (0, [dict(start=48, end=62, daily_n=2),
                            dict(start=63, end=90, daily_n=4)]),
        "J_delay@40_4pd": (0, [dict(start=40, end=90, daily_n=4)]),
        "K_delay@45_4pd": (0, [dict(start=45, end=90, daily_n=4)]),
    }
    for name, (ns, imp) in SCENARIOS.items():
        run(name, ns, imp)
