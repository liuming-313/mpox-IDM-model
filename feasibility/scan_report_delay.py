"""Scan: time-varying reporting delay (report_delay_time_varying) vs curve fit.

Uses the OFFICIAL `report_delay_time_varying` / `report_delay_t0` / `_t1` / `_tau`
params added to population.py `_apply_policy`. Each scenario runs 1 seed (300),
10M pop, 244 days, frozen params otherwise.

Metric: cumulative MAE over the growth region (day 0..120), plus peak-day shift.
Observed LAC: total=2337, peak day=92.
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


def run(name, t0, t1, tau):
    p = dict(BASE_PARAMS)
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    p["report_delay_time_varying"] = True
    p["report_delay_t0"] = t0
    p["report_delay_t1"] = t1
    p["report_delay_tau"] = tau
    t0w = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=300,
                     cache_label=f"rd_{name}", policy=POLICY, importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(300)
    pop._rng = np.random.default_rng(300)
    pop.seed_infection(20, mask=mask)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    peak = int(np.argmax(sim))
    total = float(sim.sum())

    cum_s = np.cumsum(sim); cum_o = np.cumsum(OBS)
    mae_cum = float(np.mean(np.abs(cum_s[:120]/total - cum_o[:120]/OBSTOT)))

    print(f"{name:<22} total={total:6.0f} peak={peak:3d} "
          f"| obs_peak={OBS_PEAK} | cumMAE0-120={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0w:.0f}s)", flush=True)
    return peak, total, mae_cum


if __name__ == "__main__":
    print(f"观测: total={OBSTOT:.0f}, peak={OBS_PEAK}。cumMAE 越小越好，peak 越接近 92 越好", flush=True)
    SCENARIOS = {
        "fixed5_baseline":   (5.0, 5.0, 40.0),
        "tv_12to5_tau40":    (12.0, 5.0, 40.0),
        "tv_15to5_tau40":    (15.0, 5.0, 40.0),
        "tv_15to5_tau60":    (15.0, 5.0, 60.0),
        "tv_20to5_tau50":    (20.0, 5.0, 50.0),
    }
    for name, (t0, t1, tau) in SCENARIOS.items():
        run(name, t0, t1, tau)
