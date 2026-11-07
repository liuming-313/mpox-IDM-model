"""Refined scan around the sweet spot: seed0 + sustained importation.

Previous scan found seed0_imp_1/2/4/2/0 -> peak=93, total=2568, cumMAE=0.1071
(peak right, but total +10% high and cumMAE high). seed8 -> cumMAE low but total
too low. Here we sweep importation *intensity* and *shape* to find the point
that hits peak~=92 AND total~=2337 AND cumMAE minimized.

Metrics: peak, total, cumMAE(0-120), composition.
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
                     cache_label=f"refine_{name}", policy=POLICY, importations=importations)
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

    print(f"{name:<28} total={total:6.0f} peak={peak:3d} "
          f"| obs_peak={OBS_PEAK} | cumMAE0-120={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0w:.0f}s)", flush=True)
    return peak, total, mae_cum


if __name__ == "__main__":
    print(f"观测: total={OBSTOT:.0f}, peak={OBS_PEAK}。", flush=True)

    # Refined sweep: vary (a) seed size 0 vs small, (b) importation intensity,
    # (c) importation onset/shape. Focus on peak~=92 AND total~=2337.
    # Shapes use "rise then taper" mirroring LAC (slow May -> Jun-Jul surge -> Aug taper).
    SCENARIOS = {
        # ---- vary seed size around seed0 sweet spot ----
        "seed0_imp_1/2/3/2/0": (0, [dict(start=15, end=44, daily_n=1),
                                     dict(start=45, end=74, daily_n=2),
                                     dict(start=75, end=104, daily_n=3),
                                     dict(start=105, end=134, daily_n=2)]),

        "seed0_imp_1/2/4/1/0": (0, [dict(start=15, end=44, daily_n=1),
                                     dict(start=45, end=74, daily_n=2),
                                     dict(start=75, end=104, daily_n=4),
                                     dict(start=105, end=134, daily_n=1)]),

        "seed0_imp_1/3/4/2/0": (0, [dict(start=15, end=44, daily_n=1),
                                     dict(start=45, end=74, daily_n=3),
                                     dict(start=75, end=104, daily_n=4),
                                     dict(start=105, end=134, daily_n=2)]),

        "seed0_imp_1/2/5/3/1": (0, [dict(start=15, end=44, daily_n=1),
                                     dict(start=45, end=74, daily_n=2),
                                     dict(start=75, end=104, daily_n=5),
                                     dict(start=105, end=134, daily_n=3)]),

        # ---- small seed variants to lift total without moving peak too early ----
        "seed2_imp_1/2/3/2/0": (2, [dict(start=15, end=44, daily_n=1),
                                     dict(start=45, end=74, daily_n=2),
                                     dict(start=75, end=104, daily_n=3),
                                     dict(start=105, end=134, daily_n=2)]),

        "seed3_imp_1/2/3/2/0": (3, [dict(start=15, end=44, daily_n=1),
                                     dict(start=45, end=74, daily_n=2),
                                     dict(start=75, end=104, daily_n=3),
                                     dict(start=105, end=134, daily_n=2)]),

        # ---- later onset to shift peak later while keeping total ----
        "seed0_imp_1/2/4/2_late": (0, [dict(start=20, end=44, daily_n=1),
                                        dict(start=45, end=74, daily_n=2),
                                        dict(start=75, end=104, daily_n=4),
                                        dict(start=105, end=134, daily_n=2)]),

        "seed0_imp_1/2/4/2_later": (0, [dict(start=25, end=44, daily_n=1),
                                         dict(start=45, end=74, daily_n=2),
                                         dict(start=75, end=104, daily_n=4),
                                         dict(start=105, end=134, daily_n=2)]),
    }
    for name, (ns, imp) in SCENARIOS.items():
        run(name, ns, imp)
