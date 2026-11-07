"""Scan sustained importation (exogenous case inflow during the outbreak).

User's mechanism (literature-supported): during a real outbreak, new cases are
continually imported (travel, cross-region spread), especially in the early
summer when international travel was high. This sustained inflow:
  (a) smooths the epidemic curve (fixes the "fast rise + abrupt stop" artifact
      of a purely-endogenous sex_party engine), and
  (b) shifts the peak later toward the observed LAC peak (day 92).

Literature anchors for importation:
  - England (Lancet Reg Health Eur 2025): used real importation data, ~6/month.
  - Canada 3 cities (J Infect Dis 2024): calibrated 1-3 "exposed" introductions.
  - US DC (PMC11323315): ~11 high-activity importations/day during the surge.
  - LAC itself: first case 5/19 (travel-related), then local transmission
    confirmed 6/24; international travel surged May-Jul 2022.

Each scenario = 1 seed (300), 10M pop, 244 days. The day-0 seed_infection(20)
is REPLACED by a small seed + a sustained importation schedule whose shape
mirrors the observed LAC "slow May -> accelerate Jun-Jul -> taper Aug".

Metric: peak day + cumulative MAE vs observed (total=2337, peak=92).
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
                     cache_label=f"imp_{name}", policy=POLICY, importations=importations)
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

    print(f"{name:<30} total={total:6.0f} peak={peak:3d} "
          f"| obs_peak={OBS_PEAK} | cumMAE0-120={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0w:.0f}s)", flush=True)
    return peak, total, mae_cum


if __name__ == "__main__":
    print(f"观测: total={OBSTOT:.0f}, peak={OBS_PEAK}。cumMAE 越小越好，peak 越接近 92 越好", flush=True)
    # Sustained importation schedules: (name, seed_day0, [importation segments]).
    # Shapes mirror "slow May -> accelerate Jun-Jul -> taper Aug".
    SCENARIOS = {
        # 20 seed day0 (current baseline), no importation
        "base_20seed_noimp": (20, []),

        # small seed (4) + sustained importation, moderate total inflow
        "seed4_imp_2/4/9/4/0": (4, [dict(start=20, end=44, daily_n=2),
                                    dict(start=45, end=74, daily_n=4),
                                    dict(start=75, end=104, daily_n=9),
                                    dict(start=105, end=134, daily_n=4)]),

        # small seed (4) + lighter sustained inflow
        "seed4_imp_1/2/4/2/0": (4, [dict(start=20, end=44, daily_n=1),
                                    dict(start=45, end=74, daily_n=2),
                                    dict(start=75, end=104, daily_n=4),
                                    dict(start=105, end=134, daily_n=2)]),

        # no day0 seed, pure sustained importation (early shape test)
        "seed0_imp_1/2/4/2/0": (0, [dict(start=15, end=44, daily_n=1),
                                    dict(start=45, end=74, daily_n=2),
                                    dict(start=75, end=104, daily_n=4),
                                    dict(start=105, end=134, daily_n=2)]),

        # small seed (8) + light sustained
        "seed8_imp_1/2/4/2/0": (8, [dict(start=20, end=44, daily_n=1),
                                    dict(start=45, end=74, daily_n=2),
                                    dict(start=75, end=104, daily_n=4),
                                    dict(start=105, end=134, daily_n=2)]),

        # small seed (8) + minimal sustained (~6/month style, England anchor)
        "seed8_imp_0.2pd": (8, [dict(start=20, end=134, daily_n=1)]),
    }
    for name, (ns, imp) in SCENARIOS.items():
        run(name, ns, imp)
