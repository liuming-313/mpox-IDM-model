"""Fix composition degradation of the seed0+sustained-importation scheme.

30-seed validation showed peak fixed (median 95 vs obs 92) but composition
degraded: homo median 0.841 (obs 0.775), hetero 0.044 (obs ~0.070),
female 0.039 (obs ~0.055). Cause: importation_mask = homo_male_mcp routes ALL
imported cases into the homo-MSM core, so hetero/female bridging collapses.

Here we test fixes that restore composition WITHOUT re-breaking the peak:
  A. importation mask widened to MSM+bi-male (bi bridge restores hetero spillover)
  B. importation mask = random (full population)
  C. keep homo importation but add a tiny uniform seed (2-4) back
  D. homo importation + separate bi-male importation stream

Single seed 300 for speed; best candidate will be re-validated 30-seed.
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

IMPORTATIONS = [
    dict(start=15, end=44, daily_n=1),
    dict(start=45, end=74, daily_n=2),
    dict(start=75, end=104, daily_n=4),
    dict(start=105, end=134, daily_n=1),
]


def _load_obs():
    src = open(os.path.join(ROOT, "population.py"), encoding="utf-8").read()
    start = src.find("Case_data = ") + len("Case_data = ")
    end = src.find("probability_log", start)
    return np.array(eval(src[start:end].rstrip().rstrip(",").strip()))


OBS = _load_obs()
OBSTOT = OBS.sum()
OBS_PEAK = int(np.argmax(OBS))


def run(name, n_seed_day0, seed_mask_name, importations, import_mask_name):
    p = dict(BASE_PARAMS)
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    pop = Population(n=POPULATION_SIZE, params=p, seed=300,
                     cache_label=f"fixcomp_{name}", policy=POLICY, importations=importations)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    def _mask(nm):
        return {
            'homo_msm': (g == 0) & (o == 'homo') & mcp,
            'msm_bi': (g == 0) & ((o == 'homo') | (o == 'bi')) & mcp,
            'msm_all': (g == 0) & mcp,
            'random': np.ones(pop.n, dtype=bool),
            'all_mcp': mcp,
        }[nm]
    imp_mask = _mask(import_mask_name)
    pop.importation_mask = imp_mask
    np.random.seed(300)
    pop._rng = np.random.default_rng(300)
    if n_seed_day0 > 0:
        pop.seed_infection(n_seed_day0, mask=_mask(seed_mask_name))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    peak = int(np.argmax(sim))
    total = float(sim.sum())
    cum_s = np.cumsum(sim); cum_o = np.cumsum(OBS)
    mae_cum = float(np.mean(np.abs(cum_s[:120]/total - cum_o[:120]/OBSTOT)))

    print(f"{name:<32} total={total:6.0f} peak={peak:3d} cumMAE={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0:.0f}s)" if False else
          f"{name:<32} total={total:6.0f} peak={peak:3d} cumMAE={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f}", flush=True)
    return peak, total, mae_cum


if __name__ == "__main__":
    print(f"观测: total={OBSTOT:.0f}, peak={OBS_PEAK} | 目标 homo=.775 bi=.126 hetero~.070 female~.055", flush=True)
    SCENARIOS = {
        # baseline: homo-only importation (the 30-seed-degraded scheme)
        "imp_homo_only": (0, 'homo_msm', IMPORTATIONS, 'homo_msm'),

        # A. widen importation to MSM+bi male
        "imp_msm_bi": (0, 'homo_msm', IMPORTATIONS, 'msm_bi'),

        # B. importation to all MCP (all orientations, both sexes)
        "imp_all_mcp": (0, 'homo_msm', IMPORTATIONS, 'all_mcp'),

        # C. keep homo importation + tiny uniform seed back
        "imp_homo_seed2": (2, 'random', IMPORTATIONS, 'homo_msm'),
        "imp_homo_seed4": (4, 'random', IMPORTATIONS, 'homo_msm'),

        # D. homo importation + separate bi-male importation stream
        "imp_homo_plus_bi": (0, 'homo_msm',
                             IMPORTATIONS + [dict(start=45, end=104, daily_n=1)],
                             'msm_bi'),
    }
    for name, (ns, smn, imp, imn) in SCENARIOS.items():
        t0 = time.time()
        run(name, ns, smn, imp, imn)
