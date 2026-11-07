"""30-seed stability validation for the chosen importation scheme.

Chosen scheme: seed0 (no day-0 injection) + sustained importation
    start=15,end=44,daily_n=1  (late May, slow)
    start=45,end=74,daily_n=2  (Jun, accelerate)
    start=75,end=104,daily_n=4 (Jul, surge)
    start=105,end=134,daily_n=1 (Aug, taper)

Single-seed(300) gave: total=2555, peak=93, homo=.796 bi=.134 hetero=.070 female=.064.
Here we run 30 seeds to check whether peak/total/composition are stable,
not a lucky single-seed draw.
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


def run_seed(seed):
    p = dict(BASE_PARAMS)
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"val30v7_{seed}", policy=POLICY, importations=IMPORTATIONS)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    # NO seed_infection (seed0 scheme)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    return dict(
        seed=seed,
        peak=int(np.argmax(sim)),
        total=float(sim.sum()),
        daily=sim,
        homo=s.get('episode_homo_ratio', float('nan')),
        bi=s.get('episode_bi_ratio', float('nan')),
        hetero=s.get('episode_hetero_ratio', float('nan')),
        female=s.get('episode_female_known_ratio', float('nan')),
    )


if __name__ == "__main__":
    import statistics
    seeds = list(range(100, 130))
    print(f"观测: total={OBSTOT:.0f}, peak={OBS_PEAK}。30-seed 验证 f7.5+mixmf.30+bibi1.5", flush=True)
    results = []
    for s in seeds:
        t0 = time.time()
        r = run_seed(s)
        r['wall'] = time.time() - t0
        results.append(r)
        print(f"seed={s:3d} total={r['total']:6.0f} peak={r['peak']:3d} "
              f"homo={r['homo']:.3f} bi={r['bi']:.3f} hetero={r['hetero']:.3f} "
              f"female={r['female']:.3f} ({r['wall']:.0f}s)", flush=True)

    peaks = [r['peak'] for r in results]
    totals = [r['total'] for r in results]
    homos = [r['homo'] for r in results]
    bis = [r['bi'] for r in results]
    heteros = [r['hetero'] for r in results]
    females = [r['female'] for r in results]
    print("\n=== 30-seed summary ===", flush=True)
    print(f"peak:   median={statistics.median(peaks)} mean={np.mean(peaks):.1f} "
          f"min={min(peaks)} max={max(peaks)}  (obs={OBS_PEAK})", flush=True)
    print(f"total:  median={statistics.median(totals):.0f} mean={np.mean(totals):.0f} "
          f"std={np.std(totals):.0f}  (obs={OBSTOT:.0f})", flush=True)
    print(f"homo:   median={statistics.median(homos):.3f}  (obs=0.775)", flush=True)
    print(f"bi:     median={statistics.median(bis):.3f}  (obs=0.126)", flush=True)
    print(f"hetero: median={statistics.median(heteros):.3f}  (obs~0.070)", flush=True)
    print(f"female: median={statistics.median(females):.3f}  (obs~0.055)", flush=True)
    n_below = sum(1 for p in peaks if p < OBS_PEAK)
    print(f"peak < 92 的 seed 数: {n_below}/{len(peaks)}", flush=True)
    # save per-seed daily for later plotting
    import csv, os as _os
    out = os.path.join(ROOT, "outputs", "v7_final_bibi1.2")
    _os.makedirs(out, exist_ok=True)
    for r in results:
        with open(os.path.join(out, f"seed{r['seed']}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["day", "daily_episode_cases"])
            for d, v in enumerate(r['daily']):
                w.writerow([d, v])
    print(f"per-seed daily saved to {out}", flush=True)
