"""Fine-tune the male-bisexual weight to bring bi% from 0.145 down to 0.126.

Current frozen scheme (f=7.5 + mixmf.30 + bibi=1.5) gives 30-seed bi=0.145
(target 0.126, +1.9pp). bi is driven primarily by oneoff_orientation_bias[4]
(male-bisexual one-off weight). Sweep it 1.2-1.5 (5 seeds each) to find the
value that hits bi~0.126 while keeping homo/hetero/female near target.

All other params = frozen (read from main.py BASE_PARAMS, then apply the
f=7.5 + mixmf.30 + bibi override).
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


def run_one(seed, bi_weight):
    p = dict(BASE_PARAMS)
    # frozen f=7.5 + mixmf.30
    p["oneoff_p"] = 1.0 / (1.0 + 0.5 * 7.5)
    p["party_ht_top_mean"] = 7.5
    p["mix_mf_anal_frac"] = 0.30
    p["oneoff_orientation_bias"] = dict(p["oneoff_orientation_bias"])
    p["oneoff_orientation_bias"][4] = bi_weight
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    pop = Population(n=POPULATION_SIZE, params=p, seed=seed,
                     cache_label=f"bi_{bi_weight}_{seed}", policy=POLICY, importations=IMPORTATIONS)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    return dict(peak=int(np.argmax(sim)), total=float(sim.sum()),
                homo=s.get('episode_homo_ratio', float('nan')),
                bi=s.get('episode_bi_ratio', float('nan')),
                hetero=s.get('episode_hetero_ratio', float('nan')),
                female=s.get('episode_female_known_ratio', float('nan')))


if __name__ == "__main__":
    import statistics
    print(f"观测: total={OBSTOT:.0f} peak={OBS_PEAK} | 目标 bi=.126 homo=.775 hetero~.070 female~.055", flush=True)
    SEEDS = [300, 301, 302, 303, 304]
    for bw in [1.2, 1.3, 1.35, 1.4, 1.5]:
        totals, peaks, homos, bis, hes, fes = [], [], [], [], [], []
        t0 = time.time()
        for sd in SEEDS:
            r = run_one(sd, bw)
            totals.append(r['total']); peaks.append(r['peak'])
            homos.append(r['homo']); bis.append(r['bi'])
            hes.append(r['hetero']); fes.append(r['female'])
        print(f"bibi={bw}: total_med={statistics.median(totals):.0f} "
              f"peak_med={statistics.median(peaks):.0f} "
              f"homo={statistics.median(homos):.3f} bi={statistics.median(bis):.3f} "
              f"hetero={statistics.median(hes):.3f} female={statistics.median(fes):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
