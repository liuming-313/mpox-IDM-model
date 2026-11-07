"""Joint scan: lift total toward 2280 AND improve hetero/female composition,
while keeping peak~=92 stable.

User's logic: composition bias (homo 0.841 vs 0.775, hetero/female low) is a
*spread-depth* effect — with sustained importation into the MSM core, the virus
hasn't fully spilled to bi/hetero/female layers. So first unify total+timing,
then fine-tune composition. This scan does BOTH jointly (single seed 300 for
speed), varying:

  (a) f-factor (oneoff_p + party_ht_top_mean scale) -> total / peak
  (b) mix_mf_anal_frac  (male->female anal)        -> female/hetero
  (c) oneoff_orientation_bias[4] (male-bi weight)  -> bi bridge -> hetero
  (d) p_bi_m_to_msm     (bi male orientation)      -> bi->hetero spillover

Metric: total (target 2280), peak (target 92), homo/bi/hetero/female.
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


def run(name, overrides):
    p = dict(BASE_PARAMS)
    # apply f-factor scaling to oneoff_p and party_ht_top_mean
    f = overrides.pop("f", None)
    if f is not None:
        # oneoff mean = 0.5*f -> p = 1/(1+mean); top_mean = 1.0*f
        mean = 0.5 * f
        p["oneoff_p"] = 1.0 / (1.0 + mean)
        p["party_ht_top_mean"] = 1.0 * f
    for k, v in overrides.items():
        if k == "oneoff_orientation_bias_bi":
            p["oneoff_orientation_bias"] = dict(p["oneoff_orientation_bias"])
            p["oneoff_orientation_bias"][4] = v
        else:
            p[k] = v
    p["n_days"] = SIM_DAYS + 1
    p["verbose"] = False
    t0 = time.time()
    pop = Population(n=POPULATION_SIZE, params=p, seed=300,
                     cache_label=f"joint_{name}", policy=POLICY, importations=IMPORTATIONS)
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == "homo") & mcp
    pop.importation_mask = mask
    np.random.seed(300)
    pop._rng = np.random.default_rng(300)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pop.run_simulation(SIM_DAYS)
        s = pop.summary() if hasattr(pop, "summary") else {}

    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    peak = int(np.argmax(sim))
    total = float(sim.sum())
    cum_s = np.cumsum(sim); cum_o = np.cumsum(OBS)
    mae_cum = float(np.mean(np.abs(cum_s[:120]/total - cum_o[:120]/OBSTOT)))

    print(f"{name:<30} total={total:6.0f} peak={peak:3d} cumMAE={mae_cum:.4f} "
          f"homo={s.get('episode_homo_ratio', float('nan')):.3f} "
          f"bi={s.get('episode_bi_ratio', float('nan')):.3f} "
          f"hetero={s.get('episode_hetero_ratio', float('nan')):.3f} "
          f"female={s.get('episode_female_known_ratio', float('nan')):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return peak, total, mae_cum


if __name__ == "__main__":
    print(f"观测: total={OBSTOT:.0f} peak={OBS_PEAK} | 目标 homo=.775 bi=.126 hetero~.070 female~.055", flush=True)
    SCENARIOS = {
        # baseline (current frozen scheme)
        "base_f7.0": {"f": 7.0},

        # ---- lift total via f, check peak ----
        "f7.4": {"f": 7.4},
        "f7.8": {"f": 7.8},
        "f8.2": {"f": 8.2},

        # ---- f up + spread depth (mix_mf + bi weight) ----
        "f7.4_mixmf.25": {"f": 7.4, "mix_mf_anal_frac": 0.25},
        "f7.4_mixmf.25_bibi1.5": {"f": 7.4, "mix_mf_anal_frac": 0.25,
                                   "oneoff_orientation_bias_bi": 1.5},
        "f7.8_mixmf.25": {"f": 7.8, "mix_mf_anal_frac": 0.25},
        "f7.8_mixmf.25_bibi1.5": {"f": 7.8, "mix_mf_anal_frac": 0.25,
                                   "oneoff_orientation_bias_bi": 1.5},

        # ---- keep f=7.0, only composition knobs ----
        "f7.0_mixmf.25_bibi1.5": {"f": 7.0, "mix_mf_anal_frac": 0.25,
                                   "oneoff_orientation_bias_bi": 1.5},
        "f7.0_mixmf.30_bibi1.5": {"f": 7.0, "mix_mf_anal_frac": 0.30,
                                   "oneoff_orientation_bias_bi": 1.5},
    }
    for name, ov in SCENARIOS.items():
        run(name, dict(ov))
