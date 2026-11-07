"""
A/B test: heavy-tail activity stratification for the one-off layer.

Compares, on identical population + seed + policy:
  A) oneoff_activity_stratified = False  (old uniform baseline)
  B) oneoff_activity_stratified = True   (new 6-level heavy-tail, CDC-aligned)

Both scenarios use N MSM index seeds (no external importations), matching the
new main.py baseline, so the only difference is the one-off activity tail.

Reports total infections, episode size, orientation mix, and one-off-layer
contribution, so we can judge whether the heavy tail raises endogenous
amplification (larger final size) while preserving MSM dominance.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population

POP_SIZE = int(os.environ.get("HT_POP", 2_000_000))
SIM_DAYS = int(os.environ.get("HT_DAYS", 244))
BASE_SEED = int(os.environ.get("HT_SEED", 42))
N_SEEDS = int(os.environ.get("HT_NSEEDS", 20))

# ---- import a minimal copy of main.py's params (avoid running main.py's loop)
def load_params():
    import main as _m  # noqa: this executes main's top-level only (loop is guarded by SCENARIOS at module import? NO)
    # NOTE: main.py runs its scenario loop at import time -> dangerous.
    # Instead, hardcode the key transmission params here (mirror main.py).
    return {
        "oneoff_active_frac": 0.01,
        "oneoff_p": 0.15,
        "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10,
        "sexual_anal_receptive": 0.28,
        "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035,
        "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7,
        "party_core_intensity": 0.7,
        "social_contact_scale": 10,
        "social_same_sex_assort": 0.9,
        "n_days": SIM_DAYS + 1,
        "verbose": False,
    }


POLICY = [
    dict(name='phase1_growth', start=0, end=91,
         contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95, social_mult=0.90,
         test_mult=0.70, trace_mult=0.80,
         test_pathway_prob=0.75, test_seek_delay_mean_days=5.0,
         test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
    dict(name='phase2_control', start=92, end=170,
         contact_mult=0.65, oneoff_mult=0.65, party_mult=0.42, social_mult=1.05,
         test_mult=1.50, trace_mult=1.20,
         test_pathway_prob=0.90, test_seek_delay_mean_days=4.0,
         test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
    dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
         contact_mult=0.70, oneoff_mult=0.85, party_mult=0.47, social_mult=1.20,
         test_mult=1.20, trace_mult=1.00,
         test_pathway_prob=0.85, test_seek_delay_mean_days=4.5,
         test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
]


def run(stratified: bool):
    params = load_params()
    params["oneoff_activity_stratified"] = stratified
    label = "ht" if stratified else "uniform"

    pop = Population(n=POP_SIZE, params=params, seed=BASE_SEED,
                     cache_label=f"htab_{label}", policy=POLICY, importations=[])

    # seed N homo_male_mcp index cases
    g = pop.genders
    o = pop.orientations
    mcp = pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(BASE_SEED)
    pop._rng = np.random.default_rng(BASE_SEED)
    pop.seed_infection(N_SEEDS, mask=mask)

    t0 = time.time()
    pop.run_simulation(SIM_DAYS)
    dt = time.time() - t0

    summ = pop.summary()
    cum_inf = int(summ.get("cum_inf_total",
                   summ.get("cum_inf_homo", 0) + summ.get("cum_inf_bi", 0) + summ.get("cum_inf_hetero", 0)))

    # one-off layer contribution
    oneoff_cnt = int(getattr(pop, "layer_infection_counts", {}).get("oneoff", 0))

    return {
        "stratified": stratified,
        "cum_inf": cum_inf,
        "episode": int(summ.get("episode_total", summ.get("episode_cases", 0))),
        "homo": float(summ.get("episode_homo_ratio", summ.get("cum_inf_homo_ratio", 0))),
        "bi": float(summ.get("episode_bi_ratio", summ.get("cum_inf_bi_ratio", 0))),
        "hetero": float(summ.get("episode_hetero_ratio", summ.get("cum_inf_hetero_ratio", 0))),
        "female": float(summ.get("episode_female_known_ratio", summ.get("cum_inf_female_known_ratio", 0))),
        "oneoff_contrib": oneoff_cnt,
        "layers": dict(getattr(pop, "layer_infection_counts", {})),
        "dt": dt,
    }


def main():
    print(f"A/B test: POP={POP_SIZE}, SEED={BASE_SEED}, NSEEDS={N_SEEDS}, DAYS={SIM_DAYS}")
    results = {}
    for stratified in (False, True):
        print(f"\n=== Running stratified={stratified} ===")
        r = run(stratified)
        results["stratified" if stratified else "uniform"] = r
        print(f"  cum_inf={r['cum_inf']}, episode={r['episode']}, "
              f"homo={r['homo']:.3f}, bi={r['bi']:.3f}, hetero={r['hetero']:.3f}, "
              f"female={r['female']:.3f}, oneoff_contrib={r['oneoff_contrib']}, dt={r['dt']:.0f}s")
        print(f"  layers={r['layers']}")

    print("\n===== SUMMARY =====")
    u = results["uniform"]
    s = results["stratified"]
    print(f"{'metric':<18}{'uniform':>12}{'stratified':>12}{'ratio':>10}")
    for key in ("cum_inf", "episode", "oneoff_contrib"):
        uv, sv = u[key], s[key]
        print(f"{key:<18}{uv:>12}{sv:>12}{(sv/max(uv,1)):>10.2f}")
    for key in ("homo", "bi", "hetero", "female"):
        print(f"{key:<18}{u[key]*100:>11.2f}%{s[key]*100:>11.2f}%")


if __name__ == "__main__":
    main()
