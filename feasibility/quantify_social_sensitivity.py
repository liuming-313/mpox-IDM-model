"""Quantify the social layer's contribution and the sensitivity of its 3 P0
out-of-bound parameters, before re-anchoring them to literature bounds.

Two parts:
  A) BASELINE: one 10M run -> layer_infection_counts share + social-layer
     (layer_code 6) infected-person composition by sex/orientation.
  B) SENSITIVITY: one-at-a-time parameter changes vs baseline (single seed 300):
       social_contact_scale:  10 -> 6 -> 2.5 -> 1.0
       social_same_sex_assort: 0.9 -> 0.85 -> 0.80 -> 0.75
       community_bridge_scale: 1.5 -> 1.0

Outputs CSV + printed summary to outputs/social_sensitivity/.
"""
import os, sys, time, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population, Case_data

POP_SIZE = 10_000_000
SIM_DAYS = 244
SEED = 300

SOCIAL_LAYER_CODE = 6  # from txlog_layer_map in population.py

def base_params():
    return {
        "oneoff_active_frac": 0.01, "oneoff_p": 0.15, "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10, "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
        "sexual_anal_receptive": 0.28, "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035, "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7, "party_core_intensity": 0.7,
        "social_contact_scale": 10, "social_same_sex_assort": 0.9,
        "community_bridge_scale": 1.5,
        "party_heavy_tail": True, "party_ht_frac": 0.05,
        "party_ht_top_mean": 18.0, "party_ht_top_clip": 30,
        "n_days": SIM_DAYS + 1, "verbose": False,
    }

def make_policy():
    return [
        dict(name='phase1_growth', start=0, end=91,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95, social_mult=0.90,
             test_mult=0.70, trace_mult=0.80, test_pathway_prob=0.75,
             test_seek_delay_mean_days=5.0, test_seek_delay_sd_days=3.0, lab_turnaround_days=2),
        dict(name='phase2_control', start=92, end=170,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70, social_mult=1.05,
             test_mult=1.50, trace_mult=1.20, test_pathway_prob=0.90,
             test_seek_delay_mean_days=4.0, test_seek_delay_sd_days=2.0, lab_turnaround_days=2),
        dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70, social_mult=1.20,
             test_mult=1.20, trace_mult=1.00, test_pathway_prob=0.85,
             test_seek_delay_mean_days=4.5, test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]

def run_one(params, seed, cache_label):
    pop = Population(n=POP_SIZE, params=params, seed=seed,
                     cache_label=cache_label, policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    sim = np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)
    return sim, pop

def summarize_layer(pop):
    """Return dict: per-layer share + social-layer sex/orientation composition."""
    lc = pop.layer_infection_counts
    total = sum(lc.values())
    out = {"total_layers": total}
    for layer, cnt in lc.items():
        out[f"layer_{layer}"] = cnt
        out[f"layer_{layer}_share"] = cnt / total if total else 0.0

    # social-layer infected-person composition via txlog dst
    n = int(pop.txlog_i)
    if n > 0:
        layer = pop.txlog_layer[:n]
        dst = pop.txlog_dst[:n]
        social_dst = dst[layer == SOCIAL_LAYER_CODE]
        if social_dst.size:
            sg = pop.genders[social_dst]
            so = pop.orientations[social_dst]
            out["social_infected_n"] = int(social_dst.size)
            out["social_infected_male"] = float((sg == 0).mean())
            out["social_infected_female"] = float((sg == 1).mean())
            out["social_infected_homo"] = float((so == "homo").mean())
            out["social_infected_bi"] = float((so == "bi").mean())
            out["social_infected_hetero"] = float((so == "hetero").mean())
        else:
            out["social_infected_n"] = 0
    else:
        out["social_infected_n"] = 0
    return out

if __name__ == "__main__":
    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    cobs = np.cumsum(obs)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "social_sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    results = []

    # ---- A) BASELINE ----
    print("=== A) BASELINE (top_mean=18, scale=10, assort=0.9, bridge=1.5) ===")
    t0 = time.time()
    sim, pop = run_one(base_params(), SEED, f"soc_base_{SEED}")
    tot = sim.sum()
    csim = np.cumsum(sim)
    crmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
    s = pop.summary()
    layer_sum = summarize_layer(pop)
    print(f"total={tot:.0f} cRMSE={crmse:.0f} homo={s.get('episode_homo_ratio',0):.1%} "
          f"hetero={s.get('episode_hetero_ratio',0):.1%} female={s.get('episode_female_ratio',0):.2%}")
    print(f"  layers (total={layer_sum['total_layers']}):")
    for k in sorted(layer_sum):
        if k.startswith("layer_") and k.endswith("_share") and layer_sum[k] > 0.005:
            print(f"    {k.replace('layer_','').replace('_share',''):>14} = {layer_sum[k]:.1%}")
    if layer_sum.get("social_infected_n", 0):
        print(f"  social-layer infected composition (n={layer_sum['social_infected_n']}): "
              f"male={layer_sum.get('social_infected_male',0):.1%} female={layer_sum.get('social_infected_female',0):.1%} | "
              f"homo={layer_sum.get('social_infected_homo',0):.1%} bi={layer_sum.get('social_infected_bi',0):.1%} "
              f"hetero={layer_sum.get('social_infected_hetero',0):.1%}")
    base_total = tot
    base_hetero = s.get('episode_hetero_ratio', 0.0)
    base_female = s.get('episode_female_ratio', 0.0)
    print(f"  [baseline done in {time.time()-t0:.0f}s]")

    # ---- B) SENSITIVITY ----
    print("\n=== B) SENSITIVITY (single seed 300) ===")
    print(f"{'param':>24} {'value':>6} {'total':>6} {'delta%':>7} {'homo':>6} {'hetero':>7} {'female':>7}")

    def run_sens(label, params, key, val):
        p = dict(params)
        p[key] = val
        sim, pop = run_one(p, SEED, f"soc_{label}_{int(val*100)}_{SEED}")
        tot = sim.sum()
        s = pop.summary()
        h = s.get('episode_homo_ratio', 0.0)
        ht = s.get('episode_hetero_ratio', 0.0)
        f = s.get('episode_female_ratio', 0.0)
        d = (tot - base_total) / base_total * 100
        print(f"{key:>24} {val:>6.2f} {tot:>6.0f} {d:>7.1f}% {h:>6.1%} {ht:>7.1%} {f:>7.2%}", flush=True)
        results.append(dict(param=key, value=val, total=tot, homo=h, hetero=ht, female=f))

    # social_contact_scale
    for val in [6.0, 2.5, 1.0]:
        run_sens("scale", base_params(), "social_contact_scale", val)
    # social_same_sex_assort
    for val in [0.85, 0.80, 0.75]:
        run_sens("assort", base_params(), "social_same_sex_assort", val)
    # community_bridge_scale
    for val in [1.0]:
        run_sens("bridge", base_params(), "community_bridge_scale", val)

    csv_path = os.path.join(out_dir, "social_sensitivity.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["param", "value", "total", "homo", "hetero", "female"])
        w.writeheader()
        # prepend baseline row
        w.writerow(dict(param="baseline", value=10.0, total=base_total, homo=s.get('episode_homo_ratio',0),
                        hetero=base_hetero, female=base_female))
        w.writerows(results)
    print(f"\nsaved -> {csv_path}")
