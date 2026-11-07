"""Generate publication-quality simulated-vs-observed epidemic curve comparison.

Runs N seeds of the FINAL locked configuration (20 uniform MSM seeds,
heavy-tail ON, long_tail2 policy) and plots:
  - Left panel:  daily incidence (observed bars + simulated mean + 1-sd band)
  - Right panel: cumulative incidence (observed vs simulated mean + band)

Output: PNG files under outputs/curve_compare/
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from population import Population, Case_data

POP_SIZE = 10_000_000
SIM_DAYS = 244
START_DATE = "2022-05-01"

def base_params():
    return {
        "oneoff_active_frac": 0.01, "oneoff_p": 0.15, "oneoff_age_gap": 80,
        "oneoff_mf_anal_frac": 0.10, "oneoff_activity_stratified": True,
        "oneoff_orientation_bias": None,
        "sexual_anal_receptive": 0.28, "sexual_anal_insertive": 0.06,
        "sexual_vaginal": 0.035, "sexual_oral": 0.0175,
        "p_bi_m_to_msm": 0.7, "party_core_intensity": 0.7,
        "social_contact_scale": 10, "social_same_sex_assort": 0.9,
        "party_heavy_tail": True, "party_ht_frac": 0.05,
        "party_ht_top_mean": 20.0, "party_ht_top_clip": 30,
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

def run_one(seed):
    pop = Population(n=POP_SIZE, params=base_params(), seed=seed,
                     cache_label=f"curvefig_{seed}", policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(20, mask=mask)
    pop.run_simulation(SIM_DAYS)
    return np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)

def month_ticks(ndays):
    # day0 = 2022-05-01; label every ~30 days
    ticks, labels = [], []
    months = ["May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan"]
    for d in range(0, ndays + 1, 30):
        mi = (d // 30)
        if mi < len(months):
            ticks.append(d)
            labels.append(months[mi])
    return ticks, labels

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="outputs/curve_compare")
    args = ap.parse_args()

    obs = np.asarray(Case_data, dtype=float)[:SIM_DAYS]
    obs_total = obs.sum()
    cobs = np.cumsum(obs)
    obs_peak = int(np.argmax(obs))

    # collect simulated curves
    curves = []
    print("Running seeds...", flush=True)
    for i in range(args.seeds):
        seed = 300 + i * 11
        t0 = time.time()
        curves.append(run_one(seed))
        print(f"  seed {seed}: total={curves[-1].sum():.0f} ({time.time()-t0:.0f}s)", flush=True)

    curves = np.array(curves)          # (seeds, days)
    sim_mean = curves.mean(axis=0)
    sim_sd = curves.std(axis=0)
    sim_cum = np.cumsum(curves, axis=1)
    sim_cum_mean = sim_cum.mean(axis=0)
    sim_cum_sd = sim_cum.std(axis=0)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # Left: daily incidence
    ax = axes[0]
    x = np.arange(SIM_DAYS)
    ax.bar(x, obs, color="#B5D4F4", width=1.0, label="Observed cases (episode date)", zorder=1)
    ax.plot(x, sim_mean, color="#993C1D", lw=2.0, label="Simulated mean", zorder=3)
    ax.fill_between(x, sim_mean - sim_sd, sim_mean + sim_sd,
                    color="#F0997B", alpha=0.35, label="Simulated ±1 SD", zorder=2)
    ax.axvline(obs_peak, color="#5F5E5A", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(f"Date (from {START_DATE})")
    ax.set_ylabel("Daily new cases")
    ax.set_title("Daily incidence")
    ticks, labels = month_ticks(SIM_DAYS)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(0, SIM_DAYS)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Right: cumulative incidence
    ax = axes[1]
    ax.plot(x, cobs, color="#185FA5", lw=2.0, label="Observed cumulative", zorder=3)
    ax.plot(x, sim_cum_mean, color="#993C1D", lw=2.0, label="Simulated cumulative", zorder=3)
    ax.fill_between(x, sim_cum_mean - sim_cum_sd, sim_cum_mean + sim_cum_sd,
                    color="#F0997B", alpha=0.3, label="Simulated ±1 SD", zorder=2)
    ax.axvline(obs_peak, color="#5F5E5A", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(f"Date (from {START_DATE})")
    ax.set_ylabel("Cumulative cases")
    ax.set_title("Cumulative incidence")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(0, SIM_DAYS)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Mpox epidemic curve: simulated vs observed (LAC, {args.seeds} seeds, "
                 f"median total {np.median(curves.sum(axis=1)):.0f} vs observed {obs_total:.0f})",
                 fontsize=12, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(args.outdir, "curve_compare.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "curve_compare_300dpi.png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")
    print(f"Simulated total: mean={sim_mean.sum():.0f} median={np.median(curves.sum(axis=1)):.0f} "
          f"sd={curves.sum(axis=1).std():.0f} | observed total={obs_total:.0f}")
    print(f"Simulated peak day: mean={np.argmax(sim_mean)} | observed peak day={obs_peak}")
