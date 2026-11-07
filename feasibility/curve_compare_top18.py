"""Formal simulated-vs-observed epidemic curve comparison for the CURRENT
top_mean=18 baseline.

Reuses main.py's `params` (top_mean=18, gi_sd=5.0, social_contact_scale=10,
etc.) and the 3-phase policy (party_mult 0.95/0.70/0.70) so the figure and
fit metrics reflect the exact configuration that produces the ~2280-case
median, rather than the stale hard-coded top_mean=20 config in the older
curve_compare_plot.py.

Runs N seeds of the FINAL locked configuration (20 uniform MSM seeds) and
outputs:
  - Left panel:  daily incidence (observed bars + simulated mean + 1-sd band)
  - Right panel: cumulative incidence (observed vs simulated mean + band)
  - A CSV of per-seed and aggregate fit metrics:
      total, peak-day error, daily RMSE, cumulative RMSE,
      25/50/75/90% cumulative-share days (obs vs sim),
      late-tail shares (day>120, day>150)

Reproducibility: each seed uses np.random.seed(seed) + a fresh
default_rng(seed) assigned to pop._rng before seed_infection, matching the
fixed pattern used by verify_reproducibility.py.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import params, SIM_DAYS, POPULATION_SIZE, INITIAL_INFECTIONS
from population import Population, Case_data

START_DATE = "2022-05-01"


def make_policy():
    # Identical to main.py's 3-phase policy (party_mult 0.95/0.70/0.70).
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
    pop = Population(n=POPULATION_SIZE, params=params, seed=seed,
                     cache_label=f"curve18_{seed}", policy=make_policy(), importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(INITIAL_INFECTIONS, mask=mask)
    pop.run_simulation(SIM_DAYS)
    return np.asarray(pop.daily_episode_cases[:SIM_DAYS], dtype=float)


def month_ticks(ndays):
    months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"]
    ticks, labels = [], []
    for d in range(0, ndays + 1, 30):
        mi = d // 30
        if mi < len(months):
            ticks.append(d); labels.append(months[mi])
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
    obs_pct = {p: int(np.searchsorted(cobs, p / 100 * obs_total))
               for p in [25, 50, 75, 90]}
    obs_tail120 = 100 * obs[120:].sum() / obs_total
    obs_tail150 = 100 * obs[150:].sum() / obs_total

    curves, rows = [], []
    print("Running seeds...", flush=True)
    for i in range(args.seeds):
        seed = 300 + i * 11
        t0 = time.time()
        sim = run_one(seed)
        curves.append(sim)
        tot = sim.sum()
        pk = int(np.argmax(sim))
        csim = np.cumsum(sim)
        s_pct = {p: int(np.searchsorted(csim, p / 100 * tot)) if tot > 0 else -1
                 for p in [25, 50, 75, 90]}
        daily_rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
        c_rmse = float(np.sqrt(np.mean((csim - cobs) ** 2)))
        rows.append(dict(seed=seed, total=tot, peak=pk,
                         peak_err=pk - obs_peak, daily_rmse=daily_rmse,
                         crmse=c_rmse, d25=s_pct[25], d50=s_pct[50],
                         d75=s_pct[75], d90=s_pct[90],
                         tail120=100 * sim[120:].sum() / tot,
                         tail150=100 * sim[150:].sum() / tot))
        print(f"  seed {seed}: total={tot:.0f} peak={pk} cRMSE={c_rmse:.0f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    curves = np.array(curves)
    sim_mean = curves.mean(axis=0)
    sim_sd = curves.std(axis=0)
    sim_cum_mean = np.cumsum(curves, axis=1).mean(axis=0)
    sim_cum_sd = np.cumsum(curves, axis=1).std(axis=0)

    os.makedirs(args.outdir, exist_ok=True)

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(SIM_DAYS)

    ax = axes[0]
    ax.bar(x, obs, color="#B5D4F4", width=1.0, label="Observed cases (episode date)", zorder=1)
    ax.plot(x, sim_mean, color="#993C1D", lw=2.0, label="Simulated mean", zorder=3)
    ax.fill_between(x, sim_mean - sim_sd, sim_mean + sim_sd,
                    color="#F0997B", alpha=0.35, label="Simulated \u00b11 SD", zorder=2)
    ax.axvline(obs_peak, color="#5F5E5A", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(f"Date (from {START_DATE})")
    ax.set_ylabel("Daily new cases")
    ax.set_title("Daily incidence")
    ticks, labels = month_ticks(SIM_DAYS)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(0, SIM_DAYS); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(x, cobs, color="#185FA5", lw=2.0, label="Observed cumulative", zorder=3)
    ax.plot(x, sim_cum_mean, color="#993C1D", lw=2.0, label="Simulated cumulative", zorder=3)
    ax.fill_between(x, sim_cum_mean - sim_cum_sd, sim_cum_mean + sim_cum_sd,
                    color="#F0997B", alpha=0.3, label="Simulated \u00b11 SD", zorder=2)
    ax.axvline(obs_peak, color="#5F5E5A", ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel(f"Date (from {START_DATE})")
    ax.set_ylabel("Cumulative cases")
    ax.set_title("Cumulative incidence")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(0, SIM_DAYS); ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"Mpox epidemic curve: simulated vs observed (LAC, {args.seeds} seeds, "
                 f"median total {np.median(curves.sum(axis=1)):.0f} vs observed {obs_total:.0f})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(args.outdir, "curve_compare_top18.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(args.outdir, "curve_compare_top18_300dpi.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- aggregate report ----
    totals = np.array([r["total"] for r in rows])
    print("\n" + "=" * 78)
    print(f"Observed : total={obs_total:.0f} peak_day={obs_peak} "
          f"pct(25/50/75/90)={obs_pct[25]}/{obs_pct[50]}/{obs_pct[75]}/{obs_pct[90]} "
          f"tail120={obs_tail120:.1f}% tail150={obs_tail150:.1f}%")
    print(f"Simulated: total mean={totals.mean():.0f} median={np.median(totals):.0f} "
          f"std={totals.std():.0f}")
    for key, label in [("peak_err", "peak-day error (sim-obs)"),
                       ("daily_rmse", "daily RMSE"), ("crmse", "cumulative RMSE")]:
        vals = np.array([r[key] for r in rows])
        print(f"  {label:<26}: mean={vals.mean():.1f} median={np.median(vals):.1f}")
    for p in [25, 50, 75, 90]:
        vals = np.array([r[f"d{p}"] for r in rows])
        print(f"  {p}% cumulative day       : obs={obs_pct[p]}  sim mean={vals.mean():.1f} "
              f"(median {np.median(vals):.0f}, lead {np.median(vals) - obs_pct[p]:+.0f}d)")
    for t in [120, 150]:
        vals = np.array([r[f"tail{t}"] for r in rows])
        obs_tail = obs_tail120 if t == 120 else obs_tail150
        print(f"  late tail (day>{t}) share   : obs={obs_tail:.1f}%  "
              f"sim mean={vals.mean():.1f}% (median {np.median(vals):.1f}%)")

    # ---- CSV ----
    import csv
    csv_path = os.path.join(args.outdir, "curve_compare_top18_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {os.path.join(args.outdir, 'curve_compare_top18.png')}")
    print(f"Metrics CSV: {csv_path}")
