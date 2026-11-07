"""Regenerate the main epicurve figure from the NEW scheme (seed0 + sustained
importation), using the 30-seed validation runs in outputs/v5_importation_val30.

Reads per-seed daily CSV -> computes daily/cumulative mean + 95% interval ->
renders the two-panel figure (daily + cumulative), annotated with peak timing
and totals. This supersedes make_fig_main_epicurve.py (which read the old
v4_daily_trajectory baseline).

Output: outputs/figures/fig_main_epicurve_final.png + .pdf
"""
import os
import ast
import csv
import glob
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(ROOT, "outputs", "v7_final_bibi1.2")
OUTDIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUTDIR, exist_ok=True)

POPULATION_PY = os.path.join(ROOT, "population.py")
START_DATE = "2022-05-01"


def read_case_data(path):
    def eval_seq(node):
        if isinstance(node, ast.List):
            return [ast.literal_eval(i) for i in node.elts]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return eval_seq(node.left) + eval_seq(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            l, r = eval_seq(node.left), eval_seq(node.right)
            if isinstance(l, list) and isinstance(r, int):
                return l * r
            if isinstance(l, int) and isinstance(r, list):
                return l * r
        raise ValueError(ast.dump(node))
    mod = ast.parse(open(path, encoding="utf-8").read())
    for node in mod.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "Case_data":
                    return [float(v) for v in eval_seq(node.value)]
    raise RuntimeError("Case_data not found")


def load_seed_matrix():
    files = sorted(glob.glob(os.path.join(TRAJ, "seed*.csv")))
    mat = []
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        arr = np.array([float(r["daily_episode_cases"]) for r in rows])
        mat.append(arr)
    mat = np.array(mat)
    print(f"loaded {mat.shape[0]} seeds x {mat.shape[1]} days")
    return mat


def month_ticks(ndays):
    ticks, labels = [], []
    months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan"]
    for d in range(0, ndays + 1, 30):
        mi = d // 30
        if mi < len(months):
            ticks.append(d)
            labels.append(months[mi])
    return ticks, labels


def main():
    mat = load_seed_matrix()
    nseed, ndays = mat.shape

    obs = np.asarray(read_case_data(POPULATION_PY), dtype=float)[:ndays]
    obs_cum = np.cumsum(obs)
    OBSERVED_TOTAL = obs.sum()

    daily_mean = mat.mean(axis=0)
    daily_p025 = np.percentile(mat, 2.5, axis=0)
    daily_p975 = np.percentile(mat, 97.5, axis=0)
    cum = np.cumsum(mat, axis=1)
    cum_mean = cum.mean(axis=0)
    cum_p025 = np.percentile(cum, 2.5, axis=0)
    cum_p975 = np.percentile(cum, 97.5, axis=0)

    x = np.arange(ndays)
    obs_peak = int(np.argmax(obs))
    sim_peak = int(np.argmax(daily_mean))

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    # panel a: daily
    ax = axes[0]
    ax.bar(x, obs, color="#B5D4F4", width=1.0, label="Observed (episode date)", zorder=1)
    ax.plot(x, daily_mean, color="#993C1D", lw=2.0, label="Simulated median", zorder=3)
    ax.fill_between(x, daily_p025, daily_p975, color="#F0997B", alpha=0.35,
                    label="Simulated 95% interval", zorder=2)
    ax.axvline(obs_peak, color="#5F5E5A", ls="--", lw=1.0, alpha=0.6)
    ax.axvline(sim_peak, color="#993C1D", ls="--", lw=1.0, alpha=0.6)
    ymax = max(obs.max(), daily_p975.max()) * 1.1
    ax.set_ylim(0, ymax)
    ax.text(obs_peak + 1, ymax * 0.90, f"obs peak day {obs_peak}",
            fontsize=8, color="#5F5E5A", va="top")
    ax.text(sim_peak + 1, ymax * 0.78, f"sim peak day {sim_peak}",
            fontsize=8, color="#993C1D", va="top")
    ax.set_xlabel(f"Date (from {START_DATE})")
    ax.set_ylabel("Daily cases")
    ax.set_title("Daily incidence", loc="left", fontweight="bold")
    ticks, labels = month_ticks(ndays)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(0, ndays)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    # panel b: cumulative
    ax = axes[1]
    ax.plot(x, obs_cum, color="#185FA5", lw=2.0, label="Observed cumulative", zorder=3)
    ax.plot(x, cum_mean, color="#993C1D", lw=2.0, label="Simulated median", zorder=3)
    ax.fill_between(x, cum_p025, cum_p975, color="#F0997B", alpha=0.3,
                    label="Simulated 95% interval", zorder=2)
    ax.axhline(OBSERVED_TOTAL, color="#185FA5", lw=1.2, ls=":", alpha=0.8)
    ax.text(ndays * 0.98, OBSERVED_TOTAL * 1.02, f"observed total {OBSERVED_TOTAL:.0f}",
            fontsize=8, color="#185FA5", ha="right", va="bottom")
    ax.set_xlabel(f"Date (from {START_DATE})")
    ax.set_ylabel("Cumulative cases")
    ax.set_title("Cumulative incidence", loc="left", fontweight="bold")
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_xlim(0, ndays)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.text(0.02, 0.97, "b", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    fig.suptitle(f"Mpox epidemic curve: simulated vs observed (LAC 2022) — "
                 f"median total {cum_mean[-1]:.0f} vs observed {OBSERVED_TOTAL:.0f}",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_main_epicurve_final.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"observed: total={obs.sum():.0f} peak_day={obs_peak}")
    print(f"simulated: total_median={cum_mean[-1]:.0f} peak_day={sim_peak}")
    print(f"Saved {os.path.join(OUTDIR, 'fig_main_epicurve_final.png')}")


if __name__ == "__main__":
    main()
