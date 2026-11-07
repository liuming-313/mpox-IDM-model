"""Publication-quality main figure: observed vs simulated epidemic curve.

Reads the CURRENT frozen baseline daily trajectory (outputs/v4_daily_trajectory)
and produces a two-panel figure (Spicknall MMWR 2022 / Endo Science 2022 style):
  a) daily episode cases: observed bars + simulated median + 95% interval
  b) cumulative episode cases: observed + simulated median + 95% interval

Also annotates observed/simulated totals and peak timing.

Output: outputs/figures/fig_main_epicurve.png + .pdf
"""
import os
import ast
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(ROOT, "outputs", "v4_daily_trajectory")
SUMMARY = os.path.join(TRAJ, "daily_trajectory_summary_by_day.csv")
OUTDIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUTDIR, exist_ok=True)

POPULATION_PY = os.path.join(ROOT, "population.py")
START_DATE = "2022-05-01"
OBSERVED_TOTAL = 2280.0


def read_case_data(path):
    """Extract the literal Case_data list without importing population.py."""
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
    rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8-sig")))
    dates = [datetime.fromisoformat(r["date"]) for r in rows]
    ndays = len(rows)

    obs = np.asarray(read_case_data(POPULATION_PY), dtype=float)[:ndays]
    obs_cum = np.cumsum(obs)

    daily_mean = np.array([float(r["daily_episode_cases_mean"]) for r in rows])
    daily_p025 = np.array([float(r["daily_episode_cases_p025"]) for r in rows])
    daily_p975 = np.array([float(r["daily_episode_cases_p975"]) for r in rows])
    cum_mean = np.array([float(r["cumulative_episode_cases_mean"]) for r in rows])
    cum_p025 = np.array([float(r["cumulative_episode_cases_p025"]) for r in rows])
    cum_p975 = np.array([float(r["cumulative_episode_cases_p975"]) for r in rows])

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
    ax.bar(x, obs, color="#B5D4F4", width=1.0, label="Observed (episode date)",
           zorder=1)
    ax.plot(x, daily_mean, color="#993C1D", lw=2.0, label="Simulated median", zorder=3)
    ax.fill_between(x, daily_p025, daily_p975, color="#F0997B", alpha=0.35,
                    label="Simulated 95% interval", zorder=2)
    ax.axvline(obs_peak, color="#5F5E5A", ls="--", lw=1.0, alpha=0.6)
    ax.text(obs_peak + 2, ax.get_ylim()[1] * 0.92, f"obs peak\nday {obs_peak}",
            fontsize=8, color="#5F5E5A", va="top")
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
                 f"final simulated total {cum_mean[-1]:.0f} vs observed {OBSERVED_TOTAL:.0f}",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_main_epicurve.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"observed: total={obs.sum():.0f} peak_day={obs_peak}")
    print(f"simulated: total_mean={cum_mean[-1]:.0f} peak_day={sim_peak}")
    print(f"Saved {os.path.join(OUTDIR, 'fig_main_epicurve.png')}")


if __name__ == "__main__":
    main()
