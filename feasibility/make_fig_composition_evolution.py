"""Publication-quality composition-evolution panel figure.

Reads the CURRENT frozen baseline daily trajectory and produces a four-panel
figure showing how the case composition evolves over the outbreak:
  a) daily episode cases (simulated median + 95% interval + observed)
  b) cumulative episode cases
  c) sex composition (male / female, %)
  d) sexual-orientation composition (homo / bi / hetero, %)

Panels c/d overlay the observed final targets as dashed lines, illustrating the
"MSM core -> bisexual bridge -> heterosexual spillover -> female tail" mechanism.

Output: outputs/figures/fig_composition_evolution.png + .pdf
"""
import os
import ast
import csv
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAJ = os.path.join(ROOT, "outputs", "v4_daily_trajectory")
SUMMARY = os.path.join(TRAJ, "daily_trajectory_summary_by_day.csv")
LONG = os.path.join(TRAJ, "daily_trajectory_long.csv")
OUTDIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUTDIR, exist_ok=True)

POPULATION_PY = os.path.join(ROOT, "population.py")
MIN_CUM_FOR_COMP = 25.0

# observed final targets (LAC 2022-12-30 known-orientation table)
TARGETS = {
    "homo": 1560 / 2012,
    "bi": 253 / 2012,
    "hetero": (125 + 74) / 2012,
    "female_known": 74 / 2012,
}


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


def median_by_day(long_rows, metric, min_total):
    grouped = defaultdict(list)
    for r in long_rows:
        if float(r["cumulative_episode_cases"]) < min_total:
            continue
        grouped[int(r["day"])].append(float(r[metric]))
    return [np.median(grouped[d]) if grouped[d] else np.nan for d in range(ndays)]


def median_ratio_by_day(long_rows, num, den, min_total):
    grouped = defaultdict(list)
    for r in long_rows:
        if float(r["cumulative_episode_cases"]) < min_total or float(r[den]) == 0:
            continue
        grouped[int(r["day"])].append(float(r[num]) / float(r[den]) * 100.0)
    return [np.median(grouped[d]) if grouped[d] else np.nan for d in range(ndays)]


summary_rows = list(csv.DictReader(open(SUMMARY, encoding="utf-8-sig")))
long_rows = list(csv.DictReader(open(LONG, encoding="utf-8-sig")))
dates = [datetime.fromisoformat(r["date"]) for r in summary_rows]
ndays = len(summary_rows)

obs = np.asarray(read_case_data(POPULATION_PY), dtype=float)[:ndays]
obs_cum = np.cumsum(obs)

daily_mean = np.array([float(r["daily_episode_cases_mean"]) for r in summary_rows])
daily_p025 = np.array([float(r["daily_episode_cases_p025"]) for r in summary_rows])
daily_p975 = np.array([float(r["daily_episode_cases_p975"]) for r in summary_rows])
cum_mean = np.array([float(r["cumulative_episode_cases_mean"]) for r in summary_rows])
cum_p025 = np.array([float(r["cumulative_episode_cases_p025"]) for r in summary_rows])
cum_p975 = np.array([float(r["cumulative_episode_cases_p975"]) for r in summary_rows])

male_med = median_ratio_by_day(long_rows, "cumulative_episode_male", "cumulative_episode_cases", MIN_CUM_FOR_COMP)
female_med = median_ratio_by_day(long_rows, "cumulative_episode_female", "cumulative_episode_cases", MIN_CUM_FOR_COMP)
homo_med = median_by_day(long_rows, "cum_ratio_homo", MIN_CUM_FOR_COMP)
bi_med = median_by_day(long_rows, "cum_ratio_bi", MIN_CUM_FOR_COMP)
hetero_med = median_by_day(long_rows, "cum_ratio_hetero", MIN_CUM_FOR_COMP)
homo_med = [v * 100 for v in homo_med]
bi_med = [v * 100 for v in bi_med]
hetero_med = [v * 100 for v in hetero_med]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

# a) daily
ax = axes[0, 0]
ax.plot(dates, daily_mean, color="#E23B3B", lw=2.0, label="Model median")
ax.fill_between(dates, daily_p025, daily_p975, color="#E23B3B", alpha=0.18, label="95% interval")
ax.plot(dates, obs, color="#222222", lw=1.3, label="Observed")
ax.set_ylabel("Daily cases")
ax.legend(frameon=False, fontsize=8)
ax.text(0.0, 1.03, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")

# b) cumulative
ax = axes[0, 1]
ax.plot(dates, cum_mean, color="#E23B3B", lw=2.0, label="Model median")
ax.fill_between(dates, cum_p025, cum_p975, color="#E23B3B", alpha=0.18, label="95% interval")
ax.plot(dates, obs_cum, color="#222222", lw=1.6, label="Observed")
ax.axhline(OBSERVED_TOTAL := 2280.0, color="#222222", lw=1.1, ls="--", label="Observed final")
ax.set_ylabel("Cumulative cases")
ax.legend(frameon=False, fontsize=8)
ax.text(0.0, 1.03, "b", transform=ax.transAxes, fontsize=13, fontweight="bold")

# c) sex composition
ax = axes[1, 0]
ax.plot(dates, male_med, lw=1.8, label="Male", color="#4E79A7")
ax.plot(dates, female_med, lw=1.8, label="Female", color="#B07AA1")
ax.axhline(100 - TARGETS["female_known"] * 100, color="#4E79A7", lw=1.1, ls="--", alpha=0.9)
ax.axhline(TARGETS["female_known"] * 100, color="#B07AA1", lw=1.1, ls="--", alpha=0.9)
ax.set_ylabel("Percent of cases")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.text(0.0, 1.03, "c", transform=ax.transAxes, fontsize=13, fontweight="bold")

# d) orientation composition
ax = axes[1, 1]
ax.plot(dates, homo_med, lw=1.8, label="Homosexual", color="#E6D900")
ax.plot(dates, bi_med, lw=1.8, label="Bisexual", color="#F28E2B")
ax.plot(dates, hetero_med, lw=1.8, label="Heterosexual", color="#59A14F")
ax.axhline(TARGETS["homo"] * 100, color="#E6D900", lw=1.1, ls="--", alpha=0.9)
ax.axhline(TARGETS["bi"] * 100, color="#F28E2B", lw=1.1, ls="--", alpha=0.9)
ax.axhline(TARGETS["hetero"] * 100, color="#59A14F", lw=1.1, ls="--", alpha=0.9)
ax.set_ylabel("Percent of cases")
ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="center right")
ax.text(0.0, 1.03, "d", transform=ax.transAxes, fontsize=13, fontweight="bold")

for ax in axes.flat:
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle("Case composition evolution — MSM core → bisexual bridge → heterosexual/female spillover",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUTDIR, f"fig_composition_evolution.{ext}"), dpi=300,
                bbox_inches="tight")
plt.close(fig)
print(f"Saved {os.path.join(OUTDIR, 'fig_composition_evolution.png')}")
