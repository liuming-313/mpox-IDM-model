"""Generate publication-quality tornado plot for the one-way sensitivity analysis.

Reads outputs/sensitivity/table_s8_final.csv (FINAL locked parameter set,
seed=300, one-at-a-time 3-level sweeps around the locked baseline) and produces
a horizontal tornado chart of total-case deviation from the baseline.

The tornado shows, per parameter, the range of total episodes spanned by the
low and high literature-derived bounds. Parameters are sorted by spread
(descending), highlighting the transmission-intensity knobs that dominate
(receptive/insertive anal, infectious period) versus the structural
assumptions that barely move total.

Output: outputs/figures/tornado_sensitivity_final.png + .pdf
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "outputs", "sensitivity", "table_s8_final.csv")
OUTDIR = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUTDIR, exist_ok=True)

# human-friendly labels (parameter -> display name)
LABELS = {
    "incubation_lognorm_mu": "incubation μ (lognormal)",
    "incubation_lognorm_sigma": "incubation σ (lognormal)",
    "infectious_period_gamma_scale": "infectious period scale",
    "sexual_anal_receptive": "receptive anal β",
    "sexual_anal_insertive": "insertive anal β",
    "sexual_vaginal": "vaginal β",
    "sexual_oral": "oral β",
    "social_p_eff": "social per-contact p_eff",
    "party_touch_frac": "party touch fraction",
    "oneoff_mf_anal_frac": "one-off MF anal fraction",
    "monitor_sexual_contact_mult": "monitoring sexual-contact mult",
    "bi_same_sex_frac": "bi same-sex fraction",
    "social_same_sex_assort": "social same-sex assortativity",
    "mix_mf_anal_frac": "MIX MF anal fraction (M→F)",
    "mix_fm_anal_frac": "MIX FM anal fraction (F→M)",
    "party_mix_target_mean": "MIX party target mean",
    "oneoff_orientation_bias_bi": "one-off bi-male weight",
}

CATEGORY_COLOR = {
    "natural_history": "#185FA5",   # blue
    "per_act": "#D85A30",           # red-orange
    "nonsexual": "#639922",         # green
    "structural": "#7A5EA8",        # purple
}


def main():
    rows = []
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # baseline total
    base = [r for r in rows if r["parameter"] == "_baseline_"][0]
    base_total = float(base["total"])

    # group by parameter (exclude baseline), get low/high total
    by_param = {}
    cat_of = {}
    for r in rows:
        p = r["parameter"]
        if p == "_baseline_":
            continue
        if p not in LABELS:
            continue
        by_param.setdefault(p, []).append((float(r["value"]), float(r["total"])))
        cat_of[p] = r["category"]

    entries = []
    for p, vals in by_param.items():
        vals.sort(key=lambda x: x[0])  # sort by value ascending
        lo_total = vals[0][1]
        hi_total = vals[-1][1]
        lo_val = vals[0][0]
        hi_val = vals[-1][0]
        dev_lo = lo_total - base_total
        dev_hi = hi_total - base_total
        entries.append({
            "param": p,
            "label": LABELS[p],
            "lo_val": lo_val, "hi_val": hi_val,
            "dev_lo": dev_lo, "dev_hi": dev_hi,
            "spread": abs(dev_hi - dev_lo),
            "cat": cat_of[p],
        })

    # sort by spread descending
    entries.sort(key=lambda e: e["spread"], reverse=True)

    n = len(entries)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(11, 0.5 * n + 1.8))

    # compute symmetric x-limits so the label area on the left is reserved
    xmax = max(abs(e["dev_lo"]) for e in entries) * 1.18
    xmax = max(xmax, max(abs(e["dev_hi"]) for e in entries) * 1.18)

    for i, e in enumerate(entries):
        color = CATEGORY_COLOR.get(e["cat"], "#888780")
        left = min(e["dev_lo"], e["dev_hi"])
        width = abs(e["dev_hi"] - e["dev_lo"])
        ax.barh(y[i], width, left=left, height=0.62, color=color, alpha=0.78,
                edgecolor="none", zorder=2)
        ax.plot([0, 0], [y[i] - 0.35, y[i] + 0.35], color="#5F5E5A", lw=1.0, zorder=3)
        # annotate low/high totals (clamp inside the plot to avoid overlap)
        ax.text(left - 10, y[i], f"{left + base_total:.0f}", va="center", ha="right",
                fontsize=8, color="#444441")
        ax.text(left + width + 10, y[i], f"{left + width + base_total:.0f}", va="center",
                ha="left", fontsize=8, color="#444441")

    ax.set_xlim(-xmax, xmax)

    ax.axvline(0, color="#5F5E5A", lw=1.2, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([e["label"] for e in entries], fontsize=10)
    ax.set_xlabel("Deviation from baseline total cases (baseline = %.0f)" % base_total,
                  fontsize=11)
    ax.set_title("One-way sensitivity: total-case deviation across literature bounds\n"
                 "(final locked parameters, seed = 300)",
                 fontsize=11, fontweight="bold", loc="left")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=CATEGORY_COLOR["per_act"], label="per-act transmission"),
        Patch(color=CATEGORY_COLOR["natural_history"], label="natural history"),
        Patch(color=CATEGORY_COLOR["nonsexual"], label="nonsexual contact"),
        Patch(color=CATEGORY_COLOR["structural"], label="structural assumption"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=False,
              bbox_to_anchor=(0.0, -0.16))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"tornado_sensitivity_final.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    print(f"baseline total = {base_total:.0f}")
    print(f"{'parameter':<30} {'low':>6} {'high':>6} {'dev_lo':>8} {'dev_hi':>8} {'spread':>8}")
    for e in entries:
        print(f"{e['label']:<30} {e['lo_val']:>6} {e['hi_val']:>6} "
              f"{e['dev_lo']:>8.0f} {e['dev_hi']:>8.0f} {e['spread']:>8.0f}")
    print(f"\nSaved to {os.path.join(OUTDIR, 'tornado_sensitivity_final.png')}")


if __name__ == "__main__":
    main()
