"""Composition-sensitivity figure: how each parameter shifts the sexual
orientation / sex composition of cases, independent of total magnitude.

The tornado plot only shows TOTAL-case sensitivity, which obscures the key
structural knobs (e.g. one-off bi-male weight) that dominate the homo/bi/
hetero/female split without moving total much. This figure plots, for each
parameter, the deviation of the female fraction (and homo fraction) across
its literature bounds.

Outputs: outputs/figures/composition_sensitivity_final.png + .pdf
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
    "natural_history": "#185FA5",
    "per_act": "#D85A30",
    "nonsexual": "#639922",
    "structural": "#7A5EA8",
}


def main():
    rows = []
    with open(CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    base = [r for r in rows if r["parameter"] == "_baseline_"][0]
    base_female = float(base["female"]) * 100
    base_homo = float(base["homo"]) * 100
    base_hetero = float(base["hetero"]) * 100

    # group by parameter
    by_param = {}
    cat_of = {}
    for r in rows:
        p = r["parameter"]
        if p == "_baseline_" or p not in LABELS:
            continue
        by_param.setdefault(p, []).append(
            (float(r["value"]), float(r["female"]) * 100, float(r["homo"]) * 100,
             float(r["hetero"]) * 100))
        cat_of[p] = r["category"]

    entries = []
    for p, vals in by_param.items():
        vals.sort(key=lambda x: x[0])
        lo_f = vals[0][1]; hi_f = vals[-1][1]
        lo_h = vals[0][2]; hi_h = vals[-1][2]
        lo_ht = vals[0][3]; hi_ht = vals[-1][3]
        entries.append({
            "param": p, "label": LABELS[p], "cat": cat_of[p],
            "f_lo": lo_f - base_female, "f_hi": hi_f - base_female,
            "h_lo": lo_h - base_homo, "h_hi": hi_h - base_homo,
            "ht_lo": lo_ht - base_hetero, "ht_hi": hi_ht - base_hetero,
            "f_spread": abs((hi_f - base_female) - (lo_f - base_female)),
        })

    # sort by female spread descending (composition relevance)
    entries.sort(key=lambda e: e["f_spread"], reverse=True)
    n = len(entries)
    y = np.arange(n)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 0.5 * n + 1.8), sharey=True)

    for ax, key in zip(axes, [("f", "female"), ("h", "homo"), ("ht", "hetero")]):
        k, title = key
        for i, e in enumerate(entries):
            color = CATEGORY_COLOR.get(e["cat"], "#888780")
            lo = e[f"{k}_lo"]; hi = e[f"{k}_hi"]
            left = min(lo, hi); width = abs(hi - lo)
            ax.barh(y[i], width, left=left, height=0.62, color=color, alpha=0.78,
                    edgecolor="none", zorder=2)
            ax.plot([0, 0], [y[i] - 0.35, y[i] + 0.35], color="#5F5E5A", lw=0.8, zorder=3)
        ax.axvline(0, color="#5F5E5A", lw=1.1, ls="--", zorder=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([e["label"] for e in entries], fontsize=9)
    axes[1].set_xlabel("deviation from baseline (percentage points)", fontsize=10)

    fig.suptitle(
        f"Composition sensitivity: sex/orientation split across literature bounds\n"
        f"(baseline: homo={base_homo:.1f}%, hetero={base_hetero:.1f}%, female={base_female:.1f}%)",
        fontsize=12, fontweight="bold", x=0.02, ha="left")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=CATEGORY_COLOR["per_act"], label="per-act transmission"),
        Patch(color=CATEGORY_COLOR["natural_history"], label="natural history"),
        Patch(color=CATEGORY_COLOR["nonsexual"], label="nonsexual contact"),
        Patch(color=CATEGORY_COLOR["structural"], label="structural assumption"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", fontsize=8, frameon=False,
               ncol=4, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"composition_sensitivity_final.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"baseline: homo={base_homo:.1f}% hetero={base_hetero:.1f}% female={base_female:.1f}%")
    print(f"{'parameter':<30} {'fem_lo':>7} {'fem_hi':>7} {'hom_lo':>7} {'hom_hi':>7}")
    for e in entries:
        print(f"{e['label']:<30} {e['f_lo']:>7.2f} {e['f_hi']:>7.2f} "
              f"{e['h_lo']:>7.2f} {e['h_hi']:>7.2f}")
    print(f"\nSaved to {os.path.join(OUTDIR, 'composition_sensitivity_final.png')}")


if __name__ == "__main__":
    main()
