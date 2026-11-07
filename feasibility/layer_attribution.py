"""
Layer-attribution diagnostic (read-only, does NOT modify main.py / population.py).

Runs ONE seed of the model and reports how many cumulative infections each
transmission layer produced, plus the orientation composition of the
secondary infections arising on each layer (to locate where MSM cases leak
to heterosexuals).

Layer set (from population.py `self.layers`):
    stable, household_close, social, oneoff, sex_party, sex_party_touch,
    small_event, large_event

External importations (the 742 schedule) are applied via _apply_importations
and are NOT attributed to any layer; we report them separately.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from population import Population

# ---- reuse main.py parameter block (import without executing the run loop) ----
import importlib.util

_main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")


def _load_main_params():
    spec = importlib.util.spec_from_file_location("main_mod", _main_path)
    m = importlib.util.module_from_spec(spec)
    # main.py executes the run loop on import, so we only want the params dict.
    # Instead, exec the file but guard the run loop: simplest is to exec only the
    # params block by reading the file and execing up to the SCENARIOS loop.
    src = open(_main_path, "r", encoding="utf-8").read()
    cut = src.index("SCENARIOS = [")
    head = src[:cut]
    ns = {}
    # main.py imports population and matplotlib; we already have them, but exec head
    # is safe because it only defines params/helpers. It also imports at top; re-import ok.
    exec(compile(head, _main_path, "exec"), ns)

    # importations are defined inside main.py's SCENARIOS loop (not module-level),
    # so reproduce the 742-schedule here verbatim.
    importations = [
        dict(start=0, end=30, daily_n=2),
        dict(start=31, end=70, daily_n=4),
        dict(start=71, end=110, daily_n=9),
        dict(start=111, end=150, daily_n=4),
        dict(start=151, end=244, daily_n=0),
    ]
    return ns["params"], importations


def main():
    n = int(os.environ.get("DIAG_N", "2000000"))
    seed = int(os.environ.get("DIAG_SEED", "42"))
    days = 244

    params, importations = _load_main_params()
    params["n_days"] = days + 1

    label = f"diag_{n // 1_000_000}M"

    t0 = time.time()
    pop = Population(n=n, params=params, seed=seed, cache_label=label,
                     policy=None, importations=importations)
    print(f"[build] population built in {time.time() - t0:.1f}s", flush=True)

    # Use the same MSM-only importation mask as main.py's homo_male_mcp scenario.
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    pop.importation_mask = (g == 0) & (o == 'homo') & mcp

    np.random.seed(100001)
    pop._rng = np.random.default_rng(100001)

    pop.run_simulation(days)
    print(f"[run] {days} days in {time.time() - t0:.1f}s", flush=True)

    # ---- total infections & importations ----
    total_inf = int(np.sum(pop.infections_by_day["total"]))
    total_import = sum(
        int(imp.get("daily_n", 0)) * (int(imp.get("end", 0)) - int(imp.get("start", 0)) + 1)
        for imp in importations
    )
    layer_total = dict(pop.layer_infection_counts)
    layer_sum = sum(layer_total.values())

    print("\n================ LAYER ATTRIBUTION ================")
    print(f"population            = {n}")
    print(f"seed                  = {seed}")
    print(f"simulated days        = {days}")
    print(f"cumulative infections = {total_inf}")
    print(f"scheduled importations= {total_import} (not attributed to any layer)")
    print(f"sum over layers       = {layer_sum}")
    print(f"unattributed (total - layers) = {total_inf - layer_sum}")
    print("\n--- cumulative infections by layer (sorted) ---")
    order = sorted(layer_total.items(), key=lambda kv: -kv[1])
    for layer, cnt in order:
        share = 100.0 * cnt / max(layer_sum, 1)
        print(f"  {layer:<18} {cnt:>8}   ({share:5.2f}% of layer-sum)")

    # ---- orientation composition of each layer's secondary infections ----
    # We don't have per-layer orientation logged, so approximate: recompute by
    # instrumenting is not possible read-only. Instead report global orientation
    # composition and note that layer-level orientation requires instrumentation.
    s = pop.summary()
    print("\n--- global orientation composition (episode/case) ---")
    for k in ("episode_homo_ratio", "episode_bi_ratio", "episode_hetero_ratio",
              "episode_female_known_ratio"):
        if k in s:
            print(f"  {k:<28} = {s[k]:.4f}")

    # Save CSV
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "feasibility", "layer_attribution.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8-sig") as f:
        f.write("layer,cumulative_infections,share_of_layer_sum\n")
        for layer, cnt in order:
            f.write(f"{layer},{cnt},{100.0 * cnt / max(layer_sum, 1):.4f}\n")
        f.write(f"importations,{total_import},NA\n")
        f.write(f"TOTAL,{total_inf},NA\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
