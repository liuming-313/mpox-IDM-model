"""测试传染期 21 天 vs 27 天对时间尺度和 total 的影响。

假设验证：CDC Spicknall 用 27 天传染期（vs 当前 21 天）。
预期（理论）：传染期延长 → R0 增大 → total 涨，但代际时间主要由潜伏期决定，
时间尺度（首例/峰值/50%日）基本不变或略提前。
"""
import sys, time, io, csv, os
import numpy as np
from contextlib import redirect_stdout

sys.path.insert(0, '.')
from population import Population, Case_data

src = open("main.py", encoding="utf-8").read()
ps = src.find("params = {") + len("params = ")
pe = src.find("params[", ps)
CT = {'target_homo': 1560/2012, 'target_bi': 253/2012, 'target_hetero': (125+74)/2012,
      'target_female_known': 74/2012, 'target_homo_male': 1560/2012, 'target_bi_male': 253/2012}
params = eval(src[ps:pe].rstrip(), {'__builtins__': {}, 'SIM_DAYS': 244, 'CALIBRATION_TARGETS': CT}, {})
pls = src.find("policy = [")
ple = src.find("]\n", pls) + 1
policy = eval(src[pls+len("policy = "):ple], {'__builtins__': {}, 'dict': dict, 'SIM_DAYS': 244}, {})

POP_SIZE = 10_000_000
SIM_DAYS = 244
SEED = 100001

obs = np.asarray(Case_data, dtype=float)


def key_times(a):
    a = np.asarray(a, dtype=float)
    nz = np.where(a > 0)[0]
    if nz.size == 0:
        return dict(first=None, peak=None, p50=None, p90=None, last=None, total=0.0)
    cum = np.cumsum(a)
    total = cum[-1]
    return dict(first=int(nz[0]), peak=int(np.argmax(a)),
                p50=int(np.searchsorted(cum, total*0.5)),
                p90=int(np.searchsorted(cum, total*0.9)),
                last=int(nz[-1]), total=float(total))


def run_case(mean_days, label):
    """跑一个传染期 = mean_days 的模拟。gamma(shape=9, scale=mean/9)。"""
    p = dict(params)
    # 保持 shape=9，scale = mean/9 → 均值 = mean
    p["infectious_period_gamma_shape"] = 9.0
    p["infectious_period_gamma_scale"] = mean_days / 9.0
    # clip 上限相应放宽到 21-42（保留 27 天均值的合理尾部）
    p["infectious_period_clip"] = (14, 42)

    pop = Population(n=POP_SIZE, params=p, seed=42,
                     cache_label="10M_homo_male_mcp", policy=policy, importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(SEED)
    pop._rng = np.random.default_rng(SEED)
    pop.seed_infection(20, mask=mask)
    with redirect_stdout(io.StringIO()):
        pop.run_simulation(SIM_DAYS)
    sim_ep = np.asarray(getattr(pop, "daily_episode_cases", []), dtype=float)
    # 构成
    total_inf = int(np.sum(sim_ep)) if sim_ep.size else 0
    return sim_ep


def summarize(sim_ep):
    sk = key_times(sim_ep)
    ok = key_times(obs)
    return {
        "first": sk["first"], "peak": sk["peak"], "p50": sk["p50"],
        "p90": sk["p90"], "last": sk["last"], "total": sk["total"],
        "first_off": (sk["first"] or 0) - ok["first"],
        "peak_off": (sk["peak"] or 0) - ok["peak"],
        "p50_off": (sk["p50"] or 0) - ok["p50"],
        "p90_off": (sk["p90"] or 0) - ok["p90"],
        "last_off": (sk["last"] or 0) - ok["last"],
    }


def main():
    cases = [
        (21.0, "传染期21天(现状)"),
        (27.0, "传染期27天(CDC)"),
    ]
    results = []
    for mean_d, label in cases:
        t0 = time.time()
        sim_ep = run_case(mean_d, label)
        s = summarize(sim_ep)
        s["case"] = label
        s["time_s"] = round(time.time()-t0, 1)
        results.append(s)
        print(f"[{label}] total={s['total']:.0f} 首例{s['first_off']:+d}d "
              f"峰值{s['peak_off']:+d}d 50%{s['p50_off']:+d}d 90%{s['p90_off']:+d}d "
              f"末例{s['last_off']:+d}d ({s['time_s']}s)", flush=True)

    os.makedirs("outputs/timing", exist_ok=True)
    with open("outputs/timing/infectious_period_test.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case", "total", "first", "peak", "p50", "p90", "last",
                                          "first_off", "peak_off", "p50_off", "p90_off", "last_off", "time_s"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\n已保存 outputs/timing/infectious_period_test.csv")


if __name__ == "__main__":
    main()
