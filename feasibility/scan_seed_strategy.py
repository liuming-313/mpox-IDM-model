"""测试 seed 播种策略对「早期过陡」时间偏移的影响。

对比方案：
  A. 现状：day 0 一次性播 20 例 MSM（seed_infection）
  B. 延后播种：day 15 一次性播 20 例
  C. 渐进引入：day 0-15 每天 1 例（importations），共 ~16 例
  D. 渐进 + 首日少量：day 0 播 4 例 + day 0-15 每天 1 例

量化指标：首例日、峰值日、累计50%/90%日、total、构成。
"""
import sys, time, io, csv, os
import numpy as np
from contextlib import redirect_stdout

sys.path.insert(0, '.')
from population import Population, Case_data

# 提取 main.py 的 params + policy
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


def run_case(name, seed_n=None, seed_day=0, import_daily=None, import_range=None):
    pop = Population(n=POP_SIZE, params=params, seed=42,
                     cache_label="10M_homo_male_mcp", policy=policy, importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(SEED)
    pop._rng = np.random.default_rng(SEED)

    # 播种方案
    if import_daily is not None:  # 渐进引入（importations 每天 daily_n 例）
        start, end = import_range
        pop.importations = [dict(start=start, end=end, daily_n=import_daily)]
        # 首日少量 seed 点燃（真实疫情 5/16 才 4 例 MSM，此处 day 0 少量）
        if seed_n:
            pop.seed_infection(seed_n, mask=mask)
    elif seed_day > 0:  # 延后播种：seed_day 当天一次性引入 seed_n 例
        pop.importations = [dict(start=seed_day, end=seed_day, daily_n=seed_n)]
        pop.seed_infection(0, mask=mask)  # 不播种子
    else:
        pop.seed_infection(seed_n, mask=mask)

    with redirect_stdout(io.StringIO()):
        pop.run_simulation(SIM_DAYS)
    sim_ep = np.asarray(getattr(pop, "daily_episode_cases", []), dtype=float)
    return sim_ep


def summarize(sim_ep):
    sk = key_times(sim_ep)
    ok = key_times(obs)
    # 构成
    total = max(sk["total"], 1.0)
    return {
        "first_offset": sk["first"] - ok["first"],
        "peak_offset": sk["peak"] - ok["peak"],
        "p50_offset": sk["p50"] - ok["p50"],
        "p90_offset": sk["p90"] - ok["p90"],
        "last_offset": sk["last"] - ok["last"],
        "total": sk["total"],
    }


def main():
    cases = [
        ("A_现状_day0_20", dict(seed_n=20, seed_day=0)),
        ("B_延后_day15_20", dict(seed_n=20, seed_day=15)),
        ("C_渐进_day0-15_每天1", dict(seed_n=4, import_daily=1, import_range=(0, 15))),
        ("D_渐进_day15-30_每天1", dict(seed_n=4, import_daily=1, import_range=(15, 30))),
    ]
    results = []
    for name, kw in cases:
        t0 = time.time()
        sim_ep = run_case(name, **kw)
        s = summarize(sim_ep)
        s["case"] = name
        s["time_s"] = round(time.time()-t0, 1)
        results.append(s)
        print(f"[{name}] 首例偏移{s['first_offset']:+d}d 峰值{s['peak_offset']:+d}d "
              f"50%{s['p50_offset']:+d}d 90%{s['p90_offset']:+d}d total={s['total']:.0f} "
              f"({s['time_s']}s)", flush=True)

    # 写 CSV
    os.makedirs("outputs/timing", exist_ok=True)
    with open("outputs/timing/seed_strategy_scan.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case", "first_offset", "peak_offset",
                                          "p50_offset", "p90_offset", "last_offset", "total", "time_s"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\n已保存 outputs/timing/seed_strategy_scan.csv")


if __name__ == "__main__":
    main()
