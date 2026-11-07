"""诊断模拟曲线 vs 观测曲线的时间偏移（提前 2 周问题）。

量化指标：
  - 峰值日 / 峰值高度
  - 累计达 50% / 90% 的日期
  - 末个非零病例日（尾巴长度）
  - 首个病例日

对比观测 Case_data 与模拟 daily_episode_cases，输出偏移天数。
"""
import sys, time
import numpy as np
from contextlib import redirect_stdout
import io

sys.path.insert(0, '.')
from population import Population, Case_data

# 从 main.py 提取 params（用 find 截取 params = {...} 块并 eval，避免 import main 触发顶层模拟）
import re
_main_src = open("main.py", encoding="utf-8").read()
_p_start = _main_src.find("params = {") + len("params = ")
_p_end = _main_src.find("params[", _p_start)
_EVAL_G = {"__builtins__": {}, "SIM_DAYS": 244,
           "CALIBRATION_TARGETS": {"target_homo": 1560/2012, "target_bi": 253/2012,
                                   "target_hetero": (125+74)/2012, "target_female_known": 74/2012,
                                   "target_homo_male": 1560/2012, "target_bi_male": 253/2012}}
params = eval(_main_src[_p_start:_p_end].rstrip(), _EVAL_G, {})
# 补充 main.py 里引用的常量
SIM_DAYS = 244
POPULATION_SIZE = 10_000_000
INITIAL_INFECTIONS = 20
pop_seed = 42

# 提取 policy（三阶段干预政策，影响曲线下降形状）
_pol_start = _main_src.find("policy = [")
_pol_end = _main_src.find("]\n", _pol_start) + 1
policy = eval(_main_src[_pol_start + len("policy = "):_pol_end], {"__builtins__": {}, "dict": dict, "SIM_DAYS": 244}, {})

obs = np.asarray(Case_data, dtype=float)


def key_times(arr):
    a = np.asarray(arr, dtype=float)
    nz = np.where(a > 0)[0]
    if nz.size == 0:
        return dict(peak_day=None, peak_h=None, total=0.0,
                    first=None, last=None, p50=None, p90=None)
    cum = np.cumsum(a)
    total = cum[-1]
    return dict(
        peak_day=int(np.argmax(a)), peak_h=float(np.max(a)), total=float(total),
        first=int(nz[0]), last=int(nz[-1]),
        p50=int(np.searchsorted(cum, total * 0.5)),
        p90=int(np.searchsorted(cum, total * 0.9)),
    )


def run_and_extract(seed=100001):
    pop = Population(n=POPULATION_SIZE, params=params, seed=pop_seed,
                     cache_label="10M_homo_male_mcp", policy=policy,
                     importations=[])
    # 复用 main.py 的 mask 逻辑
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(INITIAL_INFECTIONS, mask=mask)
    with redirect_stdout(io.StringIO()):
        pop.run_simulation(SIM_DAYS)
    sim_ep = np.asarray(getattr(pop, "daily_episode_cases", []), dtype=float)
    return sim_ep


def main():
    t0 = time.time()
    print("运行模拟（提取 episode 时序）...")
    sim_ep = run_and_extract(seed=100001)
    print(f"模拟耗时 {time.time()-t0:.1f}s")

    ok = key_times(obs)
    sk = key_times(sim_ep)
    T = min(obs.size, sim_ep.size)

    print("\n=== 关键时间点对比 ===")
    rows = [
        ("首个病例日", ok["first"], sk["first"]),
        ("峰值日", ok["peak_day"], sk["peak_day"]),
        ("峰值高度", ok["peak_h"], sk["peak_h"]),
        ("累计达50%日", ok["p50"], sk["p50"]),
        ("累计达90%日", ok["p90"], sk["p90"]),
        ("末个非零日", ok["last"], sk["last"]),
        ("累计总数", ok["total"], sk["total"]),
    ]
    print(f"{'指标':<14} {'观测':>8} {'模拟':>8} {'偏移(天/%)':>12}")
    for name, o, s in rows:
        if name == "累计总数":
            off = f"{(s-o)/o*100:+.1f}%"
        elif name == "峰值高度":
            off = f"{(s-o)/o*100:+.1f}%"
        elif o is None or s is None:
            off = "n/a"
        else:
            off = f"{s-o:+d}天"
        print(f"{name:<14} {o:>8} {s:>8} {off:>12}")

    # 保存对比 CSV
    import csv, os
    os.makedirs("outputs/timing", exist_ok=True)
    with open("outputs/timing/timing_offset.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "observed", "simulated", "offset"])
        for name, o, s in rows:
            w.writerow([name, o, s, (s - o if isinstance(o, (int, float)) and isinstance(s, (int, float)) else "")])
    print("\n结果已保存 outputs/timing/timing_offset.csv")


if __name__ == "__main__":
    main()
