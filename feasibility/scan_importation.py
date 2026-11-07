"""测试「低水平持续输入」对早期慢启动 + 晚期尾巴的联合修复。

设计（文献锚点）：
  - 初始 seed：4 例 MSM（英国 5/16 首现 4 例 MSM）
  - 持续输入：低水平 daily_n，模拟隐匿传播 + 反复输入
    文献：英格兰 ~6例/月≈0.2/天；加拿大 2-5例 imported；美国 DC surge期11例/天(太高)

方案扫描（daily_n 全程持续）：
  A: seed=4, 输入 0.2/天（英格兰 6例/月）
  B: seed=4, 输入 0.5/天
  C: seed=4, 输入 1.0/天
  D: seed=4, 输入 0.5/天 仅早期 day0-60（慢启动后停止）
  E: 现状对照 seed=20 无输入

量化：首例、5/20/50例日、峰值、total、末例。
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


def milestones(a):
    a = np.asarray(a, dtype=float)
    nz = np.where(a > 0)[0]
    if nz.size == 0:
        return dict(first=None, peak=None, total=0.0, last=None,
                    m5=None, m20=None, m50=None, m100=None)
    cum = np.cumsum(a)
    total = cum[-1]
    return dict(
        first=int(nz[0]), peak=int(np.argmax(a)), total=float(total), last=int(nz[-1]),
        m5=int(np.searchsorted(cum, 5)), m20=int(np.searchsorted(cum, 20)),
        m50=int(np.searchsorted(cum, 50)), m100=int(np.searchsorted(cum, 100)),
    )


def run_case(n_seed, import_daily, import_range, seed=SEED):
    pop = Population(n=POP_SIZE, params=params, seed=42,
                     cache_label="10M_homo_male_mcp", policy=policy,
                     importations=[dict(start=import_range[0], end=import_range[1],
                                        daily_n=import_daily)] if import_daily else [])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(n_seed, mask=mask)
    with redirect_stdout(io.StringIO()):
        pop.run_simulation(SIM_DAYS)
    return np.asarray(getattr(pop, "daily_episode_cases", []), dtype=float)


def main():
    om = milestones(obs)
    print(f"观测: 首例{om['first']} 5例@{om['m5']} 20例@{om['m20']} 50例@{om['m50']} "
          f"100例@{om['m100']} 峰值{om['peak']} total={om['total']:.0f} 末例{om['last']}")

    cases = [
        ("A_seed4_输入0.2全程", 4, 0.2, (0, 243)),
        ("B_seed4_输入0.5全程", 4, 0.5, (0, 243)),
        ("E_seed20_无输入(现状)", 20, 0.0, (0, 243)),
    ]
    results = []
    for name, n_seed, imp, rng_ in cases:
        t0 = time.time()
        sim_ep = run_case(n_seed, imp, rng_)
        m = milestones(sim_ep)
        m["case"] = name
        m["time_s"] = round(time.time()-t0, 1)
        results.append(m)
        print(f"[{name}] 首例{m['first']} 5例@{m['m5']} 20例@{m['m20']} "
              f"50例@{m['m50']} 100例@{m['m100']} 峰值{m['peak']} "
              f"total={m['total']:.0f} 末例{m['last']} ({m['time_s']}s)", flush=True)

    os.makedirs("outputs/timing", exist_ok=True)
    with open("outputs/timing/importation_scan.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case", "first", "m5", "m20", "m50", "m100",
                                          "peak", "total", "last", "time_s"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\n已保存 outputs/timing/importation_scan.csv")


if __name__ == "__main__":
    main()
