"""测试「慢启动」假设：少量 seed 能否修复早期过陡。

真实疫情：1 例(day18) → 5例(day33) → 20例(day43) → 指数加速
模型现状：20 seed day0 同时爆发 → 首例 day6（提前12天）

假设：把 seed 从 20 减到 2-6 例，模拟真实「慢启动」，
首例日和早期斜率会贴合观测（但 total 可能缩水，需重标定）。

量化：首例日、累计达5/10/20/50例日、峰值日、total、灭绝情况。
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
obs_cum = np.cumsum(obs)


def milestones(a):
    """返回累计达 [5,10,20,50,100] 例的日期，以及首例、峰值、total、末例。"""
    a = np.asarray(a, dtype=float)
    nz = np.where(a > 0)[0]
    if nz.size == 0:
        return dict(first=None, peak=None, total=0.0, last=None, extinct=True,
                    m5=None, m10=None, m20=None, m50=None, m100=None)
    cum = np.cumsum(a)
    total = cum[-1]
    r = dict(
        first=int(nz[0]), peak=int(np.argmax(a)), total=float(total), last=int(nz[-1]),
        extinct=(total < 50),
        m5=int(np.searchsorted(cum, 5)), m10=int(np.searchsorted(cum, 10)),
        m20=int(np.searchsorted(cum, 20)), m50=int(np.searchsorted(cum, 50)),
        m100=int(np.searchsorted(cum, 100)),
    )
    return r


def run_case(n_seed, seed):
    pop = Population(n=POP_SIZE, params=params, seed=42,
                     cache_label="10M_homo_male_mcp", policy=policy, importations=[])
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    mask = (g == 0) & (o == 'homo') & mcp
    pop.importation_mask = mask
    np.random.seed(seed)
    pop._rng = np.random.default_rng(seed)
    pop.seed_infection(n_seed, mask=mask)
    with redirect_stdout(io.StringIO()):
        pop.run_simulation(SIM_DAYS)
    sim_ep = np.asarray(getattr(pop, "daily_episode_cases", []), dtype=float)
    return sim_ep


def main():
    # 观测里程碑（参照）
    om = milestones(obs)
    print(f"观测里程碑: 首例{om['first']} 5例@{om['m5']} 10例@{om['m10']} 20例@{om['m20']} "
          f"50例@{om['m50']} 100例@{om['m100']} 峰值{om['peak']} total={om['total']:.0f}")

    # 每个 seed 数量跑多个随机种子，看灭绝率和稳定性
    cases = [2, 4, 6]
    seeds_per = 2  # 每个数量跑 2 个 seed（先快速验证，确认方向后再扩大）
    results = []
    for n_seed in cases:
        for sd in range(seeds_per):
            seed = 100001 + sd * 1000
            t0 = time.time()
            sim_ep = run_case(n_seed, seed)
            m = milestones(sim_ep)
            m["n_seed"] = n_seed
            m["run_seed"] = seed
            m["time_s"] = round(time.time()-t0, 1)
            results.append(m)
            status = "灭绝" if m["extinct"] else f"total={m['total']:.0f}"
            print(f"[n_seed={n_seed}, seed={seed}] 首例{m['first']} "
                  f"5例@{m['m5']} 10例@{m['m10']} 20例@{m['m20']} "
                  f"50例@{m['m50']} 峰值{m['peak']} {status} ({m['time_s']}s)", flush=True)

    os.makedirs("outputs/timing", exist_ok=True)
    with open("outputs/timing/slow_start_scan.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["n_seed", "run_seed", "first", "m5", "m10", "m20",
                                          "m50", "m100", "peak", "total", "last", "extinct", "time_s"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print("\n已保存 outputs/timing/slow_start_scan.csv")


if __name__ == "__main__":
    main()
