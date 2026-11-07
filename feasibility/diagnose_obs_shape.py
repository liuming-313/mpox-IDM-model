"""Diagnose the exact observed Case_data shape to guide phase-timing adjustment.

Prints the observed LAC daily case series key landmarks (first non-zero day,
growth milestones as % of total, peak day and its calendar date) and maps them
to the current policy phase boundaries. This gives the evidence for shifting
phase2 behaviour-contraction onset earlier (LAC behaviour change / vaccine
rollout began in July, not Aug 1).

Read-only; no model run.
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datetime import date, timedelta

START = date(2022, 5, 1)


def _load_obs():
    src = open(os.path.join(ROOT, "population.py"), encoding="utf-8").read()
    start = src.find("Case_data = ") + len("Case_data = ")
    end = src.find("probability_log", start)
    return np.array(eval(src[start:end].rstrip().rstrip(",").strip()))


def main():
    obs = _load_obs()
    total = obs.sum()
    peak = int(np.argmax(obs))
    cum = np.cumsum(obs)

    print(f"观测 Case_data: len={len(obs)}, total={total:.0f}, peak_day={peak}")
    print(f"  peak 日历日期 = {(START + timedelta(days=peak)).isoformat()} "
          f"(峰值 {obs[peak]} 例/天)")
    print(f"  first non-zero day = {int(np.argmax(obs > 0))} "
          f"({(START + timedelta(days=int(np.argmax(obs > 0)))).isoformat()})")

    # growth milestones
    print("\n增长里程碑（累计病例占比）:")
    for pct in [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0]:
        day = int(np.argmax(cum >= total * pct / 100.0))
        dstr = (START + timedelta(days=day)).isoformat()
        print(f"  {pct:5.1f}% -> day {day:3d} ({dstr})")

    # phase boundaries mapping
    print("\n当前 policy phase 边界（day0 = 2022-05-01）:")
    for name, s, e in [("phase1_growth", 0, 91), ("phase2_control", 92, 170),
                       ("phase3_postpeak", 171, 244)]:
        ds = (START + timedelta(days=s)).isoformat()
        de = (START + timedelta(days=e)).isoformat()
        print(f"  {name:<16} day {s:3d}-{e:3d} = {ds} ~ {de}")


if __name__ == "__main__":
    main()
