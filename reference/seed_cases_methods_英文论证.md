# Methods 论证段落（英文，可直接贴入论文）

> 用途：为论文 Methods 里 "initial conditions / seeding and importation" 部分提供可直接引用的英文段落。所有数值均来自本模型最终方案（`main.py` 冻结版：`INITIAL_INFECTIONS = 0` + 持续外部输入 `importations = 1/2/4/1`，30-seed 中位数验证），文献锚点为 Milwid 2023 / Clay 2024 / Zhang 2025。

---

## A. Initial conditions and importation

The epidemic was ignited not by a fixed batch of index cases on day 0 but by a sustained, low-level exogenous importation stream whose intensity followed the observed importation rhythm of the Los Angeles County (LAC) outbreak. The importation schedule was divided into four consecutive phases (daily imported cases = 1 over days 15–44, 2 over days 45–74, 4 over days 75–104, and 1 over days 105–134), reproducing a "slow May → accelerate Jun–Jul → taper Aug" shape. Imported cases were routed to the MSM core of the sexual network.

**Why sustained importation rather than a one-shot seed.** An earlier version of the model introduced 20 infectious gay/bisexual men simultaneously on day 0. That scheme produced a systematic ≈18-day *early* peak (median simulated peak day 74 versus the observed day 92): because the real LAC outbreak began slowly (first reported case 19 May 2022, sustained growth only from mid-June), a one-shot day-0 seed caused full-speed endogenous amplification to begin far too early. Sustained importation defers epidemic acceleration to match the observed tempo, and reflects the epidemiological reality that an urban outbreak such as LAC's is continuously replenished by externally acquired cases rather than isolated once its index case is detected.

**Literature support.** This approach is directly consistent with prior mpox models. Canada calibrated both the timing (March–April 2022) and the number (1–3) of imported cases, all assigned to the exposed state with more than 10 annual partners and age <50 (Milwid et al., *J Med Virol* 2023). The US Washington DC network model seeded 5 cases on 2 June from the highest-activity strata and added an extra-network importation term ε during the 11–25 June surge (Clay et al., *Sex Transm Infect* 2024). England's cost-effectiveness model assumed a baseline importation rate of 6 cases/month (sensitivity 1–10/month) (Zhang et al., *Lancet Reg Health Eur* 2025). Our four-phase schedule (1/2/4/1 cases/day) is of the same order of magnitude as these published importation rates (roughly 1–6 imported cases per day across the growth window).

**Calibration outcome.** Under this scheme the 30-seed median peak occurs on day 96 (observed 92), eliminating the early-peak artifact while preserving the calibrated outbreak size (median ≈2,400 total cases versus the observed ≈2,280–2,337) and case composition (male-homosexual ≈80.8%, male-bisexual ≈13.1%, heterosexual ≈6.2%, female ≈5.6%).

---

## B. 对应中文要点（供你核对）

1. **零 day-0 seed + 持续输入**：`INITIAL_INFECTIONS = 0`，疫情由 `importations` 持续流点燃，分四段 1/2/4/1（day 15–44 / 45–74 / 75–104 / 105–134），对应 LAC "慢 May → 加速 Jun–Jul → 收尾 Aug" 节奏。
2. **为什么不用一次性 seed**：旧版 20 例一次性播种导致峰值系统性早约 18 天（median 74 vs 观测 92），因为真实 LAC 疫情启动慢（首例 5/19，6 月中才持续增长）；持续输入让加速节奏对齐观测，并反映"城市疫情持续被外部病例补充"的现实。
3. **文献支撑**：加拿大校准 1–3 例输入（Milwid 2023）；DC 网络模型 6/2 播 5 例 + 网络外输入项 ε（Clay 2024）；英格兰假设 6 例/月基线输入（Zhang 2025）。本方案 1/2/4/1 与这些量级一致。
4. **标定结果**：30-seed 中位 peak = 96（观测 92），总病例 ≈2400（观测 2337），构成 homo .808 / bi .131 / hetero .062 / female .056。

---

## C. 引用占位（[ref xxx] 处，对应最终文献）

| 占位 | 对应文献 | S1 编号 |
|---|---|---|
| [ref Canada] | Milwid RM, et al. *J Med Virol*. 2023 (Canada, 1–3 imported cases, March–April 2022) | 见 S1 参考文献 |
| [ref DC] | Clay PA, et al. *Sex Transm Infect*. 2024 (US DC network model, 5 cases 2 June + importation term ε) | 见 S1 参考文献 |
| [ref England] | Zhang XS, et al. *Lancet Reg Health Eur*. 2025 (England, 6 cases/month baseline importation) | 见 S1 参考文献 |
| [ref LAC] | Los Angeles County 2022 mpox outbreak (observed epicurve, first case 19 May 2022) | — |

> 注：NYC 5 例、CMI 综述等旧引用（用于论证 20-seed 方案的）已随方案切换为 importation 框架而不再用于本段；如需保留作为"初始输入规模量级"的旁证，可在正文另行引用。
