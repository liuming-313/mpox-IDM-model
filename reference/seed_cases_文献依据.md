# 初始病例引入（importation）的数量、时间与文献依据

> 目的：回答"模型里疫情如何点燃、持续输入有多少、什么时候进入模型、有没有文献依据"。用于论文 Methods 里 justify `INITIAL_INFECTIONS = 0` + `importations = 1/2/4/1` 分段持续输入的取值与时间安排。
>
> **关键结论（先写结论）**：最终方案采用「零 day-0 seed + 持续外部输入流」，输入强度按 LAC 真实引入节奏分四段（day 15–44 / 45–74 / 75–104 / 105–134 = 每日 1 / 2 / 4 / 1 例）。这替代了早期"20 例一次性播种"方案——后者导致峰值系统性提前约 18 天（median 74 vs 观测 92），因为真实 LAC 疫情是"6 月初逐例零星输入、6 月中下旬才转入社区传播"的慢启动过程。

---

## 1. 为什么从"一次性 seed"改成"持续输入"

### 1.1 旧方案的失败模式（峰值错位）

- 早期版本用 `INITIAL_INFECTIONS = 20`（20 例 MSM 在 day 0 一次性播种）。
- 结果：峰值中位数 day 74，比观测 day 92 **提前约 18 天**。
- 根因：一次性 seed 让内生放大从 day 0 就全速启动，而真实 LAC 疫情启动很慢（首例 6/2、持续增长 6 月中才确立）。

### 1.2 真实疫情的启动节奏（关键锚点）

LAC 2022 年 5–6 月逐例时间线（权威来源见 §5.5）：

| 日期 | 事件 | 性质 |
|---|---|---|
| 5/27 | 加州首例（Sacramento，非 LAC） | 州级输入 |
| **6/2** | **LAC 首例**，近期旅行史 | 输入病例 |
| 6/8 | LAC 第 2 例，近期旅行 | 输入 |
| 6/10 | LAC 第 3、4 例，均有旅行史 | 输入 |
| 6/14 | CDC 报告美国出现"无旅行史"社区传播 | 转入社区传播 |
| 6/24 | LAC 官方确认"本地传播" | 本地传播确立 |

→ **真实疫情是"逐例零星输入"启动的**：前 4 例均有旅行史，每次 1–2 例、间隔约一周，而非某天集中暴发；本地社区传播到 6 月中下旬才确立。

### 1.3 持续输入如何解决峰值错位

- 用 `importations` 持续流替代 day-0 一次性 seed，输入强度跟随 LAC 引入节奏（慢 May → 加速 Jun–Jul → 收尾 Aug）。
- 结果（30-seed 中位数）：peak day 96（观测 92，仅差 4 天），26/30 seed 覆盖观测 92；总病例 ≈2400（观测 2337）。
- 机制上可解释：城市疫情持续被外部病例补充，内生放大在引入后约 2–3 周才主导。

---

## 2. 建模文献里对"初始感染/输入"的设定

直接对标本模型 importation 的证据：

| 文献 | 初始/输入设定 | 方式 |
|---|---|---|
| **加拿大 SEIR**（Milwid et al., *J Med Virol* 2023）| **1–3 例输入**，暴露态、>10 伴侣/年、年龄 <50 | 校准输入日期（2022 年 3–4 月）与数量 |
| **美国 DC 网络模型**（Clay et al., *Sex Transm Infect* 2024）| 6/2 播 **5 例**（最高活跃层）+ surge 期**网络外输入项 ε**（11–25 Jun）| 一次性 seed + 持续外部输入 |
| **英格兰成本效益模型**（Zhang et al., *Lancet Reg Health Eur* 2025）| **6 例/月**基线输入（敏感性 1–10/月）| 持续输入流 |
| **NYC 疫苗优先序模型**（*Vaccines* 2023）| 5 例，5/14 | 一次性 seed |
| **荷兰 fading 模型**（*J Infect Dis* 2024）| 反事实场景 20 例 / 1 例/天 输入 very-high 层 | 场景假设 |

**结论**：主流文献要么用"少量一次性 seed + 持续输入项"（DC），要么直接用"持续输入流"（英格兰 6 例/月）。本模型 1/2/4/1 例/天的四段输入，与这些量级一致（增长窗口内约 1–6 例/天），且明确把"持续外部输入"作为正式机制而非敏感性场景。

---

## 3. 输入 schedule 的四段设计依据

| 段 | day 范围 | 每日输入 | 对应 LAC 节奏 |
|---|---|---|---|
| 1 | 15–44 | 1 | 慢启动（5 月中–6 月初，逐例零星输入）|
| 2 | 45–74 | 2 | 加速（6 月，输入增多）|
| 3 | 75–104 | 4 | 高峰（6 月底–7 月，引入高峰，占新病例 20–30%）|
| 4 | 105–134 | 1 | 收尾（8 月，输入回落）|

- 输入强度量级对齐：英格兰 6 例/月 ≈ 0.2 例/天（偏保守），DC 每天 11 例高活跃输入（偏激进）；本方案峰值段 4 例/天居中。
- 引入高峰锚点：phylodynamic 研究（medRxiv 2025, PMC13276399）估计 LAC 引入高峰在 2022 夏季、引入占新病例 20–30%。
- 输入人群：全部经 `importation_mask` 路由到 MSM core，与文献"seed 落在 GBMSM 内"一致（§5.1）。

---

## 4. 与最终方案各参数的对应关系（冻结版）

| 参数 | 值 | 说明 |
|---|---|---|
| `INITIAL_INFECTIONS` | 0 | day-0 seed 关闭，疫情由持续输入点燃 |
| `importations` | 1/2/4/1（day 15–134）| 四段持续输入，见 §3 |
| `oneoff_p` | 0.2105 | 输入方案下经 f=7.5 标定（均值 3.75 伴侣/活跃个体/天）|
| `party_ht_top_mean` | 7.5 | 输入方案下经 f=7.5 标定 |
| `mix_mf_anal_frac` | 0.30 | 修复输入集中 MSM core 后 hetero/female 外溢 |
| `oneoff_orientation_bias`[MB] | 1.2 | 微调 bi 占比到 13.4% |

---

## 5. seed/输入的人群与活跃度设置

### 5.1 人群：几乎所有文献的 seed 都在 MSM/GBMSM 内

- 意大利个体模型（PMC10937928）、加拿大 2-city（PMC9828961）、英格兰 GBMSM（PubMed 37708908）、美国 DC（PMC11323315）等，seed/imported cases 均默认 GBMSM。
- **结论**：本模型 `importation_mask` 把输入病例放在 MSM core，与文献完全一致。

### 5.2 活跃度分层

| 文献 | 性活动分层 | 输入设置 |
|---|---|---|
| 荷兰 fading（PMC11272055）| MSM 4 层 | 反事实场景 20 例/1 例·天 输入 very-high 层 |
| 英格兰 GBMSM（PubMed 37708908）| low/high risk | 83.8% 感染来自 high-risk GBMSM 性伴侣 |
| 意大利个体模型（PMC10937928）| 年伴侣数 4 层（HR=30+/年占 2.5%）| — |
| 美国 DC（PMC11323315）| 6 activity group | surge 期每天 11 例额外高活跃个体（group 5-6）|
| 加拿大 SEIR（Milwid 2023）| — | 1–3 例输入，暴露态、>10 伴侣/年 |

**说明**：文献常把 seed/输入放在"高活跃层"（荷兰 very-high、DC group 5-6）。本模型输入经 `importation_mask` 进入 MSM core（未额外偏向高活跃子层），放大主要由网络 heavy-tail 内生完成——这比一次性把输入塞进高活跃核心更稳健（避免过冲，见早期 A/B 测试：一次性 20 例全进高活跃核心曾超调 +56%）。

---

## 5.5 LAC 早期病例逐例时间线（权威来源，2026-11-05 补充）

> 用于 justify "分段引入 + 持续输入" 的引入曲线设计。关键修正：**5 月 27 日是加州首例（Sacramento），不是 LAC 首例**。LAC 首例是 **6 月 2 日**。

### 5.5.1 权威时间线（LAC 2022 年 5–6 月）

| 日期 | 事件 | 性质 | 权威来源 |
|---|---|---|---|
| 5 月 20 日 | CDC HAN 0466 发布，LAC 通报"无确诊/疑似病例" | 全美疫情预警 | CDC HAN 0466 (5/20) |
| 5 月 27 日 | **加州首例**（Sacramento，非 LAC），有欧洲旅行史；LAC 仍"无病例" | 州级输入 | CDPH Health Advisory 5/27 |
| **6 月 2 日** | **LAC 首例**：成人居民，近期旅行 + 已知密接，有症状居家隔离 | LAC 输入病例 | lacounty.gov 官方通告 |
| **6 月 8 日** | **LAC 第 2 例**：成人居民，近期旅行 | 输入 | ABC7 / CBS 洛杉矶 |
| **6 月 10 日** | **LAC 第 3、4 例**：2 例均有近期旅行史 | 输入 | ABC7 |
| 6 月 14 日 | CDC HAN 0468 更新：美国出现"社区传播"，部分病例无旅行史 | 转入社区传播 | CDC HAN 0468 |
| 6 月 24 日 | LAC 官方确认"本地传播"（部分新病例无国际/州外旅行史）| 本地传播确立 | ABC7 |

### 5.5.2 关键结论（可直接写入 Methods/Supplementary）

1. **LAC 疫情是"逐例零星输入"启动的**：6/2 首例 → 6/8 第 2 例 → 6/10 第 3、4 例，**前 4 例均有近期旅行史**（输入病例），每次 1–2 例、间隔约一周，而非某一天集中暴发。
2. **本地社区传播直到 6 月中下旬才确立**：6/14 CDC 才报告美国出现"无旅行史"的社区传播，6/24 LAC 官方确认本地传播。这给了一个关键锚点——**模型里"内生放大"应该在引入后约 2–3 周才主导**，而非 day 0 立即放大。
3. **LAC 总病例 2281 例**（CDC MMWR 73/24，tecoivirimat 研究）——这正是本模型校准目标 total≈2280 的**权威来源**。
4. **引入持续到疫情高峰**：phylodynamic 研究（medRxiv 2025, PMC13276399）估计 LAC 引入高峰在 2022 年夏季、引入占新病例 20–30%。

### 5.5.3 对模型引入曲线的启示（已落实）

- 旧的"day 0 一次性 20 例"与真实"6/2 起逐例零星引入"不符，是峰值提前的根因之一。
- 已改为：**零 day-0 seed + 6/2 起低强度持续输入（四段 1/2/4/1 例/天，暴露态引入）**。
- 时间锚点：day 0 ≈ 2022-05-01 → LAC 首例 6/2 = day 32（落在输入第 1 段 day 15–44 内）；社区传播确立 6/24 = day 54（进入第 2 段 day 45–74）。

### 5.5.4 权威引用清单（可进参考文献）

1. CDC Health Alert Network HAN 0466. Monkeypox Virus Infection in the United States and Other Non-endemic Countries—2022. May 20, 2022.
2. California Department of Public Health. Healthcare Provider Monkeypox Health Advisory. May 27, 2022.
3. Los Angeles County Department of Public Health. "Los Angeles County Confirms First Presumptive Case of Monkeypox." June 2, 2022. (lacounty.gov)
4. CDC Health Alert Network HAN 0468. Updated Case-finding Guidance: Monkeypox Outbreak—United States, 2022. June 14, 2022.
5. Los Angeles County Department of Public Health. ABC7 Los Angeles monkeypox timeline (May–June 2022).
6. "Successful Distribution of Tecovirimat During the Peak of the Mpox Outbreak — Los Angeles County, June 2022–January 2023." CDC MMWR 2024;73(24). (2281 LAC cases)
7. Viral introductions and return to baseline sexual behaviors maintain low-level mpox incidence in Los Angeles County, USA, 2023–2024. medRxiv 2025. (PMC13276399)

---

## 6. 参考文献

1. UK Health Security Agency. Community transmission of monkeypox in the United Kingdom, April to May 2022. *Eurosurveillance* 2022;27(22). (PMC9164677)
2. Emergence of mpox in the post-smallpox era — a narrative review. *Clinical Microbiology and Infection* 2023.
3. Modeling the impact of vaccine dose prioritization strategies during the 2022 mpox outbreak (NYC). *Vaccines* 2023.
4. Milwid RM, et al. *J Med Virol*. 2023. (Canada SEIR, 1–3 imported cases, March–April 2022)
5. Clay PA, et al. *Sex Transm Infect*. 2024. (US DC network model, 5 cases + importation term ε)
6. Zhang XS, et al. Cost-effectiveness of vaccination strategies to control future mpox outbreaks in England. *Lancet Reg Health Eur*. 2025.
7. The Fading of the Mpox Outbreak Among MSM: A Mathematical Modelling Study (Netherlands). *J Infect Dis* 2024. (PMC11272055)
8. Estimates of mpox effective reproduction number in Spain, April–August 2022. (PMC10369440)
9. Transmission dynamics and effect of control measures on the 2022 outbreak of mpox among GBMSM in England. *Lancet Infect Dis* 2024. (PubMed 37708908)
10. The decline of the 2022 Italian mpox epidemic: Role of behavior changes and control strategies. (PMC10937928)
11. Modelling the impact of vaccination and sexual behaviour adaptations on mpox cases in the USA. (PMC11323315)
12. Maximizing the impact of limited vaccine supply... 2-city modelling analysis (Canada). *CMAJ* 2022. (PMC9828961)
13. Transmission of monkeypox/mpox virus: a narrative review (引用 Endo et al. 与 Brand et al. 的重尾证据). (PMC10107822)
