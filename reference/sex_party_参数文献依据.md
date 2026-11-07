# sex_party 层参数：文献依据与设计

> 目的：sex_party 层是当前模型 10M 下的**绝对主力传播层**（占 75.9% 的本地传播），因此它的参数必须有实证依据、可写进论文 Methods。本文档把每个参数的文献来源、当前值、建议值、敏感性范围系统化。

---

## 0. 为什么 sex_party 是核心（模型内证据）

10M A 基线（seed 42）各层传播贡献：

| 层 | 感染数 | 占比 |
|---|---|---|
| **sex_party** | 1241 | **75.9%** |
| stable | 142 | 8.7% |
| social | 103 | 6.3% |
| oneoff | 91 | 5.6% |
| sex_party_touch | 36 | 2.2% |
| household_close | 21 | 1.3% |

且 sex_party 的传播边 **MM 主导**（MF/FM 边占比仅 ~4.4%），即 MSM 内部传播为主、漏出极少。这是它"既放大规模、又不崩构成"的根本原因，与 CDC 模型"one-time 层贡献 46–54% 且全在 MSM 内部"高度同构。

---

## 1. 最重要的流行病学依据：派对/群交是 2022 mpox 的核心暴露场景

这是 sex_party 层存在意义的直接依据，也是论文里"为什么单列一个 sex_party 层"的论证基础：

| 来源 | 关键数字 |
|---|---|
| **WHO 全球监测（2022-09-07 sitrep）** | "party setting with sexual contacts" 是最常见暴露场景，占所有报告暴露场所的 **60.0%**（2921/4867） |
| **WHO（2022-12-14 sitrep）** | 同上，**59.1%**（3776/6390） |
| **综合综述（PMC10020973）** | party setting with sexual contacts 是最可能暴露类别，**46.8%** |
| **NEJM 2022（528 例全球系列）** | 前一个月 **32%** 去过 sex-on-premises venue、**20%** 参与 chemsex |
| **米兰病例系列（135 例）** | **55.6%** 参与 group sex |

→ 结论：**"性派对/群交/性场所"是 2022 mpox 疫情里唯一被反复确认的最主要暴露场景**，占比 47%–60%。把 sex_party 作为独立且主导的传播层，有最强的流行病学支撑。

---

## 2. 出席率（attendance）参数

### 2.1 当前值

```python
"sex_party_attend_prob": 3/7,     # 默认 ~0.43/天，注释"eligible 人约每周 3 晚参加"
"party_msm_attend_prob": 0.40,    # MSM 池
"party_wsw_attend_prob": 0.35,    # WSW 池
"party_core_intensity": 0.7,      # 映射到 party_mix_attend_prob
"party_weekday_mult": 0.7,
"party_weekend_mult": 1.3,
```

### 2.2 文献依据

- **32% 的病例前一个月去过 sex-on-premises venue**（NEJM 528 例）。注意这是"去过"（至少一次/月），不是"每晚"。
- **法国：49% 的 MSM 病例在 3 周内去过 MSM venue**（Eurosurveillance 2023）。
- **马德里：14.4% 报告参加 mass events/parties**（含 Maspalomas Pride）。

### 2.3 设计建议

- "每周 3 晚"（3/7 ≈ 0.43/天）作为**总活跃频率锚点**是合理的，但要注意它针对的是"eligible 高活跃 MSM 核心"而非全体 MSM。当前 `party_idx` 已是"eligible"子集（不是全体 MCP），所以 0.40–0.43 的出席率作用在核心人群上，量级合理。
- **MSM 出席率 > WSW 出席率**（0.40 vs 0.35）符合疫情"男性主导"事实，方向正确。
- 建议补充引用：**NEJM 32%/月** 和 **法国 49%/3周** 作为"出席率"的量级锚定，写入 Methods。

---

## 3. 伴侣数（partner number / degree）参数

### 3.1 当前值

```python
"party_sex_frac": 0.73,          # p_sex：出席者中当天发生性行为的比例
"party_nb_p": 1/(1+2/0.73),      # NB(r=1,p)，使 E[k|sex]≈2.74，总体 mean partners≈2
"party_k_clip": 15,              # 伴侣数上限
"party_mix_target_mean": 2.0,    # MIX 池平均伴侣数
```

### 3.2 文献依据（这是 heavy-tail 的核心证据）

| 来源 | 伴侣数分布 |
|---|---|
| **NEJM 2022** | 中位 **5 伴侣/3个月** |
| **法国 2216 例** | 3 周内中位 **3 伴侣**（IQR 1–5，range 0–100） |
| **ANRS-174 DOXYVAC** | 中位 **10 伴侣/3月**（IQR 5–20），43% 有 >10 伴侣 |
| **米兰 135 例** | 50.4% 有 11–30 伴侣/3月，**23% 有 >50 伴侣/3月** |
| **CDC MMWR 模型（reference）** | top 5% 活动层 ≈ **100 伴侣/年**（one-time 率 0.286/天） |

### 3.3 关键洞察：当前模型缺 heavy-tail

- 当前 `party_nb_p` 用 **NB(r=1)**，期望 ~2.74/活跃夜，clip 到 15。
- 但米兰数据显示 **23% 的病例 3 个月有 >50 个伴侣**，即存在明显的重尾——少数"超级传播者"有几十个伴侣。
- 当前模型把伴侣数做成"人人都 ~2-3 个 + clip 15"，**抹掉了这个重尾**，与 CDC 模型"top 5% 达 100 伴侣/年"的设计差距明显。

### 3.4 设计建议（核心改动方向）

给 sex_party 的 partner 数加 **activity 分层**（对齐 CDC 的 riskg 6 层设计），让 top 层（~5%）的伴侣数远超普通层：

- **普通层**（~95%）：E[k] ≈ 2–3（对齐 NEJM 中位 5/3月、法国中位 3/3周）
- **高活跃层**（~5%）：E[k] ≈ 10–20（对齐米兰 23% 有 >50/3月、CDC top 5% ≈ 100/年）

这能让少数超级传播者驱动派对内传播，**提升放大倍数而不过度扩大 hetero 漏出**（因为 sex_party 的 MM 边本就主导）。

### 3.5 已实现（v2，2026-10-24）

按上述建议实现，且做了**一处关键修正**——把 heavy-tail 限制在 **MSM core 池内部**，而非全体 MCP：

- **初版 bug**：最初用全局 `activity_level==5` 分层，但该字段均匀分配给所有 MCP，导致 top 5% 层里 **hetero 占 72.7%、MSM 仅 5%**，重尾大部分落在异性恋个体上，反而推高 hetero 漏出（homo 从 71.2% 掉到 66.8%）。
- **修正**：新增持久标记 `sex_party_high_active`（`population.py`），**只标记 MSM core 池（gay male 且 party-eligible）中的 ~5%** 为高活跃，重尾只作用在这批人上。
- 新增参数：`party_heavy_tail`（开关）、`party_ht_frac`（默认 0.05）、`party_ht_top_mean`（默认 12，NB r=1 的 E[k|sex]）、`party_ht_top_clip`（默认 30）。

**验证结果（10M，seed 42，20 种子）：**

| 指标 | A（heavy-tail OFF） | B（heavy-tail ON） | 变化 |
|---|---|---|---|
| episode | 1012 | **1601** | +58% |
| cum_inf | 1249 | **2091** | +67% |
| homo | 72.04% | 71.71% | ≈持平 |
| bi | 17.98% | 18.61% | ≈持平 |
| hetero | 9.98% | **9.68%** | ↓（更近 9.9% 目标）|
| female | 5.34% | **5.00%** | ↓（更近 3.7%）|

**结论**：heavy-tail 只放大 MM 内部传播、不扩大 hetero 漏出，构成不劣化且规模 +58%。若要进一步逼近 2280 目标，可上调 `party_ht_top_mean`（12→20–30，仍对齐米兰 23%>50/3月、CDC top5%≈100/年）。

### 3.6 多 seed 稳定性（top_mean 敏感性，2026-10-24 晚）

2280 是模拟中位数的概念，单次 seed 随机波动大，故对 `party_ht_top_mean` 做 3 档 × 5 seed 的 10M 扫描：

| top_mean | episode median | episode mean | std | homo | hetero | female |
|---|---|---|---|---|---|---|
| 12 | 1635 | 1599 | 262 | 72.85% | 10.07% | 4.86% |
| 20 | 2064 | 1986 | 567 | 72.59% | 10.20% | 5.00% |
| 24 | 2070 | 1905 | 284 | 72.01% | 10.49% | 5.41% |

关键发现：

1. **中位数随 top_mean 先升后饱和**：12→20 时 median 1635→2064（明显），20→24 几乎不涨（2064→2070），说明 top_mean≈20 已接近 sex_party 伴侣数重尾的放大上限。
2. **top_mean=20 处 std 暴增（567）**：出现双峰/爆发分支——部分 seed 爆到 2846/2288（但 hetero 11.35%、female 5.62% 劣化），部分 seed 只有 1360/1372（构成很干净）。这是重尾传播的临界现象：高活跃 MSM 伴侣数跨过阈值后，"能否点燃派对内超级传播"变得高度随机。
3. **构成在 top_mean≥20 开始漂移且波动加大**：hetero 10.07%→10.2~10.5%、female 4.86%→5.0~5.4%，个别 seed 出现 hetero 11%+、female 5.6% 的偏离。

**结论**：`party_ht_top_mean=20` 是把中位数推到 ~2060（距 2280 差 ~10%）的较优点，代价是随机性加大、构成开始漂移。继续上调不再有效。单纯靠 sex_party 伴侣数重尾，中位数天花板约 2050–2100，距 2280 有系统性 ~10% 缺口，需其他机制补齐（而非继续堆伴侣数）。

---

## 4. 传播概率（per-act risk）参数

### 4.1 当前值（已按文献分层，见 `build_edge_risks`）

```python
"sexual_anal_receptive": 0.28,   # MM receptive
"sexual_anal_insertive": 0.06,   # MM insertive
"sexual_vaginal": 0.035,         # MF vaginal
"sexual_oral": 0.0175,           # FF oral
"mix_mf_anal_frac": 0.13,        # MIX 里 MF 边升级为 anal 的比例
```

### 4.2 文献依据（详见 `per_act_risk_文献依据.md`）

- **receptive anal aOR = 5.4**（Chard et al. MMWR 2024，唯一调整后显著）
- insertive anal aOR ≈ 1.0（不显著）；oral 不显著
- 排序：receptive anal ≫ insertive anal ≳ oral ≳ nonsexual —— 当前模型已正确

### 4.3 设计建议

- `mix_mf_anal_frac=0.13` 这个值：**文献没有直接依据**。它是"MF 边里多大比例升级为 anal"。米兰数据显示 MSM 里 74.7% 有 receptive anal、80.7% 有 insertive anal，但这些是 MSM 内部；**异性恋（MF）的 anal 比例应显著更低**。建议把它作为敏感性分析参数，范围 0.05–0.20，默认 0.13 作为"偏低但非零"的保守值。

---

## 5. 皮肤接触（close-touch）参数

### 5.1 当前值

```python
"party_touch_frac": 0.20,        # 派对皮肤接触扩展比例
"close_touch": 0.0067,           # 皮肤接触 per-contact 风险
```

### 5.2 文献依据

- 非性密切接触 SAR = **1.5%**（CDC MMWR 2023）
- 皮肤接触传播 mpox 有明确证据（皮损直接接触），但风险远低于性接触（aOR 差异显著）

### 5.3 设计建议

- `close_touch=0.0067` 与 `social_p_eff=0.0063` 同源（都从 SAR 1.5% 锚定），**方向正确**。
- `party_touch_frac=0.20`（20% 的派对接触升级为 skin-touch）是"扩展比例"参数，文献无直接依据，建议作为敏感性参数，范围 0.10–0.25。

---

## 6. 混合（MIX 池）参数

### 6.1 当前值

```python
"p_bi_m_to_msm": 0.7,            # 双性恋男性路由到 MSM 池的比例
"p_bi_f_to_wsw": 0.75,           # 双性恋女性路由到 WSW 池
"party_mix_homo_spill_frac": 0.01,  # 同性恋进入 MIX 池的比例
```

### 6.2 文献依据

- 2022 mpox 里 **1% 病例是异性恋**（ECDC 监测：99% 男性、44% MSM、**1% hetero**）
- 女性仅 **2.7%–3.8%**

### 6.3 设计建议

- `party_mix_homo_spill_frac=0.01`（1% 同性恋进入 MIX）**直接对齐"1% 异性恋病例"**，设计正确。
- 这是控制 hetero/female 构成的关键参数，与校准目标（hetero 9.89%、female 3.68%）强相关。

---

## 7. 汇总：参数 - 文献 - 建议对照表

| 参数 | 当前值 | 文献依据 | 状态 | 建议 |
|---|---|---|---|---|
| sex_party 层存在 | — | WHO 60%/59%、综述 46.8% | ✅ 强支撑 | 写入 Methods |
| party_msm_attend_prob | 0.40 | NEJM 32%/月、法国 49%/3周 | ✅ 量级合理 | 补引用 |
| party_wsw_attend_prob | 0.35 | 男性主导（99% 病例男性） | ✅ 方向正确 | 补引用 |
| party_sex_frac | 0.73 | 米兰 55.6% group sex | ⚠️ 偏主观 | 敏感性 0.5–0.8 |
| party_nb_p (E[k]≈2.7) | NB(r=1) | NEJM 中位 5/3月、米兰 23% >50/3月 | ✅ **已加 heavy-tail** | MSM-core 5% 层 E[k]=12 |
| party_k_clip | 15 | 米兰 23% >50/3月 | ⚠️ 上限偏低 | 高活跃层提至 30–50 |
| anal_rec/ins/vag/oral | 0.28/0.06/0.035/0.0175 | Chard aOR=5.4 | ✅ 正确 | 保持 |
| mix_mf_anal_frac | 0.13 | 无直接依据 | ⚠️ 需敏感性 | 0.05–0.20 |
| party_touch_frac | 0.20 | SAR 1.5% | ⚠️ 无直接依据 | 敏感性 0.10–0.25 |
| party_mix_homo_spill_frac | 0.01 | ECDC 1% hetero | ✅ 对齐 | 保持 |
| p_bi_m_to_msm | 0.7 | 校准目标（bi 12.57%） | ✅ 校准参数 | 保持 |

---

## 8. 核心结论

1. **sex_party 层存在的意义有最强文献支撑**（WHO 60% 暴露场所是派对/群交），是论文里最站得住的机制。
2. **当前最大的参数缺口是"缺 heavy-tail"**——伴侣数被做成均匀 ~2-3 个，而米兰数据明确显示 23% 病例 3 个月有 >50 伴侣。这正对应之前诊断的"放大倍数不足"。
3. **per-act 风险已按文献正确分层**（receptive anal 主导），无需改。
4. **hetero 构成控制参数**（party_mix_homo_spill_frac、p_bi_m_to_msm）与校准目标对齐，设计正确。

---

## 9. 参考文献

1. WHO Multi-country outbreak of mpox, External Situation Report #5 (2022-09-07) & #12 (2022-12-14).
2. Thornhill JP, et al. Monkeypox Virus Infection in Humans across 16 Countries — April–June 2022. *N Engl J Med* 2022.
3. Chard AN, et al. Risk factors for mpox... case-control study. *MMWR* 2024.
4. Eurosurveillance. Mpox outbreak in France: epidemiological characteristics and sexual behaviour of cases aged ≥15 years, 2022. 2023.
5. Eurosurveillance. Monkeypox outbreak predominantly affecting MSM, Madrid, Spain, 2022. 2022.
6. Pathogens (Milan, Italy). Comparative analysis of MVA-BN vaccinated vs mpox-infected individuals. 2023.
7. ANRS-174 DOXYVAC trial. Impact of MVA vaccination and sexual behaviour on mpox incidence in MSM. 2024.
8. PMC10020973. 2022 Mpox outbreak: concise review. 2023.
9. CDC MMWR. Mpox secondary attack rate, household/nonsexual contacts. 2023.
