# 模型参数-来源-依据总表（Supplementary Table）

> 本文档是论文补充材料（Supplementary Table S1）的底稿，对齐 `main.py` 当前生效值（2026-11-07 冻结版：零 seed + 持续输入 1/2/4/1，f=7.5，bibi=1.2）。
> 每个参数给出：当前值、来源类型、文献依据（可引用）、以及投稿时的处理建议。
>
> **来源类型四档**（模仿高分建模文章 Supplementary Table 的通行写法）：
> - **① 文献直接锚定**：有明确文献/官方报告给点估计，直接引用。
> - **② 文献间接锚定**：有文献给比值/量级/范围，通过换算或取区间中点得到。
> - **③ 换算推导**：由文献频率/比率经标准公式推导（如月度频率→日概率）。
> - **④ 校准（calibrated）/结构假设**：通过拟合观测目标反推，或为模型结构的必要假设；投稿时声明搜索范围 + 敏感性分析兜底。

---

## 一、自然史参数（疾病进程时序）

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `incubation_lognorm_mu` | 2.108 | ① | Madewell et al., *Emerg Infect Dis* 2023;29(4):818-821. 美国 12 辖区 mpox 潜伏期 lognormal 拟合，中位 5.6 d（95% CrI 4.3–7.8） |
| `incubation_lognorm_sigma` | 0.254 | ① | 同上 |
| `incubation_clip` | (3, 17) d | ① | Madewell 2023 潜伏期范围截断 |
| `symptom_onset_delay_range` | (1, 4) d | ① | CDC 临床描述："皮疹通常在症状出现后 1–4 天出现" |
| `infectious_period_gamma_shape` | 9.0 | ① | Spicknall et al., *MMWR* 2022;71(35):1131-1135（CDC 官方参考模型，传染期均值 27 d）；荷兰 fading 模型 14–28 d（Xiridou et al., *J Infect Dis* 2024）。本模型取 gamma 均值 ≈21 d（介于 CDC 27d 与荷兰 14-28d 之间） |
| `infectious_period_gamma_scale` | 2.33 | ① | 同上（shape×scale = 21 d） |
| `infectious_period_clip` | (14, 28) d | ① | 荷兰 fading 模型传染期截断 |
| `gi_mean` | 9.0 | ① | Madewell 2023 串行间隔均值 8.5 d（7.3–9.9）取整；英格兰 8.82 d（5.22–25.81）量级一致 |
| `gi_sd` | 5.0 | ① | Madewell 2023 SD 5.0（4.0–6.4） |
| `gi_max` | 30 | ④ | 世代间隔截断上限（结构假设） |

## 二、传播概率（per-act / per-contact risk）

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `sexual_anal_receptive` | 0.28 | ① | 荷兰 fading 模型每次性接触传播概率 βs∈[0.25, 0.40]（Xiridou et al., *J Infect Dis* 2024 / medRxiv 2023），取区间内 0.28。**关键流行病学佐证**：Chard et al., *MMWR* 2024;73(40) 多辖区病例对照，无套接受性肛交 OR=5.4（p=0.031，唯一调整后显著）——证实接受性肛交是最主要的传播暴露 |
| `sexual_anal_insertive` | 0.06 | ② | 插入方风险 ≈ 接受方 1/4（Chard et al. 2024 插入方 aOR≈1.0 不显著，比值落在 0.15–0.35） |
| `sexual_vaginal` | 0.035 | ② | 女性占比 2.7%（Oakley 2023）；阴道接触 aOR=0.41（Ward 家庭研究）；取肛交风险的 ~1/8 |
| `sexual_oral` | 0.0175 | ② | 口交 aOR 不显著（Chard 2024），风险最低，取肛交 ~1/16 |
| `social_p_eff` | 0.0063 | ① | **唯一有精确外部锚定**：非性密切接触次代发病率 SAR=1.5%（CDC MMWR 2023），p_eff=1-(1-0.015)^(1/2.4)≈0.0063。另见次代发病率系统综述（*BMC* 2024/PMC13274034）：非性接触 SAR 1.3%、性接触 SAR 11.4%（8.5 倍差异） |
| `social_p0_cap` | 0.03 | ② | per-contact 非性接触风险上限（量级合理） |

> 详见 `per_act_risk_文献依据.md`、`性传播层参数_文献锚定.md`。

## 三、一次性（one-off）性网络层

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `oneoff_active_frac` | 0.01 | ② | 归一化锚点：使分层分布的「人群加权均值」≈1%。实际活跃度由 `oneoff_activity_stratified` + CDC riskg 6 层决定（顶层 0.286/天）。CDC MMWR 2022 riskg 表 |
| `oneoff_p` | 0.2105 | ④ | NegBin(n=1,p) 伴侣数分布参数，均值=(1-p)/p≈3.75 伴侣/活跃者/天。经「接触量水平扫描 f=7.5」标定（详见 `feasibility/multi_seed_f.py`，由 f=7.0 上调以在持续输入方案下恢复疫情规模）。对应高风险活跃子集（CDC 顶层 5%）的伴侣数，落在 CDC 顶层(0.286/天)与荷兰 casual 顶层(0.7-1.0/天)之上的合理高位 |
| `oneoff_age_gap` | 20 | ④ | 同性性网络年龄同质性高，20 岁为合理上限（结构假设，原 80 无依据已改） |
| `oneoff_mf_anal_frac` | 0.10 | ② | MF 一次性接触中肛交比例。NSFG（CDC 全国性行为调查 2015-2019）：美国 15-49 岁 38.5% 男性、32.6% 女性「曾有过」异性肛交（lifetime）。「单次性行为为肛交」的比例显著低于 lifetime，0.10 为合理下限，敏感性 0.05–0.20。另 meta 分析（Han et al. 2025）肛交 OR=2.40、加州病例对照（Snyder 2024）肛交为主要暴露 |
| `oneoff_orientation_bias` | {MSM:3.0, MB:1.2, 其余<1} | ② | CDC one-time 网络本质为 MSM 主导（98% 病例 MSM，ECDC/WHO 2022）。权重归一化使人群加权均值=oneoff_active_frac。MB 权重由初始 2.0 降至 1.0（2026-11-04）再上调至 1.5、最终定在 1.2（2026-11-06），用于把 bi 病例占比从 15.0% 微调到 13.4%（目标 12.6%）；反映双性恋男性虽深嵌 MSM 网络、但未维持相对异性恋男性 10 倍的一次性活跃度超额（NYC 仅 5.6% 病例为双性恋，Kyaw et al. *MMWR* 2022）。方向有据，具体权重为经验设定，做敏感性 |
| `oneoff_activity_stratified` | True | ① | CDC MMWR riskg 6 层设计（Weiss et al. 2020） |

## 四、性网络 / 派对层（sex_party）

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `sexual_active_frac` | 0.03 | ② | MCP 性活跃边每日激活比例。英格兰技术简报 8：GBMSM 病例 34.2% 报告 4-9 伴侣/3月、28.9% 报告 10+ 伴侣/3月；加拿大 3 城 MSM 平均 10.4–13.1 伴侣/6月（*J Infect Dis* 2024）。量级合理 |
| `sexual_edge_mult` | 1.0 | ④ | 性边数量乘子（baseline 不缩放） |
| `sex_party_attend_prob` | 3/7≈0.43 | ② | 高风险 MSM 每周出席派对约 3 晚。CDC 高危 MSM 派对频率描述性数据；AMIS 调查（CDC 2022）：36.3% MSM 过去 3 月参加过性派对/俱乐部 |
| `party_weekday_mult` / `party_weekend_mult` | 0.7 / 1.3 | ④ | 周末派对强度更高（社交节奏常识，无直接文献点值，声明为结构假设） |
| `party_touch_frac` | 0.20 | ④ | 高接触事件内非性密切接触扩展比例。敏感性 0.10–0.25，无直接文献点值 |
| `party_heavy_tail` | True | ① | 米兰病例系列 23% 病例 >50 伴侣/3月；CDC top 5%≈100 伴侣/年；Endo et al. 重尾证据 |
| `party_ht_frac` | 0.05 | ① | CDC riskg 顶层占比固定 5%（荷兰 very-high、意大利 HR 2.5%、英格兰 high-risk 的合理区间内） |
| `party_ht_top_mean` | 7.5 | ④ | 顶层 E[k]（NB r=1）。经接触量扫描 f=7.5 标定（详见 `feasibility/multi_seed_f.py`，由 f=7.0 上调以在持续输入方案下恢复疫情规模）。对应高风险活跃子集单晚伴侣数，CDC 顶层 3 周 6.6 伴侣的合理外推 |
| `party_ht_top_clip` | 30 | ② | 顶层伴侣数上限，米兰 >50 伴侣/3月 的上界量级 |
| `party_mix_target_mean` | 8.0 | ② | NEJM 中位 5 伴侣/3月、法国中位 3 伴侣/3周；设男性→女性桥接强度，见 `mix_mf_anal_frac` |
| `mix_mf_anal_frac` | 0.30 | ② | 方向分流的混合性别肛交比例（男→女，女性接受方）。顺性别女性主要经阴道接触感染（外阴阴道病变为主，Thornhill et al. *N Engl J Med* 2022;387:679–691；西班牙女性肛门生殖器病变率 51.0% vs 男性 67.3%、性传播 65.7% vs 92.9%），故女性接受肛交比例低于男男路线。由 0.15 上调至 0.30（仍低于男男路线）以在持续输入方案集中病例于 MSM 核心后恢复异性恋/女性外溢（2026-11-06） |
| `mix_fm_anal_frac` | 0.05 | ② | 方向分流的混合性别肛交比例（女→男，男性接受方），量级低于男→女 |
| `party_mix_homo_spill_frac` | 0.01 | ② | ECDC 1% hetero 病例 |
| `party_core_intensity` | 0.55 | ④ | bounds 0.45–0.70 中点（无直接文献，声明为校准参数） |

## 五、稳定伴侣 / 常规伴侣层（stable）

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `p_stable_sex_general_per_day` | 0.10 | ③ | 月度性频率 3/月 → p=1-exp(-3/30)≈0.095（换算推导） |
| `p_stable_sex_one_mcp_per_day` | 0.26 | ③ | 9/月 → 1-exp(-9/30)≈0.259 |
| `p_stable_sex_both_mcp_per_day` | 0.39 | ③ | 15/月 → 1-exp(-15/30)≈0.393 |
| `hh_assort` | 0.25 | ② | 性网络同配性（assortative mixing）文献，需引 assortativity 研究 |
| `bi_same_sex_frac` | 0.10 | ④ | 双性恋同性伴侣比例（经验假设，缺直接文献） |
| `stable_mf_anal_frac` | 0 | ④ | MF 常规伴侣基本不升级为肛交（结构假设） |
| `p_bi_m_to_msm` | 0.65 | ④ | bounds 0.55–0.75 中点（无直接文献，声明为校准参数） |

## 六、社会/非性密切接触层（social）

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `social_contacts_per_case` | 2.4 | ① | CDC 报告"每病例平均识别 2.4 个非性密切接触"（278 例、662 非性接触） |
| `social_contact_window_days` | 21.0 | ① | mpox 最长潜伏期/监测期 21 天 |
| `social_contact_scale` | 1.0 | ① | 文献锚点 1.0（CDC 2.4 接触 × SAR 1.5% 已够，不需再放大）。原 10 无依据，已回调 |
| `social_same_sex_assort` | 0.80 | ② | 社会接触同性同配性文献 ≈0.80（范围 0.65–0.85） |
| `social_adjust_for_r_sus` | True | ④ | 按易感者年龄调整（结构假设） |
| `community_bridge_scale` | 1.0（禁用） | — | 无文献依据，纯 LA-specific 拟合旋钮，已禁用（baseline=1.0） |

## 七、检测 / 追踪 / 监测

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `test_delay_mode` | scheduled_delay | ① | CDC 技术报告：症状→检测→结果的分阶段延迟 |
| `test_pathway_prob_symp` | 0.85 | ② | 症状病例进入检测通路的比例（CDC 技术报告量级） |
| `test_seek_delay_mean_days` | 5.0 | ② | 症状→采样延迟（CDC 技术报告量级） |
| `test_seek_delay_sd_days` | 3.0 | ② | 同上 |
| `test_seek_delay_max_days` | 21 | ① | 延迟上限 = 潜伏期上限 21 天 |
| `LAB_TURNAROUND_DAYS` | 2 | ② | 实验室周转（CDC 报告量级） |
| `MONITORING_DAYS` | 21 | ① | mpox 最长潜伏期 21 天（CDC） |
| `monitor_sexual_contact_mult` | 0.30 | ④ | 监测期性接触风险折减（经验假设，敏感性） |
| `monitor_social_contact_mult` | 0.70 | ④ | 监测期社交接触风险折减（经验假设，敏感性） |

## 八、初始条件与引入（importation）

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| `INITIAL_INFECTIONS` | 0 | ② | day-0 seed 关闭。旧版 20 例一次性播种导致峰值系统性提前约 18 天（median 74 vs 观测 92），因真实 LAC 疫情启动慢（首例 6/2、6 月中才持续增长）。改为持续低强度输入流（见下）。详见 `seed_cases_文献依据.md` |
| `importations` | 1/2/4/1 例/天（day 15–134） | ① | 持续外源输入，分四段（daily_n=1/2/4/1 over day 15–44/45–74/75–104/105–134），对应 LAC「慢 May → 加速 Jun–Jul → 收尾 Aug」节奏。文献支撑：加拿大校准 1–3 例输入（Milwid et al. *J Med Virol* 2023）；DC 网络模型 6/2 播 5 例 + surge 期网络外输入项 ε（Clay et al. *Sex Transm Infect* 2024）；英格兰 6 例/月基线输入（Zhang et al. *Lancet Reg Health Eur* 2025）。30-seed 中位 peak=96（观测 92） |
| 输入人群 | 均匀 MSM core | ① | 文献惯例（NYC 5 例、加拿大 1–3 例、英格兰 GBMSM 模型均 seed/输入于 MSM）；经 `importation_mask` 路由到 MSM core |

## 九、干预政策时间形状（policy）—— 校准所得

| 参数 | 当前值 | 来源类型 | 文献依据（可引用） |
|---|---|---|---|
| phase1 `party_mult` | 0.95 | ④ | 曲线拟合（early growth 阶段派对强度） |
| phase2/3 `party_mult` | 0.70 | ④ | long_tail2 曲线拟合（cRMSE 193 vs baseline 296） |
| 各 phase `test_mult`/`trace_mult` | 0.70–1.50 | ④ | 干预强度假设（行为改变 + 检测加强） |
| `contact_mult`/`oneoff_mult`/`social_mult` | 0.65–1.20 | ④ | 行为改变假设（疫情后性接触下降、社交回升） |

---

## 十、最需补强的参数（投稿前优先处理）

| 优先级 | 参数 | 现值 | 现状 | 建议 |
|---|---|---|---|---|
| P1 | `oneoff_p` | 0.2105 | 校准值，伴侣数分布参数 | 补充 CDC riskg 顶层伴侣数 + 荷兰 casual 层的推导过程（已写注释，可扩展成推导附录） |
| P1 | `party_ht_top_mean` | 7.5 | 校准值 | 同上，补充「3周6.6伴侣→单晚E[k]」的语义换算说明 |
| P2 | `oneoff_orientation_bias` 权重 | 3.0/1.2/… | 方向有据、权重经验 | 声明为归一化方案，做敏感性 |
| P2 | `party_touch_frac` | 0.20 | 无直接文献 | 敏感性 0.10–0.25 兜底 |
| P2 | `monitor_*_mult` | 0.30/0.70 | 经验假设 | 敏感性兜底 |
| P3 | `bi_same_sex_frac` | 0.10 | 缺直接文献 | 引双性恋人群行为学文献或声明假设 |

---

## 十一、投稿时的 source 列建议（四档速查）

1. **直接文献** → 自然史（Madewell/Spicknall）、social 层（CDC SAR=1.5%）、riskg 分层（CDC MMWR）。
2. **间接锚定** → per-act 风险比值（肛/阴/口 OR 比值）、性活动频率（英格兰简报、加拿大 3 城）。
3. **换算推导** → stable 层月频率→日概率。
4. **校准/假设** → oneoff_p、party_ht_top_mean（f=7.5 扫描）、mix_mf_anal_frac（0.30）、policy mult、monitor mult，附敏感性范围。

> 新增文献（本次补充）：
> - 英格兰 UKHSA 技术简报 8（GBMSM 伴侣数分布 34.2% 4-9 伴侣、28.9% 10+ 伴侣/3月）
> - 加拿大 3 城性网络研究（*J Infect Dis* 2024，平均伴侣 10.4–13.1/6月，R0 2.4–2.7）
> - 次代发病率系统综述（*BMC* 2024，性接触 SAR 11.4% vs 非性 1.3%，8.5 倍）
> - Han et al. 2025 meta 分析（PMC12832785，肛交 OR=2.40、多伴侣 OR=3.36）
> - Snyder et al. *J Infect Dis* 2024 加州病例对照（肛交 aOR、无症传播）
> - Chard et al. *MMWR* 2024;73(40)（无套接受性肛交 OR=5.4，唯一调整后显著）
> - NSFG / CDC 全国性行为调查 2015-2019（38.5% 男、32.6% 女曾异性肛交）
