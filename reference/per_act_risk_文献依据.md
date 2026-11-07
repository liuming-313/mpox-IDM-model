# mpox per-act / per-contact 传播风险：文献依据与模型参数化

> 用途：为 `main.py` 里的 per-act 传播风险参数（`sexual_anal_receptive` / `sexual_anal_insertive` / `sexual_vaginal` / `sexual_oral` / `social_p_eff` / `close_touch`）提供可引用的流行病学依据，供论文 Methods / Supplementary Tables 使用。
>
> 核心问题：per-act 风险既决定传播动力学（总规模、增长速度、放大倍数），又决定病例的性别-性取向构成（homo / bi / hetero / female）。这两个维度正是模型的校准目标，因此 per-act 风险的分层必须有理有据，不能拍脑袋。

---

## 一、一个必须先澄清的生物学事实

mpox 的传播本质是**皮肤/黏膜与感染性皮损的直接接触**，**不是**像 HIV 那样的体液介导黏膜感染。这一点决定了 anal vs vaginal 的风险差异机制与 HIV 不同：

- HIV 里 anal > vaginal 主要是因为**直肠黏膜薄、易损伤、富含免疫靶细胞**，属于"黏膜易感性"差异；
- mpox 里 receptive anal 风险高，主要是因为**肛交更容易造成微损伤 + 皮损直接接触**，属于"接触强度/皮损暴露"差异，而非"体液传染"。

**文献铁证**（多处一致）：

- **NEJM 2022（Thornhill et al., 528 例 16 国病例系列）**：95% 通过性行为传播，但作者明确写 "there is no clear evidence of sexual transmission through seminal or vaginal fluids"——即**没有证据支持精液/阴道液介导传播**。
- **DW 专家访谈（Paul Hunter 等）**：mpox "almost certainly sexually transmitted"，但"即使只是裸体拥抱（naked cuddling）也是大风险"，不能用避孕套类比来防护。

**模型含义**：per-act 风险分层的正确理解是"不同性行为方式的**接触强度/皮损暴露风险**差异"，而不是"不同黏膜的感染易感性差异"。这为 anal > vaginal > oral 的排序提供了机制解释，但**比值大小必须靠流行病学 OR 数据锚定**（见下）。

---

## 二、最硬的数据：CDC 2024 病例对照研究（Chard et al., MMWR 2024）

> Chard AN, et al. *Risk of Clade II Mpox Associated with Intimate and Nonintimate Close Contact Among MSM and Transgender Adults — United States, August 2022–July 2023.* MMWR 2024;73(40).
>
> 设计：457 病例 + 1030 对照（多辖区疫苗有效性病例对照研究的子分析），logistic 回归控制疫苗接种、社会人口学特征、并发的其他亲密接触行为。

各行为的感染 OR（关键表格）：

| 亲密接触行为 | 未调整 OR (95% CI) | 调整后 aOR (95% CI) |
|---|---|---|
| **无套肛交·接受方（condomless receptive anal）** | 3.2 (1.4–7.3) | **5.4 (1.2–24.6)** ← 唯一调整后仍显著 |
| 无套肛交·插入方（condomless insertive anal） | 2.8 (1.3–6.1) | 1.0 (0.2–5.6) |
| 有套肛交（anal with condom） | 1.6 (0.4–5.8) | 0.5 (0–10.7) |
| 接受无套口交（received oral） | 2.7 (1.3–5.4) | 0.8 (0.2–3.4) |
| 给予无套口交（gave oral） | 2.0 (1.0–4.1) | 2.6 (0.6–11.2) |
| 集会中半裸接触（mass gathering, undressed touching） | 0.3 (0.1–0.7) | 0.3 (0.1–1.3) |
| **非亲密接触（护理 / 共用餐具 / 共享毛巾等）** | ~0.9–2.3 | ~1.0–4.3（多不显著） |

**核心结论**：
1. **receptive（接受方）无套肛交是唯一在调整后仍显著的风险行为，aOR=5.4**。这是 mpox 性传播的 dominant 风险行为。
2. **插入方（insertive）肛交在调整后 aOR≈1.0（不显著）**——即插入方的风险远低于接受方。
3. **口交在调整后不显著**。
4. **非亲密接触整体不显著**，风险远低于肛交。

**模型含义**：per-act 风险排序 **receptive anal ≫ insertive anal ≳ oral ≳ nonsexual** 有强文献支撑。具体比值：receptive/insertive 的 aOR 比约为 5.4/1.0，但 CI 很宽（1.2–24.6），只能作为方向性依据，不能作为精确比值。

---

## 三、补充证据：接触部位与皮损的对应关系（支持"接触强度"机制）

> Snyder RE, et al. *Sexual Exposures Associated With Mpox Infection: California, November 2022 to June 2023.* J Infect Dis 2024;229(S2):S188–S196.

- 有**肛周/直肠皮损**的病例，更常报告接受方肛交（aOR=14.4）；
- 有**阴茎皮损**的病例，更常报告插入方肛交/阴道性交（aOR=9.3）；
- 皮损部位与性行为方式高度对应 → 支持"皮损接触 = 传播"的机制，而非"体液传播"。

**模型含义**：进一步佐证 per-act 风险应按"皮损暴露部位"分层，receptive anal（肛周皮损暴露）是最高风险。

---

## 四、非性接触的 secondary attack rate（SAR）锚定

这是你 `social_p_eff` 和 `close_touch` 的锚定依据，文献非常明确：

| 场景 | SAR | 来源 |
|---|---|---|
| **非性密切接触**（nonsexual close contact） | **1.5%** | CDC MMWR 2023（Notes from the Field，DC + Tennessee） |
| 非性家庭接触 | 1.5% | 北京朝阳区 2023–24 现场研究 |
| 家庭接触（总体，含性） | 4%（英国）/ 8%（非洲 pooled） | UK 2022 家庭研究 / 系统综述 |
| 性接触（家庭内） | aOR=11 vs 非性 | UK 家庭研究 |
| 性接触 SAR（clade Ib） | ≈75% | WHO 2026 外部情况报告 64 |
| 非性家庭接触 SAR（clade Ib） | ≈20% | WHO 2026（注意：clade Ib 家庭场景，高于 2022 clade IIb） |

**关键点**：
- 2022 clade IIb 的非性 SAR ≈ 1.5%，这是你 `social_p_eff=0.0063` 的锚定依据；
- 你的换算 `p_eff = 1 - (1 - 0.015)^(1/2.4) ≈ 0.0063`（每个病例平均 2.4 个非性密切接触）已经写在 main.py 注释里，方法正确；
- UK 家庭研究还给出一个有用的信号：**女性接触者的感染风险比男性低 59%（aOR=0.41, 0.15–0.95）**——这直接支撑"MF 边（女性作为接受方）风险低于 MM 边"。

---

## 五、你模型当前参数 vs 文献依据对照

| 参数 | 你模型现值 | 文献依据 | 评价 |
|---|---|---|---|
| `sexual_anal_receptive` | 0.28 | aOR=5.4（dominant） | ✅ 方向正确，作为 anchor |
| `sexual_anal_insertive` | 0.06 | aOR 调整后≈1.0（不显著） | ✅ 0.06/0.28≈0.21 倍，与"插入方明显低于接受方"一致 |
| `sexual_vaginal` | 0.035 | 女性仅 2.7%；女性接触 aOR=0.41 | ✅ 方向正确（低于 anal），需敏感性分析确认 female 落点 |
| `sexual_oral` | 0.0175 | 口交 aOR 调整后不显著 | ✅ 最低，合理 |
| `social_p_eff` | 0.0063 | 非性 SAR 1.5%，2.4 接触/病例 | ✅ 唯一有精确外部锚定的参数 |
| `close_touch`（party touch） | ~0.0067 | 非性接触 SAR 1.5% | ⚠️ 与 social_p_eff 同量级，但**目前不分性别组合**（见下） |

---

## 六、文献给出的比值约束（可作为敏感性分析范围）

文献**不给 per-act 概率的绝对值**（那需要传播链的精确建模，现有研究做不到），只给**相对 OR 比值**。因此正确的参数化策略是：

1. **选一个 anchor**：receptive anal（文献确认它是 dominant），用你的 0.28（或 CDC 参考模型的 0.6）；
2. **其余行为按 OR 比值缩放**：
   - insertive anal ≈ 0.2–0.3 × receptive（依据：插入方 aOR≈1.0 vs 接受方 5.4，但 CI 宽，建议 0.15–0.35 作为敏感性范围）→ 你现在的 0.21 已落在范围内；
   - vaginal ≈ 0.05–0.15 × receptive（依据：女性 2.7% 占比 + 女性接触 aOR=0.41）→ 你现在 0.125 在范围内；
   - oral ≈ 0.03–0.08 × receptive（依据：口交 aOR 不显著）→ 你现在 0.0625 在范围内；
3. **nonsexual**：用 SAR 1.5% 直接锚定（不随 anchor 缩放），即 `social_p_eff` 和 `close_touch` 独立于性风险。

**你的 main.py 里 `transmission_risk_parameter_bounds`（第 289-294 行）已经定义了这个敏感性范围**，且方向与文献一致。这说明之前的校准已经隐含地做对了大部分——缺的只是把"为什么这么定"的文献依据显式写出来。

---

## 七、真正需要修的缺口（与文献对照后确认）

对照文献，你的 per-act 风险**排序和相对比值基本合理**，真正的问题是**网络层没有贯彻这个分层**：

1. **one-off 层风险未按性别组合分层**（过夜扫描暴露的硬伤）：one-off 层建边时用的是统一风险值，没有像 `build_edge_risks` 那样按 MM/FF/MF 区分。这导致提升 one-off 活跃度时，MF 边（异性恋 spillover）与 MM 边（MSM 内部）等比放大，把 hetero/female 占比冲垮。
   - **文献支撑**：女性接触 aOR=0.41（UK 家庭研究）、女性仅 2.7%（MMWR）、receptive anal 是唯一显著风险（MM 网络主导）。→ one-off 层的 MF 边应显著降权。

2. **`close_touch`（party touch）未按性别组合分层**：与 one-off 同理，party 内的皮肤接触对 MF/MM/FF 统一风险，无难度墙。

这两处就是"per-act 风险分层没有贯穿到所有传播层"的体现，也是下一步要修的。

---

## 八、参考文献清单（可直接用于论文）

1. **Thornhill JP, et al.** Monkeypox Virus Infection in Humans across 16 Countries — April–June 2022. *N Engl J Med* 2022;387:679–691.（95% 性传播；无精液/阴道液传播证据）
2. **Chard AN, et al.** Risk of Clade II Mpox Associated with Intimate and Nonintimate Close Contact Among MSM and Transgender Adults — United States, August 2022–July 2023. *MMWR* 2024;73(40).（receptive anal aOR=5.4，唯一显著）
3. **Snyder RE, et al.** Sexual Exposures Associated With Mpox Infection: California, November 2022 to June 2023. *J Infect Dis* 2024;229(S2):S188–S196.（皮损部位与性行为对应）
4. **Oakley LP, et al.** Mpox Cases Among Cisgender Women and Pregnant Persons — United States, May 11–November 7, 2022. *MMWR* 2023;72(1).（女性占 2.7%）
5. **CDC MMWR 2023.** Notes from the Field: Transmission of Mpox to Nonsexual Close Contacts — Two U.S. Jurisdictions, May 1–July 31, 2022. *MMWR* 2023;72(50).（非性 SAR 1.5%）
6. **Ward T, et al.** Mpox in UK households: estimating secondary attack rates and factors associated with transmission, May–November 2022.（家庭 SAR 4%；性接触 aOR=11；女性接触 aOR=0.41）
7. **WHO.** Multi-country outbreak of mpox — External Situation Report No. 64 (2026).（性接触 SAR≈75%，非性家庭≈20%，clade Ib）
8. **Spicknall IH, et al.** Modeling the Impact of Sexual Networks in the Transmission of Monkeypox virus Among Gay, Bisexual, and Other MSM — United States, 2022. *MMWR* 2022.（CDC 参考模型，inf.prob=0.6/0.9，act rate 分层）

---

## 九、结论（一句话）

你的 per-act 风险**排序和相对比值在文献上站得住脚**（receptive anal ≫ insertive anal ≳ oral ≳ nonsexual），唯一"拍脑袋"的部分是**绝对量级**（文献给不了唯一答案，只能做敏感性分析）；真正要修的是**把这个分层贯彻到 one-off 层和 party touch 层**——这两个层现在没按性别组合区分风险，是 hetero/female 构成崩坏的根源。
