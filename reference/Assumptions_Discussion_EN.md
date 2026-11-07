# Assumptions, Discussion, and Model Comparison — English draft

> 审稿人第 5/6/7 条质疑的回应底稿：
> - §1 显式 H1/H2 假设（第 5 条）
> - §2 政策含义 Discussion 段（第 6 条）
> - §3 与他模型对比表（第 7 条）

---

## 1. Explicit study hypotheses

We evaluated two pre-specified hypotheses about the transmission structure of the 2022 mpox outbreak in Los Angeles County:

- **H1 (primary sexual-network transmission).** Sexual contact within MSM networks — particularly high-activity sexual partnerships and group-sex events — is the dominant transmission route, and its heterogeneous, heavy-tailed structure is necessary and sufficient to explain both the outbreak size and its over-representation of gay and bisexual men.

- **H2 (secondary nonsexual bridging).** Nonsexual close contact (household and social) is a comparatively minor route that contributes a small but measurable fraction of cases, and acts principally as a bridge to women and to non-MSM men rather than as a driver of the main MSM epidemic.

These hypotheses were translated into model structure rather than tested as free parameters: H1 motivated the layered sexual network (stable / one-off / sex-party) with activity stratification, and H2 motivated a separate, low-probability social contact layer. The results below evaluate the degree to which each layer contributes to transmission and thus bear directly on both hypotheses (see the layer-composition decomposition in Results).

---

## 2. Policy implications (Discussion)

Our finding that the outbreak is attributable predominantly to high-activity sexual networks has three direct public-health implications.

**First, intervention targeting matters more than overall intensity.** Because a small, high-activity subpopulation drives most transmission, interventions that reduce partner rates among this group — rather than population-wide contact reductions — are disproportionately effective. This is consistent with the English analysis, which attributed the July 2022 downturn primarily to a 44.5% reduction in the sexual-partner rate among high-risk GBMSM rather than to vaccination.

**Second, early detection of the high-activity core is the rate-limiting step.** In our model, the epidemic's establishment is governed by whether the high-activity sex-party core becomes "ignited" into self-sustaining growth; before that threshold, cases remain sporadic. Surveillance that rapidly identifies and engages newly-diagnosed individuals in the high-activity network — for vaccination, treatment, and partner notification — is therefore more valuable than broad case-finding in the general population.

**Third, the secondary nonsexual route, though minor in case count, is the pathway to women and non-MSM men.** Preventing sustained heterosexual transmission requires interrupting household and social bridging early, which argues for contact tracing that explicitly includes non-sexual close contacts of diagnosed MSM cases.

---

## 3. Comparison with other mpox transmission models

Our model occupies a distinct position in the 2022 mpox modelling literature: an individual-based, network-structured model with explicit sexual-act-level transmission probabilities, applied to a large US county with fully endogenous (importation-free) transmission.

| Study / region | Model type | Transmission structure | Seed / input | R0 | Key finding |
|---|---|---|---|---|---|
| **This study (LAC)** | Individual-based, layered network | 5 layers (household/stable/one-off/sex-party/social); per-act probabilities by act type; 6-level MSM stratification | 20 MSM, no importation | — | High-activity sexual network drives size; nonsexual route bridges to women |
| England (Lancet ID 2023) | Structured compartmental | GBMSM split low/high risk; sexual + non-sexual routes | — | 1.41–2.17 | 83.8% of infections from high-risk GBMSM sexual partnerships; 44.5% partner-rate reduction reversed outbreak |
| NYC (Spicknall 2022) | Agent-based (Covasim) | sexual + non-sexual, network-based | 5 cases | — | Vaccine + behavior interventions averted cases |
| Canada 3-city (Xiu 2024) | Network-based | sexual networks of MSM | 2–5 imported | 2.4–2.7 | partner-degree heterogeneity drives R0 |
| Netherlands (Xiridou 2024) | Agent-based | sexual network, fading | — | — | behavior change (fading) + vaccination ended outbreak |
| Italy / 16-country (Bragazzi 2024) | Compartmental, risk-structured | high/low-risk, sexual/non-sexual | — | 1.37–3.68 | underestimation 25–93% in high-risk group |
| US DC (contribution) | Agent-based | high-activity input surge | daily input to high-activity | — | targeted input sustains epidemic |

**What distinguishes our model.** Unlike the England compartmental model (which aggregates risk into two groups), we resolve six MSM activity strata and separate one-off from stable from group-sex contacts. Unlike models that seed sustained importation (Netherlands, US DC), we assume fully endogenous transmission and still reproduce the outbreak size, demonstrating that the heavy-tailed sexual network alone is sufficient. Unlike the 16-country risk-structured models, we model per-act transmission probabilities (receptive/insertive anal, vaginal, oral) rather than an aggregate contact rate.

> **Note.** The comparison rows for NYC, Canada, Netherlands, and US DC are summarized from the references cited in `Supplementary_Material_S1.md`; exact R0 and intervention-effect values should be verified against each primary source before final submission.
