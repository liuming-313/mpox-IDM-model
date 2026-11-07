# Model Parameterization — Methods Section (English draft)

> 这是论文 Methods "Model parameterization" 小节的英文底稿，可直接改写进正文。
> 写法模仿高分建模文章（*Lancet Infect Dis* / *J Infect Dis* / *Nat Commun*）：
> 每个参数给出取值 + 来源引用，区分 literature-informed / calibrated / structural assumption。

---

## Model parameterization

### Natural history

Incubation period (exposure to symptom onset) was drawn from a lognormal distribution with parameters (μ = 2.108, σ = 0.254; median 5.6 days, 95% CrI 4.3–7.8), truncated to 3–17 days, based on a pooled analysis of 12 US jurisdictions during May–August 2022.[Madewell 2023] The delay from symptom onset to rash was set to 1–4 days, consistent with the CDC clinical description that rash typically develops within 1–4 days after symptom onset.[CDC] The infectious period followed a gamma distribution (shape 9, scale 2.33; mean ≈21 days) truncated to 14–28 days, consistent with the CDC reference model (mean infectious period 27 days)[Spicknall 2022] and the Netherlands fading model (14–28 days).[Xiridou 2024] The generation-interval distribution for R_t estimation (Cori method) used a mean of 9.0 days and SD of 5.0 days, based on the US serial-interval estimate of 8.5 days (95% CrI 7.3–9.9).[Madewell 2023]

### Per-act sexual transmission probabilities

We specified per-act transmission probabilities for four act types. The receptive-anal probability was set to 0.28, within the range βs = 0.25–0.40 estimated for the Netherlands fading model.[Xiridou 2024] This is epidemiologically supported by a multijurisdictional case–control study in which condomless receptive anal sex was associated with 5.4-fold higher odds of mpox (p = 0.031), the only intimate-contact behavior that remained significant after adjustment.[Chard 2024] The insertive-anal probability was set to 0.06 (≈1/4 of receptive), the vaginal probability to 0.035, and the oral probability to 0.0175, following the observed hierarchy of exposure-specific risk (receptive anal > insertive anal > vaginal > oral).[Chard 2024; Ward; Oakley 2023]

### Nonsexual close-contact transmission

Nonsexual close-contact transmission was parameterized to reproduce the CDC-reported average of 2.4 identified nonsexual close contacts per case and a secondary attack rate (SAR) of 1.5% among such contacts,[CDC MMWR 2023] yielding a per-contact effective probability of p_eff = 1 − (1 − 0.015)^(1/2.4) ≈ 0.0063. This is consistent with a systematic review reporting an overall nonsexual-contact SAR of 1.3% versus 11.4% for sexual contact (8.5-fold difference).[BMC 2024]

### Sexual contact network and activity stratification

The MSM core population was stratified into six activity levels (five strata of 19% and a top stratum of 5%), following the CDC riskg design.[CDC MMWR 2022; Weiss 2020] Partner-degree distributions within the sex-party layer followed a negative-binomial with a heavy tail, motivated by the Milan case series (23% of cases reporting >50 partners over 3 months),[Milan] the CDC top-stratum estimate (≈100 partners/year), and the heavy-tail transmission argument of Endo et al.[Endo] The top-stratum mean partner count and the one-off-layer partner-count parameter were jointly calibrated (see Calibration below) to reproduce the observed outbreak size and peak timing while keeping partner counts within the range reported for high-activity MSM (CDC top stratum 6.6 partners/3 weeks; Netherlands casual top stratum 0.7–1.0/day).[CDC MMWR 2022; Xiridou 2024] One-off sexual activity was oriented toward MSM and male-bisexual strata, consistent with the observation that the 2022 outbreak was overwhelmingly concentrated in MSM networks (≈98% of cases).[ECDC/WHO 2022]

### Stable-partner layer

Stable-partner sexual frequency was converted to a daily activation probability via p = 1 − exp(−f/30) for monthly frequencies f of 3, 9, and 15 per month (yielding 0.10, 0.26, and 0.39 per day for general, one-MCP, and both-MCP stable edges, respectively).[behavioral literature]

### Initial conditions and importation

The epidemic was ignited not by a fixed batch of index cases on day 0 but by a sustained, low-level exogenous importation stream whose intensity followed the observed importation rhythm of the Los Angeles County (LAC) outbreak. The importation schedule was divided into four consecutive phases (daily imported cases = 1 over days 15–44, 2 over days 45–74, 4 over days 75–104, and 1 over days 105–134), reproducing a "slow May → accelerate Jun–Jul → taper Aug" shape; imported cases were routed to the MSM core of the sexual network. This replaced an earlier scheme that introduced 20 infectious MSM simultaneously on day 0, which produced a systematic ≈18-day early peak (median simulated peak day 74 versus the observed day 92): because the real LAC outbreak began slowly (first reported case 19 May 2022, sustained growth only from mid-June), a one-shot day-0 seed caused endogenous amplification to begin far too early. Sustained importation defers epidemic acceleration to match the observed tempo and reflects the reality that an urban outbreak such as LAC's is continuously replenished by externally acquired cases. This approach is directly consistent with prior mpox models: Canada calibrated both the timing (March–April 2022) and number (1–3) of imported cases, all in the exposed state with >10 annual partners and age <50 (Milwid et al., *J Med Virol* 2023); the US Washington DC network model seeded 5 cases on 2 June from the highest-activity strata and added an extra-network importation term ε during the 11–25 June surge (Clay et al., *Sex Transm Infect* 2024); and England's cost-effectiveness model assumed a baseline importation rate of 6 cases/month (sensitivity 1–10/month) (Zhang et al., *Lancet Reg Health Eur* 2025). Under this scheme the 30-seed median peak occurs on day 96 (observed 92), eliminating the phase error.

### Calibration

The overall transmission volume was calibrated by jointly scaling the one-off partner-count parameter (oneoff_p = 0.2105; mean ≈3.75 partners/active-individual/day) and the top-stratum sex-party partner count (party_ht_top_mean = 7.5) by a factor f = 7.5, chosen to reproduce the LAC final outbreak size (≈2,280 cases) and peak timing (≈day 92) under the sustained-importation scheme. Case-composition targets (male-homosexual ≈77.5%, male-bisexual ≈12.6%, heterosexual ≈9.9%) were matched through three structural parameters with direct behavioral anchors: (i) the male-bisexual one-off participation weight (oneoff_orientation_bias = 1.2, down from an initial 2.0 then fine-tuned to restore the bisexual bridge without overshooting the 12.6% target), reflecting that bisexual men were highly embedded in MSM networks yet did not sustain a 10-fold one-off activity excess over heterosexual men (only 5.6% of NYC cases were bisexual; Kyaw et al., *MMWR* 2022); (ii) the mixed-sex party partner-count target (party_mix_target_mean = 8.0), which sets the strength of male–female bridging within mixed-sex party events; and (iii) a direction-split mixed-sex anal fraction (mix_mf_anal_frac = 0.30 male-to-female, mix_fm_anal_frac = 0.05 female-to-male), reflecting that receptive anal transmission — the dominant high-risk route — is concentrated in male–male sex, whereas cisgender women acquire mpox predominantly through vaginal contact (vulvovaginal rather than anorectal lesions); the male-to-female fraction was raised from 0.15 to 0.30 (still below the male–male anal route) to restore heterosexual/female spillover after the importation scheme concentrated cases in the MSM core. All remaining parameters were either literature-fixed or structural assumptions with sensitivity ranges reported in Supplementary Table S1.

### Sexual orientation as identity vs. behavior

Consistent with the ECDC–WHO case-definition guidance, we treated surveillance-reported sexual orientation as a self-identified attribute that is *not necessarily representative of the sex of a case's sexual partners*.[ECDC/WHO 2022] This distinction matters for calibration: in the CDC national technical report, 93.3% of cases with known orientation/sex were gay or bisexual men, 5.4% were heterosexual men, and only 1.3% were women, whereas the LAC target attributes 6.2% of cases to heterosexual men and 3.7% to women. The higher proportion of "heterosexual" men in LAC surveillance likely reflects men who self-identify as heterosexual but report male sexual contact — the CDC found that 65 of 122 cases without reported male-to-male sexual contact were ultimately excluded because male-to-male contact could not be ruled out. Accordingly, our model classifies cases by *sexual behavior* (the sex of the transmitting partner), so the model's heterosexual-male proportion is expected to be systematically lower than the self-identified LAC figure; we report this as a known calibration caveat rather than forcing the model to reproduce a self-identification artifact.

### Structural assumptions and sensitivity

Parameters lacking direct literature anchors — including the mixed-sex anal fraction (mix_mf_anal_frac = 0.30, sensitivity range reported in Supplementary Table S1), party close-touch fraction (0.20, sensitivity 0.10–0.25), and monitoring risk-reduction multipliers (0.30/0.70) — were treated as structural assumptions and varied in sensitivity analyses (Supplementary Table S1). Epidemic ignition relied on a sustained low-level exogenous importation stream (rather than a single day-0 seed), as described under "Initial conditions and importation" above.

---

## References (to be formatted)

- **Madewell ZJ, et al.** Incubation period, serial interval... *Emerg Infect Dis* 2023;29(4):818-821.
- **Spicknall IH, et al.** Modeling the impact of sexual networks... *MMWR* 2022;71(35):1131-1135.
- **Xiridou M, et al.** The fading of the mpox outbreak among MSM... *J Infect Dis* 2024 (PMC11272055).
- **Chard AN, et al.** Risk of clade II mpox associated with intimate and nonintimate close contact... *MMWR* 2024;73(40).
- **Snyder RE, et al.** Sexual exposures associated with mpox infection: California... *J Infect Dis* 2024;229(Suppl 2):S188-S196.
- **Ward T, et al.** (household transmission, vaginal contact aOR=0.41).
- **Oakley L, et al.** (female case proportion 2.7%).
- **Weiss KM, et al.** Egocentric sexual networks of men who have sex with men in the United States (ARTnet study). *Epidemics* 2020;30:100386.
- **Milwid RM, et al.** (Canada SEIR, 1–3 imported cases, March–April 2022). *J Med Virol* 2023.
- **Clay PA, et al.** (US DC network model, 5 cases 2 June + importation term ε). *Sex Transm Infect* 2024.
- **Zhang XS, et al.** Cost-effectiveness of vaccination strategies to control future mpox outbreaks in England. *Lancet Reg Health Eur* 2025.
- **Kyaw NT, et al.** Clinical and epidemiologic characteristics of mpox cases from the initial phase of the outbreak — New York City. *MMWR Morb Mortal Wkly Rep* 2022;71(52):1636–1637.
- **UKHSA.** Investigation into monkeypox outbreak in England: technical briefing 8.
- **Canada 3-city.** Characteristics of the sexual networks of MSM in Montréal, Toronto, Vancouver... *J Infect Dis* 2024.
- **Systematic review.** Secondary attack rates after mpox exposure... *BMC* 2024 (PMC13274034).
- **Han Y, et al.** Epidemiological and clinical differences... *BMC* 2025 (PMC12832785).
- **NSFG / CDC.** Heterosexual anal and oral sex... 2011-2015 / 2015-2019.
- **ECDC/WHO.** Joint situation report — monkeypox multi-country outbreak (sexual orientation ≠ sexual behavior caveat). 2022.
- **CDC.** Multi-national monkeypox outbreak: technical report 2 (4,460 cases: 93.3% gay/bi men, 5.4% heterosexual men, 1.3% women).
- **CDC MMWR.** Possible exposures among mpox patients without reported male-to-male sexual contact (65/122 excluded because MMSC could not be ruled out). 2023.
- **Thornhill JP, et al.** Monkeypox virus infection in humans across 16 countries (female case series: 89% reported sex with men, vulvovaginal lesions). *N Engl J Med* 2022;387:679-691.
