# Supplementary Material S1
## Model Parameterization and Data Sources

> This file consolidates three working documents — the parameter source master table, the English Methods parameterization draft, and the parameter source audit — into a single formal supplementary file suitable for journal submission.
>
> **Content overview**
> - **S1.1** Parameter source master table (parameter, value, source type, justification, reference)
> - **S1.2** English "Model parameterization" Methods text (ready to adapt into the main text)
> - **S1.3** Parameters requiring prioritised documentation before submission
> - **S1.4** Full reference list (formatted)
>
> Source types (four tiers, following common practice in high-impact modelling papers):
> - **① Literature-anchored (direct)** — a point estimate is available from a peer-reviewed study or an official report.
> - **② Literature-anchored (indirect)** — a ratio, magnitude, or range is available from the literature; the value is obtained by conversion or by taking a range midpoint.
> - **③ Derived by conversion** — obtained from a literature frequency/ratio via a standard formula (e.g., monthly frequency → daily probability).
> - **④ Calibrated / structural assumption** — inferred by fitting to calibration targets, or a necessary model-structure assumption; the search range is declared and sensitivity analysis is provided.

---

## S1.1 Parameter source master table

### S1.1.1 Natural history (disease-progression timing)

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `incubation_lognorm_mu` | 2.108 | ① | Madewell et al., *Emerg Infect Dis* 2023;29(4):818–821. Lognormal fit to incubation periods across 12 US jurisdictions; median 5.6 d (95% CrI 4.3–7.8) |
| `incubation_lognorm_sigma` | 0.254 | ① | Same as above |
| `incubation_clip` | (3, 17) d | ① | Truncation of the incubation distribution (Madewell 2023) |
| `symptom_onset_delay_range` | (1, 4) d | ① | CDC clinical description: rash typically appears 1–4 days after symptom onset |
| `infectious_period_gamma_shape` | 9.0 | ① | Spicknall et al., *MMWR* 2022;71(35):1131–1135 (CDC reference model, mean infectious period 27 d); Netherlands fading model 14–28 d (Xiridou et al., *J Infect Dis* 2024). Our gamma mean ≈21 d sits between these estimates |
| `infectious_period_gamma_scale` | 2.33 | ① | Same as above (shape × scale = 21 d) |
| `infectious_period_clip` | (14, 28) d | ① | Truncation of the infectious period (Netherlands fading model) |
| `gi_mean` | 9.0 | ① | Madewell 2023 serial-interval mean 8.5 d (7.3–9.9), rounded; consistent in magnitude with the England estimate of 8.82 d (5.22–25.81) |
| `gi_sd` | 5.0 | ① | Madewell 2023 SD 5.0 (4.0–6.4) |
| `gi_max` | 30 | ④ | Upper truncation of the generation interval (structural assumption) |

### S1.1.2 Per-act / per-contact transmission probabilities

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `sexual_anal_receptive` | 0.28 | ① | Netherlands fading-model per-contact transmission probability βs ∈ [0.25, 0.40] (Xiridou et al., *J Infect Dis* 2024 / medRxiv 2023); value 0.28 taken within this range. Key epidemiological support: Chard et al., *MMWR* 2024;73(40) multijurisdictional case–control study — condomless receptive anal sex OR = 5.4 (p = 0.031), the only intimate-contact behaviour significant after adjustment |
| `sexual_anal_insertive` | 0.06 | ② | Insertive risk ≈ 1/4 of receptive (Chard et al. 2024: insertive aOR ≈ 1.0, not significant; ratio within 0.15–0.35) |
| `sexual_vaginal` | 0.035 | ② | Female case proportion 2.7% (Oakley 2023); vaginal-contact aOR = 0.41 (Ward household study); ≈1/8 of anal risk |
| `sexual_oral` | 0.0175 | ② | Oral aOR not significant (Chard 2024), lowest risk; ≈1/16 of anal risk |
| `social_p_eff` | 0.0063 | ① | **The single parameter with a precise external anchor**: nonsexual close-contact secondary attack rate SAR = 1.5% (CDC MMWR 2023), so p_eff = 1 − (1 − 0.015)^(1/2.4) ≈ 0.0063. See also the systematic review of secondary attack rates (*BMC* 2024): nonsexual-contact SAR 1.3% vs sexual-contact SAR 11.4% (8.5-fold) |
| `social_p0_cap` | 0.03 | ④ | Upper bound on per-contact nonsexual risk. A structural assumption (no direct point estimate); magnitude is consistent with the nonsexual SAR anchor (1.5% over ~2.4 contacts) but the cap itself is a modelling safeguard, varied in sensitivity analysis |

### S1.1.3 One-off sexual network layer

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `oneoff_active_frac` | 0.01 | ② | Normalisation anchor such that the population-weighted mean of the stratified distribution ≈ 1%. Actual activity is determined by `oneoff_activity_stratified` together with the CDC riskg 6 strata (top stratum 0.286/day). CDC MMWR 2022 riskg table |
| `oneoff_p` | 0.2105 | ④ | Negative-binomial (n = 1, p) partner-count parameter; mean = (1 − p)/p ≈ 3.75 partners/active-individual/day. Calibrated by a contact-volume sweep (factor f = 7.5; see `feasibility/multi_seed_f.py`), raised from f = 7.0 to restore the outbreak size under the sustained-importation scheme. Corresponds to the partner count of the high-activity subset (CDC top 5% stratum), above the CDC top stratum (0.286/day) and the Netherlands casual top stratum (0.7–1.0/day) |
| `oneoff_age_gap` | 20 | ② | Same-sex sexual networks are age-assortative with a peak in partner counts and concurrency among MSM in mid-life (35–54 years) and a monotonic decline with age for heterosexual partnership groups (Brainard et al., *PLoS One* 2023;18(9):e0291001). A 20-year age difference is a reasonable upper bound; the original value of 80 was unsupported and revised |
| `oneoff_mf_anal_frac` | 0.10 | ② | Proportion of mixed-sex one-off contacts that are anal. NSFG (CDC national survey 2015–2019): 38.5% of US men and 32.6% of women aged 15–49 have *ever* had heterosexual anal sex (lifetime). The per-act proportion is substantially lower than the lifetime proportion; 0.10 is a reasonable lower bound (sensitivity 0.05–0.20). See also Han et al. 2025 meta-analysis (anal OR = 2.40) and Snyder et al. 2024 California case–control (anal sex the primary exposure) |
| `oneoff_orientation_bias` | {MSM:3.0, MB:1.2, others <1} | ② | The CDC one-time network is MSM-dominated (≈98% of cases MSM; ECDC/WHO 2022). The male-bisexual weight (MB) is set to 1.2 (initially 2.0, lowered to 1.0 on 2026-11-04, raised to 1.5 then back to 1.2 on 2026-11-06 to fine-tune the bisexual case share from 15.0% to 13.4%, near the 12.6% target), reflecting that bisexual men were highly embedded in MSM networks yet did not sustain a 10-fold one-off activity excess over heterosexual men: only 5.6% of NYC cases were bisexual (40/719) (Kyaw et al., *MMWR* 2022). WHO global data similarly record only 6.3% of orientation-known cases as bisexual men. Weights are normalised so the population-weighted mean equals `oneoff_active_frac`; the specific weights are varied in sensitivity analysis |
| `oneoff_activity_stratified` | True | ① | CDC MMWR riskg 6-stratum design (Weiss et al. 2020) |

### S1.1.4 Sex-party network layer

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `sexual_active_frac` | 0.03 | ② | Daily activation fraction of MCP sexual edges. England technical briefing 8: 34.2% of GBMSM cases reported 4–9 partners/3 months and 28.9% reported ≥10 partners/3 months; Canadian 3-city MSM mean 10.4–13.1 partners/6 months (*J Infect Dis* 2024) |
| `sexual_edge_mult` | 1.0 | ④ | Multiplier on sexual edge count (baseline unscaled) |
| `sex_party_attend_prob` | 3/7 ≈ 0.43 | ② | High-risk MSM attend parties ≈3 nights/week. CDC descriptive data on high-risk MSM party frequency; AMIS survey (CDC 2022, n = 553 high-risk MSM): 36.3% attended a sex club/sex party/bathhouse and 43.4% reported group sex in the past 3 months; England technical briefing 4 (July 2022) reported festival (37%) and sex-on-premise venue (25%) attendance among confirmed cases; Madrid case series: 14.4% of cases reported attending mass events/parties, 56.9% used dating apps, and 64.9% had casual partners (Iñigo Martínez et al., *Euro Surveill* 2022) |
| `party_msm_attend_prob` | 0.40 | ② | Anchored to AMIS 2022 (36.3% MSM attended a sex party/club/bathhouse in 3 months) and England briefing (sex-on-premise 25%, festival 37%); set slightly above the survey figure to reflect the higher-risk MSM subpopulation represented in the party layer |
| `party_wsw_attend_prob` | 0.35 | ② | Slightly below the MSM figure, consistent with women representing <5% of cases and lower participation in male-dominated party networks (WHO/CDC 2022) |
| `party_weekday_mult` / `party_weekend_mult` | 0.7 / 1.3 | ④ | Higher party intensity at weekends (social convention; direction supported by England briefing showing festival/sex-on-premise attendance concentrated on weekend/festival periods; no direct point estimate; declared as structural smoothing assumption) |
| `party_touch_frac` | 0.20 | ② | Fraction of extended nonsexual close contacts within high-contact events. Direction and magnitude anchored to CDC Thomas et al., *MMWR* 2023;72(50):1351–2, which reported a mean of 2.4 identified nonsexual close contacts per case, a nonsexual close-contact SAR of 1.5% (pooled), a household-contact SAR of 8% in endemic settings, and large gatherings (festivals) as the most common nonsexual exposure setting (40.9%). Sensitivity 0.10–0.25; no precise point estimate |
| `party_heavy_tail` | True | ① | Milan cohort: 23.0% of mpox cases reported >50 partners/3 months (vs 1.69% of controls) (Passini et al. 2023); CDC top 5% ≈ 100 partners/year; heavy-tail argument of Endo et al. |
| `party_ht_frac` | 0.05 | ① | Top-stratum fraction fixed at 5% (CDC riskg); within the range of the Netherlands very-high, Italian HR (2.5%), and England high-risk strata |
| `party_ht_top_mean` | 7.5 | ④ | Top-stratum E[k] (negative-binomial r = 1). Calibrated by the contact-volume sweep (f = 7.5; see `feasibility/multi_seed_f.py`), raised from f = 7.0 to restore the outbreak size under the sustained-importation scheme. Corresponds to the single-night partner count of the high-activity subset; a reasonable extrapolation of the CDC top-stratum 6.6 partners/3 weeks |
| `party_ht_top_clip` | 30 | ② | Upper bound on top-stratum partner count; magnitude of the Milan >50 partners/3 months upper bound |
| `party_mix_target_mean` | 8.0 | ④ | Mixed-sex party partner-count target (mean partners per attendee in the MIX pool). Calibrated to restore heterosexual bridging after the male-bisexual one-off weight was reduced (bi men are a key hetero bridge; see scan_mix_restore.py). WHO global data indicate that 13.3% of orientation-known cases were attributed to heterosexual transmission, supporting the need for a non-negligible mixed-sex bridge |
| `party_mix_homo_spill_frac` | 0.01 | ② | ECDC 1% heterosexual cases |
| `party_core_intensity` | 0.55 | ② | Core MIX/party intensity (mapped to party-mix attendance). Direction and magnitude anchored to the WHO global situation report 12 (Dec 2022), which found 59.1% (3776/6390) of reported exposure settings to be "party setting with sexual contact" — the single most common exposure setting; UK EID 2024 (n=1333) found 33% of GBMSM met a recent partner at a sex-on-premise venue/sex party; AMIS 2022 found 36.3% of high-risk MSM attended a sex club/party/bathhouse in 3 months. The midpoint 0.55 sits within this "one-third to three-fifths" range; declared as a calibrated midpoint over bounds 0.45–0.70 |
| `mix_mf_anal_frac` | 0.30 | ② | Direction-split mixed-sex anal fraction, male→female (female receptive). Cisgender women acquire mpox predominantly through vaginal contact — vulvovaginal lesions predominated in cis women whereas anorectal lesions predominated in trans women (Thornhill et al., *N Engl J Med* 2022;387:679–691); Spanish data likewise show lower anogenital-lesion rates in women (51.0% vs 67.3% in men) and lower sexual transmission (65.7% vs 92.9%). The female-receptive anal fraction is therefore lower than the MSM anal fraction; it was raised from 0.15 to 0.30 (still below the male–male route) on 2026-11-06 to restore heterosexual/female spillover after the importation scheme concentrated cases in the MSM core |
| `mix_fm_anal_frac` | 0.05 | ② | Direction-split mixed-sex anal fraction, female→male (female insertive). Biologically rare; set near zero |

### S1.1.5 Stable-partner layer

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `p_stable_sex_general_per_day` | 0.10 | ③ | Monthly frequency 3/month → p = 1 − exp(−3/30) ≈ 0.095 (conversion) |
| `p_stable_sex_one_mcp_per_day` | 0.26 | ③ | 9/month → 1 − exp(−9/30) ≈ 0.259 |
| `p_stable_sex_both_mcp_per_day` | 0.39 | ③ | 15/month → 1 − exp(−15/30) ≈ 0.393 |
| `hh_assort` | 0.25 | ② | Household/stable-partner assortativity. Direction supported by CDC Thomas et al. 2023 (home was 7.8% of nonsexual exposure settings) and social-contact assortativity literature; no precise point estimate, declared as a structural assumption |
| `bi_same_sex_frac` | 0.10 | ② | Fraction of same-sex partners among bisexual individuals. WHO global surveillance records only 6.3% of orientation-known cases as bisexual men, and a Peru case series (n = 205) found 12.7% (26/205) bisexual, of whom 96.2% were male (Sihuincha Maldonado et al., *Int J Infect Dis* 2023); both indicate that bisexual men's male-partner fraction in this outbreak is modest, supporting a low same-sex fraction (declared as a sensitivity-analyzed assumption) |
| `stable_mf_anal_frac` | 0 | ② | Mixed-sex stable partners essentially do not escalate to anal sex. Supported by Thornhill et al. 2022 (*Lancet*) showing cisgender women present predominantly with vulvovaginal lesions (anorectal lesions concentrated in trans women), so stable heterosexual transmission is vaginal-dominated |
| `p_bi_m_to_msm` | 0.65 | ② | Bisexual-male routing into MSM-like networks. Direction and magnitude anchored to the CDC technical report 2 (93.3% of orientation-known cases gay or bisexual men, 5.4% heterosexual men, 1.3% women), WHO global surveillance (85.9% of cases gay/bi/other MSM; only 6.3% of orientation-known cases bisexual men), and the UK EID 2024 prospective cohort (n=1333 GBMSM), which documented that self-identified bisexual men had lower mpox vaccination uptake yet remained embedded in GBMSM networks. Bisexual men were highly embedded in MSM networks yet retained limited heterosexual bridging, so the routing fraction >0.5; 0.65 is the midpoint of bounds 0.55–0.75, declared as a calibrated midpoint |

### S1.1.6 Social / nonsexual close-contact layer

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `social_contacts_per_case` | 2.4 | ① | CDC report (Thomas et al., *MMWR* 2023;72(50):1351–2): mean of 2.4 identified nonsexual close contacts per case (278 cases, 662 nonsexual contacts; 2.2 in DC, 3.8 in Tennessee) |
| `social_contact_window_days` | 21.0 | ① | mpox maximum incubation/monitoring period 21 days |
| `social_contact_scale` | 1.0 | ① | Literature anchor 1.0 (CDC 2.4 contacts × SAR 1.5% is sufficient without further scaling). The original value of 10 was unsupported and reverted |
| `social_same_sex_assort` | 0.80 | ④ | Same-sex social-contact assortativity ≈ 0.80 (range 0.65–0.85). The direction (homophily in social networks) is supported by the assortative-mixing literature, but no direct point estimate exists for nonsexual same-sex contact assortativity in this population; declared as a structural assumption and varied in sensitivity analysis |
| `social_adjust_for_r_sus` | True | ④ | Age-adjustment by susceptible status (structural assumption) |
| `community_bridge_scale` | — (removed) | — | The late-stage "Los-Angeles social ramp" (community_bridge_scale / start_day / ramp_days) was removed: no literature basis, low interpretability, and no effect once fixed at 1.0. The social layer now scales contacts uniformly over time via `social_contact_scale` only |

### S1.1.7 Testing / tracing / surveillance

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `test_delay_mode` | scheduled_delay | ① | CDC technical report: staged delay from symptom → test → result |
| `test_pathway_prob_symp` | 0.85 | ② | Fraction of symptomatic cases entering the testing pathway (CDC technical report magnitude) |
| `test_seek_delay_mean_days` | 5.0 | ① | Symptom-onset → specimen/diagnosis delay. Anchored to Okoli et al., *Infection* 2024 global meta-analysis (66 studies): mean symptom-onset → diagnosis 5.8 d (95% CI 4.8–6.8; 4 studies, 982 cases); Patalon et al., *Microorganisms* 2023 (Israel): median symptom → confirmatory test 6.5 d; AAD/ILDS registry (101 cases, 13 countries): median symptom → diagnosis 5 d (IQR 4–9). Our mean 5.0 d sits at the lower end of this range |
| `test_seek_delay_sd_days` | 3.0 | ① | Consistent with the AAD/ILDS IQR 4–9 d (≈ ±3 d) |
| `test_seek_delay_max_days` | 21 | ① | Delay upper bound = incubation upper bound 21 days |
| `report_delay_time_varying` | False (default) | ③ | Optional time-varying reporting-delay mode. When enabled, the symptom-onset → specimen/report delay declines continuously over calendar time, reflecting the well-documented shortening of healthcare-seeking and reporting lags during the 2022 outbreak: Hinch et al., *Sci Rep* 2024 (UK) found the symptom-onset → healthcare-presentation delay fell from **22 d in early May** to **10 d by early June** and **8 d in August**; a NYC nowcasting study (*JMIR Public Health Surveill* 2025) reported an onset → report lag of **10 d (IQR 7–14) in July, declining to 8 d in September**; and the Tennessee surveillance evaluation (*Public Health Rep* 2024) measured **5 d symptom→specimen + 3 d specimen→report ≈ 9 d symptom→interview**. Enabled as a sensitivity scenario for curve-timing alignment (see S1.3) |
| `report_delay_t0` / `_t1` / `_tau` | 12 / 5 / 40 d | ③ | Parameters of the exponential time-decay curve `delay(day) = t1 + (t0 − t1)·exp(−day/tau)`. Anchored to the literature above: early reporting delay ≈ 10–22 d decaying to ≈ 5–8 d by August |
| `LAB_TURNAROUND_DAYS` | 2 | ② | Laboratory turnaround (CDC report magnitude) |
| `MONITORING_DAYS` | 21 | ① | mpox maximum incubation/monitoring period 21 days (WHO fact sheet; CDC; UKHSA) |
| `monitor_sexual_contact_mult` | 0.30 | ② | Risk reduction on sexual contacts during monitoring. WHO/CDC explicitly advise contacts to avoid sexual activity for the 21-day monitoring period (WHO fact sheet; Manitoba mpox protocol), supporting a substantial reduction; the precise multiplier is an empirical assumption varied in sensitivity analysis |
| `monitor_social_contact_mult` | 0.70 | ② | Risk reduction on social contacts during monitoring. WHO/CDC advise contacts to reduce prolonged close contact (rather than abstain), supporting a milder reduction than for sexual contacts; precise multiplier is an empirical assumption varied in sensitivity analysis |

### S1.1.8 Initial conditions (seeding and importation)

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| `INITIAL_INFECTIONS` | 0 | ② | Day-0 seed disabled. The previous 20-MSM-seed scheme caused a systematic ~18-day *early* peak (median peak day 74 vs observed 92) because it ignited full-speed endogenous amplification from day 0, whereas the real LAC outbreak started slowly (first case 5/19, sustained growth only from mid-June). Replaced by a sustained low-level importation stream (see `importations` below) |
| `importations` | 1/2/4/1 per day (day 15–134) | ① | Sustained exogenous importation with a "slow May → accelerate Jun–Jul → taper Aug" shape (daily_n = 1/2/4/1 over day 15–44/45–74/75–104/105–134), mirroring the real LAC importation rhythm. Literature-supported: Canada calibrated 1–3 exposed importations with estimated importation date March–April 2022 (Milwid et al., *J Med Virol* 2023); the US DC network model seeded 5 cases on June 2 from the highest-activity strata plus an extra-network importation term ε during the 11–25 June surge (Clay et al., *Sex Transm Infect* 2024); England's cost-effectiveness model assumed a baseline importation rate of 6 cases/month (sensitivity 1–10/month) (Zhang et al., *Lancet Reg Health Eur* 2025). Sustained importation fixes the early-peak artifact: 30-seed median peak = 96 (observed 92) |
| Seed population | (uniform MSM core, if seed > 0) | ① | Standard practice (NYC 5 cases, Canada 1–3 cases, England GBMSM model all seed MSM); with `INITIAL_INFECTIONS = 0` the outbreak is ignited endogenously by the sustained importation stream, which is routed to the MSM core via `importation_mask` |

### S1.1.9 Intervention policy time-shape (calibrated)

| Parameter | Value | Type | Justification and reference |
|---|---|---|---|
| phase1 `party_mult` | 0.95 | ④ | Curve fitting (party intensity during early growth) |
| phase2/3 `party_mult` | 0.70 | ④ | long_tail2 curve fitting (cRMSE 193 vs baseline 296) |
| per-phase `test_mult` / `trace_mult` | 0.70–1.50 | ④ | Intervention-intensity assumptions (behaviour change + testing strengthening) |
| `contact_mult` / `oneoff_mult` / `social_mult` | 0.65–1.20 | ④ | Behaviour-change assumptions (post-outbreak sexual-contact decline, social rebound) |

---

## S1.2 Model parameterization (English Methods text)

### Natural history

The incubation period (exposure to symptom onset) was drawn from a lognormal distribution with parameters μ = 2.108, σ = 0.254 (median 5.6 days, 95% CrI 4.3–7.8), truncated to 3–17 days, based on a pooled analysis of 12 US jurisdictions during May–August 2022 (Madewell et al., 2023). The delay from symptom onset to rash was set to 1–4 days, consistent with the CDC clinical description that rash typically develops within 1–4 days after symptom onset. The infectious period followed a gamma distribution (shape 9, scale 2.33; mean ≈21 days) truncated to 14–28 days, consistent with the CDC reference model (mean infectious period 27 days; Spicknall et al., 2022) and the Netherlands fading model (14–28 days; Xiridou et al., 2024). The generation-interval distribution used for R_t estimation (Cori method) had a mean of 9.0 days and SD of 5.0 days, based on the US serial-interval estimate of 8.5 days (95% CrI 7.3–9.9; Madewell et al., 2023).

### Per-act sexual transmission probabilities

We specified per-act transmission probabilities for four act types. The receptive-anal probability was set to 0.28, within the range βs = 0.25–0.40 estimated for the Netherlands fading model (Xiridou et al., 2024). This is epidemiologically supported by a multijurisdictional case–control study in which condomless receptive anal sex was associated with 5.4-fold higher odds of mpox (p = 0.031), the only intimate-contact behaviour that remained significant after adjustment (Chard et al., 2024). The insertive-anal probability was set to 0.06 (≈1/4 of receptive), the vaginal probability to 0.035, and the oral probability to 0.0175, following the observed hierarchy of exposure-specific risk (receptive anal > insertive anal > vaginal > oral) (Chard et al., 2024; Ward et al.; Oakley et al., 2023).

### Nonsexual close-contact transmission

Nonsexual close-contact transmission was parameterized to reproduce the CDC-reported average of 2.4 identified nonsexual close contacts per case and a secondary attack rate (SAR) of 1.5% among such contacts (CDC MMWR, 2023), yielding a per-contact effective probability of p_eff = 1 − (1 − 0.015)^(1/2.4) ≈ 0.0063. This is consistent with a systematic review reporting an overall nonsexual-contact SAR of 1.3% versus 11.4% for sexual contact (an 8.5-fold difference) (BMC, 2024).

### Sexual contact network and activity stratification

The MSM core population was stratified into six activity levels (five strata of 19% and a top stratum of 5%), following the CDC riskg design (CDC MMWR, 2022; Weiss et al., 2020). Partner-degree distributions within the sex-party layer followed a negative binomial with a heavy tail, motivated by the Milan cohort in which 23.0% of mpox cases reported >50 partners over 3 months (Passini et al., 2023), the CDC top-stratum estimate (≈100 partners/year), and the heavy-tail transmission argument of Endo et al. The top-stratum mean partner count and the one-off-layer partner-count parameter were jointly calibrated (see Calibration below) to reproduce the observed outbreak size and peak timing while keeping partner counts within the range reported for high-activity MSM (CDC top stratum 6.6 partners/3 weeks; Netherlands casual top stratum 0.7–1.0/day) (CDC MMWR, 2022; Xiridou et al., 2024). One-off sexual activity was oriented toward MSM and male-bisexual strata, consistent with the observation that the 2022 outbreak was overwhelmingly concentrated in MSM networks (≈98% of cases) (ECDC/WHO, 2022).

### Stable-partner layer

Stable-partner sexual frequency was converted to a daily activation probability via p = 1 − exp(−f/30) for monthly frequencies f of 3, 9, and 15 per month (yielding 0.10, 0.26, and 0.39 per day for general, one-MCP, and both-MCP stable edges, respectively).

### Initial conditions and importation

Rather than injecting a fixed number of index cases on day 0, we ignited the epidemic with a sustained, low-level exogenous importation stream whose intensity follows the observed LAC importation rhythm ("slow May → accelerate Jun–Jul → taper Aug"; daily_n = 1/2/4/1 over day 15–44/45–74/75–104/105–134). This choice was motivated by a systematic ~18-day early-peak artifact produced by the previous day-0 seeding scheme (median simulated peak day 74 vs observed day 92): because the real LAC outbreak started slowly (first case 19 May 2022, sustained growth only from mid-June), a one-shot day-0 seed caused endogenous amplification to begin far too early. The sustained-importation approach is directly supported by prior mpox models: Canada calibrated the date (March–April 2022) and number (1–3) of imported cases, all assigned to the exposed state with >10 annual partners and age <50 (Milwid et al., *J Med Virol* 2023); the US Washington DC network model seeded 5 cases on June 2 from the highest-activity strata and added an extra-network importation term ε during the 11–25 June surge (Clay et al., *Sex Transm Infect* 2024); and England's cost-effectiveness model assumed a baseline importation rate of 6 cases/month (sensitivity 1–10/month) (Zhang et al., *Lancet Reg Health Eur* 2025). With this scheme, the 30-seed median peak is 96 days (observed 92), eliminating the phase error.

### Calibration

The overall transmission volume was calibrated by jointly scaling the one-off partner-count parameter (oneoff_p = 0.2105; mean 3.75 partners/active-individual/day) and the top-stratum sex-party partner count (party_ht_top_mean = 7.5) by a factor f = 7.5, chosen to reproduce the LAC final outbreak size (≈2,280 cases) and peak timing (≈day 92) under the sustained-importation scheme (see Initial conditions above). Case-composition targets (male-homosexual ≈77.5%, male-bisexual ≈12.6%, heterosexual ≈9.9%) were matched through three structural parameters with direct behavioural anchors: (i) the male-bisexual one-off participation weight (oneoff_orientation_bias = 1.2, down from an initial 2.0 then fine-tuned to restore the bisexual bridge without overshooting the 12.6% bisexual target), reflecting that bisexual men were highly embedded in MSM networks yet did not sustain a 10-fold one-off activity excess over heterosexual men (only 5.6% of NYC cases were bisexual; Kyaw et al., 2022); (ii) the mixed-sex party partner-count target (party_mix_target_mean = 8.0), which sets the strength of male–female bridging within mixed-sex party events; and (iii) a direction-split mixed-sex anal fraction (mix_mf_anal_frac = 0.30 male-to-female, mix_fm_anal_frac = 0.05 female-to-male), reflecting that receptive anal transmission — the dominant high-risk route — is concentrated in male–male sex, whereas cisgender women acquire mpox predominantly through vaginal contact (vulvovaginal rather than anorectal lesions); the male-to-female fraction was raised from 0.15 to 0.30 (still below the male–male anal route) to restore heterosexual/female spillover after the importation scheme concentrated cases in the MSM core. All remaining parameters were either literature-fixed or structural assumptions with sensitivity ranges reported in Supplementary Table S1.

### Sexual orientation as identity vs. behavior

Consistent with the ECDC–WHO joint surveillance definitions, we treated surveillance-reported sexual orientation as a self-identified attribute that is *not necessarily representative of the sex of a case's sexual partners* in the 21 days before symptom onset.[ECDC/WHO 2022] This distinction is material to calibration. In the CDC national technical report, 93.3% of cases with known orientation and sex were gay or bisexual men, 5.4% were heterosexual men, and only 1.3% were women (a male:female ratio of ≈4.2:1). The LAC calibration target instead attributes 6.2% of cases to heterosexual men and 3.7% to women. The higher proportion of "heterosexual" men in LAC surveillance most plausibly reflects men who self-identify as heterosexual but report male sexual contact: the CDC found that 65 of 122 cases without reported male-to-male sexual contact were ultimately excluded because male-to-male contact could not be ruled out. Because our model classifies cases by *sexual behaviour* (the sex of the transmitting partner), the model's heterosexual-male proportion is expected to be systematically lower than the self-identified LAC figure. We therefore report the male-heterosexual and female proportions as *behaviour-based* quantities and treat the residual discrepancy as a documented calibration caveat rather than forcing the model to reproduce a self-identification artefact.

### Structural assumptions and sensitivity

Parameters lacking direct literature anchors — including the party close-touch fraction (0.20, sensitivity 0.10–0.25) and the monitoring risk-reduction multipliers (0.30/0.70) — were treated as structural assumptions and varied in sensitivity analyses (Supplementary Table S1). Epidemic ignition relied on a sustained low-level exogenous importation stream (rather than a single day-0 seed); the importation schedule's shape and magnitude are calibrated to the LAC importation rhythm and are documented in the "Initial conditions and importation" section above.

---

## S1.3 Parameters requiring prioritised documentation before submission

| Priority | Parameter | Value | Status | Recommendation |
|---|---|---|---|---|
| P1 | `oneoff_p` | 0.2105 | Calibrated; partner-count distribution parameter | Provide the derivation appendix linking the CDC riskg top-stratum partner count and the Netherlands casual layer |
| P1 | `party_ht_top_mean` | 7.5 | Calibrated | Provide the semantic-conversion note "6.6 partners/3 weeks → single-night E[k]" |
| P2 | `oneoff_orientation_bias` weights | MSM 3.0 / MB 1.2 / … | Direction supported, weights empirical | Declare as a normalisation scheme; run sensitivity analysis |
| P2 | `party_touch_frac` | 0.20 | No direct point estimate | Sensitivity 0.10–0.25 |
| P2 | `monitor_*_mult` | 0.30 / 0.70 | Empirical assumption | Sensitivity analysis |
| P2 | `party_core_intensity` | 0.55 | No direct source | Declare as calibrated midpoint; sensitivity over bounds |
| P2 | `p_bi_m_to_msm` | 0.65 | No direct source | Declare as calibrated midpoint; sensitivity over bounds |
| P3 | `bi_same_sex_frac` | 0.10 | Indirect (WHO 6.3% bisexual men; Peru 12.7%) | Cite behavioural literature on bisexual populations; sensitivity analysis |

### Known calibration caveat — sexual orientation as identity vs. behaviour

The LAC "male-heterosexual" proportion (6.2%) is a **self-identified** attribute that includes men who identify as heterosexual but report male sexual contact (the CDC excluded 65 of 122 "no-MMSC" cases because male-to-male contact could not be ruled out). Our model classifies cases by **sexual behaviour**, so its male-heterosexual proportion is systematically lower. This is documented as a calibration caveat in the Methods (see "Sexual orientation as identity vs. behavior") rather than a fitted target; the behaviour-based estimate is closer to the CDC national figure (heterosexual men 5.4%, women 1.3%).

---

## S1.4 References

1. Madewell ZJ, Charniga K, Masters NB, et al. Incubation period and serial interval of mpox virus infections — 12 US jurisdictions, May–August 2022. *Emerg Infect Dis*. 2023;29(4):818–821. doi:10.3201/eid2904.221569.

2. Spicknall IH, Pollock ED, Clay PA, et al. Modeling the impact of sexual networks in the transmission of monkeypox virus among gay, bisexual, and other men who have sex with men — United States, 2022. *MMWR Morb Mortal Wkly Rep*. 2022;71(35):1131–1135. doi:10.15585/mmwr.mm7135e2.

3. Xiridou M, Miura F, Adam P, et al. The fading of the mpox outbreak among men who have sex with men: a mathematical modelling study. *J Infect Dis*. 2024. doi:10.1093/infdis/jiae403.

4. Chard AN, Dalton AF, Diallo AO, et al. Risk of clade II mpox associated with intimate and nonintimate close contact among men who have sex with men and transgender adults — United States, August 2022–July 2023. *MMWR Morb Mortal Wkly Rep*. 2024;73(40):896–901. doi:10.15585/mmwr.mm7340a2.

5. Snyder RE, Saadeh K, Tang EC, et al. Sexual exposures associated with mpox infection: California, November 2022 to June 2023. *J Infect Dis*. 2024;229(Suppl 2):S188–S196. doi:10.1093/infdis/jiad607.

6. Ward T, Christie R, Paton RS, et al. Transmission dynamics of monkeypox in the United Kingdom: contact tracing study. *BMJ*. 2022;379:e073153. doi:10.1136/bmj-2022-073153.

7. Oakley LP, Hufstetler K, O'Shea J, et al. Mpox cases among cisgender women and pregnant persons — United States, May 11–November 7, 2022. *MMWR Morb Mortal Wkly Rep*. 2023;72(1):9–14. doi:10.15585/mmwr.mm7201a2.

8. Weiss KM, Goodreau SM, Morris M, Prasad P, Ramaraju R, Sanchez T, Jenness SM. Egocentric sexual networks of men who have sex with men in the United States: results from the ARTnet study. *Epidemics*. 2020;30:100386. doi:10.1016/j.epidem.2020.100386.

9. UK Health Security Agency. Investigation into monkeypox outbreak in England: technical briefing 8. Updated 23 September 2022. London: UKHSA; 2022. Available from: https://www.gov.uk/government/publications/monkeypox-outbreak-technical-briefings/investigation-into-monkeypox-outbreak-in-england-technical-briefing-8

10. Xiu F, Flores Anato JL, Cox J, Grace D, Hart TA, Skakoon-Sparling S, et al. Characteristics of the sexual networks of men who have sex with men in Montréal, Toronto, and Vancouver: insights from Canada's 2022 mpox outbreak. *J Infect Dis*. 2024;229(Suppl 2):S293–S304. doi:10.1093/infdis/jiae033.

11. El-Saed A, Al-Fayez S, AlAmeer K, Al Zunitan M, Othman F, Farahat F, et al. Secondary attack rates after mpox exposure in multiple settings during the 2022–2024 pandemic: a systematic review. *BMC Infect Dis*. 2026;26(1):1149. doi:10.1186/s12879-026-13410-7.

12. Han Y, Li X, Wang X, Zhong Z. The epidemiological and clinical differences between patients with mpox and patients without mpox among suspected cases during the recent global mpox outbreaks: a systematic review and meta-analysis. *Front Cell Infect Microbiol*. 2026;15:1740617. doi:10.3389/fcimb.2025.1740617.

13. National Survey of Family Growth (NSFG). Heterosexual anal and oral sex among U.S. men and women, 2015–2019. Hyattsville, MD: National Center for Health Statistics, Centers for Disease Control and Prevention.

14. Centers for Disease Control and Prevention. Epidemiologic and clinical features of monkeypox cases — United States, May 17–July 22, 2022. *MMWR Morb Mortal Wkly Rep*. 2022;71(32):1018–1022.

15. Centers for Disease Control and Prevention. Monkeypox transmission among close contacts — United States, 2022. *MMWR Morb Mortal Wkly Rep*. 2023.

16. European Centre for Disease Prevention and Control / World Health Organization. Joint ECDC–WHO regional office for Europe monkeypox surveillance bulletin. 2022.

17. Endo A, Murayama H, Abbott S, et al. Heavy-tailed sexual contact networks and monkeypox epidemiology in the global outbreak, 2022. *Science*. 2022;378(6615):90–94. doi:10.1126/science.add4507.

18. Candela C, Raccagni AR, Bruzzesi E, Bertoni C, Rizzo A, Gagliardi G, et al. Human monkeypox experience in a tertiary level hospital in Milan, Italy, between May and October 2022: epidemiological features and clinical characteristics. *Viruses*. 2023;15(3):667. doi:10.3390/v15030667.

19. Passini F, Raccagni AR, Diotallevi S, Lolatto R, Bruzzesi E, Candela C, et al. Mpox outbreak 2022: a comparative analysis of the characteristics of individuals receiving MVA-BN vaccination and people diagnosed with mpox infection in Milan, Italy. *Pathogens*. 2023;12(9):1079. doi:10.3390/pathogens12091079.

20. Thornhill JP, Barkati S, Walmsley S, et al. Monkeypox virus infection in humans across 16 countries — April–June 2022. *N Engl J Med*. 2022;387(8):679–691. doi:10.1056/NEJMoa2207323.

21. Thornhill JP, Palich R, Ghosn J, et al. Human monkeypox virus infection in women and non-binary individuals during the 2022 outbreaks: a global case series. *Lancet*. 2022;400(10367):1953–1965. doi:10.1016/S0140-6736(22)02187-4.

22. Brainard J, Smith LE, Potts HWW, Rubin GJ. The relationship between age and sex partner counts during the mpox outbreak in the UK, 2022. *PLoS One*. 2023;18(9):e0291001. doi:10.1371/journal.pone.0291001.

23. Vallejo-Plaza A, Rodríguez-Cabrera F, Hernando Sebastián V, et al. Mpox (formerly monkeypox) in women: epidemiological features and clinical characteristics of mpox cases in Spain, April to November 2022. *Euro Surveill*. 2022;27(47):2200867. doi:10.2807/1560-7917.ES.2022.27.47.2200867.

24. Iñigo Martínez J, Gil Montalbán E, Jiménez Bueno S, et al. Monkeypox outbreak predominantly affecting men who have sex with men, Madrid, Spain, 26 April to 16 June 2022. *Euro Surveill*. 2022;27(27):2200471. doi:10.2807/1560-7917.ES.2022.27.27.2200471.

25. Centers for Disease Control and Prevention. Multi-national monkeypox outbreak: technical report 2. 2022. Available from: https://www.cdc.gov/poxvirus/monkeypox/response/2022/index.html

26. Centers for Disease Control and Prevention. Possible exposures among mpox patients without reported male-to-male sexual contact — United States, 2022. *MMWR Morb Mortal Wkly Rep*. 2023.

27. European Centre for Disease Prevention and Control / World Health Organization Regional Office for Europe. Joint ECDC–WHO monkeypox surveillance bulletin. 2022. (Note: "Sexual orientation is not necessarily representative of the gender of the person the case had sex with in the past 21 days, nor does it imply sexual contact and sexual transmission.")

28. Thomas ES, Madewell ZJ, Still WL, et al. Transmission of mpox to nonsexual close contacts — two U.S. jurisdictions, May 1–July 31, 2022. *MMWR Morb Mortal Wkly Rep*. 2023;72(50):1351–1352. doi:10.15585/mmwr.mm7250a3.

29. Okoli GN, Van Caeseele P, Askin N, Abou-Setta AM. A global systematic evidence review with meta-analysis of the epidemiological characteristics of the 2022 mpox outbreaks. *Infection*. 2024;52(3):901–921. doi:10.1007/s15010-023-02133-5.

30. Patalon T, Perez G, Saciuk Y, Refaeli Z, Gazit S. Mpox patient journey in Israel. *Microorganisms*. 2023;11(4):1042. doi:10.3390/microorganisms11041042.

31. Prasad S, Galvan Casas C, Strahan AG, et al. A dermatologic assessment of 101 mpox (monkeypox) cases from 13 countries during the 2022 outbreak: skin lesion morphology, clinical course, and scarring. *J Am Acad Dermatol*. 2023;88(5):1066–1073. doi:10.1016/j.jaad.2022.12.035.

32. Centers for Disease Control and Prevention. Characteristics of mpox vaccine recipients among a sample of men who have sex with men with presumed exposure to mpox (AMIS mpox survey). CDC Stacks; 2022. Available from: https://stacks.cdc.gov/view/cdc/130847

33. UK Health Security Agency. Investigation into monkeypox outbreak in England: technical briefing 4. Updated 22 July 2022. London: UKHSA; 2022. Available from: https://www.gov.uk/government/publications/monkeypox-outbreak-technical-briefings/investigation-into-monkeypox-outbreak-in-england-technical-briefing-4

34. World Health Organization. Mpox (monkeypox): fact sheet. Geneva: WHO; 2024. Available from: https://www.who.int/news-room/fact-sheets/detail/mpox

35. Ogaz D, Enayat Q, Brown JRG, et al. Mpox diagnosis, behavioral risk modification, and vaccination uptake among gay, bisexual, and other men who have sex with men, United Kingdom, 2022. *Emerg Infect Dis*. 2024;30(5):916–925. doi:10.3201/eid3005.230676.

36. World Health Organization. Multi-country outbreak of mpox: external situation report 12. Geneva: WHO; 14 December 2022. Available from: https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON398

37. Hinch R, Panovska-Griffiths J, Ward T, Charlett A, Watkins N, Fraser C. Quantification of the time-varying epidemic growth rate and of the delays between symptom onset and presenting to healthcare for the mpox epidemic in the UK in 2022. *Sci Rep*. 2024;14:19755. doi:10.1038/s41598-024-68154-8.

38. Greene SK, et al. Nowcasting to monitor real-time mpox trends during the 2022 outbreak in New York City: evaluation using reportable disease data stratified by race or ethnicity. *JMIR Public Health Surveill*. 2025;11:e56495.

39. Thomas CM, Shaffner J, Johnson R, et al. Lessons learned from implementation of mpox surveillance during an outbreak response in Tennessee, 2022. *Public Health Rep*. 2024;139(5):566–572. doi:10.1177/00333549231223710.

40. Philpott D, Hughes CM, Alroy KA, et al. Epidemiologic and clinical characteristics of monkeypox cases — United States, May 17–July 22, 2022. *MMWR Morb Mortal Wkly Rep*. 2022;71(32):1018–1022. doi:10.15585/mmwr.mm7132e3.

41. Milwid RM, Li M, Fazil A, et al. Exploring the dynamics of the 2022 mpox outbreak in Canada. *J Med Virol*. 2023;95(12):e29256. doi:10.1002/jmv.29256.

42. Clay PA, Asher JM, Carnes N, et al. Modelling the impact of vaccination and sexual behaviour adaptations on mpox cases in the USA during the 2022 outbreak. *Sex Transm Infect*. 2024;100(2):70–76. doi:10.1136/sextrans-2023-055922.

43. Zhang XS, Niyomsri S, Mandal S, et al. Cost-effectiveness of vaccination strategies to control future mpox outbreaks in England: a modelling study. *Lancet Reg Health Eur*. 2025;55:101364. doi:10.1016/j.lanepe.2025.101364.

44. Sihuincha Maldonado M, Lucchetti AJ, Paredes Pacheco RA, et al. Epidemiologic characteristics and clinical features of patients with monkeypox virus infection from a hospital in Peru between July and September 2022. *Int J Infect Dis*. 2023;129:175–180. doi:10.1016/j.ijid.2023.01.045.

45. Kyaw NT, Kipperman N, Alroy KA, et al. Clinical and epidemiologic characteristics of mpox cases from the initial phase of the outbreak — New York City, May 19–July 15, 2022. *MMWR Morb Mortal Wkly Rep*. 2022;71(52):1636–1637. doi:10.15585/mmwr.mm7152a3.
