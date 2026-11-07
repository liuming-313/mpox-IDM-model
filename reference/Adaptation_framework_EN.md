# Adaptation of the simulation framework for mpox — Methods section (English draft)

> 这是审稿人第 1 条质疑（"拿 Covasim 硬套 mpox"）的直接回应，属于 Methods 开头必写段落。
> 写法：明确区分「引擎复用」vs「传播机制重写」，逐层说明哪些是保留的、哪些是为 mpox 从零重建的。

---

## Adaptation of the simulation framework for mpox

We built the mpox transmission model on Covasim's individual-based simulation engine, but the epidemiological core was re-derived from scratch for mpox and shares no transmission assumptions with the original COVID-19 framework. We retained only the general-purpose machinery — the discrete-time agent scheduler, the contact-network abstraction, and the event system for policy interventions — and replaced every disease-specific component.

**What was retained (engine only).** The simulation loop schedules discrete daily updates over a population of 10 million individuals, tracks per-individual health states, and exposes a layered contact network (household, stable sexual, one-off sexual, sex-party, and nonsexual social layers). Policy interventions are applied through a shared event interface. None of these components encode any pathogen-specific assumption; they are generic tools applicable to any person-to-person infection.

**What was re-derived for mpox.** The following components were implemented specifically for mpox and have no counterpart in the COVID-19 framework:

1. **Natural history.** Incubation, symptom-to-rash delay, infectious period, and the generation-interval distribution were parameterized from mpox-specific estimates (Madewell et al. 2023; Spicknall et al. 2022; Xiridou et al. 2024), not from COVID-19 values. The infectious-period dynamics differ qualitatively from COVID-19 (mean ≈21 days vs. a substantially shorter SARS-CoV-2 infectious period).

2. **Sexual contact network.** A six-level activity stratification of the MSM core population (five 19% strata plus a 5% top stratum) was built following the CDC riskg design (CDC MMWR 2022; Weiss et al. 2020). A dedicated sex-party layer captures group sexual events with heavy-tailed partner-degree distributions, motivated by the Milan case series and the heavy-tail argument of Endo et al. (2022). The distinction between persistent stable-partner edges and daily-rebuilt one-off edges is explicit and is described below.

3. **Per-act transmission probabilities.** Transmission was modeled at the level of specific sexual acts — receptive anal, insertive anal, vaginal, and oral — each with its own per-act probability (Xiridou et al. 2024; Chard et al. 2024). This act-type granularity is essential for capturing the observed exposure-specific risk hierarchy (receptive anal > insertive anal > vaginal > oral) that drives the over-representation of MSM in the 2022 outbreak.

4. **Nonsexual close-contact transmission.** A separate social layer was parameterized from mpox-specific nonsexual close-contact data (CDC-reported 2.4 contacts per case and 1.5% secondary attack rate), yielding a per-contact probability of ≈0.0063.

5. **Case detection and surveillance.** Symptom-based testing, specimen collection, diagnosis, and reporting delays were modeled to match LAC surveillance definitions of "episode date" (the earliest of symptom onset, specimen collection, or report date), rather than any COVID-19 testing cascade.

This separation — a generic engine with an mpox-specific epidemiological core — is the same modular design used by other agent-based mpox models built on general-purpose simulators, and it ensures that no COVID-19-specific transmission assumption carries over into the present model.

---

## Note on "daily partner turnover" (审稿人第 2 条质疑的澄清)

The model does **not** assume that all partners are renewed daily. Only the **one-off sexual layer** rebuilds its edges each day (representing casual, non-recurring sexual contacts). The **stable-partner layer** maintains persistent edges with durations governed by the partnership model, and the **household** and **social** layers are similarly persistent. This layered design reflects the empirical distinction between recurring partnerships and casual encounters observed in MSM sexual-network surveys (Xiu et al. 2024).
