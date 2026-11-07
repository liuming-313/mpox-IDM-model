# AGENTS.md

## Project overview

This repository contains a stochastic agent-based infectious disease transmission model for mpox/STI-related public health research. The model simulates transmission across sexual-orientation and sex/gender groups using layered contact networks, including stable partnerships, one-off contacts, sex-party/MIX contacts, and nonsexual/social close-contact bridging.

The scientific goal is not to maximize predictive performance at any cost. The goal is to produce an interpretable, reproducible, and biologically plausible calibrated baseline that can reproduce observed cumulative case composition by sexual orientation and sex while preserving clear model mechanisms.

## Primary files

- `main.py`: scenario setup, calibrated parameter values, plotting, output saving, and simulation execution.
- `population.py`: population generation, contact-layer construction, transmission logic, disease progression, tracing/policy effects, and summaries.
- Calibration/validation helper scripts may be created, but they must not replace the canonical `main.py` baseline unless explicitly requested.

## Scientific calibration targets

Use these observed cumulative targets unless the user explicitly gives new values:

```text
heterosexual = 0.0931
bisexual     = 0.1272
homosexual   = 0.7796
female_known = 0.0336
```

Primary objective:

```text
MAE_pp = mean(abs(simulated_ratio - target_ratio)) * 100
```

The calibrated baseline is acceptable if the multi-seed mean MAE is around or below 1.0 percentage point, and still acceptable if it is <= 1.2 percentage points without severe imbalance in any target.

Useful soft constraints:

```text
bisexual     ≈ 11.5% to 13.8%
female_known ≈ 2.8% to 4.5%
heterosexual preferably >= 8.0%
homosexual   preferably <= 80.5%
```

Do not optimize proportions while creating unrealistic total cases. Total cases should remain in the same approximate scale as the validated baseline unless the user explicitly changes the scenario.

## Simplified calibrated parameters

Only these six parameters should be treated as tunable in the simplified baseline:

```python
p_bi_m_to_msm       = 0.66   # allowed range: 0.55 to 0.75
party_core_intensity = 0.55 # allowed range: 0.45 to 0.70
party_hetero_leak  = 0.026  # allowed range: 0.015 to 0.040
social_contact_scale = 4.20 # allowed range: 2.5 to 6.0
social_male_bias   = 0.92   # allowed range: 0.75 to 0.95
late_social_ramp   = 5.00   # allowed range: 1.0 to 6.0
```

Interpretation:

- `p_bi_m_to_msm`: routing of bisexual males into MSM-like networks versus heterosexual bridging.
- `party_core_intensity`: intensity of core sex-party/MIX transmission.
- `party_hetero_leak`: low-probability heterosexual leakage from party/MIX contexts.
- `social_contact_scale`: effective number of nonsexual close contacts.
- `social_male_bias`: male-skewed social close-contact targeting.
- `late_social_ramp`: late-stage social/community amplification, for example broader community, unstable-housing, shelter-linked, or street-linked contacts.

## Fixed structural assumptions

Do not tune these in baseline calibration unless the user explicitly asks:

```python
p_bi_f_to_wsw = 0.75
mix_mf_anal_frac = 0.13
party_mix_homo_spill_frac = 0.03
social_p_eff = 0.0063
unhoused_hetero_male_boost = 1.00
```

Do not tune `social_p_eff` together with `social_contact_scale`. `social_p_eff` is externally anchored from nonsexual close-contact secondary attack calculations. If uncertainty is needed, treat it as a sensitivity-analysis parameter, not a baseline calibration knob.

## Parameters and logic that must not be changed during calibration

Unless explicitly requested by the user, do not modify:

- Disease natural history.
- Age-linked susceptibility.
- Symptomatic probability by age.
- Mortality probability by age.
- `social_p_eff`.
- Importation schedule.
- Policy timing.
- Vaccination/susceptibility assumptions.
- Plotting/output behavior in the final `main.py`.
- The simplified six-parameter structure.

Do not add new calibration parameters to improve fit. If a new mechanism seems necessary, report it as a recommendation first and wait for approval.

## Calibration strategy

Use staged stochastic calibration. Do not perform large blind grid searches.

### Stage 0: baseline check

Run seeds 100000 to 100004 using current parameters. Save:

```text
outputs/simplified_calibration/baseline_5seed.csv
```

Record:

```text
seed
total_cases
heterosexual
bisexual
homosexual
female_known
MAE_pp
```

### Stage 1: single-parameter local screening

Change one parameter at a time using seed 100000 only.

Suggested values:

```text
p_bi_m_to_msm:        0.62, 0.66, 0.70
party_core_intensity: 0.52, 0.55, 0.58
party_hetero_leak:    0.022, 0.026, 0.030
social_contact_scale: 3.8, 4.2, 4.6, 5.0
social_male_bias:     0.90, 0.92, 0.94
late_social_ramp:     4.0, 5.0, 5.5, 6.0
```

Save:

```text
outputs/simplified_calibration/screening_single_param.csv
```

### Stage 2: small combined search

Based on Stage 1, test only 8 to 15 combined candidates using seed 100000. Do not test hundreds of combinations unless the user explicitly asks.

Save:

```text
outputs/simplified_calibration/screening_combined.csv
```

Select the best 3 candidates by MAE subject to the soft constraints.

### Stage 3: five-seed validation

Run the current baseline and best 3 candidates using seeds 100000 to 100004.

Save:

```text
outputs/simplified_calibration/validation_5seed.csv
```

Select a new candidate only if it clearly improves mean MAE and does not worsen female-known or bisexual stability.

### Stage 4: twenty-seed validation

Only if a candidate passes Stage 3, run seeds 100000 to 100019.

Adopt a candidate only if:

- mean MAE improves by at least 0.15 percentage points, or
- heterosexual ratio improves meaningfully without worsening other targets.

Save:

```text
outputs/simplified_calibration/validation_20seed.csv
```

### Stage 5: optional final 100-seed validation

Only run 100 seeds after a candidate passes Stage 4, or if the user explicitly requests it. Do not run 100 seeds for every candidate.

Save:

```text
outputs/simplified_calibration/final_100seed_validation.csv
```

## Calibration interpretation rules

Use these rules before changing parameters:

- If female-known is too high but heterosexual is too low, do not increase `party_hetero_leak` first. Prefer increasing `social_male_bias` or `late_social_ramp`.
- If bisexual is already close to target, avoid changing `p_bi_m_to_msm`.
- If total cases become too high, reduce `party_core_intensity` before changing social parameters.
- If homosexual is too high and heterosexual is too low, try slightly lowering `party_core_intensity` while increasing `social_contact_scale` or `late_social_ramp`.
- If heterosexual is too low and female-known is acceptable, increase `social_contact_scale` or `late_social_ramp` cautiously.
- If female-known becomes too high, increase `social_male_bias` or reduce `party_hetero_leak`.

## Coding rules

- Make minimal, high-confidence changes.
- Preserve existing function signatures unless the user asks for refactoring.
- Prefer backward-compatible parameter mapping over broad rewrites.
- Do not introduce new dependencies unless necessary.
- Do not delete previous outputs or logs unless they are temporary files created in the current run.
- Use unique run IDs including seed and timestamp to avoid overwriting logs.
- If a run crashes, record the seed and error message, then continue with remaining seeds.
- Do not silently ignore failed seeds.
- Keep comments in code clear and scientific, explaining why each simplified parameter exists.

## Validation and syntax checks

Before reporting completion, run:

```bash
python -m py_compile main.py population.py
```

For final calibration outputs, create or update:

```text
outputs/simplified_calibration/final_simplified_params.json
outputs/simplified_calibration/final_validation_summary.csv
outputs/simplified_calibration/final_validation_readme.md
outputs/simplified_calibration/history.csv
```

## Final reporting format

When reporting back, include:

- Whether a new parameter set was adopted or the current simplified baseline was retained.
- Final values of the six simplified parameters.
- Number of seeds run.
- Failed runs, if any.
- Mean and SD for:
  - total_cases
  - heterosexual
  - bisexual
  - homosexual
  - female_known
  - MAE_pp
- A short explanation of why the final parameter set was selected.

## Manuscript-oriented documentation

When writing comments or summaries, use language suitable for a scientific manuscript:

- “simulation-based calibration”
- “multi-seed stochastic validation”
- “observed cumulative case composition”
- “behavioral/network parameters”
- “nonsexual close-contact bridging”
- “heterosexual spillover”
- “sex-party/MIX core transmission”

Avoid overclaiming that any calibrated parameter is a directly observed true value. Describe calibrated values as plausible baseline assumptions selected to reproduce observed case composition.
