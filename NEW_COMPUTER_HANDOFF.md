# Mpox v3 model handoff for a new computer

## Canonical project location

The complete model project is stored in:

```text
D:\onedrive_personal\OneDrive\Paper\IDM
```

This directory is inside OneDrive. On a new computer, sign in to the same
OneDrive account and wait until the full `Paper\IDM` directory is downloaded.
The drive letter may differ on the new computer.

The Codex workspace directory:

```text
C:\Users\Administrator\Documents\Codex\2026-05-16\you-are-working-on-my-mpox
```

contains only `AGENTS.md`. It is not the canonical model directory.

## Frozen final v3 calibrated baseline

```python
p_bi_m_to_msm = 0.70
party_core_intensity = 0.70
social_contact_scale = 6.0
social_same_sex_assort = 0.75
late_social_ramp = 6.0
mix_mf_anal_frac = 0.13
stable_mf_anal_frac = 0.00
party_mix_homo_spill_frac = 0.01
```

Do not restore:

```text
party_hetero_leak
social_male_bias
party_mix_female_mult
```

Do not recalibrate unless explicitly required. Do not modify `population.py`
without a clear runtime bug.

## Final 100-seed validation

Location:

```text
outputs\v3_final_100seed_validation
```

Results:

```text
successful runs: 100
failed runs: 0
total_cases: mean 1209.43, SD 199.97
heterosexual: mean 0.086461, SD 0.009988
bisexual: mean 0.123275, SD 0.013384
homosexual: mean 0.790264, SD 0.017600
female_known: mean 0.042012, SD 0.006878
homo_male: mean 0.789682, SD 0.017529
bi_male: mean 0.118271, SD 0.013145
MAE_pp: mean 1.148766, SD 0.560577
MAE_pp <= 1.0: 47%
MAE_pp <= 1.2: 64%
MAE_pp <= 1.5: 83%
acceptance criteria: passed
```

Key files:

```text
outputs\v3_final_100seed_validation\final_v3_params.json
outputs\v3_final_100seed_validation\final_100seed_results.csv
outputs\v3_final_100seed_validation\final_100seed_summary.csv
outputs\v3_final_100seed_validation\final_100seed_README.md
```

## Selected Table S8 sensitivity analyses

Location:

```text
outputs\v3_table_s8_selected_sensitivity
```

All 12 scenarios across seeds 100000-100019 completed:

```text
successful runs: 240
failed runs: 0
```

Key files:

```text
outputs\v3_table_s8_selected_sensitivity\selected_table_s8_manifest.csv
outputs\v3_table_s8_selected_sensitivity\selected_table_s8_sensitivity_results.csv
outputs\v3_table_s8_selected_sensitivity\selected_table_s8_sensitivity_summary.csv
outputs\v3_table_s8_selected_sensitivity\selected_table_s8_sensitivity_README.md
outputs\v3_table_s8_selected_sensitivity\frozen_v3_baseline_params.json
```

## Important scripts

```text
main.py
population.py
run_one.py
objective.py
calibration_config.py
run_v3_table_s7_calibration.py
run_v3_table_s7_secondary_calibration.py
run_v3_table_s7_tertiary_calibration.py
run_v3_final_100seed_validation.py
run_v3_table_s8_selected_sensitivity.py
```

## Local Codex conversation backup

On the old computer, the local Codex data root is:

```text
C:\Users\Administrator\.codex
```

Relevant transcript files:

```text
C:\Users\Administrator\.codex\sessions\2026\05\16\rollout-2026-05-16T16-43-50-019e2ff4-c526-7c41-a72f-4778a5fd4f3a.jsonl
C:\Users\Administrator\.codex\sessions\2026\06\21\rollout-2026-06-21T16-39-24-019ee955-a4d8-7323-9489-33f093248dc3.jsonl
```

The first file (`2026-05-16`, thread id
`019e2ff4-c526-7c41-a72f-4778a5fd4f3a`) is the current, complete conversation
and contains the final v3 calibration, validation, sensitivity work, and the
new-computer handoff discussion. The second file is a shorter recovered/thread
continuation session. Keep both files as a readable archive.

Do not copy `auth.json` to another computer. Sign in normally on the new
computer.

## New-computer restoration checklist

1. Sign in to the same OneDrive account.
2. Download the complete `Paper\IDM` directory and mark it available offline.
3. Install Python 3.11 and PyCharm/Codex as needed.
4. Recreate or select a Python virtual environment and install the project
   dependencies.
5. Open the restored `IDM` directory as the project, not the small Codex
   workspace directory.
6. Confirm `main.py` contains the frozen parameter values above.
7. Confirm `MAKE_PLOTS = True` and `base_run_seed = 100000`.
8. Run:

```text
python -m py_compile main.py population.py
```

9. Give Codex this handoff file and `AGENTS.md` as context when starting a new
   conversation.
10. Do not rerun calibration or sensitivity simulations merely to restore the
    project; the completed CSV outputs are already saved.
