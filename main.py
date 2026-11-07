"""
Run multiple mpox-transmission scenarios.

Each scenario = pick a seed pool, vaccinate / test–trace according to
global toggles, run the model, then show summary + plots.

Files required in the same folder
---------------------------------
population.py      – core model
mpox_vax.py        – vaccination patch (optional)
mpox_testtrace.py  – testing / contact-tracing patch (optional)
"""

import time
import os
import re
from pathlib import Path

import numpy as np

# ============================================================
# Plot backend: save figures only, no PyCharm pop-up window.
# This MUST be set before importing population.py, because
# population.py imports matplotlib.pyplot as plt.
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from population import Population           # core model



# ---------------------------------------------------------------------
# Toggle public-health interventions here


# ---------------------------------------------------------------------
# Model-wide parameters
reps=1000
pop_seed=42
base_run_seed = 100001
POPULATION_SIZE = int(10000000)
SIM_DAYS        =  244
# Baseline seeding strategy (2026-11-05 revision): the previous "20 MSM index
# cases injected on day 0" scheme caused a systematic ~18-day *early* peak
# (median peak day 74 vs observed 92) because it ignited full-speed endogenous
# amplification from day 0, whereas the real LAC outbreak started slowly
# (first case 5/19, sustained growth only from mid-June). The fix replaces the
# day-0 seed with a SUSTAINED LOW-LEVEL EXOGENOUS IMPORTATION stream (see
# `importations` in _run_one_scenario) whose "slow May -> accelerate Jun-Jul ->
# taper Aug" shape mirrors the real LAC importation rhythm. This is
# literature-supported: England used real importation data (~6/month; Lancet Reg
# Health Eur 2025), Canada calibrated 1-3 "exposed" introductions (J Infect Dis
# 2024), and US DC saw ~11 high-activity importations/day during the surge
# (PMC11323315). With this scheme the 30-seed median peak = 95 (obs 92), fixing
# the phase error; composition degrades mildly (homo ~0.84 vs 0.775 target),
# accepted as a documented limitation (see Supplementary S1.1.7).
INITIAL_INFECTIONS = 0                   # day-0 seed count; replaced by sustained importation
# Plot controls. With Agg + auto-save, this is safe for long calibration runs.
MAKE_PLOTS = True
SAVE_PLOTS = True
PLOT_DPI = 300

# ---------------------------------------------------------------------
# Final validation targets from the LAC 2022-12-30 stratified case table
# ---------------------------------------------------------------------
# Denominator for orientation targets:
#   known_orientation_cases = Male-Homo + Male-Bisexual
#                           + Male-Heterosexual + Female-Heterosexual
#                           + Female-Bisexual + Female-Homosexual
# In the 2022-12-30 table, Female-Bisexual and Female-Homosexual were blank
# and are therefore treated as 0 in the known-orientation denominator:
#   known_orientation_cases = 1560 + 253 + 125 + 74 = 2012
#   Homo        = 1560 / 2012 = 0.7753
#   Bi          =  253 / 2012 = 0.1257
#   Hetero      = (125 + 74) / 2012 = 0.0989
#   Female-known=   74 / 2012 = 0.0368
#
# The "other/unknown" category (268/2280) and the dashboard "non-male gender
# identity" count (101/2280) are retained as external validation context, not
# as direct calibration knobs.  Mechanistic calibration should act through
# exposure routing and per-act transmission-risk parameters instead.
CALIBRATION_TARGETS = {
    "target_homo": 1560 / 2012,
    "target_bi": 253 / 2012,
    "target_hetero": (125 + 74) / 2012,
    "target_female_known": 74 / 2012,

    # These are kept as explicit keys because they are useful in your notes.
    # In the current observed table, female homo/bi counts are zero, so these
    # match the overall orientation targets.
    "target_homo_male": 1560 / 2012,
    "target_bi_male": 253 / 2012,
}

params = {
    # ------------------------------------------------------------------
    # Generation-interval (GI) distribution for R_t estimation (Cori method).
    # Source: mean serial interval 8.5 days (95% CrI 7.3-9.9), SD 5.0
    # (95% CrI 4.0-6.4) across 12 US jurisdictions, May-Aug 2022
    # (Madewell ZJ et al. Emerg Infect Dis 2023;29(4):818-821). The UK
    # estimate (8.82 d, 5.22-25.81) and Italy generation-time (12.5 d) are
    # consistent in magnitude. We use the US mean (8.5 d) rounded to 9.0 as
    # the GI mean, with SD 5.0 aligned to the reported US SD rather than the
    # earlier narrower assumption.
    "gi_mean": 9.0,   # 来源：Madewell 2023 串行间隔均值 8.5d 取整 9.0
    "gi_sd": 5.0,   # 来源：Madewell 2023 SD 5.0 (4.0-6.4)；was 3.0
    "gi_max": 30,   # 来源：GI 截断上限（结构假设）

    # ------------------------------------------------------------------
    # Natural-history (disease timing) parameters. These are LITERATURE-FIXED
    # (not fitted) and shared by every published mpox model. They are consumed
    # by population.py (which also carries the same defaults internally).
    #
    # Incubation period (exposure -> rash/symptom onset): lognormal, median
    # ~5.6 d (95% CrI 4.3-7.8), from 12 US jurisdictions, May-Aug 2022
    # (Madewell ZJ et al. Emerg Infect Dis 2023;29(4):818-821). Lognormal
    # mu=2.108, sigma=0.254 is the Charniga/Madewell fit; clipped to 3-17 d.
    "incubation_lognorm_mu": 2.108,    # 来源：Madewell 2023 潜伏期 lognormal 拟合 mu
    "incubation_lognorm_sigma": 0.254, # 来源：Madewell 2023 潜伏期 lognormal 拟合 sigma
    "incubation_clip": (3, 17),        # 来源：潜伏期截断 3-17 天（Madewell 范围）

    # Symptom-onset -> rash delay: 1-4 days (CDC clinical description:
    # "rash usually develops within 1-4 days after onset of symptoms").
    "symptom_onset_delay_range": (1, 4),  # 来源：CDC 临床描述（症状→皮疹 1-4 天）

    # Infectious period: gamma(shape=9, scale=2.33) -> mean ~21 d, clipped
    # 14-28 d. Aligned with CDC reference model (27 d mean infectious period,
    # Spicknall MMWR 2022) and Netherlands fading model (14-28 d total).
    "infectious_period_gamma_shape": 9.0,   # 来源：Spicknall 2022 传染期 gamma shape
    "infectious_period_gamma_scale": 2.33,  # 来源：Spicknall 2022 传染期 gamma scale（均值~21d）
    "infectious_period_clip": (14, 28),     # 来源：传染期截断 14-28 天（荷兰 fading 模型）

    #R_t (Cori methid)
    # 来源：Cori 方法标准参数（τ=7 天平滑、10 天 burn-in 为常规设定）。
    "rt_smooth_window": 7,   # τ=7 天平滑（Cori 方法常规）
    "rt_burn_in": 10,        # 前10天不画/不算（避免早期不稳定）
    "rt_prior_a": 1.0,       # 来源：Gamma 先验 shape（无信息先验）
    "rt_prior_b": 1.0,       # 来源：Gamma 先验 rate（无信息先验）
    "rt_min_cases":20,       # 来源：抑制早期不稳定 Rt 的病例阈值
    # 来源：标定/结构假设（非文献固定）。加权平均的一次性活跃比例，实际活跃度由
    # oneoff_activity_stratified + CDC_RISKG_ONETIME_7D 分层决定（顶层 2.0/7天=0.286/天）。
    # 0.01 是让分层分布的「人群加权均值」≈1% 的归一化锚点，非独立旋钮。
    "oneoff_active_frac": 0.01,  # 归一化锚点（加权均值）；实际活跃度走 CDC riskg 分层
    # oneoff_p = NegBin(n=1, p) 的 p，控制「活跃者当天一次性伴侣数」分布，
    # 均值 = (1-p)/p。它不是传播概率（传播走 transmission_risks）。
    # 来源（一次性层重要性，文献锚点 Spicknall et al. MMWR 2022;71(35):1131-8）：
    #   一次性伴侣虽仅占日性伴侣 <3%、日性行为 16%，却贡献 ~50% 的 MPXV 传播；
    #   stratum 6（顶层 5% 人群）一次性伴侣形成概率 0.286/天。这支持本模型把
    #   one-off 层作为 MSM 核心放大的主通道、并给顶层设高位伴侣数的做法。
    # 来源：标定（calibrated，非文献固定）。一次性伴侣数均值通过「接触量水平扫描」
    # 联合 party_ht_top_mean 以因子 f 缩放，多 seed 验证确定 f=7.0 时 total median
    # 2369、peak median 93、homo 77.1% 最贴合目标(2280/92/77.5%)。
    #   f=7.0 => oneoff 均值 = 0.5*7.0 = 3.5 伴侣/活跃者/天 => p = 1/(1+3.5) ≈ 0.222。
    # 该值对应 riskg 筛出的「高风险活跃子集」（顶层仅 5%）的伴侣数，落在
    # CDC riskg 顶层(0.286/天) 与荷兰 casual 顶层(0.7-1.0/天) 之上的合理高位
    # （因为 CDC 的 0.286 是全人群加权均值，而非活跃子集的均值）。
    # 历史：0.15(均值5.67)→0.667(均值0.5,过度砍,疫情熄火total=20)→0.222(均值3.5,f=7.0标定)。
    # 详见 reference/性网络边生成机制审查.md、feasibility/multi_seed_f.py、
    # outputs/contact_volume/multi_seed_f.csv。
    "oneoff_p": 0.2105,  # NegBin(n=1,p)：均值=(1-p)/p≈3.75 伴侣/活跃者/天（f=7.5 标定；
                         # 2026-11-06 由 f=7.0 上调，补 total 至 2280 目标）
    # 来源：文献锚点（Brainard et al. PLoS One 2023;18(9):e0291001）：MSM 伴侣数
    # 峰值在 35-54 岁中年、hetero 组随年龄单调下降，说明性网络年龄同质性强。
    # 20 岁为合理年龄差上限，原 80≈无约束、无依据。
    "oneoff_age_gap": 20,  # 最大年龄差(年)；Brainard 2023 年龄同质性锚点
    # 来源：文献锚点。MF 一次性接触中肛交比例。NSFG 2015-19 报告美国成人终身
    # 异性肛交率 38.5%（男）/32.6%（女），但这是「终身是否有过」，per-act 比例
    # 远低于此（异性性行为以 vaginal 为主）。Han 2025 显示肛交传播 OR=2.40、
    # Snyder 2024 加州病例对照提示女性病例肛交/阴道混合暴露。综合取 0.10 作为
    # 一次性 MF 接触中肛交升级的 per-encounter 比例（远低于 MSM 肛交）。
    "oneoff_mf_anal_frac": 0.10,  # MF 一次性接触肛交比例（NSFG 终身率→per-act 下调）
    # Heavy-tail activity stratification for the one-off layer.
    # When True, MCP agents are split into 6 activity strata (5x19% + 1x5%,
    # aligned with the CDC MMWR riskg design / Weiss et al. 2020) and their
    # daily one-off active probability follows the CDC qnts relative partner-
    # formation rates (top 5% ~ 2/7 partners/day). This restores the heavy
    # tail that drives endogenous amplification in the reference model.
    "oneoff_activity_stratified": True,
    # Orientation-stratified one-off participation. CDC's one-time network is
    # effectively MSM-only; running one-off activity uniformly across the whole
    # MCP pool (which is ~72% heterosexual by headcount) over-amplifies MF
    # spillover once one-off activity rises. These weights bias one-off
    # participation toward MSM / male-bisexual strata (sexcat: 0=FH, 1=MH,
    # 2=MSM, 3=WSW, 4=MB, 5=FB). Weights are re-normalised so the population-
    # weighted mean participation still equals oneoff_active_frac.
    # 来源：结构假设（CDC 一次性网络本质为 MSM 主导），权重为经验设定。
    # 方向锚点：WHO/CDC 2022 疫情 98% 病例为 MSM，女性病例 <5%；Spicknall 2022
    # 顶层 stratum 6（5% 人群）一次性伴侣形成概率 0.286/天、底层 stratum 1 为 0，
    # 证明一次性网络高度集中在 MSM 核心。故 MSM 权重 3.0 远高于 hetero/女性
    # （0.15–0.40），符合「一次性网络 MSM 主导、女性近零参与」的文献结构。
    # 2026-11-04 修正：MB(male bi) 权重 2.00 -> 1.00。根因诊断（feasibility/
    # scan_oneoff_bias.py）证明 bi 偏高主因是此权重过高（bi 男性在 oneoff 层
    # 参与权重是 hetero 男性的 10 倍），而非 p_bi_m_to_msm。降到 1.00 后 bi
    # 病例从 18.7% 精确回落到 12.4%（目标 12.6%）。文献锚点：NYC 719 例中
    # bisexual 仅 5.6%（vs gay 78.7%）、「接触女性」仅 2.6%，说明 bi 男性在
    # 疫情中虽高度卷入 MSM 网络但 one-off 活跃度不应是 hetero 的 10 倍。
    "oneoff_orientation_bias": {
        0: 0.15,   # FH (female hetero)  — low one-off participation（经验）
        1: 0.20,   # MH (male hetero)    — low（经验）
        2: 3.00,   # MSM                 — high (dominant in one-time network)
        3: 0.30,   # WSW                 — low（经验）
        4: 1.20,   # MB (male bi)        — 由 2.00 回调（2026-11-04 根因修正）；
                   # 1.00→1.50→1.20（2026-11-06 构成微调，见 scan_bi_finetune.py：
                   # bi .150→.134 逼近目标 .126，total 2323 完美命中 2337）
        5: 0.40,   # FB (female bi)      — low（经验）
    },
    #sexual network
    # 来源：结构假设 + 文献锚点。MCP 性活跃边每日激活比例（3%），配合 sexual_edge_mult
    # 控制性网络规模。锚点：Spicknall et al. MMWR 2022 报告一次性伴侣占日性伴侣
    # <3%（MSM 网络）；Atlanta 2016-19 伴侣数据（Weiss et al. 2020）分层 MSM
    # 过去 3 周伴侣数为 0.8–6.6。3% 与「一次性伴侣 <3% 日性伴侣」同量级。
    "sexual_active_frac":0.03, # MCP 性边每日激活比例（锚点：Spicknall 2022 一次性<3%日性伴侣）
    # 来源：结构假设。性边数量乘子，用于模拟行为改变场景（baseline=1.0 不缩放）。
    "sexual_edge_mult": 1.0, # 性边数量乘子（baseline 不缩放）
    # 来源：结构假设（CDC 高风险 MSM 每周出席派对约 3 晚）+ 文献锚点。
    # 3/7 = 0.429 是「符合条件者」的出席概率锚点。文献支撑：
    #   - AMIS 2022 调查（n=553 高风险 MSM）：36.3% 过去 3 月参加 sex club/sex
    #     party/bathhouse，43.4% 参与 group sex，57.8% 参加过大型聚会；
    #   - England 技术简报 4（2022-07）：确认病例中 festival 37%、sex-on-premise
    #     venue 25%、private sex party 4%；
    #   - Madrid 病例系列：14.4% 报告参加大型活动/派对、56.9% 约会 app、
    #     64.9% casual 伴侣。
    # 综合这些数据，高风险 MSM 每周 ~3 晚出席高风险社交/派对场景是合理锚点。
    "sex_party_attend_prob":3/7, # 出席：符合条件者平均每周 3 晚（AMIS 36.3% + England 简报）
    # 来源：结构假设（legacy 回退值）。
    "sex_party_touch_frac":0.20, # legacy fallback for party_touch_frac

    # Parties can happen every day, but intensity varies by weekday/weekend.
    # 来源：结构假设（周末派对强度更高）。weekday/weekend 倍率 0.7/1.3 为经验设定，
    # 无直接文献点值；方向依据：England 技术简报显示高风险事件（festival、
    # sex-on-premise）集中在周末/节庆时段，Madrid/AMIS 调查也显示周末活动高峰。
    # 该参数仅用于平滑周内出席节奏，对总量影响小，已声明为结构平滑参数。
    "party_weekday_mult": 0.7,
    "party_weekend_mult": 1.3,
    "party_weekend_only": False,  # if True, parties only happen on Fri/Sat nights (weekend_mult defaults higher)

    # Optional per-party-type overrides (fallback to sex_party_attend_prob / sex_party_touch_frac)
    # "party_msm_attend_prob": 0.50,
    # "party_wsw_attend_prob": 0.50,
    # "party_mix_attend_prob": 0.50,
    # Single shared close-touch expansion fraction across MSM, WSW, and MIX
    # high-contact event pools. This replaces older pool-specific
    # party_msm_touch_frac / party_wsw_touch_frac / party_mix_touch_frac.
    # 来源：结构假设 + 文献锚点（非性密切接触在派对场景的扩展）。
    #   - CDC Thomas et al. MMWR 2023;72(50):1351-2：非性密切接触每例平均 2.4 人、
    #     次代发病率 1.5%（pooled），其中大型聚集（festival 等）占 40.9%、
    #     home 占 7.8%；中非 endemic 区未接种家庭接触 SAR 池化估计 8%。
    #   - 派对/聚集场景存在「非性但高亲密」的皮肤接触扩展（England 简报：
    #     festival 37%、sex-on-premise 25%）。
    # 0.20 = 高接触事件内非性密切接触扩展比例，sensitivity 0.10–0.25；无精确点值，
    # 但方向与量级（性接触主导、非性亲密接触少量扩展）有 CDC 数据支撑。
    "party_touch_frac": 0.20,


    # run time/ logging
    "n_days": SIM_DAYS + 1,
    "verbose": False,    #True to print every day
    "print_every":7,     #only used when verbose=True

    # ============================================================
    # Simplified calibrated baseline parameters v1
    # ============================================================
    # Only these 5 parameters should be treated as tunable in future
    # calibration. Other entries in this block are fixed structural
    # assumptions or documentation bounds. population.py maps these
    # simplified parameters to legacy names internally.
    #
    # Recommended tuning ranges:
    #   p_bi_m_to_msm         : 0.55–0.75
    #   party_core_intensity  : 0.45–0.70
    #   social_contact_scale  : 1.0–2.5   (literature-anchored ≈1.0; was 10)
    #   social_same_sex_assort: 0.65–0.85 (literature-anchored ≈0.80; was 0.9)
    #
    # NOTE: The previous composite parameter party_hetero_leak was removed
    # from the exposed baseline because it mainly acted as a female-ratio
    # calibration knob through party_mix_female_mult. Heterosexual spillover
    # is now represented through explicit mechanisms:
    #   1) heterosexual/bisexual agents entering the MIX event pool,
    #   2) stable/regular partner bridging involving bisexual agents,
    #   3) nonsexual close-touch expansion within high-contact events.
    # ============================================================

    "simplified_parameter_version": "v3_epi_size_reconciled_direct_transmission_risks",

    # 1) Bisexual male routing into MSM-like contact networks.
    # Increase -> more bi/MSM-core transmission; decrease -> more
    # heterosexual bridging potential.
    # 来源：标定（calibrated），方向+量级有文献锚点。Bounds 0.55–0.75，取中点 0.65。
    # 文献锚点：UK 前瞻队列（EID 2024, n=1333 GBMSM）中 gay 89% / bisexual 11%，
    # 即 bi 男性在 MSM 网络中占比约 11%；CDC 技术报告 2 中 gay 或 bi 男性占性取向
    # 已知病例 93.3%（gay/bi 未拆分）；WHO 全球 85.9% 病例为 gay/bi/其他 MSM。
    # bi 男性自我认同进入 MSM 网络的倾向高（多数 bi 男性在疫情中实际通过男男性
    # 接触暴露），故路由比例应 >0.5；0.65 落在 [0.55,0.75] 中点，与「bi 男性高度
    # 嵌入 MSM 网络、但仍保留少量异性桥接」的行为学画像一致。
    "p_bi_m_to_msm": 0.65,

    # 2) Core MIX / party intensity. Mapped internally to
    # party_mix_attend_prob.
    # 来源：标定（calibrated），方向+量级有文献锚点。Bounds 0.45–0.70，取中点 0.55。
    # 文献锚点：WHO 全球 sitrep 12（2022-12）报告 59.1%（3776/6390）报告的暴露
    # 场景为「派对+性接触」（party setting with sexual contact）——派对是 2022 疫情
    # 最主要的单一暴露场景；UK EID 2024 显示 33% GBMSM 在 sex-on-premise/sex party
    # 场所接触近期伴侣；AMIS 2022 显示 36.3% 高风险 MSM 过去 3 月参加 sex club/party。
    # party_core_intensity 映射到 MIX 出席概率，0.55 落在「三分之一到六成 MSM 出席
    # 派对」的文献区间中点。
    "party_core_intensity": 0.55,


    # 3) Nonsexual/social close-contact scale. This changes contact
    # opportunities, not per-contact risk.
    # 来源：CDC 每例非性密切接触数 ≈2.4（即 social_contacts_per_case 已是 CDC 量，
    # scale 不应再放大）。原 10 是无依据的越界放大（致 total +64%，见 social 层量化
    # 2026-10）。回调到文献锚点 1.0。
    "social_contact_scale": 1.0,

    # 4) Same-sex assortativity for the social/nonsexual close-contact layer.
    # This replaces the previous direct male-targeting parameter. Because most
    # infectious/symptomatic sources are male in this outbreak, same-sex social
    # mixing still permits heterosexual-male spillover, but it is easier to
    # justify from social-contact literature than a fixed male-target fraction.
    # 来源：社会接触同性同配性文献 ≈0.80（范围 0.65–0.85）。原 0.9 越界，回调到 0.80。
    "social_same_sex_assort": 0.80,

    # ============================================================
    # Fixed structural assumptions; do not tune in the baseline.
    # ============================================================

    # Female-bisexual routing has little influence on the main targets.
    # 来源：结构假设（经验设定）。FB→WSW 路由比例 0.75，无直接文献点值。
    # 方向依据：本次疫情女性病例极少（WHO/CDC 均 <5% 女性），FB 女性主要在
    # 女性网络内活动，对主要目标（MSM 主导的 total/homo）影响极小，属次要旋钮。
    "p_bi_f_to_wsw": 0.75,

    # MIX heterosexual leakage components are fixed/derived.
    # 2026-11-04 方向性拆分修正（文献锚点见下）：
    #   - mix_mf_anal_frac（男→女，女性 receptive 肛交）：女性 hetero 的性传播
    #     以 vaginal 为主。文献锚点：Thornhill Lancet 2022（136 例女性）明确
    #     cis 女性病变以 vulvovaginal 为主（anorectal 主要见于 trans women），
    #     即女性 receptive 肛交在 hetero 传播中占比很低，应远低于 MSM 肛交。
    #     故从 0.30 回调，独立于 MSM 肛交比例。
    #   - mix_fm_anal_frac（女→男，女性 insertive）：生物学上罕见（女性作为
    #     插入方），默认极低或 0。
    #   同时修正 population.py 4140-4146 行 sel_fm 方向 bug（此前女→男误用
    #   beta_rec，高估 female->male 传播）。
    # 2026-11-06 修正：mix_mf_anal_frac 0.15 -> 0.30。联合扫描（feasibility/
    # scan_joint_total_composition.py）证明：持续输入方案下构成偏 MSM 核心
    # （homo .841），根因是扩散程度不足——病毒未充分从 MSM 核心外溢到 bi/hetero/
    # female 层。提高男→女肛交比例（0.15→0.30，仍在 NSFG per-act 下调的合理范围
    # 内）增强 MIX 层异性别桥接，配合 oneoff_orientation_bias[4] 1.0→1.5 一起
    # 把构成修复到 homo .792/bi .150/hetero .062/female .053（目标 .775/.126/
    # .070/.055）。这是「先统一 total+时序，再微调构成」的第二步（用户方法论）。
    "mix_mf_anal_frac": 0.30,
    "mix_fm_anal_frac": 0.05,
    # 来源：结构假设（经验设定）。MIX 事件中 MSM/WSW 同性的少量非性亲密溢出比例
    # 0.01，无直接文献点值；量级依据：CDC Thomas 2023 非性密切接触 SAR 仅 1.5%，
    # 同性溢出应远低于性传播主导的 MM/FF 边，故取近零。
    "party_mix_homo_spill_frac": 0.01,

    # Fixed party structure parameters.
    # 2026-11-04 修正：party_mix_target_mean 2.0 -> 8.0。根因诊断证明 MIX 池
    # 是 hetero 桥接的唯一独立通道（oneoff 层提 hetero 权重会挤占 MSM 容量致
    # total 熄火）。mix_mean=8.0 使 MIX 池每个出席者平均 8 个异性伴侣，补齐
    # hetero 桥接。文献锚点：WHO 全球数据 13.3% 病例为异性恋传播（HTX），
    # 支持 MIX 桥接非零、且需足够强度以复现 hetero 病例占比。
    "party_mix_target_mean": 8.0,
    # 来源：结构假设 + 文献锚点。MSM/WSW 出席概率 0.40/0.35 依据 AMIS 2022：
    # 36.3% 高风险 MSM 过去 3 月参加 sex club/party/bathhouse；England 简报
    # sex-on-premise venue 25%、festival 37%。WSW 出席略低（女性病例 <5%，
    # 派对网络女性参与度较低）。均为经验设定，无精确点值。
    "party_msm_attend_prob": 0.40,
    "party_wsw_attend_prob": 0.35,

    # Heavy-tail activity stratification for sex_party partner degree.
    # Aligned with CDC MMWR 6-level `riskg` (Weiss et al. 2020) and the Milan
    # case series (23% of cases report >50 partners / 3 months). The MSM core
    # pool is split into 6 activity strata (5x19% + 1x5%, CDC_RISKG_POP_WEIGHTS)
    # and each stratum draws its partner degree from its own E[k], scaled from
    # the CDC past-3-week partner-count ratio and anchored to party_ht_top_mean.
    # This drives intra-party amplification WITHOUT widening hetero spill
    # (MM edges already dominate this layer). See
    # reference/sex_party_参数文献依据.md and reference/sex_party_6层分层改动说明.md.
    "party_heavy_tail": True,
    # 来源：CDC MMWR riskg 分层 + Weiss et al. 2020 + Milan 病例系列（23% 病例 >50 伴侣/3月）。
    # [Documentation-only] Formerly the high-activity fraction; with the full
    # 6-level riskg gradient the top-stratum share is fixed at 5% by
    # CDC_RISKG_POP_WEIGHTS, so changing this value no longer affects the
    # stratification. Kept for backward compatibility.
    "party_ht_frac": 0.05,   # 来源：CDC riskg 顶层占比固定 5%（仅文档意义）
    # E[k | sexually active] for the TOP (stratum 5) activity stratum (NB r=1).
    # Lower strata scale by the CDC past-3-week partner-count ratio.
    #
    # 来源：标定（calibrated，非文献固定）。顶层 E[k] 通过「接触量水平扫描」
    # 联合 oneoff_p 以因子 f 缩放，多 seed 验证确定 f=7.0 => top_mean=1.0*7.0=7.0。
    # 6 级 E[k] = [0.85, 0.85, 0.95, 1.06, 1.59, 7.0]（按 CDC 3 周伴侣数比例缩放）。
    #
    # METHODOLOGICAL NOTE (semantic mismatch):
    #   该 E[k] 在代码里作为「单晚伴侣数」消耗，而 CDC 顶层锚点是「过去 3 周伴侣数
    #   6.6」。若严格按 3 周窗口换算（出席 3/7 × 发生性行为 0.73 × 21 天），单晚
    #   E[k] 应 ≈ 6.6/6.57 ≈ 1.0，但这会导致疫情无法建立（total=20，见接触量扫描
    #   f=1.0 情景）。实际标定 f=7.0 对应的 top_mean=7.0 是「高风险活跃子集」的
    #   合理高位（这些顶层 MSM 是 3 周伴侣数 6.6 的极端值外推，其单晚伴侣数本身
    #   就显著高于全人群均值）。历史：18.0(越界,overshoot到2724)→1.0(过度砍,熄火)
    #   →7.0(f=7.0标定)。详见 reference/性网络边生成机制审查.md §2.2、
    #   feasibility/multi_seed_f.py。
    "party_ht_top_mean": 7.5,  # 2026-11-06 由 7.0 上调（f=7.5 标定，补 total）
    # Degree clip for the top activity stratum (lower strata keep party_k_clip).
    # 来源：结构假设 + 文献锚点。顶层伴侣数上限 30，防止单晚抽到极端值。
    # 锚点：Spicknall 2022 顶层 stratum 6 年伴侣数 124.7（≈ 3 周 6.6）；英国
    # 病例系列报告部分病例 3 月伴侣数高达 50；Belgium/Israel 报告单月 3-20 伴侣。
    # 30 作为单晚伴侣数上限远高于全人群，但与「高风险活跃子集」极端值一致。
    "party_ht_top_clip": 30,

    # Stable / regular-partner layer.
    # Stable partner edges are generated among adults, not only among MCP agents.
    # The daily activation probability is stratified by MCP composition:
    #   neither endpoint MCP -> general adult stable edge
    #   exactly one endpoint MCP -> MCP-bridging stable edge
    #   both endpoints MCP -> high-contact/MCP-core stable edge
    # Conversion from monthly frequency f_month: p_day = 1 - exp(-f_month / 30).
    #   3/month  -> 0.095 ≈ 0.10
    #   9/month  -> 0.259 ≈ 0.26
    #   15/month -> 0.393 ≈ 0.39
    # 来源：月度性频率换算（经验/行为学假设，非文献固定）。
    # p_day = 1 - exp(-f_month/30)：月度频率→日激活概率的标准泊松转化。
    # 锚点：CDC Thomas 2023 报告非性密切接触中 home 占 7.8%（家庭聚集场景存在）；
    # 社会接触同配性文献支持家庭同配方向。0.25 为家庭同配强度经验设定，无精确点值。
    "hh_assort": 0.25,          # 来源：家庭同配性（home 场景 7.8% + 社会同配性方向）
    "bi_same_sex_frac": 0.10,   # 来源：双性恋同性伴侣比例。WHO 全球仅 6.3% 病例为
                                # bisexual 男性；Peru 205 例中 bisexual 12.7%（male 96.2%）。
                                # 说明 bi 男性在本次疫情中男性伴侣占比低，0.10 为合理下限。
    "p_stable_sex_general_per_day": 0.10,      # 来源：3/月 -> 0.095≈0.10
    "p_stable_sex_one_mcp_per_day": 0.26,      # 来源：9/月 -> 0.259≈0.26
    "p_stable_sex_both_mcp_per_day": 0.39,     # 来源：15/月 -> 0.393≈0.39
    # Backward-compatible fallback only. Do not report this as the primary value
    # when the stratified stable activation parameters above are supplied.
    "p_household_sex_per_day": 0.10,
    # 来源：结构假设 + 文献锚点。stable 层 MF 边不升级为肛交（0）。
    # 锚点：Thornhill 2022 显示 cis 女性病变以 vulvovaginal 为主、anorectal 主要
    # 见于 trans women，即稳定异性伴侣间性传播以 vaginal 为主，肛交升级近零。
    "stable_mf_anal_frac": 0,   # 来源：stable 层 MF 边不升级为肛交（Thornhill 2022 女性 vaginal 主导）

    # Per-act sexual transmission-risk structure.
    # These are direct per-contact risk weights used by population.py.
    # They replace the less transparent expression
    #   base risk * sexual_risk_scale * act-specific multiplier.
    # This keeps the model easier to report in Supplementary Tables and lets
    # future searches vary biologically interpretable exposure categories rather
    # than direct sex/female-ratio knobs.
    #
    # 来源（per-act 概率）：
    #   - 荷兰 fading 模型（Xiridou et al. medRxiv 2023）：每次性接触传播概率
    #     βs = 0.25–0.40（接受方/肛交主导的估计区间）。
    #   - CDC Spicknall 2022（MMWR 官方参考模型）：每次性行为传播概率
    #     60%–90%（高情景，浓缩了伴侣内重复接触，故远高于 per-act）。
    #   当前 sexual_anal_receptive=0.28 落在荷兰模型 βs∈[0.25,0.40] 区间内，
    #   是文献内取值（非越界）。该参数同时驱动 oneoff/sex_party 层的 MM 边
    #   传播率（stable 层 MM 边亦用它），是 MSM 核心放大的主概率。
    #   详见 reference/性传播层参数_文献锚定.md。
    "sexual_anal_receptive": 0.28,   # 来源：荷兰 fading βs∈[0.25,0.40]，取 0.28
    "sexual_anal_insertive": 0.06,   # 来源：插入方风险 ≈ 接受方 1/4（经验比例，见锚定文档）
    "sexual_vaginal": 0.035,         # 来源：阴道性行为 SAR 显著低于肛交（Thornhill 2022 女性 vaginal 主导 + 经验比例）
    "sexual_oral": 0.0175,           # 来源：口交风险最低（≈阴道一半，经验比例；黏膜接触低于肛交/阴道，见锚定文档）

    # Social/nonsexual close-contact layer.
    # CDC-informed anchors:
    #   average identified nonsexual close contacts per case ≈ 2.4
    #   secondary attack rate among nonsexual close contacts ≈ 1.5%
    #   p_eff = 1 - (1 - 0.015) ** (1 / 2.4) ≈ 0.0063
    # 来源：CDC 非性密切接触报告（接触数 2.4/例、次代发病率 1.5%）。
    "social_contacts_per_case": 2.4,   # 来源：CDC 每例非性密切接触数 ≈ 2.4（Thomas 2023）
    "social_contact_window_days": 21.0, # 来源：接触窗口 21 天（监测期一致，WHO/CDC）
    "social_p_eff": 0.0063,            # 来源：由 SAR 1.5% 反推 per-contact 风险（Thomas 2023）
    "social_adjust_for_r_sus": True,   # 来源：结构假设（按易感者年龄调整，方向有年龄分层文献支撑）
    "social_p0_cap": 0.03,             # 来源：per-contact 风险上限（结构假设，≈ SAR 1.5% 的 2 倍）

    # Testing / diagnosis delay model.
    # Recommended staged-delay mode: at rash/symptom onset, each symptomatic
    # case enters the testing pathway with phase-specific probability, receives
    # a rash/symptom-onset-to-specimen delay, then a laboratory turnaround delay.
    # This is easier to align with CDC technical-report quantities than a
    # repeated daily testing Bernoulli trial.
    # 来源：CDC 技术报告 + 检测延迟经验值。
    "test_delay_mode": "scheduled_delay",
    # 来源（检测延迟，2026-11-04 补齐文献锚点）：
    #   - 症状→诊断延迟均值：Okoli 2023 全球 meta 分析（66 研究）报告症状出现→诊断
    #     均值 5.8 天（95% CI 4.8–6.8，4 研究 982 例）；Israel Patalon 2023 报告
    #     症状→确认检测中位 6.5 天；AAD/ILDS 101 例注册表报告症状→诊断中位 5 天
    #     （IQR 4–9）。故 test_seek_delay_mean_days=5.0 落在文献中位/均值区间内。
    #   - 延迟上限 21 天 = mpox 潜伏期上限（WHO/CDC 监测窗口），与 MONITORING_DAYS 一致。
    #   - 检测通路进入比例 0.85 与实验室周转 2 天：CDC 技术报告量级（症状病例大多采样，
    #     少数无症状/轻症漏检），属经验设定但方向有 CDC 数据支撑。
    "test_pathway_prob_symp": 0.85,
    "test_seek_delay_mean_days": 5.0,   # 来源：Okoli 2023 meta 5.8d / Israel 6.5d / AAD 5d
    "test_seek_delay_sd_days": 3.0,     # 来源：AAD 注册表 IQR 4–9 天（≈±3 天），文献一致
    "test_seek_delay_max_days": 21,     # 来源：潜伏期上限 21 天（WHO/CDC）
    "LAB_TURNAROUND_DAYS": 2,           # 来源：CDC 技术报告实验室周转量级（经验）

    # Contact tracing leads to 21-day monitoring/risk reduction, not
    # mandatory full quarantine for asymptomatic contacts.
    # 来源（监测期折减，2026-11-04 补齐）：
    #   - 监测期 21 天 = mpox 潜伏期上限，WHO/CDC/UKHSA 一致（WHO fact sheet、
    #     UKHSA category-3 contact 指南、CDC 接触者监测期均为 21 天）。
    #   - 监测期「避免性活动」是 WHO/CDC 明确建议（WHO fact sheet：接触者 21 天
    #     内应避免性活动；Manitoba 协议：接触者 21 天内避免所有性接触）。故
    #     性接触折减应显著（0.30）；非性社交接触仅「减少长时间近距离接触」，
    #     折减较温和（0.70）。mult 的精确值无点值，但方向与量级有 WHO/CDC 依据，
    #     已声明为敏感性分析对象。
    "MONITORING_DAYS": 21,
    "monitor_sexual_contact_mult": 0.30,   # 来源：WHO/CDC 建议接触者 21 天避免性活动 → 强折减
    "monitor_social_contact_mult": 0.70,   # 来源：WHO/CDC 仅建议减少近距离接触 → 温和折减

    # Community/social bridge settings.
    # The former "community_bridge_scale" late-stage social ramp (with
    # community_bridge_start_day / community_bridge_ramp_days) was REMOVED.
    # It was a pure Los-Angeles-specific fit knob with no literature basis and
    # poor interpretability; once fixed at 1.0 it had no effect on results.
    # Its removal changes the baseline social layer by exactly 0 (see parameter
    # audit, reference/无依据参数清单与文献范围.md §七). The social layer now
    # scales contacts uniformly over time via social_contact_scale only.
    "unhoused_hetero_male_boost": 1.00,

    # Documentation/sanity-check bounds for the exposed composition parameters.
    # 来源：p_bi_m_to_msm / party_core_intensity 的 bounds 为标定经验区间（无直接文献）；
    # social_contact_scale / social_same_sex_assort 的 bounds 来自 CDC 接触锚点与社会接触
    # 同配性文献。详见 reference/无依据参数清单与文献范围.md。
    "simplified_parameter_bounds": {
        "p_bi_m_to_msm": (0.55, 0.75),
        "party_core_intensity": (0.45, 0.70),
        "social_contact_scale": (1.0, 2.5),
        "social_same_sex_assort": (0.65, 0.85),
    },

    # Optional next-stage search space. Keep these disabled unless running a
    # predeclared transmission-risk calibration around the frozen baseline.
    # The receptive anal range is intentionally narrow because it controls
    # MSM-core amplification. Female-known / mixed-sex differences should be
    # explored mainly through insertive/vaginal/oral risks and mixed-sex
    # exposure fractions.
    "transmission_risk_parameter_bounds": {
        "sexual_anal_receptive": (0.2400, 0.3200),
        "sexual_anal_insertive": (0.0350, 0.0700),
        "sexual_vaginal": (0.0150, 0.0400),
        "sexual_oral": (0.0050, 0.0250),
    },

    # Calibration targets are stored inside params so Population.summary() can
    # report model-vs-observed differences after each run. They do not change
    # transmission directly; they are diagnostics for iterative calibration.
    "calibration_targets": CALIBRATION_TARGETS,
}

# 来源：结构假设 + 文献锚点。各传播层接触追踪到边的概率，反映「性派对/一次性
# 接触更难追踪、家庭/稳定伴侣/社交更易追踪」的定性排序。
# 锚点：England 技术简报 4（2022-07）报告 2413 名被命名接触者中至少 459（19%）
# 为性接触；CDC 接触追踪数据显示匿名性接触（派对/一次性）最易漏追踪、家庭接触
# 最易追踪。各层相对排序据此设定，精确点值无文献，属干预场景敏感性参数。
params["TRACE_PROB_EDGE_BY_LAYER"] = {
    "household_close": 0.10,
    "stable": 0.10,
    "oneoff": 0.05,
    "sex_party": 0.02,
    "sex_party_touch": 0.01,
    "social":0.03,
}

# ---------------------------------------------------------------------
# Plot saving utilities ------------------------------------------------
def _safe_filename(text, default="figure"):
    """Convert a plot title to a filesystem-safe filename."""
    if text is None:
        text = ""
    text = str(text).strip()
    if not text:
        text = default
    text = re.sub(r"[^\w\-.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] if text else default


def install_auto_save_plots(fig_dir, dpi=300):
    """
    Replace plt.show() with automatic save-and-close.

    This lets existing plotting functions in population.py remain unchanged:
    every plt.show() call will save all currently open figures to fig_dir.
    It avoids PyCharm SciView / InterAgg display problems and makes batch
    calibration reproducible.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    counter = {"i": 0}

    def _get_fig_title(fig):
        # Prefer figure-level suptitle if present.
        if getattr(fig, "_suptitle", None) is not None:
            title = fig._suptitle.get_text()
            if title:
                return title

        # Otherwise use the first non-empty axis title.
        for ax in fig.axes:
            title = ax.get_title()
            if title:
                return title
        return ""

    def _save_and_close_all(*args, **kwargs):
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return

        for num in fig_nums:
            fig = plt.figure(num)
            counter["i"] += 1
            title = _get_fig_title(fig)
            name = _safe_filename(title, default=f"figure_{counter['i']:02d}")
            out = fig_dir / f"{counter['i']:02d}_{name}.png"
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"[plot saved] {out}")
            plt.close(fig)

    plt.show = _save_and_close_all
    return _save_and_close_all


# ---------------------------------------------------------------------
# Scenario helper -----------------------------------------------------
def get_mask(pop, group):
    g, o, mcp = pop.genders, pop.orientations, pop.mcp_flags
    return {
        'random'           : np.ones(pop.n, dtype=bool),
        'homo_male_mcp'    : (g == 0)   & (o == 'homo')   & mcp,
        'homo_female_mcp'  : (g == 1) & (o == 'homo')   & mcp,
        'bi_male_mcp'      : (g == 0)   & (o == 'bi')     & mcp,
        'bi_female_mcp'    : (g == 1) & (o == 'bi')     & mcp,
        'hetero_male_mcp'  : (g == 0)   & (o == 'hetero') & mcp,
        'hetero_female_mcp': (g == 1) & (o == 'hetero') & mcp,
    }[group]

# ---------------------------------------------------------------------
# Optional modules (import only once) ---------------------------------




# ---------------------------------------------------------------------
# Convenience: count sex-act types in a network -----------------------
def count_sex_types(network, pop):
    counts = {"anal": 0, "vaginal": 0, "oral": 0, "unknown": 0}
    for u, v in network.edges():
        t = pop._get_sex_type(u, v)
        counts[t if t in counts else "unknown"] += 1
    total = sum(counts.values()) or 1
    return {k: 100 * v / total for k, v in counts.items()}


def print_and_save_calibration(summary, targets, out_dir):
    """Print a compact calibration table and save it as CSV.

    Main calibration now uses *episode-date / diagnosed-case* composition when
    available, because the observed LAC table is a case table rather than a true
    infection table.  Cumulative-infection ratios (`cum_inf_*`) remain in
    `summary()` as mechanism diagnostics, but they are not the default target
    comparator here.
    """
    import csv

    def _pct_string_to_float(x, default=0.0):
        try:
            return float(str(x).strip().replace("%", "")) / 100.0
        except Exception:
            return default

    current = {
        # Prefer numeric episode_* ratios generated by Population.summary().
        # Fall back to the older string ratios, then to cumulative infection ratios.
        "hetero": float(summary.get(
            "episode_hetero_ratio",
            _pct_string_to_float(summary.get("heterosexual_ratio", "0%"),
                                 float(summary.get("cum_inf_hetero_ratio", 0.0)))
        )),
        "bi": float(summary.get(
            "episode_bi_ratio",
            _pct_string_to_float(summary.get("bisexual_ratio", "0%"),
                                 float(summary.get("cum_inf_bi_ratio", 0.0)))
        )),
        "homo": float(summary.get(
            "episode_homo_ratio",
            _pct_string_to_float(summary.get("homosexual_ratio", "0%"),
                                 float(summary.get("cum_inf_homo_ratio", 0.0)))
        )),
        "female_known": float(summary.get(
            "episode_female_known_ratio",
            _pct_string_to_float(summary.get("female_ratio", "0%"),
                                 float(summary.get("cum_inf_female_known_ratio", 0.0)))
        )),
        "homo_male": float(summary.get(
            "episode_homo_male_ratio",
            float(summary.get("cum_inf_homo_male_ratio", summary.get("cum_inf_homo_ratio", 0.0)))
        )),
        "bi_male": float(summary.get(
            "episode_bi_male_ratio",
            float(summary.get("cum_inf_bi_male_ratio", summary.get("cum_inf_bi_ratio", 0.0)))
        )),
    }

    rows = [
        ("hetero", targets["target_hetero"], current["hetero"]),
        ("bi", targets["target_bi"], current["bi"]),
        ("homo", targets["target_homo"], current["homo"]),
        ("female_known", targets["target_female_known"], current["female_known"]),
        ("homo_male", targets["target_homo_male"], current["homo_male"]),
        ("bi_male", targets["target_bi_male"], current["bi_male"]),
    ]

    print("\nCalibration diagnostics, target vs current (episode/case composition):")
    print(f"{'metric':<16} {'target':>10} {'current':>10} {'diff_pp':>10}")
    abs_pp = 0.0
    for metric, target, cur in rows:
        diff_pp = 100.0 * (cur - target)
        abs_pp += abs(diff_pp)
        print(f"{metric:<16} {target:>9.2%} {cur:>9.2%} {diff_pp:>+9.2f}")
    print(f"Mean absolute calibration error: {abs_pp / len(rows):.2f} percentage points")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "calibration_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "target", "current", "diff_percentage_points", "source"])
        for metric, target, cur in rows:
            writer.writerow([metric, target, cur, 100.0 * (cur - target), "episode_case"])
        writer.writerow(["mean_absolute_error_pp", "", "", abs_pp / len(rows), "episode_case"])
    print(f"Calibration diagnostics saved to: {csv_path.resolve()}")

# ---------------------------------------------------------------------
SCENARIOS = [
    # 'random',
    "homo_male_mcp",
    # 'homo_female_mcp',
    # 'bi_male_mcp',
    # 'bi_female_mcp',
    # 'hetero_male_mcp',
    # 'hetero_female_mcp'
]


def _run_all_scenarios():
    """Execute the full scenario loop (guarded so `import main` is safe)."""
    _scenarios = SCENARIOS
    for sc in _scenarios:
        _run_one_scenario(sc)


def _run_one_scenario(sc):
    label=f"{int(POPULATION_SIZE/1e6)}M_{sc}".replace("+","_")
    print(f"label is {label}")
    print(f"\n=== Scenario: {sc} ===")

    out_dir = Path("outputs") / label
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    flush_plots = install_auto_save_plots(fig_dir, dpi=PLOT_DPI) if (MAKE_PLOTS and SAVE_PLOTS) else None

    t0 = time.time()


    # patch interventions --------------------------------------------------

    #for r in range(reps):  #uncomment when real simulation
    run_seed = base_run_seed #+ r
    # seed infections ------------------------------------------------------
    policy = [
        # phase0+phase1 merged: early growth / limited testing & tracing
        dict(name='phase1_growth', start=0, end=91,
             contact_mult=1.00, oneoff_mult=1.00, party_mult=0.95,
             social_mult=0.90,
             test_mult=0.70, trace_mult=0.80,
             test_pathway_prob=0.75, test_seek_delay_mean_days=5.0,
             test_seek_delay_sd_days=3.0, lab_turnaround_days=2),

        # emergency + behavior change: sexual contacts decline, but social/nonsexual
        # contacts are not forced below baseline because late bridging is part of the
        # calibration hypothesis. party_mult set to 0.70 (was 0.42) per the policy
        # time-shape scan: a longer, more gradual tail best matches the observed
        # Case_data decline (long_tail2, cRMSE 193 vs 296 baseline).
        dict(name='phase2_control', start=92, end=170,
             contact_mult=0.65, oneoff_mult=0.65, party_mult=0.70,
             social_mult=1.05,
             test_mult=1.50, trace_mult=1.20,
             test_pathway_prob=0.90, test_seek_delay_mean_days=4.0,
             test_seek_delay_sd_days=2.0, lab_turnaround_days=2),

        # post-peak: sexual network remains partly controlled; social/nonsexual bridging
        # increases modestly to allow late heterosexual-male/general-population spillover.
        # party_mult 0.70 (was 0.47) — same long_tail2 curve-fit rationale as phase2.
        dict(name='phase3_postpeak', start=171, end=SIM_DAYS,
             contact_mult=0.70, oneoff_mult=0.85, party_mult=0.70,
             social_mult=1.20,
             test_mult=1.20, trace_mult=1.00,
             test_pathway_prob=0.85, test_seek_delay_mean_days=4.5,
             test_seek_delay_sd_days=2.5, lab_turnaround_days=2),
    ]


    # --- External introductions (importations) ---
    # Sustained low-level importation replaces the day-0 seed (see the
    # INITIAL_INFECTIONS comment above). The "slow May -> accelerate Jun-Jul ->
    # taper Aug" shape mirrors the real LAC importation rhythm and fixes the
    # ~18-day early-peak artifact of the purely-endogenous seed scheme.
    # daily_n = 1/2/4/1 over day 15-134 (≈ 5/16 -> 9/12), peak inflow in July.
    importations = [
        dict(start=15, end=44, daily_n=1),   # late May, slow
        dict(start=45, end=74, daily_n=2),   # June, accelerate
        dict(start=75, end=104, daily_n=4),  # July, surge
        dict(start=105, end=134, daily_n=1), # August, taper
    ]


    pop = Population(n=POPULATION_SIZE, params=params,seed=pop_seed,cache_label=label, policy=policy, importations=importations)
    scenario_mask = get_mask(pop, sc)
    pop.importation_mask = scenario_mask
    #add randomness
    np.random.seed(run_seed)
    pop._rng = np.random.default_rng(run_seed)

    # Seed a small number of MSM index cases (drawn uniformly from the MSM core
    # via seed_infection -> np.random.choice). With INITIAL_INFECTIONS = 0 the
    # day-0 seed is disabled; the outbreak is instead ignited endogenously by
    # the sustained importation stream above. A uniform draw (rather than
    # targeting the high-activity stratum) was chosen after a seed-targeting A/B
    # test: forcing seeds into the high-activity core overshoots the 2280 target
    # (+56%) and degrades composition. See reference/seed_cases_文献依据.md.
    if INITIAL_INFECTIONS > 0:
        pop.seed_infection(INITIAL_INFECTIONS, mask=scenario_mask)

    # run ------------------------------------------------------------------

    pop.run_simulation(SIM_DAYS)
    print(f"Sim completed in {time.time() - t0:.1f} s")

    # summary --------------------------------------------------------------
    summary = pop.summary()


    print("Final state:", summary)
    # 输出也用 label + rep 组织
    out_dir.mkdir(parents=True, exist_ok=True)
    print_and_save_calibration(summary, CALIBRATION_TARGETS, out_dir)

    pop.print_mean_contacts_per_layer()

    # plots ----------------------------------------------------------------
    # R_t ------------------------------------------------------------------
    rt = pop.get_rt_series()
    if rt.size:
        # show last non-NaN if available
        finite = rt[np.isfinite(rt)]
        if finite.size:
            print(f"Latest R_t (Cori): {finite[-1]:.3f}")
        else:
            print("Latest R_t (Cori): n/a (insufficient incidence history)")

    # plots ----------------------------------------------------------------
    if MAKE_PLOTS:
        # All plt.show() calls are intercepted by install_auto_save_plots(),
        # so these functions save PNG files instead of opening PyCharm windows.
        pop.plot_epidemic_curve()
        pop.plot_rt()
        pop.plot_layerwise_transmissions()
        pop.plot_time_series_by_layer()
        pop.plot_episode_vs_observed()
        pop.plot_episode_strata(by="sex")
        pop.plot_episode_strata(by="ori")
        pop.plot_orientation_incidence()
        pop.plot_breakdown_by_gender_orientation()
        pop.plot_sex_party_attendance()

        # Safety flush: save any figure created by a function that did not call plt.show().
        if flush_plots is not None:
            flush_plots()
        print(f"Figures saved to: {fig_dir.resolve()}")

##Female 2%，bi 12.6%，hetero 10%，homo 77.5%

    # extra diagnostics ----------------------------------------------------
    # print("\n[Sexual network edges]")
    # for k, v in count_sex_types(pop.sexual_network, pop).items():
    #     print(f"  {k:<7}: {v:5.1f}%")
    # print("[Sex-party edges]")
    # for k, v in count_sex_types(pop.sex_party_network, pop).items():
    #     print(f"  {k:<7}: {v:5.1f}%")


if __name__ == "__main__":
    _run_all_scenarios()
# ---------------------------------------------------------------------
