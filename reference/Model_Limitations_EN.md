# Model Limitations — English Discussion draft

> 用途：放进论文 Discussion / Limitations 小节的英文底稿，诚实说明两个已知的形状偏差及其根因。
> 与 `Supplementary_Material_S1.md` 的 S1.3（结构假设）和「完全内生传播」假设呼应。

---

## Limitations (Discussion section draft)

Our model reproduces the final outbreak size, peak timing, and case composition of the 2022 Los Angeles County mpox epidemic, but two aspects of the epidemic *shape* are simplified relative to the observed curve, and we describe them here for transparency.

**First, the early growth phase rises faster than observed.** The simulated curve reaches its first 50% of cumulative cases roughly two weeks earlier than the surveillance data (first reported case at day 6 vs day 18; 50% cumulative at day ~85 vs day 104). This reflects a structural property of our agent-based sexual network: the high-activity "sex-party" core exhibits threshold (all-or-nothing) behaviour, in which the epidemic either ignites rapidly once this core is seeded or fails to establish at all. In reality, penetration of the high-activity MSM core was likely gradual, producing a smoother early ramp. We found that a slower, more realistic early ramp (achieved by seeding fewer, more gradually introduced cases) could not be combined with the correct final size under a fully endogenous-transmission assumption: reducing the seed to 4–6 cases reproduced the early slope but collapsed the final size to ~950–2,000 cases, while the 20-case seed required to reach 2,280 inevitably accelerates the early phase. We therefore prioritised accurate final size and composition over the early-growth slope.

**Second, the epidemic tail is truncated.** The simulated epidemic extinguishes around day 170, whereas the observed outbreak persisted at a low level (2–7 cases/day) through day ~460. Our model assumes fully endogenous transmission with no exogenous importation, so once the sexual network is exhausted the epidemic stops. The observed long tail most plausibly reflects continued low-level introduction and residual transmission within small, high-activity subgroups that our model does not represent. Re-introducing a modest importation stream would extend the tail, but at the cost of re-introducing parameters without direct literature anchors, which we chose to avoid in favour of a parsimonious, fully-endogenous baseline.

These two simplifications are two faces of the same modelling choice: a stylised, threshold-like sexual-core transmission engine that is calibrated to the outbreak's terminal size and composition rather than to its full time-course. We view the close match on total size (median 2,288 vs 2,280 observed), peak day (median 92 vs 92 observed), and composition (male-homosexual proportion 77% vs 77.5% observed) as the primary validity evidence, and flag the early-ramp and long-tail discrepancies as limitations that future work could address by modelling gradual core penetration and exogenous importation.

---

## 附：可选的一句话精简版（供正文 Discussion 首句使用）

> "Two shape-level discrepancies remain — a faster-than-observed early ramp and a truncated late tail — both attributable to our stylised, threshold-like sexual-core transmission engine under a fully endogenous (no-importation) assumption; we prioritised accurate terminal size and composition over the full time-course."
