# social 层 3 个 P0 参数的文献锚定论证

> 模型重建第 4 步的**前置论证**（用户选择：先坐实边界，不碰代码）。
> 目标：给 `social_contact_scale`、`social_same_sex_assort`、`community_bridge_scale`
> 三个"超界凑数值"找到**文献支撑的正确边界**，判断哪个该回调、哪个能坐实、哪个该删。

---

## 0. 一句话结论

| 参数 | 现值 | 代码自述边界 | 文献能坐实的值 | 判定 |
|---|---|---|---|---|
| `social_contact_scale` | **10** | 2.5–6.0 | **1.0**（不该放大，见 §1）| ❌ 必须回调，且应=1.0 |
| `social_same_sex_assort` | **0.9** | 0.65–0.85 | **0.75–0.85**（homophily 支撑，见 §2）| ⚠️ 略超，回调到 0.80 即可坐实 |
| `community_bridge_scale` | **1.5** | 1.0 | **1.0**（无文献，见 §3）| ❌ 应删除/固定 1.0 |

---

## 1. `social_contact_scale`：当前值架空了 CDC 锚点，应=1.0

### 1.1 代码里已经写死的锚点

`main.py` 里 social 层有两个**外部精确锚定**的参数（见 §5 引用）：

```python
"social_contacts_per_case": 2.4,     # CDC 美国两辖区：平均 2.4 个非性接触/患者
"social_p_eff": 0.0063,              # 由 SAR 1.5% 反推：1-(1-0.015)^(1/2.4)
```

出处：**CDC MMWR Notes from the Field**（Tennessee + DC，2022-05-01~07-31）：
- 278 例 mpox 患者，662 个非性接触 → **平均 2.4 个/患者**
- 非性接触续发率（SAR）= **1.5%**（10/662 有症状）

### 1.2 问题：`social_contact_scale=10` 把 SAR 偷偷放大到 ~15%

`population.py` `_update_social_layer` 的实际公式：

```
λ = (contacts_per_case / D) × contact_scale × late_mult
  = (2.4 / 21) × 10 × late_mult
  ≈ 1.14 × late_mult   # 每天每个传染源的非性接触数
```

而正确的 CDC 锚定应该是：

```
λ = (2.4 / 21) × 1.0 × late_mult ≈ 0.114  # 每天 ~0.11 个非性接触
```

**`social_contact_scale=10` 意味着**：每个传染源每天的非性接触数被放大 10 倍，
等价于把 CDC 的 1.5% SAR 抬到 ~15%。这**直接违反了代码自己的纪律**（注释明确写
"social_p_eff 是外部锚定、不应和 social_contact_scale 一起调"）。

### 1.3 为什么当初会被推到 10

这是早期校准为了**凑 hetero/female 漏出**而引入的兜底旋钮。social 层是模型里
唯一的"非性传播"通道，当异性恋比例、女性比例不够时，就调大 `social_contact_scale`
让更多病例从 MSM 核心漏到一般人群。但这是**用放大一个"文献定死的 SAR"来凑数**，
本质上和审稿人批评的"拍脑袋"同源。

### 1.4 结论

- **`social_contact_scale` 应固定为 1.0**，让 social 层回到 CDC 的"2.4 非性接触 +
  1.5% SAR"外部锚定。
- 损失的 hetero/female 漏出，应通过**有据的机制**补回（如 stable 层伴侣桥接、
  one-off 层的 MSM→bisexual 路由），而不是放大 SAR。
- 这会**显著降低 social 层贡献**，因此必然需要重校准其他层补偿 —— 这正是为什么
  第 4 步不能"只改这一个参数"。

---

## 2. `social_same_sex_assort`：0.9 略超，0.80 可坐实

### 2.1 文献支撑（方向正确）

- **社会学同质性（homophily）**：McPherson, Smith-Lovin & Cook (2001) 经典结论 ——
  人倾向与同性别/同年龄者交往。
- **McCormick, Salganik & Zheng 的 latent nonrandom mixing model**：成年男性"demonstrate
  a high proportion of their ties to other males"，同性混合显著。
- **POLYMOD（Mossong et al. 2008）**：8 国 97,904 个接触，性别接触矩阵显示男性同性
  接触占主导（尤其成年男性）。

### 2.2 但"0.9"这个具体数值无文献

- 文献支持的是**"成年男性同性社交混合显著"这一方向**，但没有一篇给出精确的
  "90% 同性别"这个数。
- POLYMOD 的性别同配性没有单独提炼成 assortativity 系数；McCormick 的 mixing matrix
  是定性的（"high proportion"），不是 0.9。
- **0.9 意味着"近乎完全同性混合"**，这在一般社交（非性）接触里过于极端 —— 一般人群
  的社交接触不可能 90% 是同性别（同事、家人、邻里必然有跨性别接触）。

### 2.3 结论

- 方向有据（homophily），但 **0.9 太极端**。
- 代码自述边界 0.65–0.85 是合理的：**回调到 0.80** 既保留"男性同性社交混合占主导"
  的 homophily 结论，又不至于"近乎完全同性隔离"到不可信。
- 可以用 McCormick/McPherson 的 homophily 文献 + POLYMOD 性别接触矩阵作为引用，
  但数值本身是"在 homophily 文献区间内的合理假设"，需要如实表述为假设而非测量值。

---

## 3. `community_bridge_scale`：无文献，应固定 1.0

### 3.1 现状

```python
"community_bridge_scale": 1.5,   # 代码注释自己写"baseline 应=1.0"
"community_bridge_start_day": 74,
"community_bridge_ramp_days": 170,
```

机制：day 74 起，170 天内线性 ramp 到 `late_mult = 1 + (1.5-1)×ramp`，即最多 +50%
的 social 接触。

### 3.2 问题

- **没有任何文献对应**"mpox 疫情第 74 天起社区社交接触放大 50%"。
- 代码注释自己承认："Keep community_bridge_scale at 1.0 in the baseline... Any value
  above 1.0 should be treated as an explicit sensitivity scenario, not as a primary
  calibrated parameter."
- 这是**早期校准为了"晚期 tail 拉长"而引入的纯拟合旋钮**，和审稿人批评的"为了凑
  曲线而加参数"完全一致。

### 3.3 结论

- **应固定为 1.0**（关闭这个 ramp），回到代码自己声明的 baseline 纪律。
- 如果曲线拟合需要"晚期 tail"，应通过**有据的行为改变机制**（如 MSM 行为改变、
  检测-隔离效率提升）实现，而不是一个无来源的"社区放大 ramp"。

---

## 4. 连带影响：第 4 步不是"改 3 个参数"，是"重新校准"

这三个参数都是**通过架空外部锚定值来凑 hetero/female 漏出和总规模**的。把它们
拉回文献边界后：

1. **social 层贡献显著下降**（scale 10→1，接触数降 10 倍；bridge 1.5→1，去掉 +50% ramp）
2. **hetero/female 漏出会下降**（assort 0.9→0.80 增加跨性别接触，方向相反，需实测）
3. **总规模会下降**，需要用有据的机制补回

因此，**在动手改代码前，必须先量化 social 层当前贡献 + 这三个参数各自的敏感性**，
才能设计"拉回边界 + 有据补偿"的完整方案。这是第 4 步的下一步动作。

---

## 5. 引用清单（可直接用于论文 Methods / Supplementary）

### social_contacts_per_case = 2.4 + SAR 1.5%
- **CDC MMWR Notes from the Field**: Transmission of Mpox to Nonsexual Close Contacts —
  Two U.S. Jurisdictions, May 1–July 31, 2022. (278 cases, 662 nonsexual contacts,
  average 2.4/case; nonsexual SAR 1.5%).

### 非性接触 SAR 的独立佐证
- **Packer et al. 2024, Epidemiol Infect**: Mpox in UK households — household SAR 4%,
  sexual contact aOR 11-fold, female aOR 0.41.
- **2022–2024 系统综述（BMC Infect Dis 2026）**: 62 研究 8712 暴露，非性接触 SAR
  **1.34%**（vs 性接触 11.35%，p<0.001），非性接触占 83.17% 但性接触贡献大部分续发。

### 同性社交混合（homophily）
- **McPherson, Smith-Lovin & Cook (2001)**: Birds of a Feather: Homophily in Social
  Networks. Annu Rev Sociol.
- **McCormick, Salganik & Zheng**: How Many People Do You Know? (latent nonrandom
  mixing model，成年男性同性社交显著).
- **Mossong et al. (2008, PLoS Med)**: POLYMOD 社交接触调查（8 国 97,904 接触）。

### 性网络同配性（assortativity，区别于社交层）
- **加拿大 3 城 Engage Cohort（J Infect Dis 2024）**: MSM 性网络 assortativity
  0.67–0.78（这是**性接触**网络的同配性，不是社交层，但可作为"同性混合"的量级参考）。

---

## 6. 待办（第 4 步的实际执行，需你确认后启动）

1. 跑一次带 `layer_infection_counts` 的 10M 基线，量化 social 层当前贡献
   （占总病例 / hetero 漏出 / female 漏出的比例）。
2. 对三个参数做**单参数敏感性**（在各自边界附近），确认"拉回边界"各自的影响量级。
3. 基于量化结果，设计"拉回边界 + 有据补偿"方案（优先用 stable 层桥接 / one-off 层
   路由补回漏出，而非重新放大 social）。
4. 重跑多 seed 校准，确认规模/构成/曲线仍命中目标。
