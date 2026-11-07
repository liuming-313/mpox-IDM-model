# CDC MMWR 猴痘性网络模型 —— 机制拆解与移植方案

> 目标：把 "Modeling the Impact of Sexual Networks in the Transmission of Monkeypox virus Among Gay, Bisexual, and Other Men Who Have Sex With Men — United States, 2022"（MMWR 2022;71(35):1133-1137，DOI 由 CDC 发布）的机制拆解清楚，并对照本项目 `main.py` / `population.py` 的差距，给出**最小改动、只补核心引擎**的移植方案。
>
> 参考代码位置：`reference/mpx_networkmodel_mmwr/`（已从 zip 解压）。

---

## 一、CDC 模型的核心思想（一句话）

> **传播异质性不来自"调高感染概率 β"，而来自"网络结构的重尾（heavy tail）"。**

具体地说，CDC 模型里少数高活跃 MSM（top 5%，约 100 伴侣/年）通过 **one-time（一次性）伴侣** 形成高度连通的网络，贡献了约一半的传播，即使 one-time 伴侣只占每日伴侣总数的 <3%、每日性行为的 16%。

---

## 二、网络结构：三层重叠动态性网络

代码用 EpiModel/STERGM 建了 3 层网络，共享同一批 10000 个 MSM 节点（`fit_networks.R`）：

| 层 | 英文 | 性质 | 度约束 | 平均持续时间 |
|---|---|---|---|---|
| 主要伴侣 | main | 稳定长期关系 | `bd(maxout=1)` 最多 1 个 | 407 天 |
| 临时伴侣 | casual / pers | 并发短期关系 | `bd(maxout=2)` 最多 2 个 | 166 天 |
| 一次性伴侣 | one-time / instant | 每天重新形成的瞬时关系 | 无（按分层 rate） | 1 天（瞬时） |

三层通过 **degree 依赖** 耦合：一个人在 main 层的度影响他在 casual 层形成关系的概率（`assign_degree` 函数把 main/pers 度作为另一层的节点属性）。

### 关键：6 个活动分层 `riskg`（heavy-tail 的来源）

`base_nw_msm()` 给每个节点分配 `riskg`（风险组），共 6 层：

```r
riskg <- sample(apportion_lr(num, 1:6, c(rep(0.19, 5), 0.05)))
# 前 5 层各占 19%，最高第 6 层占 5%
```

每层的 **one-time 伴侣形成率** 不同（`fit_networks.R` 的 `qnts`）：

```r
qnts <- c(0.0000, 0.007/7, 0.038/7, 0.071/7, 0.221/7, 2/7)
# 第 6 层（top 5%）= 2/7 ≈ 0.286/天 ≈ 100 伴侣/年
```

这就是 heavy-tail：**5% 的人拥有 ~100 伴侣/年，而最低层几乎为 0**。

---

## 三、传播概率是怎么设置的（最该学的一点）

### 3.1 三个核心参数（`run_network_models_mmwr.R`）

```r
freeparam1 <- 0.6   # 低传播：每次性行为感染概率（高传播 = 0.9）
freeparam3 <- 27    # 平均传染期（天），i.to.r.rate = 1/27
```

### 3.2 核心公式：per-act → per-partnership 换算（`network_modules.R:517`）

```r
del$finalProb <- 1 - (1 - del$transProb)^del$actRate
```

其中 `actRate` 是**每个伴侣关系每天发生几次性行为**：

```r
act.rate.main    = 1.54/7   # 主要伴侣：每 7 天 1.54 次
act.rate.casual  = 0.96/7   # 临时伴侣：每 7 天 0.96 次
act.rate.instant = 1        # 一次性伴侣：当天 1 次
```

**含义**：把"单次性行为概率 β"通过性行为频率折叠成"这段关系内的整体传播概率"。例如主要伴侣关系：`1-(1-0.6)^(1.54/7) ≈ 0.183`，而不是直接 0.6。

### 3.3 疫苗接种修正（`network_modules.R:511`）

```r
del$transProb[del$statusSus=="v"] <- inf.prob * (1 - vacc.effect)  # vacc.effect=0.85
```

### 3.4 最终判定（`network_modules.R:520`）

```r
transmit <- rbinom(nrow(del), 1, del$finalProb)  # 每条 discordant 边一次伯努利
```

---

## 四、传播逐日流程（`infect_msm2`）

每天：
1. **找 discordant 边**：三层里"一端感染（i/im/a）、一端易感（s/v）"的边；
2. **算 finalProb** 并抽样判定传播；
3. **记录来源**：`secInfs`（每个感染者传了多少人）、`infGen`（感染代数）、分层计数（`nInfsMain`/`nInfsPers`/`nInfsInst`）。

**注意**：状态机是 e→a→i→r（`progress_msm`），其中 **a（无症状感染）不传播**。

---

## 五、行为变化怎么实现（`simnet_msm`）

**不是降低 β，而是降低 one-time 层的边形成系数**：

```r
if(at == 75 & i == 3){adjust <- log(new.num*0.6) - log(old.num)}  # 第 75 天 one-time 层砍 40%
```

对应论文结论：one-time 伴侣减少 40% → 感染人数减少 20%-31%。

---

## 六、论文关键结论（PDF 摘录）

- one-time 伴侣占每日伴侣 <3%、每日性行为 16%，却贡献 **46%-54%** 传播；
- 低传播场景：one-time 54.0%、casual 33.2%、main 12.9%；
- 高传播场景：one-time 45.6%、casual 38.8%、main 15.6%；
- **初始种子 = 10 个 high-activity 病例**（不是持续输入）；
- 最高活动层 ~100 伴侣/年；
- 明确声明：**regular reintroductions 未建模**（这正是本项目 742 输入方案的关键 benchmark）。

---

## 七、与本项目模型的差距对照

| 维度 | CDC 模型 | 本项目当前 | 差距 |
|---|---|---|---|
| 活动分层 | 6 层，top 5% 达 100 伴侣/年 | 无分层 | ❌ |
| one-time 率 | top 层 0.286/天 | `oneoff_active_frac=0.01`（1% 活跃）| ❌ |
| 核心网络度 | heavy-tail | stable 层一对一（55% 覆盖）；`mcp_mean=0.1` 且该 MCP 网络已废弃 | ❌ |
| 感染概率 | 0.6/0.9 per-act | `sexual_anal_receptive=0.28` 等分性行为 per-act | ⚠️ 更细 |
| act rate | 显式 1.54/7、0.96/7、1 | `sexual_active_frac`（边活跃概率，概念不同）| ⚠️ |
| 种子 | 10 个 high-activity | 742 次外部输入 | ❌ **最大问题** |
| 行为变化 | 砍 one-time 形成率 | policy 里 `oneoff_mult` | ✅ 已有 |
| 来源追踪 | secInfs/infGen/层贡献 | 无 | ❌ |

---

## 八、本项目"病例不是少数人群弄的"的根因（代码级证据）

1. **stable 层是一对一伴侣**：`_create_household_partners` 里 `partner_of[i]=j; partner_of[j]=i`，55% 成年人有 1 个稳定伴侣，**无并发、无重尾**。
2. **含 `mcp_mean=0.1` 的 MCP 多重伴侣网络已被废弃**：`_load_or_build_mcp_legacy`（`population.py:1307`）在 `__init__` 里**不再被调用**，`_update_sexual_network`（`population.py:2748`）遇到 `hh_u` 就直接返回 household 网络。
3. **one-off 层被压到 1% 活跃**：`oneoff_active_frac=0.01`。
4. **`seed_infection` 被注释**（`main.py:549`），传播起点全靠 742 次 importation。

**结论**：模型里"少数高活跃人群"这个群体被结构性抹掉了，所以疫情只能靠外部输入硬灌。

---

## 九、移植方案（最小改动，只补核心引擎）

### 改动 1：给 one-off 层加 6 层 activity 分层（核心）

模仿 CDC `riskg`，把 MCP 人群按 `(0.19, 0.19, 0.19, 0.19, 0.19, 0.05)` 分 6 层，每层用不同的 one-off 活跃度和伴侣数：

```python
# 建议映射（对齐 CDC qnts）
strata_active_frac = [0.000, 0.001, 0.005, 0.010, 0.031, 0.286]
strata_oneoff_p    = [0.50,  0.30,  0.20,  0.15,  0.10,  0.05]
```

### 改动 2：恢复 high-activity 种子替代 742 输入

```python
pop.seed_infection(10, mask=get_mask(pop, "homo_male_mcp"))
# 优先从 top 分层抽取
```

### 改动 3（可选）：引入 per-act → per-partnership act rate

```python
final_p = 1 - (1 - p_act) ** act_rate
```

### 改动 4（可选）：加来源追踪

`secInfs` / `infGen` / 层贡献，复现论文 Figure 1 的三层传播分解。

---

## 十、验证脚本

`feasibility/audit_endogenous_amplification.py` —— 不改 `main.py`/`population.py`，通过 monkeypatch 注入 heavy-tail one-off + 少量种子，对比：

- A：当前 742 输入（基线参考）
- B：10 个 high-activity 种子 + heavy-tail
- C：50 个 high-activity 种子 + heavy-tail

输出到 `outputs/feasibility/endogenous_amplification_screen.csv`，观察**灭绝率**与**自持爆发概率**。
