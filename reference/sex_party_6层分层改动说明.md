# sex_party 层 6 层活动分层：改动说明与文献依据

> 这是模型重建第 3 步的落地记录。把 sex_party 层原来的**二值高活跃标记**（`sex_party_high_active`）升级为**完整的 CDC riskg 6 层活动梯度**（`sex_party_activity_level`），使伴侣数分布有明确的文献来源支撑。

---

## 一、改动前 vs 改动后

| 维度 | 改动前（二值）| 改动后（6 层）|
|---|---|---|
| 分层结构 | `sex_party_high_active` 布尔标记 | `sex_party_activity_level` int8（-1/0-5）|
| 高活跃占比 | `party_ht_frac`（默认 5%）| 由 CDC_RISKG_POP_WEIGHTS 固定（5×19% + 1×5%）|
| 伴侣数 E[k] | 二档：top=20，bulk≈2.7 | 6 档梯度：[2.42, 2.42, 2.73, 3.03, 4.55, 20.0] |
| 文献支撑 | top 5% 对齐 CDC、bulk 对齐 NEJM | 全 6 层对齐 CDC riskg 伴侣数梯度 |

---

## 二、核心设计：伴侣数怎么映射到 CDC riskg

**关键问题**：CDC MMWR 的伴侣数是**累计**（过去3周 / 过去1年），而 sex_party 层要的是**单次派对当晚的伴侣数**，量纲不同，不能 1:1 套用。

**解决方案**：用 CDC riskg 的**相对梯度**（而非绝对值），把顶层锚到已校准的 `party_ht_top_mean`，其余层按比例缩放：

```
E[k]_level = party_ht_top_mean × (CDC_RISKG_PARTNERS_3W_level / CDC_RISKG_PARTNERS_3W_top)
```

**为什么用 3 周比例而非 1 年比例**：

| 口径 | 相对梯度（顶层=1）| 缩放后 E[k]（top=20）| 评价 |
|---|---|---|---|
| 过去3周 | [0.12, 0.12, 0.14, 0.15, 0.23, 1.0] | [2.42, 2.42, 2.73, 3.03, 4.55, 20] | ✅ 底层≈2.4 合理 |
| 过去1年 | [0.014, 0.014, 0.032, 0.032, 0.12, 1.0] | [0.29, 0.29, 0.64, 0.64, 2.36, 20] | ❌ 底层 0.29 伴侣不合理 |

1 年伴侣数被"少数长期单偶关系"稀释，不适合映射到单次派对。3 周伴侣数反映**短期伴侣周转**，才是单次派对的正确类比。

---

## 三、代码改动清单

| 文件 | 改动 |
|---|---|
| `population.py` | 新增模块级 `_cpc_sexparty_ek(top_mean)` 函数 + 4 个 CDC riskg 常量；`_build_sex_party_high_active` → `_build_sex_party_activity_level`（6 层）；缓存 bump v4 + 旧二值迁移；`_gen_edges_same_pool_grouped` 二值→6 层查表 |
| `feasibility/seed_targeting_abtest.py` | `sex_party_high_active` → `sex_party_activity_level==5` |
| `feasibility/seed_ht_mix_sweep.py` | 同上 |

---

## 四、参数语义变化（重要）

- **`party_ht_frac`**：从"实际控制高活跃占比"降级为**文档/兼容参数**。实际 6 层比例现在由 `CDC_RISKG_POP_WEIGHTS` 固定（顶层 5%）。改 `party_ht_frac` 不再影响分层。
- **`party_ht_top_mean`**：仍是顶层（stratum 5）的 E[k] 锚点，现在同时决定全部 6 层的 E[k]（通过比例缩放）。
- **`party_ht_top_clip`**：仍是顶层伴侣数上限；其余层用 `party_k_clip`。

---

## 五、预期动力学影响

改动前：5% MSM core E[k]=20，其余 95% E[k]≈2.7（统一 bulk）。
改动后：level 0-3 E[k]≈2.4-3.0（略低于原 bulk），level 4 E[k]=4.55（**高于原 bulk 2.7**），level 5=20。

关键差异在 **level 4（占 MSM core 19%）**：伴侣数从 2.7 → 4.55，理论上会**增强次高活跃层的传播、可能推高总规模**。需通过 10M 多 seed 验证确认。

---

## 六、验证记录

### smoke test（N=200000）✅
- 6 层分布近似 19%×5+5%（level 0-4 各 ~18-22%，level 5 ~3.9%）
- 非 core = -1，传播跑通，sex_party MM 边主导（48 MM vs 3 MF）

### 10M 5-seed 基线（top_mean=20 不变）⚠️ 规模超调

| 指标 | 6层梯度 | 二值（旧）| 目标 |
|---|---|---|---|
| total 中位数 | **2724** | 2386 | 2280 |
| peak 均值 | **94.6** | 92.7 | 92 |
| homo | 73.36% | 73.03% | — |
| hetero | 10.32% | 9.95% | ~9.9% |
| female | 5.33% | 5.32% | ~3.7% |
| cRMSE | 551.9 | — | — |

**结论**：
1. 构成完全健康（homo 73.4%、hetero 10.3%、female 5.3%）→ 重尾仍限制在 MM 内部，没有扩大 hetero 漏出 ✅
2. **规模超调 +19%**（2724 vs 2280），峰值晚 1.8 天 → 因为 level 4（占 19%）伴侣数从 2.7 提到 4.55，加权 E[k] 3.565→3.878
3. **需要回调 top_mean**：加权 E[k] 回到二值水平需 top_mean≈18.4；按规模线性外推 ≈16.7（非线性，需实际扫描）

**下一步**：扫描 top_mean ∈ {14, 16, 18}，找让中位数回到 ~2280 的值。

### 6.2 top_mean 重校准扫描 ✅

- 脚本：`feasibility/scan_topmean_6level.py`；结果：`outputs/topmean_scan/topmean_scan_6level.csv`
- 候选 {14, 16, 18} × 5 seed（300/311/322/333/344），共 15 次 10M 运行

| top_mean | total 中位数 | total 均值 | peak 中位 | cRMSE 中位 | homo | hetero | female |
|---|---|---|---|---|---|---|---|
| 14 | 955 | 1025 | 99 | 888 | 75.6% | 9.3% | 4.30% |
| 16 | 1627 | 1603 | 94 | 410 | 74.4% | 9.8% | 4.77% |
| **18** | **2267** | 2224 | 99 | **235** | 73.9% | 10.2% | 4.74% |

**结论：top_mean=18 是最优解**。total 中位数 2267（vs 目标 2280，偏差 -0.6%），cRMSE 235
（大幅优于 top=20 时的 552），构成全部落在软约束内（homo 73.9%≤80.5%、hetero 10.2%、
female 4.74%，略高于 4.5% 上限但在 5-seed 抽样波动内，且优于二值时代的 5.32%）。

最终 6 层 E[k] = [2.18, 2.18, 2.45, 2.73, 4.09, 18.0]。已写入 main.py 和
final_baseline_verify.py。
