# IDM 项目 Codex 交接文档

更新时间：2026-10-12（Asia/Shanghai）

## 1. 项目位置与当前状态

- 项目根目录：`D:\personal\OneDrive\Paper\IDM`
- 当前分支：`master`
- Git 工作区：干净，无未提交变更
- 注意：仓库当前没有正式 commit，不能依赖 commit hash 判断代码版本；请使用下方 SHA-256 指纹。
- 当前主模型：final E baseline，已完成规模和病例构成校准。
- 当前目标不是重新校准传播模型，而是在冻结 final E 的前提下验证逐日轨迹与外部输入病例时序。
- `main.py` 当前 `MAKE_PLOTS = True`，没有遗留批处理临时修改。

核心文件指纹：

| 文件 | SHA-256 |
|---|---|
| `main.py` | `758F2FC591A1064FED1AFC5F7418155913E4505B599A9B76C728AB0B7617A048` |
| `population.py` | `5C4CC165035A103143607A7C8787363BC7193081911FB33EE276B9AE39ED9400` |
| `run_v3_final_E_100seed_validation.py` | `CA14CF1956F2E22DD0A8E297AB5E3B3E4BB099B4D948FEFE53C9861F24B88DF3` |

新账号开始前建议运行：

```powershell
Get-FileHash -Algorithm SHA256 `
  'D:\personal\OneDrive\Paper\IDM\main.py', `
  'D:\personal\OneDrive\Paper\IDM\population.py', `
  'D:\personal\OneDrive\Paper\IDM\run_v3_final_E_100seed_validation.py'

git -C 'D:\personal\OneDrive\Paper\IDM' status --short
```

## 2. 冻结的 final E 参数

以下参数是当前 final E baseline，不要在外部输入病例时序诊断中改变：

```text
p_bi_m_to_msm              = 0.70
party_core_intensity       = 0.70
mix_mf_anal_frac           = 0.13
stable_mf_anal_frac        = 0.00
party_mix_homo_spill_frac  = 0.01
sexual_anal_receptive      = 0.2800
sexual_anal_insertive      = 0.0600
sexual_vaginal             = 0.0350
sexual_oral                = 0.0175
social_contact_scale       = 10.0
social_same_sex_assort     = 0.90
community_bridge_scale     = 1.5
```

同时冻结：疾病自然史、检测、监测、疫苗、易感性、政策时间、close-touch、event、人口学和网络混合参数。

运行设置：

```text
POPULATION_SIZE = 10,000,000
SIM_DAYS        = 244
模型 day 0      = 2022-05-01
常用验证 seeds  = 100000–100004 或 100000–100019
```

重要实现细节：`Population.run_simulation(244)` 实际执行 day 0–243；数组长度为 245，因此导出中可以存在未执行 step 的 day 244 尾部槽位。该点已审计，不是日期错位的原因。

## 3. 当前外部输入病例机制

`main.py` 当前计划：

```python
importations = [
    dict(start=0, end=30, daily_n=2),
    dict(start=31, end=70, daily_n=4),
    dict(start=71, end=110, daily_n=9),
    dict(start=111, end=150, daily_n=4),
    dict(start=151, end=SIM_DAYS, daily_n=0),
]
```

`population.py::_apply_importations()` 使用 `start <= d <= end`，即首尾日都包含。计划总量：

```text
31×2 + 40×4 + 40×9 + 40×4 = 742
```

机制说明：

- 输入对象从当天仍易感且符合 `importation_mask` 的人中无放回抽取。
- 当前场景为 `homo_male_mcp`，输入对象被该 mask 限定。
- 抽中者经 `_expose_batch()` 进入暴露状态，并非当天确诊病例。
- 后续仍经历潜伏、传染、症状和检测过程。
- `pop.seed_infection(...)` 当前已注释，疫情主要由持续外部输入启动。
- 外部输入应解释为不确定的时间强迫过程，不是对真实输入病例数的估计。

## 4. 观察数据与目标

`population.py` 中的 `Case_data` 是 LAC dashboard 按发病日期统计的逐日病例序列。

- `Case_data[0]`：2022-05-01
- 首个非零病例：day 18，即 2022-05-19
- 全序列长度：460 天，至 2023-08-03
- 当前模拟比较使用前 245 个日期槽位
- 观察峰值：2022-08-01，58 例/日
- 未发现观察序列与模拟日期轴之间的意外偏移

终点目标文件：

`outputs\final_figures_editable_package\final_E_targets.csv`

```text
observed total reported cases = 2280
known orientation denominator = 2012
heterosexual                  = 0.0989065606
implied male heterosexual     = 0.0621272366
female known                  = 0.0367793241
bisexual                      = 0.1257455268
homosexual                    = 0.7753479125
```

## 5. 已完成：100-seed 逐日轨迹验证

目录：

`outputs\v3_final_E_daily_trajectory_validation`

关键文件：

- `daily_trajectory_long.csv`：全部 seed 的逐日长表
- `daily_trajectory_summary_by_day.csv`：逐日均值、SD、2.5% 和 97.5% 分位数
- `daily_trajectory_run_summary.csv`：运行状态
- `per_seed_daily\`：逐 seed 轨迹
- `make_daily_trajectory_figures.py`：当前绘图脚本
- `figures\daily_trajectory_validation_summary.png`
- `figures\daily_trajectory_validation_summary.pdf`

当前图形约定：

- 不显示总标题和描述性子标题，仅显示 `a)`、`b)`、`c)`、`d)`。
- a：每日发病日期病例数。
- b：累计发病日期病例数。
- c：累计病例性别构成。
- d：累计病例性取向构成。
- 主模型线使用 100 个 seed 的中位数，阴影为 2.5%–97.5% 分位区间。
- 每日/累计病例面板叠加真实 `Case_data`。
- 构成面板叠加观察终点比例虚线。
- 累计病例不足 25 时隐藏构成比例，避免小分母跳变。

建议中文图注：

> 图 X 模型逐日轨迹验证。a) 每日发病日期病例数；b) 累计发病日期病例数；c) 累计病例的性别构成；d) 累计病例的性取向构成。红色实线表示100个随机种子模拟结果的中位数，粉色阴影表示2.5%至97.5%分位区间。黑色实线表示洛杉矶县监测数据中按发病日期统计的每日病例序列，黑色虚线表示累计报告病例终点目标。c) 和 d) 中的彩色虚线表示观察数据中的终点构成比例。

## 6. 已完成：导入时序筛选和 20-seed 验证

目录：

`outputs\v3_final_E_importation_timing_reconciliation`

该批次已完成，`COMPLETED.txt` 存在，失败 seed 为 0。

关键审计：

- `timing_alignment_audit.md`
- 日期轴没有发现 bug。
- 5-seed 基线中位峰值为 2022-08-04，较观察峰值晚 3 天。

5-seed screen 测试了 baseline 加 6 个时序候选。随后仅验证：

- `candidate_0`：Current final E baseline
- `candidate_4`：Lower early, moved to rising phase

20-seed 结果：

| 指标 | baseline | candidate_4 |
|---|---:|---:|
| 中位峰值日期 | 2022-08-04 | 2022-08-04 |
| 峰值日期误差 | +3 天 | +3 天 |
| 每日 RMSE | 6.0562 | 5.7031 |
| 累计 RMSE | 230.3420 | 243.8613 |
| 总病例均值 | 2311.85 | 1854.55 |
| 总病例 SD | 271.39 | 245.79 |
| 构成 MAE_pp | 0.7198 | 1.2482 |

结论：`candidate_4` 虽略微改善每日 RMSE，但没有改善峰值日期，且明显低估终点规模、累计 RMSE更差、构成误差超过 1.2 pp，因此不能替代 baseline。

主要文件：

- `importation_timing_screen_results.csv`
- `importation_timing_screen_summary.csv`
- `importation_timing_validation20_results.csv`
- `importation_timing_validation20_summary.csv`
- `summary.csv`
- `candidate_importation_schedules.json`
- `run_importation_timing_reconciliation.py`
- `logs\`
- `per_seed_daily\`

## 7. 已完成：整体提前 4/7 天诊断

目录：

`outputs\v3_final_E_importation_earlier_shift_diagnostic`

该批次已完成 15/15 个 seed，失败为 0，`COMPLETED.txt` 存在。每个方案使用 seeds 100000–100004。

| 方案 | 窗口内输入总数 | 峰值误差 | 每日 RMSE | 累计 RMSE | 总病例均值 | MAE_pp |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 742 | +3 天 | 6.4845 | 344.2971 | 2441.0 | 0.5889 |
| shift -4 days | 734 | +6 天 | 6.4930 | 293.9101 | 2285.8 | 0.9678 |
| shift -7 days | 728 | +5 天 | 6.8136 | 262.9974 | 2401.6 | 0.6030 |

结论：提前 4 天或 7 天都没有改善峰值日期误差，均不是真正优于 baseline。由于提前到 day 0 之前的输入被截断，两者总量分别降至 734 和 728，这也是下一项“精确 742 重分配”任务的动机。

主要文件：

- `earlier_shift_results.csv`
- `earlier_shift_summary.csv`
- `earlier_shift_README.md`
- `candidate_schedules.json`
- `run_earlier_shift_diagnostic.py`
- `logs\`
- `diagnostics\selection_diagnostics.csv`

## 8. 尚未执行：精确 742 次形状重分配

预定输出目录当前不存在：

`outputs\v3_final_E_importation_shape_redistribution`

这是新账号最自然的下一步。要求只改变外部输入的时间形状，每个候选的计划输入总数必须严格等于 742。

候选：

```text
B0 baseline
day 0–30: 2/day
day 31–70: 4/day
day 71–110: 9/day
day 111–150: 4/day
day 151–244: 0/day

R1 toward June
day 0–30: 1/day
day 31–61: 5/day
day 62–70: 4/day
day 71–110: 9/day
day 111–150: 4/day
day 151–244: 0/day

R2 toward July
day 0–30: 1/day
day 31–60: 4/day
day 61–70: 5/day
day 71–91: 10/day
day 92–110: 9/day
day 111–150: 4/day
day 151–244: 0/day

R3 split June/July
day 0–30: 1/day
day 31–45: 5/day
day 46–60: 4/day
day 61–70: 5/day
day 71–76: 10/day
day 77–110: 9/day
day 111–150: 4/day
day 151–244: 0/day

R4 stronger delayed redistribution
day 0–14: 0/day
day 15–30: 1/day
day 31–44: 4/day
day 45–60: 5/day
day 61–70: 5/day
day 71–90: 10/day
day 91–110: 9/day
day 111–150: 4/day
day 151–244: 0/day
```

必须在运行前按 `start <= d <= end` 逐块计算总数；任一候选不等于 742 时禁止运行。

第一阶段仅运行 5-seed screen：10M population，seeds 100000–100004。兼容性验证通过时复用已有 `candidate_0` baseline，不要重跑。

评价重点：

- 2022-05-01 至 2022-06-30 的 daily MAE/RMSE。
- 2022-05-19 至 2022-06-30 的 daily MAE/RMSE。
- 截至 6月30日、7月15日、7月31日的 cumulative MAE/RMSE。
- 6月1日、6月15日、7月1日、7月15日、8月1日的观察/模拟累计病例。
- 达到终点目标 5%、10%、25%、50% 的日期及模拟减观察差值。
- 峰值、全轨迹、最终规模和构成指标。

选择规则：不使用加权总分。分别对早期轨迹 RMSE、累计 RMSE、峰值误差、终点规模误差和构成 MAE 排名。候选只有在早期累计 RMSE有实质改善、早期 daily RMSE不明显变差、峰值不明显恶化、全程 daily RMSE不超过 baseline 约 2%、终点规模在 2280 的 ±10%、构成仍合格时，才可推荐。

不要自动运行 20-seed。只有 5-seed screen 明确显示某一候选更好，才选择最多一个候选，并在用户明确同意后运行 seeds 100000–100019；baseline 20-seed 应优先复用已有结果。

预定输出：

```text
candidate_schedule_manifest.csv
candidate_schedules.json
shape_screen_results.csv
shape_screen_summary.csv
shape_screen_rankings.csv
early_trajectory_metrics.csv
shape_redistribution_README.md
comparison_figure.png
comparison_figure.pdf
logs\
diagnostics\
```

## 9. 批处理约束与可靠运行方式

- 不要永久修改 `main.py` 或 `population.py`。
- 推荐读取 `main.py` 原文，为每个 seed 生成隔离的临时脚本，并通过 `PYTHONPATH` 指向项目根目录。
- 临时脚本内设置 `MAKE_PLOTS=False`；项目 `main.py` 始终保持 `MAKE_PLOTS=True`。
- 每个 seed 使用独立 daily CSV、stdout 和 stderr。
- 已完整落盘的 seed 必须跳过；完整性检查至少应确认 245 行且最后一行 `day == 244`。
- 中断后只补跑缺失或不完整 seed。
- 机器有 32 个逻辑处理器、约 31.6 GB RAM。10M 模拟单进程约占 1.7–1.9 GB；建议最多 5–6 个 worker。
- 单个 10M seed 通常约 5–8 分钟，取决于并行负载。
- 后台运行建议用 PowerShell `Start-Process -WindowStyle Hidden`，并把控制器 stdout/stderr 重定向至输出目录。
- 批次结束后写 `COMPLETED.txt`、completed/failed seed 清单。
- 批次执行期间只报告计算状态，不做持续科学解释，也不要中途调参。

可复用实现参考：

- `outputs\v3_final_E_importation_timing_reconciliation\run_importation_timing_reconciliation.py`
- `outputs\v3_final_E_importation_earlier_shift_diagnostic\run_earlier_shift_diagnostic.py`

## 10. 给新账号的建议首条消息

可直接粘贴：

> 请先阅读 `D:\personal\OneDrive\Paper\IDM\CODEX_PROJECT_HANDOFF.md`，并只读核对其中三个核心文件的 SHA-256、`main.py` 的 final E 参数、`MAKE_PLOTS=True`、基线外部输入总数 742，以及现有批次的 `COMPLETED.txt`。不要重新校准模型，不要永久修改 `main.py` 或 `population.py`。下一步是执行文档第 8 节的“精确742次外部输入形状重分配”5-seed screen；先生成并验证 candidate manifest，所有候选必须严格等于742，兼容时复用 baseline。不要自动运行20-seed验证。

## 11. 最简项目结论

final E baseline 已能较好再现终点规模和病例构成；20-seed baseline 总病例均值 2311.85，构成 MAE约 0.72 个百分点。模型峰值相对观察约晚 3 天，而不是早。整体平移输入计划未带来可靠改善。下一步应保持 742 次输入总量不变，仅把早期 May 输入重分配到 June/July，重点检验早期累计轨迹，而不是再次调传播或混合参数。
