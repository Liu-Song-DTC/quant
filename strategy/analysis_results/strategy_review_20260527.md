# D:\quant 量化选股系统 — 深度审查报告

**审查日期**: 2026-05-27
**审查范围**: 策略逻辑正确性、因子体系、风控、性能、代码质量
**审查方式**: 全量代码审计 + 运行时行为分析

---

## 一、总体评价

设计先进、实现严谨的 A 股多因子系统。多因子 + 缠论 + ML 三层融合 + EvolutionGuard 自进化 + 动态因子选择，在策略层面已具备实盘竞争力。主要问题集中在：**配置一致性、硬编码参数、缺少测试覆盖、性能还有优化空间**。

---

## 二、关键问题（按严重度排序）

### 🔴 严重

#### 1. `min_factor_families: 1` 实际禁用了因子家族分散化

**位置**: `factor_config.yaml:12` + `dynamic_factor_selector.py:234-253`

YAML 配置 `min_factor_families: 1`，但检查逻辑是 `if len(used_families) < min_factor_families`。由于至少会选到 1 个因子家族，这个条件**永远不会触发**。因子家族分散化机制完全失效。

**影响**: 动态因子选择可能在单一家族内选 3 个高相关因子（如 mom_20d、mom_60d、rel_strength 全来自 momentum），导致信号高度共线。

**修复**: 改为 `min_factor_families: 2`。

#### 2. `min_ic_1f` 是死代码

**位置**: `dynamic_factor_selector.py:107,325`

配置 `min_ic_1f: 0.03` 被加载到类属性 `self.min_ic_1f`，但在 IC 计算、因子过滤、因子选择的任何环节都**没有被引用**。完全无效的配置项，可能是重构遗留。

#### 3. 动态阈值在实盘和回测间存在差异

**位置**: `signal_engine.py:265-268`（刚优化）

混合模式下历史 bar 使用 `_calculate_default_factor`（MOM/REV/SHARPE），而回测全量模式使用 `_select_factor`（行业 + 动态因子）。两种路径产生的历史分数分布不同 → rolling quantile 阈值不同 → 最新 bar 的 buy/sell 判定可能不同。

**影响**: 实盘信号与回测信号在边界情况可能不一致。虽然验证测试通过了（score/buy/sell 相同），但这是单次测试，边缘情况（如 IC 刚过门槛的因子）可能有差异。

**建议**: 增加对比验证脚本，在回测数据上对两种模式跑批量对比，统计 buy/sell 一致率。

### 🟡 重要

#### 4. 70+ 硬编码魔数分布在 portfolio.py 中

**位置**: portfolio.py 全文

止损比例、冷却期、动量阈值、乖离阈值、排名权重、行业惩罚等 **70+ 个数值全部硬编码**在 Python 代码中，而非从 `factor_config.yaml` 读取。回测调参时需要改代码，容易出错且不利于 EvolutionGuard 自动优化。

**典型例子**:
- 均值回归退出冷却期: `14` 天 (line 293)
- 追高禁止 mom_60d 阈值: `0.50` (line 367)
- B3 跌幅拒绝阈值: `-0.03` (line 330)
- 行业惩罚: 医药 `-0.05`、化工 `-0.03` (lines 725-728)
- 时间止损按买点分层: `[(40, -0.08), (30, -0.05), (20, -0.05)]` (lines 1120-1124)

**建议**: 逐步迁移到配置文件，或至少集中到类的 constants 区。

#### 5. `max_single_weight` 配置不一致

**位置**: `factor_config.yaml:617` vs `:632`

- `portfolio.max_single_weight: 0.12`（全局权重上限）
- `portfolio.selection.max_single_weight: 0.18`（选股阶段权重上限）

两个值含义不同但名称相似，容易混淆。实际使用时 `selection.max_single_weight` 用于选股排名后的权重裁剪，`max_single_weight` 用于最终仓位上限。但两个值 12% vs 18% 差距较大，导致选股阶段可以配 18% 但最终被裁到 12%。

**建议**: 统一命名，或添加注释说明两者的关系和优先级。

#### 6. 监控数据不持久化

**位置**: `monitor.py`

`PipelineMonitor` 的所有数据（timings、counters、memory_peaks、IC history）纯内存存储。进程重启后全部丢失。

**影响**: 
- 无法追踪跨交易日的 IC 衰减趋势
- 实盘长期运行后无法复盘性能趋势
- 因子 IC 历史随进程重启而消失

**建议**: 每次实盘运行结束后将 `monitor.report()` 序列化为 JSON 追加到日志文件。

#### 7. 缺少单元测试

未找到 `tests/` 目录或任何测试文件。关键模块（因子计算、信号生成、组合构建）无测试覆盖。一个边界 bug（如 NaN 处理）可能在数千只股票中触发难以排查的错误。

#### 8. 时间止损分层逻辑硬编码且耦合买点类型

**位置**: `portfolio.py:1120-1124`

```python
{1: (40, -0.08), 2: (30, -0.05), 3: (20, -0.05)}
```

B1(抄底)拿 40 天、B3(趋势加速)拿 20 天，这个分层逻辑合理但参数硬编码。而且 B1/B2/B3 的判定依赖缠论买点质量，如果缠论买点识别有偏差，止损策略也跟着偏差。

### 🟢 建议

#### 9. `signal_engine.py` 2949 行，严重超长

一个文件包含：因子选择、缠论增强、分数装配、信号构造、基本面评分、风格评分。建议拆分为：
- `signal_engine.py` — 信号生成主流程
- `factor_selector.py` — 因子选择逻辑（合并 `_select_factor` 等）
- `signal_scorer.py` — 分数装配 + 动态阈值

#### 10. Chan 理论参数敏感度未验证

**位置**: `chan_theory.py` (2067 行)

缠论内部有大量硬编码阈值：
- B3 突破确认: `close > zg * 1.02` (2%)
- B2 搜索窗口: 5-50 bar
- 底分型质量权重: strength 25%, vol 10%, spike 15%, EMA 15%, divergence 20%, stroke 15%

这些参数对信号的敏感性从未经过系统验证（IC 分析、参数扫描）。报告中提到的 chan_buy_point IC≈0 值得警惕。

#### 11. 回测参数偏低

`slipage: 0.0001`（万 1）对 A 股小票偏低。A 股卖出印花税 0.05% 未在佣金中体现。建议：
- 佣金改为万 2.5 + 卖出时额外 0.05% 印花税
- 滑点按市值/流动性分档（大盘 0.0001，小盘 0.0005）

---

## 三、已确认的合理设计

以下设计经审查后确认为正确且有价值的：

| 设计 | 评价 |
|------|------|
| fork 前释放 factor_df/stock_data_dict 避免 COW 爆炸 | 关键的内存优化，不这样做会 OOM |
| walk-forward IC 验证 + 预计算缓存 | 避免过拟合，实盘用缓存 O(1) 查询 |
| 动态调仓周期 (牛市 30/中性 20/熊市 15) | 牛市捂股熊市灵活，逻辑正确 |
| 波动率缩放上限 0.75（只降不升） | 保守的风控设计，合理 |
| 组合止损 `emergency_exposure=0.30` | 回撤 8% 降到 30% 敞口，风控到位 |
| 缠论分级止盈 (S1 减半/S2 减半/S3 全清) | 避免一次性踏空，合理 |
| 均值回归退出 (mom_60d>0.60 / dist_ma60>0.50) | 防止高位接盘，阈值合理 |
| stock_pool 质量过滤 (ST/科创板/流动性) | 关键的风险前置过滤 |
| MultiTimeframeAnalyzer 向量化 (整数索引代替 strftime) | 周线/月线计算性能优秀 |

---

## 四、性能优化状态

| 阶段 | 优化前 | 优化后 | 方法 |
|------|--------|--------|------|
| 因子预计算 | 逐 bar 查 DB | 预计算缓存 O(1) | `precompute_all_factor_selections` |
| 信号生成(回测) | 逐 bar 循环 | 向量化装配 | `_vectorized_score_assembly` |
| 信号生成(实盘) | ~3.5h | **~2.5min** (82x) | `latest_only` 混合模式 (刚优化) |
| fork 前内存 | 3-5GB×workers | 释放后 fork | `del factor_df; gc.collect()` |
| 周线/月线 | strftime 字符串 | 整数编码 | `year*100+week` |

**剩余优化空间**:
- 实盘信号生成可进一步多进程并行 (factor_df 已释放，COW 可控)
- `_collect_bar_scalars` 混合模式中历史 bar 的 `_calculate_default_factor` 调用可进一步向量化

---

## 五、改进优先级

### 本周建议做

1. **修复 `min_factor_families: 2`** — 改一行配置，立刻生效
2. **增加 `latest_only` 对比验证** — 在回测数据上对比两种模式的 buy/sell 一致率
3. **监控数据持久化** — 实盘 report JSON 追加写入文件

### 本月建议做

4. **portfolio.py 参数外部化** — 至少把止损、时间止损、均值回归相关的硬编码值迁到 YAML
5. **缠论信号 IC 验证** — 对 chan_buy_score、chan_sell_score 做完整 IC 分析
6. **回测佣金修正** — 加入印花税

### 后续迭代

7. 拆分 `signal_engine.py`
8. 添加单元测试（至少覆盖 factor_calculator 和 signal_engine 的纯函数部分）
9. 缠论参数扫描/敏感性分析

---

*报告生成时间：2026-05-27*
