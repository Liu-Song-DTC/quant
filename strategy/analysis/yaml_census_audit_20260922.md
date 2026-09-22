# yaml×census 全量对账 + 评分配置路径审计 (2026-09-22)

用户9/18指令的架构审查落地件: "目前架构是否合理, 有没有优化的地方"。
方法: 36个零文档参数逐键追reader → 分组裁决; 评分配置消费链逐行追读 →
验证CSV+factor_df离线探针裁决。**零生产改动, fp 6f1cb6b1|0 未动。**

## 1. 36参数分组裁决

| 组 | 参数 | 裁决 | 证据 |
|---|---|---|---|
| A 死yaml键 | chan_gate_enabled, chan_gate_min_count, buy_sell_points, divergence_threshold, consolidation_discount, alignment_boost, top_divergence_exit, volume_divergence_exit, zhongyin_penalty | 零reader, yaml在fp内**不可删**, 仅文档化 | 全仓grep零命中 |
| A' 双源镜像 | `indicator_params`整节 | factor_preparer走硬编码`get_default_params()`(factor_calculator.py:1085), signal_engine:324读yaml — **值逐位一致**, 双源真相=架构债, 当前零影响 | 两处值比对全等 |
| B 死旋钮 | ti_boost_scale(signal_engine:296), chan_bonus_buy_point(portfolio:290), chan_bonus_sl2(289), fv_low/fv_high/fv_exposure_min/fv_exposure_max(portfolio:187-190), use_static_candidates(selector:408), evolution_guard模块 | 赋值后零读; 全并入census死旋钮名录 | 逐键read trace |
| C 机制关闭 | dynamic_factor整节(train_window/forward_period/top_n_factors/min_ic_dates/ic_decay_factor/min_train_samples/min_factor_count/min_factor_families/extra_candidate_factors) | factor_mode=fixed → enabled=False, 生产日志 DYN命中 0/6,658,619 | 生产日志line 63 |
| D 实践惰性 | composite_bias(IND_信号=0), V41(311,340行0买卖), volatility_adaptive_mult(回测cost={}), 退出栈全套(exit_mode='simple'三层闸) | census已裁决或本审计量化 | census文档 |
| E 实活已覆盖 | entry_chan_gate(E-N5), chan_div两阈值(sell侧66事件量化), mtf折扣(机制级), sell_threshold(末班8+11笔), indicator_params(见A') | 全部已有bracket/量化记录 | census+执行层末班 |
| F 实活+本审计新量化 | **全局industry_factors(377概念)** vs **季度标定(401概念)** | 见§2-4: 现状架构经IC探针裁决为**合理** | 本审计探针 |

## 2. 评分配置消费链 (fixed模式全链)

`_select_factor` → fixed模式 → `_calculate_default_factor`:
- **P0**: `_get_specific_industry`(概念解析: STOCK_CONCEPT_MAP风格过滤 + inception PIT gate + 季度配置成员资格) → 熊/弱趋势(regime==-1或trend<-0.05)读**季度bear_factors**; 中性/牛市读**全局factors/bull_factors**。
- **P1**: 仅当P0未命中 — `FactorLibrary.get_scoring_factors(fallback_config=季度配置)` → 季度factors(权重归一)。
- **P2**: IC磁盘缓存 — 生产无ic_cache_*.pkl文件 → 恒空。
- **P3**: 硬编码4因子(trend_lowvol/relative_strength/low_downside/momentum_reversal)。

验证CSV(703,516 buy)路径实测: **仅全局43.7% / 仅季度35.9% / 两者皆可8.7% / 都不含11.8%**, 真P3兜底仅0.6%。

## 3. 关键证伪: "预计算浪费算力"不成立

前一分析段曾认为 bt_execution.py:919 无条件跑IC预计算(固定模式纯浪费)。
**证伪**: 该调用位于 `if use_dynamic:` 块内(bt_execution.py:874, use_dynamic = factor_mode != 'fixed');
live_init.py:94同样门控`factor_mode != 'fixed'`。生产日志0处预计算进度标记+"因子选择分布(总计1)"
来自backtest_diagnostics记录器而非预计算。**代码早已正确门控, 无修复必要。**

## 4. 核心裁决: 全局权重 vs 季度权重 IC探针 (probe_qweight_coverage_20260922.py)

问题: 全局配置自2026-03-20未实质更新(git -S), 365交集概念的中性/牛市评分跑旧权重 —
是否应把P0切到季度权重?

方法: factor_df缓存(3.5M行, 生产评分同源rank值)上复刻生产概念解析(风格过滤+inception gate+
季度成员资格), 对可切换集逐日Spearman IC(score vs future_ret), 687交易日配对。

| 权重 | 逐日IC mean | t | 正比例 |
|---|---|---|---|
| 全局 factors | **+0.0464** | 16.6 | 73.9% |
| 季度 factors | +0.0290 | 10.1 | 65.1% |
| 配对差 季度−全局 | **−0.0175** | **−5.4** | 40.8% |

分年配对差: 2021 −0.0285 / 2022 −0.0284 / 2023 −0.0051 / 2024 −0.0179 / 2025 −0.0148 / **2026 −0.0063**。
诚实注: 2021-25全局占优部分因hindsight-fit(全局配置在2026-03生成, 携带全窗口知识);
但2026是干净窗口(两配置都先于2026固定) — 季度权重仍输。

**裁决: P0中性/牛市切季度权重 = 负期望, 方向关闭。全局旧权重不是缺陷, 是特性。**
(呼应全程序一贯证据: 简单老机制胜过滚动重标定 — 与F-2/F-3/E-seq1同签名。)

## 5. 季度标定的真实价值 = 新概念覆盖

仅季度概念(36个存量+2026中报类新概念)上, 季度权重 vs P3硬编码默认:
**季度 +0.2423 (t=7.25, 95.7%正) vs 默认 +0.0089 (t=0.22), 配对差 +0.2334 (t=5.07)** —
窗口=2026年23个交易日, 对2026Q3文件为**OOS**(标定至6/30, 测试7/1~9/16)。

结论: 季度标定通道的真实增益 = **新概念权重 + 熊分支bear_factors + 概念解析gating**
(消解E-K1悖论: 冷跑+60,652的增益来自这三通道, 而非存量概念权重刷新)。
**9/30 Q4标定执行建议: 焦点=全局缺失的新概念; 存量交集概念的中性/牛市权重重标定
不改善评分(甚至不被消费) — 不必花力气。**

## 6. 架构合理性结论 + 遗留修复候选 (全部零行为影响)

架构裁决: **合理**。评分配置双层(全局静态+季度滚动)是自洽设计; 全局层扛中性/牛市、
季度层覆盖新概念与熊分支, 经IC探针验证优于反事实替代。

9/30窗口(与A.5印花税共用一次fp重锚)可选修复:
1. **季度诊断可观测性**: `_QUARTER_DIAG`计数在多进程worker内累积后随进程消亡,
   主进程bt_execution.py:1170打印恒为0 — 生产日志从未出现"行业名查找"行。
   fix: worker结束时聚合diag传回主进程(或spawn前存文件)。零行为影响。
2. **死键/死旋钮清理**: A/B组(约20个)注释归档 — 已文档化, 删除需在fp豁免节外,
   不删yaml键(指纹内节)。
3. **双源参数合一**: indicator_params删yaml节或删硬编码, 二选一。零行为影响。
4. 生产日志"固定行业因子49.5%/固定默认因子50.5%"是计数器语义artifact
   (fixed_default=入口计数, fixed_industry=P0命中), **非覆盖度指标** — 真实覆盖度
   见§2(仅0.6%真兜底)。注释澄清即可。

**不排新alpha臂** — 与census终态一致, 本审计方向探针(季度权重切换)是最后一个
疑似headroom, 已量化为负期望关闭。
