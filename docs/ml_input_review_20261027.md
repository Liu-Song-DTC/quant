# ML 输入与评分复核（2026-10-27）

## 当前真实构成

复核封存单位实验与统一评分实验的Parquet字段、配置及最后保存模型的元数据：ML白名单62项，实际基础特征30项；再生成28项两两交叉和10项市场状态交互，共68项。基础特征的生成顺序还决定前10项状态交互，不能将其理解为主动挑选的最佳10项。

白名单中32项没有出现在本次原始因子表，也未进入这两个最后保存模型：

```
fund_pe, volume_divergence, relative_volume_ratio, volume_trend,
overnight_gap_5d, amplitude_5d, amplitude_expansion,
limit_up_proximity, limit_down_proximity, volume_price_resonance,
volume_dry_up, dt_signal, northbound_signal, margin_signal,
profit_growth_accel, revenue_growth_accel, days_since_report,
earnings_yield, fund_cf_quality, debt_ratio, log_total_assets,
log_revenue, log_operating_cf, fund_roic, fund_net_margin,
vol_regime, turnover_regime, market_breadth, market_momentum_ratio,
market_volatility, market_dry_up_ratio, market_fear_index
```

有些名称在信号或组合路径有消费者，不代表它们是ML输入。不能宣称当前ML已经使用财报新鲜度和利润增长加速度，也不能直接把缺列填零冒充已实现。

`earnings_quality` 实际由换手率和价格波动率构造，未使用利润、应计或现金流。这是量价代理，不是财报盈利质量。名称和代码注释具有误导性；后续保留模型兼容性时应记录真实来源，不用改名掩盖信息缺失。

证据：`.review_runs/ml_feature_input_audit_20261027/report.json`。模型特征重要性仅对应最后保存窗口；不能拿它代表24个窗口，更不能把树模型gain当作策略收益归因。

## 评分实验与验证IC

单位实验24个滚动窗口平均日志验证IC为0.091692，统一评分实验为0.091204；最大单窗口变化0.0046。日志保留4位小数，这些统计也仅具有相应精度。IC相近不足以隔离ML和评分装配的各自作用。

统一评分臂完整回测，输入内容前后不变：最终权益861,569.19元，最大回撤16.98%，Sharpe1.3485，净成交胜率51.23%，3/5日净标记胜率45.06%/40.12%。较原基线回撤改善，但收益和整体短期胜率未改善。2026年5日胜率56.52%不能替代全期结果；这段历史也不是未使用过的样本外。

增长压缩独立臂已从相同封存源准备；它保留统一评分臂设置，仅对利润/营收增长及边际变化按百分点转换成比例后压缩。完整结果未出，不视作通过。所有这类行为改动尚未推送到主干。

## 已实现的可观测性修复

训练时记录配置特征、数值型可用特征、缺列、非数值列及生成但未请求的列；缺列不再完全静默。成功训练后补充实际训练列。模型元数据保存这份记录及验证摘要；载入旧模型会明确标记历史记录不可得，不伪造证据。

没有改变模型特征选择、标签、训练权重、树参数或预测公式。输入不变及模型保存/载入后的预测逐值一致测试通过；兼容旧元数据。加上验证标签隔离和缺失指示位回归，共4项测试通过。

后续按以下依赖推进：先核实每项因子的真实计算及可用时点，补齐少量具有可靠数据的基础特征，做独立的滚动增量检验；再比较短期净收益目标、ML与线性评分的配比及评分门槛。避免把补32列、改标签和换权重打成一个无法归因的实验。
