# 7/24 调优计划

## 当前状态
- 代码: f3e01b8 + 5ce59a2(止损不止盈) + bear_risk_fast bug修复 + peak_trail市场感知
- 结果: 总收益 287%, Sharpe 1.34, 2022/2023亏损(-4%/-16%)
- 目标: 年均40%+, 所有年份正收益

## Step 1: 跑全量诊断回测
```bash
cd /mnt/d/quant/strategy
rm -f rolling_validation_results/backtest_signals.csv
python bt_execution.py
```
等待约2h, 关注输出中的 `[portfolio DEBUG]` 段。

## Step 2: 分析12项诊断数据

### A. 熊市触发频率
- bear_risk(空仓)触发天数 — 如果太少, 需要降低触发阈值
- bear_risk_fast(预警)触发天数 — 当前仅1只仓位, 是否过紧

### B. 持仓效率
- 各regime平均持仓数/敞口
- 资金利用率: 为什么常选2只不满仓? (no_candidates / cash_tight / gate_filtered)

### C. 退出机制
- 退出原因分布: cost_stop / peak_trail / sig_sell / factor_neg 占比
- peak_trail在熊市是否有触发记录

### D. 买入质量
- B0/B4/B5 分布 per regime — B0占比是否熊市更高?
- 入选score分布 per regime — 熊市选股质量是否明显差

### E. 门槛效果
- 各regime的score_cut / rank_cut
- 被过滤股票score分布 — max(被拒) vs mean(入选) 的差距
- 如果差距小 → 需要提高门槛

### F. 因子 & BOM
- 熊市用了什么因子(top5) — 是否激活了bear_factors
- BOM产业链命中率 — 是否缺失产业链定义

### G. ML贡献
- 候选ML得分 per regime — 熊市ML得分是否偏低

## Step 3: 根据诊断决定方向

| 发现问题 | 解决方向 |
|----------|---------|
| bear_risk触发太少 | 降低market_regime_detector阈值 |
| 熊市B0占比过高 | 提高min_score门槛, 或收紧buy_threshold |
| 熊市score明显低 | bear_factors未生效, 需检查signal_engine激活条件 |
| 资金利用率低 | 放宽candidate池, 或降低eff_min_rank |
| BOM命中率低 | 补充缺失的产业链定义 |
| 当前门槛有效(差距大) | 保持, 继续观察 |

## Step 4: 验证改动
- 每次只改一个方向, 跑回测验证
- 对比基准: 287% / Sharpe 1.34
- 通过标准: 2022/2023改善且总收益不降

## Step 5: portfolio_selections.csv后分析
- B0 vs B4 vs B5 买入后 forward return 对比
- score分位数 vs forward return 相关性
- 找出真正有效的选股信号特征
