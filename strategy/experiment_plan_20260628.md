# 实验计划 2026-06-28

## 今日完成

- [x] 去 regime 离散分类, 参数统一
- [x] 因子选择: per-industry IC → 全局 IC
- [x] 参数调优: conf=0.80, abs_score=0.0, max_pos=6, buy_pct=0.45
- [x] 产业链聚焦 (主导行业 → 链内选股)
- [x] 全链路日志 pipeline_logger
- [x] Bug 修复: Gate门槛/ADV前视/重复条件/_rank排除/死代码
- [x] 实盘初始化 live_init.py (与回测等价)

## 回测状态

回测未完成 — 在信号生成阶段被 OOM kill (exit 137)。
上次成功日志摘要 (21:52):

```
股票池: 2812 只 | 科创板: 保留 | ST: 逐日判断
因子数据: 1258 时间点 × 2812 只 → 2,172,418 行
IC缓存: HIT, 跳过计算, 直接加载 1168 个日期结果
ML: xgboost未安装, 跳过
信号生成: 4 workers, 2812 只股票
```

OOM 发生在多进程信号生成阶段。可能原因: 4 worker × 全局 pivot 内存累积。

## 明天计划

### Step 0: 解决 OOM

- 降 workers: NUM_WORKERS=2
- 或: precompute_all_factor_selections 完成后释放 factor_df (已做, 确认有效)
- 或: 增加 swap / 减少日期范围加速测试

### Step 1: Exp1 新基线 (全流程 ×1)

```
目的: 所有改动合并后的基准
配置: 当前 YAML
关注: Sharpe, MaxDD, 年收益, plog报告
```

### Step 2: 组合参数扫描 (复用 Step1 信号)

```bash
python analysis/param_sweep.py --signals rolling_validation_results/backtest_signals.csv
```

| 参数 | 测试值 |
|------|--------|
| max_positions | 4, 5, 6, 8 |
| min_confidence | 0.70, 0.75, 0.80, 0.85, 0.90 |
| min_absolute_score | -0.05, 0.0, 0.05 |
| position_stop_loss | 0.07, 0.08, 0.10, 0.12 |
| portfolio_stop_loss | 0.08, 0.10, 0.12 |

### Step 3: 信号级参数 (需重跑全流程)

| Exp | buy_threshold_pct |
|-----|-------------------|
| 3A | 0.35 |
| 3B | 0.45 (已跑) |
| 3C | 0.55 |

### Step 4: 产业链开关

| Exp | 产业链 |
|-----|--------|
| 4A | ON (已跑) |
| 4B | OFF (注释 portfolio.py 链过滤) |

### 分析任务

- [ ] plog.report() — 因子归因/门控拒绝率/退出原因
- [ ] param_sweep_results.csv — 最优参数组合
- [ ] 产业链命中率统计
- [ ] 数据质量问题 (log 中 data_quality stage)

## 待提交

- [ ] live_init.py + extend_to_date (已在本地, 未推送)
- [ ] param_sweep.py (本地, 未推送)
