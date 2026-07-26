# 量化交易系统

A股多因子量化交易系统，融合缠论技术分析 + 行业级因子配置 + 宏观周期叠加 + ML预测增强。基于 Backtrader 回测框架。

**当前最优回测**: 总收益 1098%, Sharpe 1.87 (2021-2026), 5/6年正收益。

## 快速开始

```bash
# 实盘选股 (自动更新数据→生成信号→产出订单)
python strategy/run_live.py

# 单独回测
python strategy/bt_execution.py

# 更新数据
python data/data_manager.py
```

## 系统架构

```
strategy/
  core/                    # 核心引擎
    factor_calculator.py   # 因子计算 (80+ 技术指标, 统一数据源)
    signal_engine.py       # 信号生成 (4阶段向量化管线)
    portfolio.py           # 组合构建 + 风控 (成本止损/峰值回撤/CLB/HDS)
    factor_preparer.py     # 离线因子预计算
    dynamic_factor_selector.py  # Walk-forward IC 动态因子选择
    factor_library.py      # 因子质量时序追踪
    gate_scorer.py         # 4Gate 质量评分系统
    chan_theory.py         # 缠论 (笔/段/中枢/买卖点)
    market_regime_detector.py   # 市场状态检测 (牛/熊/震荡)
    macro_data.py          # 宏观数据 (M1/社融/PPI/Fed利率)
    industry_chain.py      # 产业链拓扑 (220链+127NO_CHAIN, 100%覆盖)
    bom_chain.py           # BOM 产业链质量分析
    concept_heat.py        # 概念板块热度
    ml_predictor.py        # XGBoost ML预测 (季度模型)
    alternative_data.py    # 另类数据 (龙虎榜/北向/融资)
    multi_timeframe.py     # 多周期分析 (周/月)
    config_loader.py       # 配置加载器
  config/
    factor_config.yaml     # 主配置文件 (所有参数+200+行业因子)
  analysis/
    offline_calibration.py # 离线因子标定
    analysis_framework.py  # 统一分析框架 (IC/IR/准确率/因子衰减)
  models/                  # XGBoost 季度模型
  logs/                    # 运行时日志
data/
  data_manager.py          # 数据管理 (行情/基本面/概念)
  stock_data/              # 股票数据
  macro/                   # 宏观数据缓存 (M1/社融/PPI/Fed)
trade_orders.json          # 实盘订单输出
current_positions.json     # 持仓快照
```

## 数据流

```
股票数据 → factor_calculator (80+因子)
         → signal_engine (因子×gate_quality + 基本面 + ML + BOM)
         → SignalStore (缓存)
         → portfolio (候选筛选 → 截面排名 → 仓位分配)
         → bt_execution (回测) / run_live (实盘)
         → trade_orders.json (实盘输出)
```

## 核心模块说明

### 因子计算 (factor_calculator.py)

统一因子计算源，供信号引擎和预计算共用。产出超过 80 个技术指标和复合因子，分为 7 大家族：

| 家族 | 示例因子 |
|------|----------|
| 动量 | momentum_reversal, trend_lowvol, mom_x_lowvol |
| 低波 | volatility, low_downside, inv_turnover |
| 价值 | fund_pe, fund_pb, fund_roe, fund_score |
| 质量 | turnover_stability, consolidation_breakout |
| Alpha | overnight_ret, residual_momentum |
| 量价 | wash_sale_score, volume_surge, short_reversal |
| 另类 | 北向资金, 融资融券, 龙虎榜 |

### 信号引擎 (signal_engine.py)

4阶段向量化管线：
1. 指标计算 (factor_calculator)
2. 逐Bar标量收集 (BOM, 缠论, MTF, 资金流)
3. 向量化分数装配 (factor × gate_quality + 基本面 + ML + BOM)
4. 动态阈值 + 买卖判定 (4Gate系统)

因子模式 `fixed`：使用 `industry_factors` 中 200+ 行业的手配因子（带 IC 值和权重），由离线标定生成。

### 组合构建 (portfolio.py)

**组合构造** (`build` → `_build_desired_value`):
1. 候选收集 (信号过滤 + 价格约束)
2. 产业链聚焦 (主导行业→关联概念)
3. 截面排名 (Quantile→Normal变换, 拉大尾部差距)
4. 行业内排名 + Gate质量 + 缠论结构调整
5. 权重分配 (等权 Top-N)

**风控机制**:
- 成本止损 (cost_stop, 实盘中用 `cost={}` 屏蔽)
- 峰值回撤 (peak_trail, 仅 BEAR 期)
- 连亏中断 (CLB, 4连亏→降至25%敞口)
- 硬回撤止损 (HDS, 组合回撤超限→清仓)
- 均值回归冷却 (MR cooldown)

**宏观叠加**: M1/社融/两融数据调节 FAST↔NORM，不干预 BEAR。

**连熊降级**: BEAR 连 60 天→FAST，防止 V 反行情踏空。

**FAST 动量反转**: 惩罚高动量（-abs(mom_60d)×0.10），偏好低波动稳定股。

**调仓频率**: NORM 10 天，不触发个股止损（成本止损已屏蔽），退出完全依赖调仓+熊市清仓。

### 市场状态 (market_regime_detector.py)

三个级别：
- **BEAR** (空仓): 价格+趋势确认，545天/年(42%)
- **FAST** (预警): 1只仓位严格过滤，157天/年(12%)
- **NORM** (正常): 5只满仓，605天/年(46%)

### 产业链 (industry_chain.py)

220 个链概念 + 127 NO_CHAIN 概念 = **100% 回测数据覆盖**。12 条产业链：AI/半导体/新能源车/光伏储能/机器人/低空经济/医药/军工/数字经济/消费/资源材料/基建海洋。

每链定义上下游环节，实时计算传导信号（已涨环节→未涨下游→加分）。

### 宏观数据 (macro_data.py)

内盘: M1/M2, 社融, 两融余额, PPI
外盘: Fed 利率

信号逻辑 (AND条件):
- Bull: M1加速 + 社融改善 + 两融不跌
- Bear: M1加速跌 + 社融不改善

## 回测性能

| 版本 | 总收益 | Sharpe | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|------|--------|--------|------|------|------|------|------|------|
| 起点 | 390% | 1.54 | 24% | -5% | -2% | 37% | 140% | 28% |
| 产业链修正 | 393% | 1.55 | 24% | -5% | -2% | 36% | 142% | 28% |
| macro框架 | 633% | 1.66 | 71% | +11% | -13% | 32% | 160% | 31% |
| 连熊降级 | 1047% | 1.85 | 71% | +59% | -4% | 44% | 140% | 32% |
| **当前最优** | **1098%** | **1.87** | 75% | +69% | -5% | 47% | 139% | 25% |

关键突破：
- 2022 从 -4.6% → **+69%** (连熊降级+动量反转)
- 产业链 100% 覆盖 (220链+127NO_CHAIN)
- cost_tracker 发现 + cost={} 屏蔽 (个股止损在 A 股动量策略中持续损害收益)

## 实盘运行

```bash
python strategy/run_live.py
```

三步流程:
1. 更新数据 (行情 + 宏观)
2. 生成信号 (bt_execution.py, 复用缓存)
3. 产出订单 (trade_orders.json + current_positions.json)

输出:
- `trade_orders.json`: 买卖指令 (代码/方向/数量/止损止盈价)
- `current_positions.json`: 新持仓快照
- 日志: `strategy/logs/bt_execution_YYYYMMDD_HHMMSS.log`

## 离线标定

`factor_config.yaml` 中 200+ 行业的 `factors` + `bull_factors` 由离线标定生成：

```bash
python strategy/analysis/offline_calibration.py
```

流程: 遍历全部股票 → 计算因子 → 按概念板块+市场状态分组 → Spearman IC → 选最优因子组合 → 更新配置。

**注意**: 当前只有通用因子和牛市因子，**缺少 `bear_factors`**（熊市专用因子）。这是已知提升空间。

## 分析框架

```bash
# 1. 验证数据准备 (计算 future_ret)
python strategy/analysis/signal_validator.py
# 2. 全量分析 (IC/IR/准确率/因子衰减)
python strategy/analysis/analysis_framework.py
```

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| factor_mode | fixed | 固定因子 (用行业级配置) |
| max_positions | 5 | NORM期最大持仓 |
| rebalance_days | 10 | 调仓间隔 |
| position_stop_loss | 0.08 | 个股止损 (当前 cost={} 屏蔽) |
| ml.blend_weight | 0.50 | ML预测权重 |
| 连熊降级 | 60天 | BEAR→FAST |
| FAST rank_cut | 0.80 | Fast期排名阈值 |
| FAST score_cut | 0.30 | Fast期分数阈值 |

## 当前状态

- **最新提交**: 产业链 100%覆盖 + 震荡检测全正里程碑
- **最优收益版**: 5092ed3 (1098%/Sharpe 1.87)
- **待优化**: 2023年(-4.85%), bear_factors离线标定
