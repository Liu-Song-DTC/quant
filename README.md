# 量化交易系统

A股多因子量化交易系统，融合缠论（缠中说禅）技术分析 + Walk-Forward 动态因子选择 + LLM 行业情绪分析 + 自进化引擎。基于 Backtrader 回测框架，目标 Sharpe Ratio > 1.0。

## 快速开始

```bash
# 每日实盘运行（数据更新 + 信号生成 + 交易建议 + 微信推送）
python main.py run

# 跳过数据更新（仅生成信号）
python main.py run --skip-update

# 行业情绪分析
python main.py sentiment

# 选股导出 Excel（实盘调仓用）
python export_selection.py

# 每日复盘 + 自进化
cd strategy && python analysis/chan_review.py --evolve

# 查看持仓状态
python main.py status
```

## 架构

```
main.py                          # 主入口（run/sentiment/status）
export_selection.py              # 选股结果导出 Excel（5 Sheet）
update_data.py                   # 独立数据更新脚本

data/
├── stock_data/
│   ├── config.json              # 股票池过滤配置
│   ├── backtrader_data/         # 前复权行情 (*_qfq.csv)
│   ├── raw_data/                # 原始日线（1990-至今）
│   ├── fundamental_data/        # 基本面数据（ROE/利润增速/现金流等）
│   └── stock_metadata/          # 股票列表/下载日志
├── sentiment_data/
│   ├── raw/                     # LLM 分析原始结果
│   └── processed/
│       └── rolling_sentiment.csv # 行业情绪时序
└── data_manager.py              # 数据下载/增量更新/格式转换

strategy/
├── config/
│   └── factor_config.yaml       # 策略配置（因子/组合/缠论/回测/情绪）
├── core/
│   ├── strategy.py              # 策略主类（信号+组合+情绪编排）
│   ├── signal_engine.py         # 信号引擎（动态因子/WF-IC /信号生成+Chan增强）
│   ├── portfolio.py             # 组合构建（风险预算/行业均衡/Chan感知止损）
│   ├── factor_calculator.py     # 因子计算（技术面+Alpha+基本面+Chan指标）
│   ├── factor_library.py        # 因子库注册（40+因子4大类）
│   ├── factor_preparer.py       # 因子预计算（多进程并行）
│   ├── factor_neutralizer.py    # 截面中性化（行业+市值OLS剥离）
│   ├── dynamic_factor_selector.py # Walk-Forward IC 动态因子选择
│   ├── divergence_detector.py   # MACD背离检测（顶/底/隐藏背离）
│   ├── structure_analyzer.py    # 多级别结构分析（中枢/级别/中阴/买卖点）
│   ├── chan_theory.py           # 缠论完整实现（分型/笔/段/中枢/走势类型）
│   ├── capital_flow.py          # 资金流向分析
│   ├── news_sentiment.py        # 新闻情绪量化
│   ├── market_regime_detector.py # 市场状态检测（牛/熊/震荡）
│   ├── industry_mapping.py      # 20行业分类（申万2021标准）
│   ├── stock_pool.py            # 股票池管理
│   ├── stock_pool_filter.py     # 季报过滤（ROE/利润增速）
│   ├── signal.py / signal_types.py / signal_fusion.py / signal_store.py
│   ├── fundamental.py           # 基本面数据加载
│   ├── money_flow.py            # 资金流因子
│   ├── sector_rotation.py       # 行业轮动
│   ├── cache_manager.py         # 因子缓存管理
│   ├── config_loader.py         # YAML配置加载（支持点号路径）
│   ├── diagnostics.py           # 诊断框架（因子/信号/市场/持仓）
│   ├── evolution_guard.py       # 进化守护引擎（4层验证）
│   └── ml_predictor.py          # 机器学习预测
├── sentiment/                   # LLM情绪分析
│   ├── orchestrator.py          # 编排器（采集→分析→存储→通知）
│   ├── data_collector.py        # 新闻采集（东方财富+全球）
│   ├── llm_analyzer.py          # DeepSeek API 行业情绪分析
│   ├── sentiment_store.py       # 情绪时序CSV存储
│   └── sentiment_backfill.py    # 回测情绪代理生成（申万指数）
├── analysis/                    # 分析工具
│   ├── analysis_framework.py    # 统一分析框架（6模块）
│   ├── chan_review.py           # 每日复盘 + 自进化引擎（6步法）
│   ├── signal_validator.py      # 信号验证（future_ret计算）
│   ├── single_factor_analysis.py # 单因子深度分析
│   ├── offline_calibration.py   # 离线参数校准
│   └── dashboard.py             # Streamlit 交互式看板
├── daily_review/                # 复盘输出（MD报告/JSON数据）
├── bt_execution.py              # 回测执行（Backtrader）
└── diagnose_*.py                # 诊断工具集

trade/
├── config.yaml.example          # 实盘配置模板
├── runner.py                    # 每日运行流程
├── signal_runner.py             # 轻量信号生成（实盘用）
├── recommender.py               # 交易建议生成
├── notifier.py                  # 微信推送（Server酱）
├── reporter.py                  # 报告生成
├── portfolio_state.json         # 组合状态（手动维护）
└── portfolio_state.py           # 组合状态管理
```

## 数据流

```
data_manager → backtrader_data → factor_preparer → factor_df (中性化)
                                                       ↓
                                            DynamicFactorSelector
                                            (Walk-Forward IC, 360天窗口)
                                                       ↓
                                            SignalEngine → signals
                                            (Chan增强 ×0.7~1.25)
                                                       ↓
                                        PortfolioConstructor → positions
                                        (风险预算/行业均衡/Chan止损)
                                                       ↓
                                        recommender → 微信推送
```

## 核心模块

### 1. 因子体系（40+因子）

| 类别 | 典型因子 | 说明 |
|------|------|------|
| **技术因子** | volatility, momentum, rsi, bb_width, atr, volume_ratio | 量价类，20日滚动 |
| **基本面因子** | fund_score, fund_roe, fund_profit_growth, fund_cf_to_profit | ROE/利润增速/现金流，截面压缩 |
| **Alpha因子** | skewness, kurtosis, tail_risk, overnight_ret, illiq | 收益分布/尾部风险/非流动性 |
| **复合因子** | tech_fund_combo, momentum_reversal, trend_lowvol | 多因子加权，IC验证筛选 |

所有因子经 `tanh` 压缩到 (-1,1)，确保截面可比。

### 2. Walk-Forward 动态因子选择

每个时点向前滚动训练窗口，动态选择最优因子：

```
训练窗口(360天) → 截面IC/IR → 选Top-N因子 → 当期信号生成
```

**质量过滤**：IC符号稳定性>60%，t统计量>1.0，仅保留正向IC因子。

**三种模式**：`dynamic`（纯动态）、`fixed`（纯静态）、`both`（动态优先+静态兜底，当前默认）。

### 3. 因子中性化

截面OLS剥离行业和市值偏差：

```
factor_raw ~ industry_dummies + log(market_cap) → residual = 纯因子值
```

### 4. 20行业分类

基于申万2021标准，覆盖：AI/算力、互联网/软件、半导体/光伏、新能源车/风电、军工、医药、消费、金融等20个行业，关键词+规则映射。

### 5. 缠论增强（Chan Theory）

基于缠中说禅理论，四模块联动：

| 模块 | 功能 | 关键输出 |
|------|------|------|
| `chan_theory.py` | 分型→笔→段→中枢→走势类型→买卖点 | 完整缠论结构 |
| `divergence_detector.py` | MACD顶/底/隐藏背离 | 背离强度 (0~1) |
| `structure_analyzer.py` | 多级别中枢/中阴/走势终完美 | 结构评分 |
| 信号增强乘数 | 底背离×1.25 / 顶背离×0.70 / 中阴×0.85 | 得分修正 |

**Chan感知退出**：顶背离止盈、趋势耗尽退出、买入点保护（止损放宽×2）。

### 6. 组合构建

```
weight = signal_score / volatility × IC_weight × industry_exposure
```

- **截面排名选股**：rank_pct排序，选前N只
- **行业均衡**：防止过度集中
- **波动率目标**：动态调整仓位暴露
- **市场状态调节**：牛市1.0 / 震荡0.9 / 熊市0.4
- **情绪乘数**：LLM情绪 [0.80, 1.20] 调整行业权重
- **持仓自动计算**：`n = total_equity / 10000`，范围 [3, 15]

### 7. 多层止损

| 止损类型 | 参数 | 说明 |
|------|------|------|
| 成本止损 | -18%（波动率自适应） | 跌破成本线 |
| 时间止损 | 30天 + 收益<-8% | 持仓过久无收益 |
| 移动止盈 | 最高点回撤20% | 保护已有利润 |
| 组合止损 | 回撤>15% | 暴露降至35% |
| Chan顶背离退出 | 强度>0.4 | MACD顶背离止盈 |
| Chan趋势耗尽 | alignment<-0.4 | 多级别反转退出 |
| Chan买入点保护 | 底背离区域×2 | 防洗盘震出 |

### 8. LLM 行业情绪分析

```
新闻采集(akshare) → DeepSeek API → 情绪分数[-1,1] → 权重乘数[0.80,1.20]
```

- **实盘模式**：采集当日新闻 → LLM分析 → 存储 → 微信推送
- **回测模式**：从 `rolling_sentiment.csv` 加载预计算序列
- **牛熊调节**：熊市降低正面影响50%，牛市放大20%
- **3日平滑**：减少单日噪音

### 9. 自进化引擎

每日复盘 `chan_review.py --evolve` 驱动，4层防护保证正向进化：

| 层级 | 机制 | 说明 |
|------|------|------|
| L1 统计显著性 | 同一补丁累积≥3天 | 过滤单日噪音 |
| L2 回测验证 | 约束检查 + 信号对比 | 拒绝极端变更 |
| L3 基线追踪 | 记录变更前指标 | 可对比回滚 |
| L4 自动回滚 | 监控N天后检查 | 恶化>20%自动撤销 |

### 10. 选股导出

`export_selection.py` 生成包含5个Sheet的Excel：
- **本次选股**：目标持仓/权重/股数
- **调仓对比**：买入/卖出/调整分类
- **市场状态**：牛熊/动量/趋势/风险
- **调仓历史**：建仓记录
- **信号排名Top50**：全市场最高评分

## 回测

```bash
# 1. 数据准备
cd data && python data_manager.py

# 2. 运行回测
cd strategy && python bt_execution.py

# 3. 分析结果
cd strategy && python analysis/signal_validator.py
cd strategy && python analysis/analysis_framework.py

# 4. 交互式看板
cd strategy && streamlit run analysis/dashboard.py
```

## 配置

### 策略参数 (`strategy/config/factor_config.yaml`)

| 配置块 | 关键参数 | 当前值 | 说明 |
|------|------|:----:|------|
| `factor_mode` | — | `both` | 动态优先+静态兜底 |
| `dynamic_factor` | `train_window_days` | 360 | WF-IC训练窗口 |
| | `top_n_factors` | 3 | 每行业每期选n个最优 |
| | `ic_decay_factor` | 0.92 | IC时间衰减 |
| `signal` | `buy_threshold` | 0.06 | 买入信号阈值 |
| | `sell_threshold` | -0.15 | 卖出信号阈值 |
| `portfolio` | `target_volatility` | 0.15 | 组合目标波动率 |
| | `position_stop_loss` | 0.07 | 个股成本止损 |
| | `portfolio_stop_loss` | 0.12 | 组合回撤止损 |
| `regime_multiplier` | bull/neutral/bear | 1.0/0.6/0.3 | 市场状态敞口 |
| `chan_theory` | `enabled` | true | 缠论增强总开关 |
| | `bottom_divergence_mult` | 1.25 | 底背离买入乘数 |
| | `top_divergence_mult` | 0.70 | 顶背离抑制乘数 |
| | `zhongyin_penalty` | 0.85 | 中阴状态惩罚 |
| `backtest` | `cash` | 150000 | 回测初始资金 |
| | `commission` | 0.0015 | 交易佣金 |
| | `rebalance_days` | 20 | 调仓周期 |

### 股票池 (`data/stock_data/config.json`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `exclude_st` | true | 排除ST |
| `exclude_suspended` | true | 排除停牌 |
| `exclude_star_board` | true | 排除科创板(688) |
| `max_stock_count` | -1 | 不限制数量 |
| `min_market_cap` | -1 | 不限制市值 |
| `min_amount` | -1 | 不限制成交额 |

### 实盘配置 (`trade/config.yaml`)

复制 `trade/config.yaml.example` 为 `config.yaml` 并填入：
- `start_date`: 实盘起始日期
- `notification.sckey`: Server酱推送密钥
- `proxy.host` / `proxy.auth_token`: akshare代理配置

> `config.yaml` 已加入 `.gitignore`，token不会提交到git。

### 组合状态 (`trade/portfolio_state.json`)

```json
{
  "cash": 10000.0,
  "positions": {
    "600519": {"shares": 100, "cost_price": 16.80}
  }
}
```

交易后手动更新 `cash` 和 `positions`。

## 实盘流程

1. **每日晚间** `python main.py run` → 信号+建议+微信推送
2. **调仓日**（每20交易日）：完整买入/卖出/持有清单
3. **非调仓日**：自动止损检查，触发时微信告警
4. **次日盘中** 按建议执行
5. **交易后** 更新 `portfolio_state.json`
6. `python main.py status` 验证持仓

## 关键指标

| 指标 | 目标 | 说明 |
|------|:----:|------|
| IC | >5% | 因子与未来收益的Spearman相关系数 |
| IR | >0.5 | IC均值 / IC标准差 |
| Buy Accuracy | >55% | 买入信号正向未来收益比例 |
| Sharpe | >1.0 | 策略夏普比率 |

## 分析工具

```bash
# 统一分析框架（6模块）
cd strategy && python analysis/analysis_framework.py

# 单因子深度分析
cd strategy && python analysis/single_factor_analysis.py

# 离线参数校准
cd strategy && python analysis/offline_calibration.py

# 交互式看板
cd strategy && streamlit run analysis/dashboard.py
```

## 诊断

```bash
# 系统诊断
cd strategy && python diagnose_system.py

# 信号诊断
cd strategy && python diagnose_signal.py

# 信号质量诊断
cd strategy && python diagnose_signal_quality.py

# 持仓诊断
cd strategy && python diagnose_position.py

# 市场状态诊断
cd strategy && python diagnose_regime.py
```

## 依赖

- **数据**: [akshare](https://github.com/akfamily/akshare)（东方财富A股数据）
- **回测**: [backtrader](https://github.com/mementum/backtrader)
- **LLM**: [DeepSeek API](https://platform.deepseek.com/)（可选，情绪分析用）
- **推送**: [Server酱](https://sct.ftqq.com/)（可选，微信通知用）
- **代理**: [akshare-proxy-patch](https://pypi.org/project/akshare-proxy-patch/)（可选，反爬用）
