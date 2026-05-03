# 量化交易系统

A股多因子量化交易系统，基于 Backtrader 回测框架，Sharpe Ratio 目标 > 1.0。

## 依赖
* ip池: https://ak.cheapproxy.net/
* 微信推送: https://sct.ftqq.com/
* LLM情绪分析: [DeepSeek API](https://platform.deepseek.com/) (可选)

` 自行购买并填入相应的 token `

## 快速开始

### 每日运行
```bash
# 每日运行：数据更新 + 信号生成 + 交易建议（调仓日）/ 止损检查（非调仓日）
python main.py run

# 跳过数据更新（仅生成信号）
python main.py run --skip-update

# 非调仓日强制运行
python main.py run --force

# 查看当前持仓和止损状态
python main.py status
```

### 行业情绪分析
```bash
# 运行当日情绪分析
python main.py sentiment

# 指定日期
python main.py sentiment --date 2026-05-01

# 跳过微信通知
python main.py sentiment --no-notify
```

### 回测
```bash
# 1. 数据准备（首次或扩充股票池后）
cd data && python data_manager.py

# 2. 情绪数据回填（基于申万行业指数代理）
cd strategy && python sentiment/sentiment_backfill.py

# 3. 运行回测
cd strategy && python bt_execution.py

# 4. 分析回测结果
cd strategy && python analysis/signal_validator.py
cd strategy && python analysis/analysis_framework.py
```

### 分析工具
```bash
# 统一分析框架（6个模块：数据质量、因子层、信号层、组合层、时序稳定性、优化建议）
python analysis/analysis_framework.py

# Streamlit 交互式看板（Plotly 可视化）
streamlit run analysis/dashboard.py
```

## 架构

```
data/                       # 数据层
├── stock_data/
│   ├── config.json             # 股票池过滤配置
│   ├── backtrader_data/        # 前复权行情数据 (*_qfq.csv)
│   ├── raw_data/               # 原始日线数据（1990-至今）
│   ├── fundamental_data/       # 基本面数据
│   └── stock_metadata/         # 股票列表/质量报告
├── sentiment_data/
│   └── processed/
│       └── rolling_sentiment.csv  # 行业情绪时序数据
└── data_manager.py             # 数据下载/更新

strategy/                   # 策略层
├── config/factor_config.yaml   # 策略配置（因子/组合/回测/情绪参数）
├── core/
│   ├── strategy.py             # 策略主类（信号+组合+情绪编排）
│   ├── signal_engine.py        # 信号引擎（动态因子选择/WF-IC验证/信号生成）
│   ├── portfolio.py            # 组合构建（截面排名/行业均衡/风险预算/止损）
│   ├── factor_calculator.py    # 因子计算（技术面+Alpha+基本面组合因子）
│   ├── factor_library.py       # 因子库（波动率/动量/RSI/布林带/Alpha因子注册）
│   ├── factor_preparer.py      # 因子预计算（多进程并行+截面中性化）
│   ├── factor_neutralizer.py   # 因子中性化（行业+市值截面回归剥离）
│   ├── market_regime_detector.py  # 市场状态检测（牛/熊/中性）
│   ├── industry_mapping.py     # 20行业分类+关键词映射（申万2021标准）
│   └── config_loader.py        # 配置加载
├── sentiment/                  # LLM情绪分析模块
│   ├── orchestrator.py         # 编排器（采集→分析→存储→通知，支持回测模式）
│   ├── data_collector.py       # 新闻采集（akshare东方财富+全球）
│   ├── llm_analyzer.py         # DeepSeek API 行业情绪分析
│   ├── sentiment_store.py      # 情绪时序CSV存储
│   └── sentiment_backfill.py   # 回测情绪代理生成（申万指数）
├── analysis/                   # 分析工具
│   ├── analysis_framework.py   # 统一分析框架（6模块）
│   ├── signal_validator.py     # 信号验证数据准备（future_ret计算）
│   └── dashboard.py            # Streamlit+Plotly交互看板
├── bt_execution.py             # 回测执行（Backtrader）
└── run_bt.sh                   # 回测启动脚本

trade/                      # 实盘交易层
├── config.yaml                 # 实盘配置（起始日期/推送/代理）
├── runner.py                   # 每日运行流程（含情绪分析集成）
├── signal_runner.py            # 信号生成（含情绪乘数注入）
├── recommender.py              # 交易建议生成
├── notifier.py                 # 微信推送（Server酱+行业情绪摘要）
└── portfolio_state.json        # 组合状态（手动编辑）

main.py                     # 主入口
```

### 数据流
```
data_manager → backtrader_data → factor_preparer → factor_df (中性化)
                                                         ↓
                                              DynamicFactorSelector
                                              (Walk-Forward IC验证)
                                                         ↓
                              sentiment_store → SignalEngine → signals
                                                         ↓
                                          PortfolioConstructor → positions
                                          (风险预算+行业均衡+止损)
                                                         ↓
                                          recommender → 微信推送
```

## 策略详解

### 1. 因子体系

系统内置约 40+ 因子，分为四大类：

| 类别 | 数量 | 典型因子 | 说明 |
|------|:----:|------|------|
| **技术因子** | 15+ | volatility, momentum, rsi, bb_width, atr | 量价类因子，20日滚动计算 |
| **基本面因子** | 10 | fund_score, fund_roe, fund_profit_growth, fund_revenue_growth, fund_cf_to_profit, fund_gross_margin | ROE/利润增速/营收增速/现金流等，截面压缩到 (-1,1) |
| **Alpha因子** | 11 | skewness_20, kurtosis_20, tail_risk, volatility_skew, overnight_ret, intraday_ret, gap_ratio, price_volume_corr_20, illiq_20, turnover_stability, max_ret_20 | 收益分布偏度/峰度、尾部风险、隔夜vs日内收益、跳空缺口、量价相关性、非流动性 |
| **复合因子** | 5+ | tech_fund_combo, momentum_reversal, trend_lowvol, rsi_vol_combo, bb_rsi_combo | 多因子加权组合，通过 IC 验证筛选 |

所有因子值通过 `tanh` 压缩到 (-1, 1) 区间，确保跨股票可比。基本面因子使用百分位排名压缩，技术因子使用滚动窗口标准化。

### 2. 因子选择：Walk-Forward IC 验证

系统不依赖静态因子配置，而是在每个时点动态选择最优因子：

```
训练窗口 (360天) → 计算各因子的截面 IC/IR → 选 Top-3 因子 → 用于当期的信号生成
```

**关键参数：**
- `train_window_days: 360` — 训练窗口（滚动）
- `forward_period: 10` — IC 验证的前瞻周期
- `top_n_factors: 3` — 每行业每期选择的最优因子数
- `ic_decay_factor: 0.92` — IC 衰减权重（近期更高权重）
- `min_ic_dates: 20` — 最少的 IC 日期数

**质量过滤：**
- IC 符号稳定性 > 60%
- IC 方差异常过滤（CV < 5.0）
- t 统计量 > 1.0（p ≈ 0.05）
- 仅保留正向 IC 因子

**三种因子模式：**
| 模式 | 说明 |
|------|------|
| `dynamic` | 纯动态因子选择，WF-IC 选最优 |
| `fixed` | 纯静态因子配置（config 中的 industry_factors） |
| `both` | 动态优先 + 静态兜底（当前模式） |

### 3. 因子中性化

因子在进入 IC 验证前，进行截面中性化，剥离行业和市值偏差：

```
factor_raw ~ industry_dummies + log(market_cap) → residual = 纯因子值
```

- **行业中性化**：减去行业均值，消除行业 beta
- **市值中性化**：OLS 回归剥离市值暴露
- **标准差保持**：中性化后恢复原始均值和标准差，维持量纲

### 4. 20 行业分类

基于申万 2021 行业分类标准，将全部 A 股映射到 20 个行业：

| # | 行业 | 典型关键词 |
|:-:|------|------|
| 1 | 人工智能/算力 | AI、算力、大模型、GPU、数据中心 |
| 2 | 互联网/软件 | 互联网、软件开发、云服务、信息安全 |
| 3 | 通信/计算机 | 通信设备、计算机设备、运营商 |
| 4 | 半导体/光伏 | 半导体、芯片、集成电路、光伏 |
| 5 | 电子 | 消费电子、面板、LED、光学光电子 |
| 6 | 新能源车/风电 | 新能源车、电池、锂电、风电、储能 |
| 7 | 电力设备 | 电力设备、电网、电气设备 |
| 8 | 有色/钢铁/煤炭 | 有色金属、贵金属、钢铁、煤炭 |
| 9 | 化工 | 化学制品、化学原料、农化、塑料 |
| 10 | 建材 | 水泥、玻璃、玻纤、装修建材 |
| 11 | 军工 | 军工、航天、航空装备、雷达 |
| 12 | 自动化/制造 | 自动化、通用设备、工程机械 |
| 13 | 基建/地产/石油石化 | 基础建设、房地产、石油、石化 |
| 14 | 消费 | 食品饮料、家电、纺织服装、旅游零售 |
| 15 | 医药 | 化学制药、医疗器械、中药、生物制品 |
| 16 | 传媒 | 游戏、广告、影视、出版、教育 |
| 17 | 农业 | 种植、养殖、渔业、饲料 |
| 18 | 环保/公用 | 环境治理、环保设备、燃气、水务 |
| 19 | 金融 | 银行、证券、保险、多元金融 |
| 20 | 交运 | 航运、港口、航空、机场、物流 |

每个行业配置了独立的牛市/中性/熊市因子组合，通过 Walk-Forward IC 验证产出的权重进行动态调整。

### 5. 信号生成

```python
# 核心公式
score = weighted_sum(selected_factors × IC_weights)
       + fundamental_enhancement × 0.1
       + style_factor_adjustment

# 信号判定（截面排名制）
buy = factor_value is valid AND |score| < 5.0
sell = factor_value < sell_threshold (-0.15)
```

信号层只产出 `factor_value`，实际的买卖决策由组合层通过**截面 rank_pct 排序**决定。这种设计避免了绝对阈值带来的信号稀疏问题（历史数据显示绝对阈值导致 52.7% 信号为 NONE）。

### 6. 组合构建

**风险预算模型**：每只股票的仓位由其信号质量和波动率共同决定。

```
weight = signal_score / volatility × IC_weight × industry_exposure
```

**关键机制：**
- **截面排名选股**：每个调仓日对全市场信号做 rank_pct 排序，选前 N 只
- **行业均衡**：防止过度集中单一行业，最大行业权重约束
- **波动率目标**：组合目标波动率 25%，根据实际波动率动态调整仓位暴露
- **市场状态调节**：牛市 100% 暴露、中性 90%、熊市 40%
- **情绪乘数**：LLM 情绪分析结果以 [0.80, 1.20] 乘数调整行业权重
- **价格约束**：过滤无法买入 1 手（100股）且不超 2 倍理想仓位的股票

**持仓数量自动计算**：
```
n_positions = total_equity / 10000,  范围 [3, 15]
```

### 7. 多层止损体系

| 止损类型 | 参数 | 说明 |
|------|------|------|
| **成本止损** | -18%（波动率自适应，高波放宽至27%） | 跌破成本线自动卖出 |
| **时间止损** | 30天 + 收益 < -8% | 持仓过久且无收益 |
| **移动止盈** | 从最高点回撤 20% | 保护已有利润 |
| **组合止损** | 组合回撤 > 15% | 仓位暴露降至 35%，10日逐步恢复 |

### 8. LLM 行业情绪分析

基于 DeepSeek API 的每日行业情绪分析管道：

```
新闻采集 (akshare) → LLM 批量分析 → 情绪分数 [-1, 1] → 权重乘数 [0.80, 1.20]
```

- **实盘模式**：采集当日新闻 → DeepSeek API 分析 → 存储到 CSV → 微信推送行业摘要
- **回测模式**：从 `rolling_sentiment.csv` 加载预计算情绪序列，无需 API 密钥
- **牛熊调节**：熊市降低正面情绪影响 50%，牛市放大 20%
- **滚动平滑**：3 日窗口平滑，减少单日噪音

## 配置

### 策略参数 (strategy/config/factor_config.yaml)

| 配置块 | 关键参数 | 当前值 | 说明 |
|------|------|:----:|------|
| `factor_mode` | — | `both` | 动态优先+静态兜底 |
| `dynamic_factor` | `train_window_days` | 360 | WF-IC 训练窗口 |
| | `forward_period` | 10 | IC 前瞻周期（天） |
| | `top_n_factors` | 3 | 每行业每期选 n 个最优因子 |
| | `ic_decay_factor` | 0.92 | IC 时间衰减权重 |
| `signal` | `buy_threshold` | 0.5 | 买入信号阈值 |
| | `sell_threshold` | -0.15 | 卖出信号阈值 |
| `portfolio` | `target_volatility` | 0.25 | 组合目标波动率 |
| | `position_stop_loss` | 0.18 | 个股成本止损线 |
| | `portfolio_stop_loss` | 0.15 | 组合回撤止损线 |
| | `max_single_weight` | 0.30 | 单只最大权重 |
| `regime_multiplier` | `bull/neutral/bear` | 1.0/0.9/0.4 | 市场状态仓位乘数 |
| `enhanced_stop_loss` | `time_stop_days` | 30 | 时间止损天数 |
| | `trailing_stop_pct` | 0.20 | 移动止盈回撤比例 |
| `factor_neutralization` | `enabled` | true | 截面行业+市值中性化 |
| `risk_parity` | `enabled` | true | 风险平价配置 |
| `volatility_control` | `enabled` | true | 波动率自适应控制 |
| `backtest` | `cash` | 100000 | 回测初始资金 |
| | `commission` | 0.0015 | 交易佣金 |
| | `rebalance_days` | 10 | 调仓周期（每2周） |

### 行业情绪配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用情绪分析 |
| `deepseek_api_key` | "" | DeepSeek API Key（为空时从环境变量读取） |
| `max_news_batch` | 50 | 单次 LLM 分析新闻条数上限 |
| `weight_impact.max_multiplier` | 1.20 | 正面情绪最大行业权重乘数 |
| `weight_impact.min_multiplier` | 0.80 | 负面情绪最小行业权重乘数 |
| `weight_impact.smoothing_days` | 3 | 情绪平滑天数 |
| `weight_impact.regime_adjustment` | true | 牛熊市调节情绪影响 |

### 股票池过滤 (data/stock_data/config.json)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_stock_count` | 2000 | 最大股票数量（按市值排序取前N） |
| `min_market_cap` | 30 | 最低总市值（单位：亿） |
| `min_amount` | 200 | 最低日成交额（单位：万元） |
| `exclude_st` | true | 排除 ST 股票 |
| `exclude_suspended` | true | 排除停牌股票 |

### 实盘参数

修改 `trade/config.yaml` 并填入真实信息：
- `start_date`: 实盘起始日期，用于计算调仓日
- `notification`: 微信推送配置 (Server酱 sckey)
- `proxy`: akshare 代理配置

> `config.yaml` 已加入 `.gitignore`，token 不会提交到 git。

### 组合状态 (trade/portfolio_state.json)

模板已内置在仓库中。只需要修改 `cash` 和 `positions`：

```json
{
  "cash": 10000.0,
  "positions": {}
}
```

交易后更新示例：
```json
{
  "cash": 3000.0,
  "positions": {
    "600519": {"shares": 100, "cost_price": 16.80},
    "000858": {"shares": 200, "cost_price": 15.50}
  }
}
```

- `cash`: 买入减、卖出加
- `positions`: 买入加、卖出减，清零则删除该条目

## 实盘操作流程

1. **每日晚间** 运行 `python main.py run`
2. **调仓日**（每10个交易日）：获得完整交易建议（买入/卖出/持有清单）+ 行业情绪摘要 + 微信推送
3. **非调仓日**：自动检查个股止损（18%成本止损 + 移动止盈 + 时间止损），有触发时微信告警；若上次建议为空（整手取整导致0股）则自动重新生成
4. **次日盘中** 按建议手动执行交易
5. **交易后** 手动编辑 `trade/portfolio_state.json` 更新 `cash` 和 `positions`
6. 运行 `python main.py status` 验证持仓状态

## 仓位自动计算

`max_position` 参数已消除，由 `PortfolioConstructor` 根据资金自动计算：

```
n_positions = total_equity / 10000,  范围 [3, 15]
```

| 总资产 | 持仓数 | 说明 |
|--------|--------|------|
| 3w | 3 | 最低可行 |
| 5w | 5 | |
| 10w | 10 | 优化基线 |
| 20w | 15 (cap) | |
| 30w+ | 15 (cap) | |

增资/撤资后无需调整任何参数，下次调仓自动适应。

## 关键指标

| 指标 | 目标 | 说明 |
|------|:----:|------|
| IC | >5% | 因子值与未来收益的 Spearman 相关系数 |
| IR | >0.5 | IC均值 / IC标准差 |
| Buy Accuracy | >55% | 买入信号正向未来收益比例 |
| Sharpe | >1.0 | 策略夏普比率 |

## 验证

```bash
# 实盘 vs 回测一致性
python verify_trade_vs_backtest.py

# 资金敏感性
python test_capital_sensitivity.py
```
