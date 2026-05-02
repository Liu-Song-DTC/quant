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

# Streamlit 交互式看板
streamlit run analysis/dashboard.py
```

## 架构

```
data/                       # 数据层
├── stock_data/
│   ├── config.json             # 股票池过滤配置
│   ├── backtrader_data/        # 前复权行情数据 (*_qfq.csv)
│   ├── raw_data/               # 原始日线数据
│   ├── fundamental_data/       # 基本面数据
│   └── stock_metadata/         # 股票列表/质量报告
├── sentiment_data/
│   └── processed/
│       └── rolling_sentiment.csv  # 行业情绪时序数据
└── data_manager.py             # 数据下载/更新

strategy/                   # 策略层
├── config/factor_config.yaml   # 策略配置 (因子/组合/回测/情绪参数)
├── core/
│   ├── strategy.py             # 策略主类
│   ├── signal_engine.py        # 信号引擎 (动态因子选择/信号生成)
│   ├── portfolio.py            # 组合构建 (截面排名/行业均衡/择时/自动仓位)
│   ├── factor_calculator.py    # 因子计算 (技术面+基本面组合因子)
│   ├── factor_preparer.py      # 因子预计算 (多进程并行)
│   ├── market_regime_detector.py  # 市场状态检测 (牛/熊/中性)
│   ├── industry_mapping.py     # 14行业分类 + 关键词映射
│   └── config_loader.py        # 配置加载
├── sentiment/                  # LLM情绪分析模块
│   ├── orchestrator.py         # 编排器 (采集→分析→存储→通知)
│   ├── data_collector.py       # 新闻采集 (akshare东方财富+全球)
│   ├── llm_analyzer.py         # DeepSeek API 行业情绪分析
│   ├── sentiment_store.py      # 情绪时序CSV存储
│   └── sentiment_backfill.py   # 回测情绪代理生成 (申万指数)
├── analysis/                   # 分析工具
│   ├── analysis_framework.py   # 统一分析框架 (6模块)
│   ├── signal_validator.py     # 信号验证数据准备
│   └── dashboard.py            # Streamlit 交互看板
├── bt_execution.py             # 回测执行 (Backtrader)
└── run_bt.sh                   # 回测启动脚本

trade/                      # 实盘交易层
├── config.yaml                 # 实盘配置 (起始日期/推送/代理)
├── runner.py                   # 每日运行流程
├── signal_runner.py            # 信号生成
├── recommender.py              # 交易建议生成
├── notifier.py                 # 微信推送 (Server酱)
└── portfolio_state.json        # 组合状态 (手动编辑)

main.py                     # 主入口
```

### 数据流
```
data_manager → backtrader_data → factor_preparer → factor_df
                                                         ↓
                                              DynamicFactorSelector
                                                         ↓
                              sentiment_store → SignalEngine → signals
                                                         ↓
                                          PortfolioConstructor → positions
                                                         ↓
                                          recommender → 微信推送
```

## 配置

### 股票池过滤 (data/stock_data/config.json)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_stock_count` | 2000 | 最大股票数量（按市值排序取前N） |
| `min_market_cap` | 30 | 最低总市值（单位：亿） |
| `min_amount` | 200 | 最低日成交额（单位：万元） |
| `exclude_st` | true | 排除 ST 股票 |
| `exclude_suspended` | true | 排除停牌股票 |

### 策略参数 (strategy/config/factor_config.yaml)

| 配置块 | 关键参数 | 说明 |
|--------|----------|------|
| `factor_mode` | `fixed`/`dynamic`/`both`/`reweight` | 因子选择模式 |
| `dynamic_factor` | `train_window_days: 360`, `top_n_factors: 3`, `reweight_blend: 0.5` | 动态因子选择参数 |
| `signal` | `buy_threshold: 0.5`, `sell_threshold: -0.15` | 买卖信号阈值 |
| `portfolio` | `max_single_weight: 0.30`, `position_stop_loss: 0.30` | 组合限制参数 |
| `backtest` | `cash: 100000`, `commission: 0.0015`, `rebalance_days: 20` | 回测参数 |
| `risk_parity` | `enabled: true`, `target_risk_per_stock: 0.10` | 风险平价配置 |
| `volatility_control` | `enabled: true`, `lookback_period: 20` | 波动率控制 |
| `regime_multiplier` | `bull: 1.0`, `neutral: 0.9`, `bear: 0.25` | 市场状态仓位乘数 |

### 行业情绪配置 (strategy/config/factor_config.yaml → industry_sentiment)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | true | 是否启用情绪分析 |
| `deepseek_api_key` | "" | DeepSeek API Key（为空时从环境变量 `DEEPSEEK_API_KEY` 读取） |
| `max_news_batch` | 50 | 单次 LLM 分析新闻条数上限 |
| `weight_impact.max_multiplier` | 1.20 | 正面情绪最大行业权重乘数 |
| `weight_impact.min_multiplier` | 0.80 | 负面情绪最小行业权重乘数 |
| `weight_impact.smoothing_days` | 3 | 情绪平滑天数 |
| `weight_impact.regime_adjustment` | true | 是否根据牛熊市调整情绪影响 |

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
2. **调仓日** (每20个交易日): 获得完整交易建议（买入/卖出/持有清单）+ 微信推送
3. **非调仓日**: 自动检查个股止损（30%），有触发时微信告警
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

## 行业情绪分析

系统内置基于 LLM 的行业情绪分析管道，覆盖 14 个行业分类：

```
互联网/软件、交运、军工、化工、半导体/光伏、基建/地产/石油石化、
新能源车/风电、有色/钢铁/煤炭/建材、消费/传媒/农业/环保/医药、
电力设备、电子、自动化/制造、通信/计算机、金融
```

### 工作流程
1. **新闻采集** — 通过 akshare 抓取东方财富 + 全球财经新闻
2. **LLM 分析** — 调用 DeepSeek API 批量分析，输出情绪分数 (positive/negative/neutral)
3. **分数汇总** — 加权聚合为 `[-1.0, 1.0]` 的行业情绪分数
4. **权重映射** — 转换为组合权重乘数 `[0.80, 1.20]`，正面情绪增配、负面情绪减配
5. **牛熊调节** — 熊市降低正面情绪影响，牛市放大

### 回测模式
回测期无法获取历史新闻，使用 `sentiment_backfill.py` 生成基于申万行业指数的市场情绪代理：
- 下载 124 个申万二级行业指数历史数据
- 聚合为 14 个系统行业 → EWM 动能 → 截面 z-score → tanh 压缩
- 输出 `rolling_sentiment.csv`，与实盘 LLM 情绪共用同一接口

## 关键指标

| 指标 | 目标 | 说明 |
|------|------|------|
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
