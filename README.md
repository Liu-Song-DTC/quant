# 量化交易系统

A股多因子量化交易系统，基于 Backtrader 回测框架，Sharpe Ratio 目标 > 1.0。

## 快速开始

### 每日运行
```bash
# 调仓日运行：数据更新 + 信号生成 + 交易建议
python main.py run

# 跳过数据更新（仅生成信号）
python main.py run --skip-update

# 非调仓日强制运行
python main.py run --force

# 查看当前持仓
python main.py status

# 交互式确认交易
python main.py confirm -i
```

### 回测
```bash
cd strategy && python bt_execution.py
```

## 架构

```
data/                   # 数据层
├── stock_data/
│   ├── backtrader_data/    # 前复权行情数据 (*_qfq.csv)
│   └── fundamental_data/   # 基本面数据
└── data_manager.py         # 数据下载/更新

strategy/               # 策略层
├── config/factor_config.yaml  # 策略配置 (因子/组合/回测参数)
├── core/
│   ├── strategy.py          # 策略主类
│   ├── signal_engine.py     # 信号引擎 (因子选择/信号生成)
│   ├── portfolio.py         # 组合构建 (截面排名/行业均衡/择时)
│   ├── factor_calculator.py # 因子计算
│   ├── market_regime_detector.py  # 市场状态检测
│   └── config_loader.py     # 配置加载
├── bt_execution.py          # 回测执行 (Backtrader)
└── analysis/                # 分析工具

trade/                  # 实盘交易层
├── config.yaml              # 实盘配置
├── runner.py                # 每日运行流程
├── signal_runner.py         # 信号生成 (与回测共享策略核心)
├── recommender.py           # 交易建议生成
├── reporter.py              # 报告输出
├── notifier.py              # 微信推送 (Server酱)
├── confirm.py               # 交互式交易确认
└── portfolio_state.json     # 组合状态 (可手动编辑)

main.py                 # 主入口
```

### 数据流
```
data_manager → backtrader_data → signal_runner → Strategy → SignalEngine
                                                           ↓
                                       portfolio_state ← PortfolioConstructor
                                                           ↓
                                       recommender → 交易建议 → 微信推送
```

## 配置

### 策略参数 (strategy/config/factor_config.yaml)
- `factor_mode`: 因子模式 — `fixed`/`dynamic`/`both`/`reweight`
- `backtest.max_position`: 最大持仓数 (默认 10)
- `backtest.rebalance_days`: 调仓周期 (默认 20 个交易日)
- `portfolio.position_stop_loss`: 个股止损线 (12%)

### 实盘参数 (trade/config.yaml)
- `start_date`: 实盘起始日期，用于计算调仓日
- `max_position`: 最大持仓数，与回测配置对齐
- `notification`: 微信推送配置 (Server酱)

### 组合状态 (trade/portfolio_state.json)
```json
{
  "cash": 45000.0,
  "positions": {
    "600519": {"shares": 100, "cost_price": 1680.0}
  },
  "exposure": 0.75,
  "peak_equity": 100000.0
}
```
- `cash/positions`: 手动编辑后自动生效
- `exposure/peak_equity`: 系统自动维护，持久化 EMA 平滑状态

## 实盘操作流程

1. **每日晚间** 运行 `python main.py run`
2. **调仓日** (每20个交易日): 获得完整交易建议（买入/卖出/持有清单）
3. **非调仓日**: 自动检查个股止损（12%），无异常则跳过
4. **次日盘中** 按建议手动执行交易
5. **交易后** 运行 `python main.py confirm -i` 更新持仓状态
6. 也可直接编辑 `trade/portfolio_state.json` 手动更新

## 关键指标

| 指标 | 目标 | 说明 |
|------|------|------|
| IC | >5% | 因子值与未来收益的 Spearman 相关系数 |
| IR | >0.5 | IC均值 / IC标准差 |
| Sharpe | >1.0 | 策略夏普比率 |
| Buy Accuracy | >55% | 买入信号正收益比例 |

## 当前基线

- **Sharpe Ratio: 1.064** (factor_mode=reweight, blend=0.5)
- 最大持仓: 10 只 | 调仓周期: 20 交易日
- 初始资金: 100,000 | 最低可行资金: ~30,000

## 验证

```bash
python verify_trade_vs_backtest.py
```
