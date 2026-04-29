# 量化交易系统

A股多因子量化交易系统，基于 Backtrader 回测框架，Sharpe Ratio 目标 > 1.0。

## 依赖
* ip池: https://ak.cheapproxy.net/
* 微信推送: https://sct.ftqq.com/
  
` 自己购买填写并相应的token `

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

### 回测
```bash
cd strategy && python bt_execution.py

# 资金敏感性测试
python test_capital_sensitivity.py
```

## 架构

```
data/                   # 数据层
├── stock_data/
│   ├── backtrader_data/    # 前复权行情数据 (*_qfq.csv)
│   └── fundamental_data/   # 基本面数据
└── data_manager.py         # 数据下载/更新 (代理配置从 trade/config.yaml 读取)

strategy/               # 策略层
├── config/factor_config.yaml  # 策略配置 (因子/组合/回测参数)
├── core/
│   ├── strategy.py          # 策略主类
│   ├── signal_engine.py     # 信号引擎 (因子选择/信号生成)
│   ├── portfolio.py         # 组合构建 (截面排名/行业均衡/择时/自动仓位)
│   ├── factor_calculator.py # 因子计算
│   ├── market_regime_detector.py  # 市场状态检测
│   └── config_loader.py     # 配置加载
├── bt_execution.py          # 回测执行 (Backtrader)
└── analysis/                # 分析工具

trade/                  # 实盘交易层
├── config.yaml              # 实盘配置 (起始日期/推送)
├── runner.py                # 每日运行流程 (调仓/止损)
├── signal_runner.py         # 信号生成 (与回测共享策略核心)
├── recommender.py           # 交易建议生成
├── reporter.py              # 报告输出
├── notifier.py              # 微信推送 (Server酱/Markdown格式)
└── portfolio_state.json     # 组合状态 (手动编辑)

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
- `backtest.rebalance_days`: 调仓周期 (默认 20 个交易日)
- `portfolio.position_stop_loss`: 个股止损线 (12%)

### 实盘参数 (trade/config.yaml)
- `start_date`: 实盘起始日期，用于计算调仓日
- `notification`: 微信推送配置 (Server酱 sckey)
- `proxy`: akshare 代理配置

### 组合状态 (trade/portfolio_state.json)

首次使用前手动创建此文件，填入你的初始资金。

**你需要写的内容（只需要 cash 和 positions）：**
```json
{
  "cash": 10000.0,
  "positions": {}
}
```

交易后手动更新，例如持有 3 只股票后：
```json
{
  "cash": 3000.0,
  "positions": {
    "600519": {"shares": 100, "cost_price": 16.80},
    "000858": {"shares": 200, "cost_price": 15.50},
    "300750": {"shares": 100, "cost_price": 28.00}
  }
}
```

- `cash`: 买入减、卖出加
- `positions`: 买入加、卖出减，清零则删除该条目


## 实盘操作流程

1. **每日晚间** 运行 `python main.py run`
2. **调仓日** (每20个交易日): 获得完整交易建议（买入/卖出/持有清单）+ 微信推送
3. **非调仓日**: 自动检查个股止损（12%），有触发时微信告警
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
| 3w | 3 | 最低可行 (历史回测 Sharpe~1.0) |
| 5w | 5 | |
| 10w | 10 | 优化基线 (Sharpe 1.064) |
| 20w | 15 (cap) | |
| 30w+ | 15 (cap) | |

增资/撤资后无需调整任何参数，下次调仓自动适应。

## 关键指标

| 指标 | 目标 | 说明 |
|------|------|------|
| IC | >5% | 因子值与未来收益的 Spearman 相关系数 |
| IR | >0.5 | IC均值 / IC标准差 |
| Sharpe | >1.0 | 策略夏普比率 |

## 当前基线

- **Sharpe Ratio: 1.064** (factor_mode=reweight, blend=0.5)
- 调仓周期: 20 交易日 | 初始资金: 100,000
- 最低可行资金: \~30,000 (Sharpe~1.0)

## 验证

```bash
# 实盘 vs 回测一致性
python verify_trade_vs_backtest.py

# 资金敏感性
python test_capital_sensitivity.py
```
