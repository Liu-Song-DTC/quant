# A股量化选股系统

多因子、缠论结构、市场状态与滚动 ML 相结合的研究和选股系统。历史结果不代表未来收益。

## 当前入口

生产每日链（交易日 17:30 后）：

```bash
bash strategy/run_daily_live.sh
```

旧入口 `strategy/run_live.py`、`strategy/run_live_daily.sh`、`strategy/cron_run.sh` 和
`main.py run` 已统一到这条链。不要使用旧 `data_manager.py` 或 `update_data.py` 刷新生产行情。

## 账户与成交

- `trade_orders.json`：项目根目录下的计划订单，不代表已成交。
- `current_positions.json`、`live_account.json`：项目根目录下的实际持仓和现金。
- `live_portfolio_state.json`：跨交易日的风控、调仓和板块状态。
- 账户只按券商成交回报更新；价格与股数不能用次日开盘价推定。
- 未完成订单持续保留；未导入上一批成交、拒单或撤单回报时，不生成叠加的新订单。
- 同日重复运行重放原订单和原 run_id；多文件写入中断后自动恢复。

导入成交回报（不会出新订单）：

```bash
.venv/bin/python strategy/reconcile_live_account.py --fills strategy/qmt_bridge/fills_today.csv
```

回报字段、部分成交口径、费用与迁移说明见 [运行说明](docs/review_20261025.md)。

## 回测与验证

正式回测会覆盖 `strategy/rolling_validation_results/`。研究和优化使用隔离快照：

```bash
.venv/bin/python strategy/analysis/review_validation.py --name experiment_name --run
```

完整股票池、完整时间段、模型重训均在 `.review_runs/experiment_name/` 完成。
研究只读现有行情，不刷新生产数据。固定信号的执行重放还必须恢复相同的市场和行业上下文，
不能把缓存路径和首次生成路径的差异当作策略改进。

短期选股统一评估次日开盘入场后 1/3/5/10 个市场交易日的扣费盯市收益；
无法退出的样本仍保留并另行标记，不能通过删除亏损样本提高胜率。
账户交易胜率使用完整持仓生命周期的扣费现金流，包含加减仓。

## 快速回归测试

无需额外安装 pytest：

```bash
PYTHONDONTWRITEBYTECODE=1 QUANT_ALT_NO_AUTOREFRESH=1 \
  .venv/bin/python -B -m unittest discover -s strategy/tests -p test_review_regressions.py -v
```

测试数据中的随机标签不应有稳定预测能力；`REGRESSION_RANDOM_DATA` 日志不是正式训练结果。

## 可靠性边界

行业、ST、概念历史数据仍有已登记的 PIT 限制，见 [披露](docs/disclosure.md)。
回测成交模型仍是近似，需要持续以实际成交、费用、拒单和停牌记录校准。
回测变化必须区分纠错与策略改进；历史反复研究的区间不能重新宣称为未使用过的样本外。
