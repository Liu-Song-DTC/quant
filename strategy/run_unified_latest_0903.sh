#!/bin/bash
# 统一跑 (2026-09-03): 最新完整数据 + 指纹缓存键诚实重算 + 验证 + 分析
#   数据: K线5493文件(5174尾9/2+293北交所8/14) + 基本面5585(报告期20260630)
#         + 另类数据9/2 + concept_hist补全(60板块→9/2) + 龙虎榜历史5240只全上榜日
#   配置: factor_mode fixed / reject_all true / fromdate 2021-01-01 / todate 2026-09-02
#   因子: 数据指纹键与所有旧缓存不同 -> 首次现算并落盘新缓存 (约1h+)
#   串行: 遵守 OOM 约束, 单独一个重任务
set -eu
cd /mnt/d/quant/strategy
PY=/mnt/d/quant/.venv/bin/python
mkdir -p logs rolling_validation_results models
TS=$(date +%m%d_%H%M)
MAIN="logs/unified_latest_${TS}.log"
exec >> "$MAIN" 2>&1
echo "=== UNIFIED START $(date) (最新数据统一跑) ==="

# 1) 回测截止日更新到最新数据日
sed -i "s/^  todate:.*/  todate: '2026-09-02'/" config/factor_config.yaml
grep -n "^  todate:" config/factor_config.yaml

# 2) 回测: 因子指纹缓存键命中失败 -> 从数据现算
rm -f rolling_validation_results/backtest_signals.csv rolling_validation_results/portfolio_selections.csv
"$PY" bt_execution.py 2>&1 | tee "logs/unified_bt_${TS}.log" || echo "!!! bt 非零退出码 $?"
echo "--- 结果行 ---"
grep -E "最终净值|Sharpe:|最大回撤:" "logs/unified_bt_${TS}.log" | tail -5
echo "--- 因子缓存行 ---"
grep -E "数据指纹|因子缓存|计算因子|构建于" "logs/unified_bt_${TS}.log" | head -6

# 3) 验证数据准备 (算 future_ret)
"$PY" analysis/signal_validator.py 2>&1 | tail -6

# 4) 分析框架 (IC/胜率/推荐)
"$PY" analysis/analysis_framework.py 2>&1 | tee "logs/unified_analysis_${TS}.log" | tail -45

# 5) 归档产物
[ -f rolling_validation_results/backtest_signals.csv ] && cp -p rolling_validation_results/backtest_signals.csv "rolling_validation_results/backtest_signals.unified${TS}.csv"
[ -f rolling_validation_results/portfolio_selections.csv ] && cp -p rolling_validation_results/portfolio_selections.csv "rolling_validation_results/portfolio_selections.unified${TS}.csv"
[ -f rolling_validation_results/trade_realized.csv ] && cp -p rolling_validation_results/trade_realized.csv "rolling_validation_results/trade_realized.unified${TS}.csv"
[ -f rolling_validation_results/validation_results.csv ] && cp -p rolling_validation_results/validation_results.csv "rolling_validation_results/validation_results.unified${TS}.csv"

echo "=== UNIFIED DONE $(date) ==="
