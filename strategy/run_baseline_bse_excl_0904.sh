#!/bin/bash
# 权威基线重跑 (2026-09-04): 北交所排除生效后的新基线
#   与9/3统一跑(882,340/252.94%/Sharpe 1.3460/dd 14.53%)唯一差异 = 股票池排除北交所(用户指令)
#   开关: exclude_fundamental_flaws=false (实验#27已证负贡献, 不启用)
#   串行约束: 单独一个重任务, detached 启动 (setsid nohup)
set -eu
cd /mnt/d/quant/strategy
PY=/mnt/d/quant/.venv/bin/python
mkdir -p logs
TS=$(date +%m%d_%H%M)
MAIN="logs/baseline_bse_excl_${TS}.log"
exec >> "$MAIN" 2>&1
echo "=== BASELINE_BSE_EXCL START $(date) ==="
grep -n "exclude_fundamental_flaws" config/factor_config.yaml
grep -n "^  todate:" config/factor_config.yaml

rm -f rolling_validation_results/backtest_signals.csv rolling_validation_results/portfolio_selections.csv
"$PY" bt_execution.py 2>&1 || echo "!!! bt 非零退出码 $?"
echo "--- 结果行 ---"
grep -E "最终净值|Sharpe:|最大回撤:" "$MAIN" | tail -5
echo "=== BASELINE_BSE_EXCL END $(date) ==="
