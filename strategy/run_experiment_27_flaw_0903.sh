#!/bin/bash
# 实验#27 基本面硬伤过滤 (2026-09-03): exclude_fundamental_flaws=true
#   与基线唯一差异 = 配置开关 (factor_config.yaml 修改会触发因子缓存指纹失效 -> 现算, 诚实)
#   基线: 882,340 / 252.94% / Sharpe 1.3460 / dd 14.53% (9/3统一跑)
#   串行约束: 单独一个重任务, detached 启动 (setsid nohup, 防池fork时刻被杀)
set -eu
cd /mnt/d/quant/strategy
PY=/mnt/d/quant/.venv/bin/python
mkdir -p logs
TS=$(date +%m%d_%H%M)
MAIN="logs/exp27_flaw_${TS}.log"
exec >> "$MAIN" 2>&1
echo "=== EXP27 START $(date) ==="
grep -n "exclude_fundamental_flaws" config/factor_config.yaml
grep -n "^  todate:" config/factor_config.yaml

rm -f rolling_validation_results/backtest_signals.csv rolling_validation_results/portfolio_selections.csv
"$PY" bt_execution.py 2>&1 || echo "!!! bt 非零退出码 $?"
echo "--- 结果行 ---"
grep -E "最终净值|Sharpe:|最大回撤:" "$MAIN" | tail -5
echo "--- 拒绝统计 ---"
grep -E "fund_flaw|reject_" "$MAIN" | tail -8
echo "=== EXP27 END $(date) ==="
