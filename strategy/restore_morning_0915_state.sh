#!/bin/bash
# 恢复9/15晨态生产文件 (R2b取证后执行, 2026-09-17)
# 前提: R2b进程已退出。目标: rolling_validation_results回到晨态(1,728,548锚点),
#       C5批次A_repro臂以此态为基线。
set -e
RVD=/mnt/d/quant/strategy/rolling_validation_results
STATE=/mnt/d/quant/strategy/state_0915_todate0915

# 1. 晨态信号CSV (改名回 backtest_signals.csv)
cp "$STATE/backtest_signals.todate0915.csv" "$RVD/backtest_signals.csv"
# 2. sidecar: 晨态代码指纹 (warmup=0)
cp "$RVD/.signal_code_fp.morning0915.bak" "$RVD/.signal_code_fp"
# 3. 其余晨态产物
for f in equity_curve.csv portfolio_selections.csv regime_state.csv trade_realized.csv yaogu_watchlist.csv; do
  cp "$STATE/$f" "$RVD/$f"
done

echo "恢复完成:"
md5sum "$RVD/backtest_signals.csv" "$STATE/backtest_signals.todate0915.csv"
cat "$RVD/.signal_code_fp"
tail -2 "$RVD/equity_curve.csv"
