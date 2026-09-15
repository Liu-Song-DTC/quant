#!/bin/bash
# E-seq1a烟测臂运行协议 (2026-09-15): swap→跑→记录→复原, 生产文件零残留
# 用法: bash analysis/run_seq1_arm_20260915.sh <arm_name> [gru_w] [gru_pkl] [mode]
#   arm_name=A(基线, 不动CSV) | B(注入gru_w)
#   mode=all(默认, 全行注入) | buys(仅buy行注入, 隔离持仓churn通道)
#   gru_pkl默认=gru_panelchain2_daily.pkl (P1-3f机制终版)
set -e
cd /mnt/d/quant/strategy
ARM="${1:?arm_name}"
W="${2:-0.32}"
PKL="${3:-rolling_validation_results/gru_panelchain2_daily.pkl}"
MODE="${4:-all}"
RVD=rolling_validation_results
SNAP="$RVD/seq1_arms/$ARM"
mkdir -p "$SNAP"
TS=$(date +%Y%m%d_%H%M%S)

# ---- 快照受run影响的生产产物 ----
for f in portfolio_selections.csv trade_realized.csv equity_curve.csv regime_state.csv yaogu_watchlist.csv backtest_signals.csv .signal_code_fp; do
  if [ -f "$RVD/$f" ]; then cp -p "$RVD/$f" "$SNAP/pre_$f"; fi
done

# ---- 注入臂: 替换signals CSV ----
if [ "$ARM" != "A" ]; then
  /mnt/d/quant/.venv/bin/python analysis/edit_signals_gru_20260915.py "$W" "$PKL" "$MODE"
  GRUF="backtest_signals.gru_w${W}"; [ "$MODE" = "buys" ] && GRUF="${GRUF}_buys"; [ "$MODE" = "mlchan" ] && GRUF="${GRUF}_mlchan"
  cp -p "$RVD/backtest_signals.csv" "$SNAP/pre_backtest_signals.prod.csv"
  cp -p "$RVD/$GRUF.csv" "$RVD/backtest_signals.csv"
  echo "注入完成 w=$W pkl=$PKL mode=$MODE"
fi

# ---- 跑回测 (复用路径: 指纹/数据未变 → 不重生成信号) ----
/mnt/d/quant/.venv/bin/python bt_execution.py 2>&1 | tee "$SNAP/run_$TS.log"

# ---- 记录四指标 ----
grep -E "最终净值|Sharpe:|最大回撤" "$SNAP/run_$TS.log" | tee "$SNAP/metrics.txt"

# ---- 产物快照 + 复原 ----
for f in portfolio_selections.csv trade_realized.csv equity_curve.csv regime_state.csv yaogu_watchlist.csv; do
  if [ -f "$RVD/$f" ]; then cp -p "$RVD/$f" "$SNAP/post_$f"; cp -p "$SNAP/pre_$f" "$RVD/$f"; fi
done
cp -p "$SNAP/pre_backtest_signals.csv" "$RVD/backtest_signals.csv"
echo "复原完成: $ARM"
