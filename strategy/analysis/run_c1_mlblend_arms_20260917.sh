#!/bin/bash
# C1: ML通道blend权重臂 (2026-09-19 v2注入) — CSV注入→纯回测→记录→复原, 生产零残留
# 用法: bash analysis/run_c1_mlblend_arms_20260917.sh <pct1> [<pct2> ...]   (整数百分数: 30 50 55)
# v2注入(edit_signals_mlblend_v2_20260919.py): A项(alt_market, date纯函数)逐日重算
# 并完整剥离 — mask行与non-mask行的A均原样保留 → 纯ML权重变更, v1的
# A×(1-w1)/(1-w0)重缩放confound已消除。identity烟测(w1=0.4)内置于v2脚本。
set -e
cd /mnt/d/quant/strategy
RVD=rolling_validation_results
ARMS=arms_20260917
PY=/mnt/d/quant/.venv/bin/python
TS=$(date +%Y%m%d_%H%M%S)
PRODS="portfolio_selections.csv trade_realized.csv equity_curve.csv regime_state.csv yaogu_watchlist.csv"

for PCT in "$@"; do
  TAG="$ARMS/C1_mlblend_$(printf %03d "$PCT")"
  INJ="backtest_signals.mlw${PCT}.csv"
  mkdir -p "$TAG"
  echo "===== C1 w=${PCT}% ====="
  for f in $PRODS backtest_signals.csv; do [ -f "$RVD/$f" ] && cp -p "$RVD/$f" "$TAG/pre_$f"; done
  # 注入 (python接受float权重, v2: A项精确剥离)
  $PY analysis/edit_signals_mlblend_v2_20260919.py "$($PY -c "print($PCT/100)")"
  [ -f "$RVD/$INJ" ] || { echo "注入产物缺失: $INJ"; exit 1; }
  cp -p "$RVD/$INJ" "$RVD/backtest_signals.csv"
  # 跑
  $PY bt_execution.py 2>&1 | tee "$TAG/run_$TS.log" | grep -E "最终净值|Sharpe:|最大回撤|复用已有信号" | tee "$TAG/metrics.txt"
  # 快照+复原
  for f in $PRODS; do [ -f "$RVD/$f" ] && cp -p "$RVD/$f" "$TAG/post_$f" && cp -p "$TAG/pre_$f" "$RVD/$f"; done
  cp -p "$TAG/pre_backtest_signals.csv" "$RVD/backtest_signals.csv"
  rm -f "$TAG/pre_backtest_signals.csv" "$TAG/post_backtest_signals.csv" "$RVD/$INJ"
  echo "完成: $TAG"
done
echo "=== C1全臂完成 ==="
