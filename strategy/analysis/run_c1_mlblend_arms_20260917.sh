#!/bin/bash
# C1: ML通道blend权重臂 (2026-09-17) — CSV注入→纯回测→记录→复原, 生产零残留
# 用法: bash analysis/run_c1_mlblend_arms_20260917.sh <pct1> [<pct2> ...]   (整数百分数: 30 50 55)
# 基线(9/15数据态, 晨流锚点): 1,728,548/591.42%/1.8402/19.98% (w0=0.4生产)
# 已知混淆(2026-09-17分析): CSV只有最终adjusted_score, ML blend之后还有
# 另类数据加性项A(alt_market≤0.15+龙虎榜×0.30)混入 — 逆向剥离把A并入s_clean,
# 重blend后A被(1-w1)/(1-w0)重缩放(如w1=0.3→A×1.17)。identity(w1=0.4)代数精确,
# 非identity臂=blend重权+A重缩放的混合物, 裁决时按此口径解读。
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
  # 注入 (python接受float权重)
  $PY analysis/edit_signals_mlblend_20260917.py "$($PY -c "print($PCT/100)")"
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
