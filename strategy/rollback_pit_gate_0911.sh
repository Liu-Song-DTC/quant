#!/bin/bash
# 阶段6c回滚 (仅当PIT gate结果不可接受时用): 还原产物到基线态
# 代码回滚用 git checkout strategy/core/signal_engine.py strategy/bt_execution.py
# 之后重跑bt_execution.py自会重生成基线信号(fp不匹配)。
set -u
cd /mnt/d/quant
RVR=strategy/rolling_validation_results
for f in backtest_signals portfolio_selections trade_realized equity_curve; do
    if [ -f "$RVR/$f.prePIT0911.csv" ]; then
        cp "$RVR/$f.prePIT0911.csv" "$RVR/$f.csv"
        echo "[restore] $f.csv"
    fi
done
[ -f "$RVR/.signal_code_fp.prePIT0911" ] && cp "$RVR/.signal_code_fp.prePIT0911" "$RVR/.signal_code_fp" && echo "[restore] .signal_code_fp"
echo "代码回滚(如需要): git checkout -- strategy/core/signal_engine.py strategy/bt_execution.py"
