#!/bin/bash
# 阶段6c(2026-09-11): PIT概念gate全链跑 — signal_engine加入concept_inception gate
# 基线: 979,885/291.95%/1.4094/18.33% (磁盘当前=基线态)
# 协议: 备份map+anchors+signals+sidecar(.prePIT0911) → 全链跑(因子+信号+回测,
#       signal_engine在指纹列表→全量重生成~90min) → grep四指标。
# 回滚(否决时): 还原代码(git)+还原备份产物+重跑基线。串行, 不要并行(会oom)。
set -u
cd /mnt/d/quant
VENV=.venv/bin/python
RVR=strategy/rolling_validation_results

# ---------- 0) 一次性备份 ----------
if [ ! -f "$RVR/backtest_signals.prePIT0911.csv" ]; then
    cp "$RVR/backtest_signals.csv" "$RVR/backtest_signals.prePIT0911.csv"
    cp "$RVR/.signal_code_fp" "$RVR/.signal_code_fp.prePIT0911" 2>/dev/null
    cp "$RVR/portfolio_selections.csv" "$RVR/portfolio_selections.prePIT0911.csv"
    cp "$RVR/trade_realized.csv" "$RVR/trade_realized.prePIT0911.csv"
    cp "$RVR/equity_curve.csv" "$RVR/equity_curve.prePIT0911.csv" 2>/dev/null
    echo "[backup] signals+sidecar+anchors -> .prePIT0911"
fi
# 基线四指标留档
BASELINE_NAV=$(tail -1 "$RVR/equity_curve.csv" | cut -d, -f2)
echo "[baseline] equity_curve尾值=$BASELINE_NAV (基线979,885) 957,435 979,885"

# ---------- 1) 全链跑 (PIT gate) ----------
echo "=== PIT gate 全链跑 开始 $(date '+%H:%M:%S') ==="
$VENV strategy/bt_execution.py > strategy/logs/bt_execution_pit_0911.log 2>&1
RC=$?
echo "=== 全链结束 rc=$RC $(date '+%H:%M:%S') ==="
grep -E "最终净值|Sharpe|最大回撤|总收益" strategy/logs/bt_execution_pit_0911.log | tail -8
grep -E "季度配置诊断" strategy/logs/bt_execution_pit_0911.log | tail -2
echo "=== done $(date '+%H:%M:%S') ==="
