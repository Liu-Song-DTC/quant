#!/bin/bash
# #348 P0 对照 (2026-10-23): 生产代码裸跑 (无#348) × 新再生信号 → P0 产物.
# 用途: P1(iso+#348) ≠ anchor 时, P1−P0 = 纯 #348 效应; P0−anchor = 信号再生噪声.
# 前置: 信号已再生 (sidecar=27e6fc54), P1 已存 .fix348_iso 标签.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/fix348_p0.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
ANCHOR_TAG=anchor_1022
ARMTAG=fix348_p0

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

echo "[$(date '+%F %T')] P0 启动, 等待其它 bt 退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py/ {f=1} END{exit !f}'; do sleep 20; done
sleep 5

# 裸跑: 生产 bt_execution.py (含#349, 无#348), 复用再生信号
SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG"); SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)
echo "[$(date '+%F %T')] === P0 裸跑 (生产代码, 复用再生信号) ===" >> $LOG
env QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_fix348_p0.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "⛔ P0 rc=$rc" >> $LOG; exit $rc; fi
grep -q "复用已有信号" "$LOGD/bt_fix348_p0.log" || { echo "⛔ P0 未复用信号" >> $LOG; exit 3; }
[ "$(md5sum "$SIG" | cut -d' ' -f1)" = "$SIGMD50" ] && [ "$(stat -c %Y "$SIG")" = "$SIGMT0" ] \
  || { echo "⛔ P0 信号CSV变化" >> $LOG; exit 4; }

NAV0=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
MD50=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$ARMTAG"; done
"$PY" /mnt/d/quant/.arms/datastate.py rec fix348_p0 fix348_p0 "$NAV0" "$MD50" \
  "P0对照: 生产代码×再生信号 (隔离#348效应/再生噪声)" >> $LOG 2>&1
echo "[$(date '+%F %T')] ✅ P0 完成: NAV=$NAV0 md5=$MD50 → .$ARMTAG" >> $LOG
