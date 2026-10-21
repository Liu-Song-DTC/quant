#!/bin/bash
# identityf (10-21): 验证 QUANT_BP2_NORM_ONLY 钩子默认空=yaml(false) 零改动.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/identityf.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base1020bp3
TARGET=d6365c1a7a9c098eb98834638382cd07
echo "[$(date '+%F %T')] identityf 启动" >> $LOG
# 只认 comm=python* 的进程: pgrep -f 会匹配到"命令行文本含 bt_execution.py"的包装 shell,
# 10-21 曾因此把自身 heredoc 包装进程当回测, 死锁 10min. awk 的 /正则/ 不以文件形式存在.
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
env QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_identityf.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "⚠ rc=$rc" >> $LOG; exit $rc; fi
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.identityf" 2>/dev/null; done
MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
echo "[$(date '+%F %T')] identityf NAV=$NAV md5=${MD5:0:8}" >> $LOG
if [ "$MD5" != "$TARGET" ]; then echo "⛔ identityf 未复现 d6365c1a" >> $LOG; else echo "✅ identityf 逐位复现 d6365c1a" >> $LOG; fi
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] identityf 完毕, 产物还原基线" >> $LOG
