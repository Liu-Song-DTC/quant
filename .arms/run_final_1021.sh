#!/bin/bash
# final_1021 (2026-10-21 19:30): d2 收尾 — zg 软罚双侧 + rank_weight_cap 左括号.
# 动机: zg0 在 mp3 锚下崩到 650,331 (−334k), 是迄今最强承重旋钮, 0.20 是否内点未知;
#   rank_weight_cap 右括号 0.35 已拒 (+17.6k/(b)✗/MDD恶化), 左括号 0.20 未测.
# 基线 = prod_mp3_1021 (984,331.43 / 8ea9a88d).
# 判据 (a)ΔNAV≥20k (b)逐年 (c)票级α (d)机制 (e)(f)(g)胜率; 独有 top1 ex-top.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/final_1021.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_mp3_1021
TARGET_MD5=8ea9a88d

echo "[$(date '+%F %T')] final_1021 启动, 等bt退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
for f in $PROD; do
  [ -f "$RVD/$f.$BASE_TAG" ] || { echo "⚠ 缺基线快照 $f.$BASE_TAG" >> $LOG; exit 1; }
done
BMD5=$(md5sum "$RVD/equity_curve.csv.$BASE_TAG" | cut -d' ' -f1)
case "$BMD5" in
  $TARGET_MD5*) : ;;
  *) echo "⛔ 基线快照 md5=${BMD5:0:8} 非 8ea9a88d — 中止" >> $LOG; exit 2 ;;
esac
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ]; then
    echo "[$(date '+%F %T')] $tag 已有产物, 跳过" >> $LOG; return 0
  fi
  for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
  echo "[$(date '+%F %T')] === $tag: ${*:-identity} ===" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 "$@" "$PY" -u bt_execution.py > "$LOGD/bt_$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ⚠ $tag rc=$rc (bt_$tag.log)" >> $LOG; return $rc
  fi
  if ! grep -q "复用已有信号" "$LOGD/bt_$tag.log"; then
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号 — 中止" >> $LOG; exit 3
  fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ] || \
     [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ]; then
    echo "[$(date '+%F %T')] ⛔ $tag 信号CSV变化 — 中止" >> $LOG; exit 4
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec final_1021 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm zg10_5 QUANT_ZG_PEN=0.10
run_arm zg40_5 QUANT_ZG_PEN=0.40
run_arm rwc20_5 QUANT_RANK_WEIGHT_CAP=0.20

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] final_1021 完毕, 产物已还原基线(mp3)态" >> $LOG
