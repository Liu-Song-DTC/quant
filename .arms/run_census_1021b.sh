#!/bin/bash
# census_1021b (2026-10-21): mp=3 新锚下的旋钮复验 (用户裁定"所有历史调优重想一遍").
#
# 背景: mp4→mp3 采纳后锚 711,609→984,331. mp3 直接改变:
#   - slack=(max_positions−n)/max_positions 的分母 (1/3,2/3 vs 1/4,1/2,3/4) ⇒ slk 面必复验;
#   - 槽位竞争强度 (FAST cap 与其同族) ⇒ fec 面复验;
#   - 单票权重放大 +33% ⇒ ea2/bp2 加成观感变化 ⇒ 复验.
# 另: E-LAD chan floor (0.60/0.45/0.15) 自 #365 修复后从未复扫, cfall0 先做灭活探针.
#
# 基线 = prod_mp3_1021 (yaml max_positions=3 直跑, 双闸应逐位复现 8ea9a88d).
# 预注册判据同 census_1020/1021a: (a) ΔNAV≥+20k; (b) 逐年同号或最大单年<60%;
# (c) 臂独有 fwd10 α ≥ 基线独有 (n≥30; n<30 信息性); (d) 机制一致.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1021b.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_mp3_1021
TARGET_MD5=8ea9a88d

echo "[$(date '+%F %T')] census_1021b 启动, 等bt退出…" >> $LOG
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
  "$PY" "$SCRATCH/datastate.py" rec census_1021b "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm id3
run_arm slk15  QUANT_SLACK_CAP=1.5
run_arm slk20  QUANT_SLACK_CAP=2.0
run_arm fec0   QUANT_FAST_ENTRY_CAP=0
run_arm fec2   QUANT_FAST_ENTRY_CAP=2
run_arm ea2old QUANT_EA2_RATIO=1.15
run_arm cfall0 QUANT_CHAN_FLOOR_NORM=0 QUANT_CHAN_FLOOR_FAST=0 QUANT_CHAN_FLOOR_BEAR=0
run_arm bp2z   QUANT_BP2_BOOST=0
run_arm bp2h   QUANT_BP2_BOOST=0.30

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1021b 完毕, 产物已还原基线(mp3)态" >> $LOG
