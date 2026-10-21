#!/bin/bash
# census_1021d2 (2026-10-21): 评分层已采纳旋钮的单侧灭活/放宽复验 (mp3 锚).
# 用户裁定 "历史所有调优重新思考". 先做**移除/放宽**单侧: 若行为等价 ⇒ 旋钮惰性;
# 若变化明显 ⇒ 再补另一侧括号. 相比 8 臂全扫, 5 臂先筛, 降低重复扫参/过拟合.
#   min_rank_pct 0.3 → 0.2 / 0.4
#   turnover_bonus 0.1 → 0
#   rank_within_w 0.3 → 0.0
#   zg 软罚 0.20 → 关闭
# 基线 = prod_mp3_1021 (984,331.43 / 8ea9a88d).
# 判据: (a)ΔNAV≥20k (b)逐年 (c)票级α (d)机制 (e)(f)(g)胜率 (winrate_pair p<0.05);
#       ★新增: 若 NAV 增益集中于单笔 (arm-unique 少且 top1 大), 必做 ex-top 反事实,
#       剥离后 ≤ 基线 ⇒ 彩票, 不采纳.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1021d2.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_mp3_1021
TARGET_MD5=8ea9a88d

echo "[$(date '+%F %T')] census_1021d2 启动, 等bt退出…" >> $LOG
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
  "$PY" "$SCRATCH/datastate.py" rec census_1021d2 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm mrp20_4 QUANT_MIN_RANK_PCT=0.2
run_arm mrp40_4 QUANT_MIN_RANK_PCT=0.4
run_arm tb0_4   QUANT_TURNOVER_BONUS=0
run_arm rw0_4   QUANT_RANK_WITHIN_W=0.0
run_arm zg0_4   QUANT_ZG_FILTER=0

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1021d2 完毕, 产物已还原基线(mp3)态" >> $LOG
