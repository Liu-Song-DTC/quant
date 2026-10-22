#!/bin/bash
# 选股胜率轴 确认臂 (2026-10-22 深夜): cap 剂量镜像.
#
# 目的: 区分两种解释
#   (a) "新入场越少越好"          → cap=1 应不差于 cap=2 (单调)
#   (b) "保留前2名(第2名是最强桶)" → cap=1 应明显更差 (内部峰在 2)
# 池级证据 (.arms/logs/pool_rank_1022.log): 非持仓候选行序
#   R1 w5 40.6 / R2 53.7 / R3 38.5 / R4 42.6; R2 vs R3 w5 McNemar p=0.0086
#   R1 vs R2 w5 p=0.0105 (R2 最好) ⇒ 预期 cap=1 丢掉 R2, 应更差.
#
# 单臂 cap1 (QUANT_NEW_CAP=1), 复用生产信号 CSV (portfolio-only 指纹豁免).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/capdose_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
BASE_NAV=984331.4340947575
BASE_MD5=350208d1

echo "[$(date '+%F %T')] capdose_1022 启动, 等待其它 bt 退出…" >> $LOG
while pgrep -f 'run_selcap_1022.sh|run_w730_1022.sh|bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py' > /dev/null 2>&1; do sleep 20; done
sleep 5

PMD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "${PMD5:0:8}" != "$BASE_MD5" ]; then
  echo "⛔ 生产产物 md5=${PMD5:0:8} != base ${BASE_MD5}; 中止." >> $LOG
  exit 5
fi
BASE_NAV_CHK=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
if [ "$BASE_NAV_CHK" != "$BASE_NAV" ]; then
  echo "⛔ 基线快照 NAV=$BASE_NAV_CHK != $BASE_NAV; 中止." >> $LOG; exit 7
fi

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
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号 — 中止 (bt_$tag.log)" >> $LOG; exit 3
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
  "$PY" "$SCRATCH/datastate.py" rec capdose_1022 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm cap1 QUANT_NEW_CAP=1

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] capdose_1022 完毕, 产物已还原基线态" >> $LOG
