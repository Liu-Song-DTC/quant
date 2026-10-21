#!/bin/bash
# census_1021c (2026-10-21): mp3 新锚下"结构性耦合阈值"复扫.
#
# 动机(读码, 10-21 15:52): mp3 使
#   max_single=rank_weight_cap=0.25 成为约束 — 3选日 slack=0 ⇒ 实投 0.75/0.85;
#   industry_max_weight=0.40 — 2只同行业权重 2/3=67% > 40%, binding 更重;
#   base_exposure 0.85 — mp3 降低了实投敞口 ⇒ 必须用 be75 控制臂回答
#   "mp3 的增益是不是只是降敞口", 否则机制归因不成立.
# 这三者都是"结构性变更后依赖其阈值的闸须同臂重标定"的教科书案例(industry cap 注释 2105 已有先例).
#
# 基线 = prod_mp3_1021 (984,331.43 / 8ea9a88d).
# 预注册同前: (a) ΔNAV≥+20k; (b) 逐年同号或单年<60%; (c) 臂独有 fwd10 α ≥ 基线独有(n≥30);
# (d) 机制一致. 另: 若 be75 能解释 mp3 的大部分增益 ⇒ mp3 机制归因改写, 且两臂择优.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1021c.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_mp3_1021
TARGET_MD5=8ea9a88d

echo "[$(date '+%F %T')] census_1021c 启动, 等bt退出…" >> $LOG
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
  "$PY" "$SCRATCH/datastate.py" rec census_1021c "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm rwc20  QUANT_RANK_WEIGHT_CAP=0.20
run_arm rwc30  QUANT_RANK_WEIGHT_CAP=0.30
run_arm rwc35  QUANT_RANK_WEIGHT_CAP=0.35
run_arm imw50  QUANT_INDUSTRY_CAP=0.50
run_arm imw100 QUANT_INDUSTRY_CAP=1.0
run_arm be75   QUANT_BASE_EXPOSURE=0.75
run_arm tel3   QUANT_EXP_TEL=1
cp -f /tmp/exposure_tel.csv "$SCRATCH/armout/exposure_tel_mp3.csv" 2>/dev/null || true

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1021c 完毕, 产物已还原基线(mp3)态" >> $LOG
