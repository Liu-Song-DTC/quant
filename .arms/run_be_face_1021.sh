#!/bin/bash
# be_face_1021 (2026-10-21): mp3 锚下 base_exposure 向下双侧包围.
#
# 发现: be75 (0.85→0.75) = 1,096,503 (+112,172, +11.4%), Sharpe 1.384 vs 1.294,
#   MDD −18.28% vs −18.43%, 逐年 5 好 1 平 (2026 −0.2pp), 日/5日/交易胜率均升.
#   Jacc 0.98 ⇒ 主要是资金路径差异而非换股彩票 — 但 +11% 远超 1 只独有票的解释力.
# 读码: mp3 下 target_exposure≥0.75 在纯调仓日常被 max_single 截断 ⇒ be75 与 0.85
#   理论等价; 差异只能来自**建仓期 partial fill / 1手门槛 / 账本路径** ⇒ 必须用面形
#   与隔离臂证明不是单调路径伪影. 面形向下扩展 {0.80,0.70,0.65,0.60}.
# 基线 = prod_mp3_1021 (984,331.43 / 8ea9a88d). 旧锚 be095/be100 是向上臂, 不复用.
# 判据 (a) ΔNAV≥+20k; (b) 逐年; (c) 票级; (d) 机制; (e)(f)(g) 胜率 (winrate_pair p<0.05).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/be_face_1021.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_mp3_1021
TARGET_MD5=8ea9a88d

echo "[$(date '+%F %T')] be_face_1021 启动, 等bt退出…" >> $LOG
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
  "$PY" "$SCRATCH/datastate.py" rec be_face_1021 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm be80 QUANT_BASE_EXPOSURE=0.80
run_arm be70 QUANT_BASE_EXPOSURE=0.70
run_arm be65 QUANT_BASE_EXPOSURE=0.65
run_arm be60 QUANT_BASE_EXPOSURE=0.60

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] be_face_1021 完毕, 产物已还原基线(mp3)态" >> $LOG
