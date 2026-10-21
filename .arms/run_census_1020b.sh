#!/bin/bash
# 旋钮普查 census_1020 第二段 (2026-10-21 00:0x)
# 第一段教训: mp3/mp5/fec0 标签与 10/14-15 旧臂同名被"已有产物"误跳过(已删旧产物);
# bp3=585,917(+219k) ⇒ bp 面必须补 bp4 才能定峰。
# 本段 = 等第一段链退出 → mp3/mp5/fec0/bp4 串行 → 还原基线。
# 基线快照 $f.base1020fix 由第一段创建, 这里只校验存在。
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1020.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
SIG=$RVD/backtest_signals.csv
DS_LABEL=census_1020

echo "[$(date '+%F %T')] 第二段启动, 等第一段链+bt退出…" >> $LOG
while pgrep -f "run_census_1020.sh" > /dev/null; do sleep 30; done
while pgrep -f "bt_execution.py" > /dev/null; do sleep 30; done
for f in $PROD; do
  [ -f "$RVD/$f.base1020fix" ] || { echo "[$(date '+%F %T')] ⚠ 缺基线快照 $f.base1020fix" >> $LOG; exit 1; }
done
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ]; then
    echo "[$(date '+%F %T')] $tag 已有产物, 跳过" >> $LOG; return 0
  fi
  local now=$(date +%H%M)
  if [ "$now" -ge 400 ] && [ "$now" -le 2000 ]; then
    echo "[$(date '+%F %T')] ⚠ 过04:00截止, $tag 不启动" >> $LOG; return 0
  fi
  for f in $PROD; do cp -p "$RVD/$f.base1020fix" "$RVD/$f"; done
  echo "[$(date '+%F %T')] === $tag: $* ===" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 "$@" "$PY" -u bt_execution.py > "$LOGD/bt_$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ⚠ $tag rc=$rc (bt_$tag.log)" >> $LOG; return $rc
  fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ]; then
    echo "  ⚠ $tag 重写了信号CSV(不应发生!)" >> $LOG
    SIGMT0=$(stat -c %Y "$SIG")
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm mp3  QUANT_MAX_POSITIONS=3
run_arm mp5  QUANT_MAX_POSITIONS=5
run_arm fec0 QUANT_FAST_ENTRY_CAP=0
run_arm bp4  QUANT_BP_CONFIRM_MIN=4

for f in $PROD; do cp -p "$RVD/$f.base1020fix" "$RVD/$f"; done
echo "[$(date '+%F %T')] 第二段完毕, 产物已还原基线态" >> $LOG
