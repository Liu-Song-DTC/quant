#!/bin/bash
# 旋钮普查 census_1020d (2026-10-21): 旧采纳 yaml 评分闸 + 新env钩子复验.
# 接 census_1020c. 基线 = 修复+bp3 (md5 d6365c1a).
# 动机: 这些旋钮全部在 2026-09/10 的加性+前视+冻结bug时代采纳, 修复后从未复验:
#   · bp2_score_boost 0.45   (E-K1, 9/6)
#   · bp0bi0_penalty 0.05    (H-bi0, 10/10)
#   · entry_chan_gate bearhard / no_chan_penalty −0.20 (E-N5, 9/8)
#   · rank_decay 0.15        (核心评分, 从未 bracket)
# 第一臂 identity (无env) 必须逐位复现 d6365c1a, 否则中止 — 证明新钩子零改动.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1020d.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
SIG=$RVD/backtest_signals.csv
BASE_TAG=base1020bp3
DS_LABEL=census_1020d
TARGET_MD5=d6365c1a7a9c098eb98834638382cd07

echo "[$(date '+%F %T')] census_1020d 启动, 等 census_1020c + bt 退出…" >> $LOG
while pgrep -af "[r]un_census_1020c.sh" > /dev/null; do sleep 30; done
while pgrep -af "[b]t_execution.py" > /dev/null; do sleep 20; done

# 等上一段还原基线并校验
CUR=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$CUR" != "$TARGET_MD5" ]; then
  if [ "$CUR" == "$(md5sum "$RVD/equity_curve.csv.$BASE_TAG" | cut -d' ' -f1)" ]; then
    for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
  else
    echo "[$(date '+%F %T')] ⚠ 生产md5=$CUR 既非bp3也非快照, 拒绝跑臂" >> $LOG; exit 1
  fi
fi
[ -f "$RVD/equity_curve.csv.$BASE_TAG" ] || { echo "⚠ 缺基线快照" >> $LOG; exit 1; }
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)

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
  if [ "$tag" == "identity" ] && [ "$MD5" != "$TARGET_MD5" ]; then
    echo "[$(date '+%F %T')] ⛔ identity 未复现 d6365c1a (得 ${MD5:0:8}) — 新钩子非零改动, 中止" >> $LOG
    exit 2
  fi
}

run_arm identity
run_arm bp2_0    QUANT_BP2_BOOST=0
run_arm bp0bi0_0 QUANT_BP0BI0_PEN=0
run_arm ecg_off  QUANT_ENTRY_CHAN_GATE=off
run_arm ncp0     QUANT_NO_CHAN_PEN=0
run_arm rk0      QUANT_RANK_DECAY=0

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1020d 完毕, 产物已还原基线(bp3)态" >> $LOG
