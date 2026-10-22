#!/bin/bash
# 换手率/ATR压缩 日型三臂 (2026-10-22 深夜). 预注册: winrate_PLAN.md §14.
#   step0 身份闸: 裸跑(无env)必须逐位复现 anchor_1022 (NAV 1,321,718.0546206068 /
#     md5 cf222d55d4a81fcdf9d84fab2e531fc2) — 钩子默认零行为证明.
#   step1 lowto1022: QUANT_LOWTO_SKIP=1 (top2最高换手<1.5% → 跳过日)
#   step2 atrz1022 : QUANT_ATRZ_SKIP=1 MEAN=-1.0 (top2均值atr_z20<-1.0 → 跳过日)
#   step3 deadw1022: QUANT_DEADW_SKIP=1 (死水日=换手max<1.5 且 atr_z20均值<0)
# 每臂: 恢复 base_tr1022 产物 → env 跑 → 复用信号检查 → 存 .{tag} → 台账.
# 判据 (§14): w5/tW 升且 w20 不降; NAV ≥ 1,171,718; Sharpe ≥ 1.49; 笔/年 ≥ 25.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/deadfest_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
ANCHOR_TAG=anchor_1022
CTRL_NAV=1321718.0546206068
CTRL_MD5=cf222d55d4a81fcdf9d84fab2e531fc2

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

restore_base () {
  for f in $PROD; do
    [ -f "$RVD/$f.$BASE_TAG" ] || { echo "⛔ 缺基线快照 $f.$BASE_TAG" >> $LOG; exit 6; }
    cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"
  done
}

run_arm () {  # $1=tag $2=打印名 $3..=env vars; 前置: 已 restore_base
  local TAG=$1 NAME=$2; shift 2
  local SIG=$RVD/backtest_signals.csv
  local SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
  local SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] === $TAG: $NAME ===" >> $LOG
  env "$@" QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_$TAG.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] ⛔ $TAG rc=$rc (bt_$TAG.log)" >> $LOG; exit $rc; fi
  if ! grep -q "复用已有信号" "$LOGD/bt_$TAG.log"; then
    echo "[$(date '+%F %T')] ⛔ $TAG 未复用信号" >> $LOG; exit 3; fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ] || \
     [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ]; then
    echo "[$(date '+%F %T')] ⛔ $TAG 信号CSV变化" >> $LOG; exit 4; fi
  local NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  local NLOW NDL NDW NSEL
  NLOW=$(grep -c "\[LOWTO-Day\]" "$LOGD/bt_$TAG.log" 2>/dev/null || true)
  NDL=$(grep -c "\[DEADW-Day\]" "$LOGD/bt_$TAG.log" 2>/dev/null || true)
  NDW=$(grep -c "\[ATRZ-Day\]" "$LOGD/bt_$TAG.log" 2>/dev/null || true)
  NSEL=$(($(wc -l < "$RVD/portfolio_selections.csv") - 1))
  echo "[$(date '+%F %T')] $TAG NAV=$NAV md5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1 | cut -c1-8) " \
       "触发 lowto=$NLOW atrz=$NDW deadw=$NDL nSel=$NSEL (anchor: $CTRL_NAV)" >> $LOG
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$TAG" 2>/dev/null; done
  "$PY" "$SCRATCH/datastate.py" rec deadfest_1022 "$TAG" "$NAV" \
    "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" \
    "$NAME; 触发lowto=$NLOW/atrz=$NDW/deadw=$NDL nSel=$NSEL" >> $LOG 2>&1
}

echo "[$(date '+%F %T')] deadfest 启动, 等待其它 bt 退出…" >> $LOG
while pgrep -f 'bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py' > /dev/null 2>&1; do sleep 20; done
sleep 5

# ---- step0: 身份闸 (裸跑 = anchor_1022) ----
BASE_NAV_CHK=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
if [ "$BASE_NAV_CHK" != "984331.4340947575" ]; then
  echo "⛔ 基线快照 NAV=$BASE_NAV_CHK != 984331.4340947575; 中止." >> $LOG; exit 7
fi
restore_base
SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0); SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)
echo "[$(date '+%F %T')] === step0 身份闸: 裸跑 (无 QUANT_* 钩子) ===" >> $LOG
env QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_identity_1022.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "⛔ 身份闸 rc=$rc" >> $LOG; exit $rc; fi
grep -q "复用已有信号" "$LOGD/bt_identity_1022.log" || { echo "⛔ 身份闸未复用信号" >> $LOG; exit 3; }
[ "$(md5sum "$SIG" | cut -d' ' -f1)" = "$SIGMD50" ] || { echo "⛔ 身份闸信号CSV变化" >> $LOG; exit 4; }
NAV0=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
MD50=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$NAV0" = "$CTRL_NAV" ] && [ "$MD50" = "$CTRL_MD5" ]; then
  echo "[$(date '+%F %T')] ✅ 身份闸过: 裸跑=anchor_1022 逐位 ($NAV0)" >> $LOG
else
  echo "[$(date '+%F %T')] ⛔ 身份闸失败 NAV=$NAV0 md5=$MD50 (期望 $CTRL_NAV/$CTRL_MD5); 钩子非零行为?" >> $LOG
  restore_anchor; exit 8
fi

# ---- step1-3: 三臂 ----
restore_base
run_arm lowto1022 "LOWTO top2最高换手<1.5%" QUANT_LOWTO_SKIP=1 QUANT_LOWTO_MAXTR=1.5
restore_base
run_arm atrz1022  "ATRZ top2均值atr_z20<-1.0" QUANT_ATRZ_SKIP=1 QUANT_ATRZ_MEAN=-1.0
restore_base
run_arm deadw1022 "DEADW 死水日(换手<1.5 且 atrz<0)" QUANT_DEADW_SKIP=1 QUANT_DEADW_MAXTR=1.5 QUANT_DEADW_ATRZ=0.0

# ---- step4: 判读 ----
echo "[$(date '+%F %T')] === 判读: winrate_arm anchor_1022 vs 三臂 ===" >> $LOG
"$PY" "$SCRATCH/probe/winrate_arm_1022.py" anchor_1022 lowto1022 atrz1022 deadw1022 \
  >> "$LOGD/deadfest_read_1022.log" 2>&1 || echo "⚠ 判读脚本 rc=$?" >> $LOG
cat "$LOGD/deadfest_read_1022.log" >> $LOG
echo "[$(date '+%F %T')] ✅ deadfest 完成; 生产=anchor_1022 (trap)" >> $LOG
