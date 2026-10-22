#!/bin/bash
# 组合日型臂: AH日型跳过 + te×动量日型跳过 (2026-10-22 晚).
#
# 两钩子互相独立 (day-type 探针: 非AH内 te_hi 45.3 vs te_lo 55.7; te_hi内 AH 21.4 vs
# 非AH 45.3), 组合臂检验叠加增量. 起点与单臂相同: base_tr1022 产物态 + yaml cap2,
# 对照 = .ahday 产物 (NAV 1,321,718.05).
# 判据: 相对 ahday 胜率再升 (w5/w20/tW) 且 NAV 不跌超噪声地板 150k、Sharpe 不降.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
TAG=ahtemom
LOG=$LOGD/ahtemom_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
ANCHOR_TAG=new2
CTRL_NAV=1321718.0546206068

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

echo "[$(date '+%F %T')] ahtemom 启动, 等待其它 bt / temom 退出…" >> $LOG
while pgrep -f 'bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py|run_temom_1022\.sh|run_ahday_1022\.sh' > /dev/null 2>&1; do sleep 20; done
sleep 5

for f in $PROD; do
  [ -f "$RVD/$f.$BASE_TAG" ] || { echo "⛔ 缺基线快照 $f.$BASE_TAG" >> $LOG; exit 6; }
done
BASE_NAV_CHK=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
if [ "$BASE_NAV_CHK" != "984331.4340947575" ]; then
  echo "⛔ 基线快照 NAV=$BASE_NAV_CHK != 984331.4340947575; 中止." >> $LOG; exit 7
fi

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done

SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

echo "[$(date '+%F %T')] === $TAG: AH_DAY_SKIP=1 + TEMOM_SKIP=1 TE=0.85 MOM=0.10 (yaml cap2) ===" >> $LOG
env QUANT_AH_DAY_SKIP=1 QUANT_TEMOM_SKIP=1 QUANT_TEMOM_TE=0.85 QUANT_TEMOM_MOM=0.10 \
    QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_$TAG.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] ⛔ rc=$rc (bt_$TAG.log)" >> $LOG; exit $rc; fi
if ! grep -q "复用已有信号" "$LOGD/bt_$TAG.log"; then
  echo "[$(date '+%F %T')] ⛔ 未复用信号 — 指纹被破坏? (bt_$TAG.log)" >> $LOG; exit 3
fi
if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ] || \
   [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ]; then
  echo "[$(date '+%F %T')] ⛔ 信号CSV变化" >> $LOG; exit 4
fi

NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
NAHD=$(grep -c "\[AH-Day\]" "$LOGD/bt_$TAG.log" 2>/dev/null || true)
NTEM=$(grep -c "\[TEMOM-Day\]" "$LOGD/bt_$TAG.log" 2>/dev/null || true)
NSEL=$(($(wc -l < "$RVD/portfolio_selections.csv") - 1))
echo "[$(date '+%F %T')] $TAG NAV=$NAV md5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1 | cut -c1-8) " \
     "AH触发=$NAHD TEMOM触发=$NTEM nSel=$NSEL (ahday: NAV=$CTRL_NAV)" >> $LOG
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$TAG" 2>/dev/null; done
"$PY" "$SCRATCH/datastate.py" rec ahtemom_1022 "$TAG" "$NAV" \
  "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" \
  "AH_DAY_SKIP=1 + TEMOM_SKIP=1 TE=0.85 MOM=0.10 + yaml cap2; AH触发=$NAHD TEMOM触发=$NTEM nSel=$NSEL" >> $LOG 2>&1

if [ "$NAHD" -eq 0 ] || [ "$NTEM" -eq 0 ]; then
  echo "[$(date '+%F %T')] ⛔ 钩子未触发 (AH=$NAHD TEM=$NTEM) — 臂无效" >> $LOG; exit 8
fi
echo "[$(date '+%F %T')] ✅ $TAG 完成, 产物存 .$TAG; 生产还原到 anchor new2" >> $LOG
