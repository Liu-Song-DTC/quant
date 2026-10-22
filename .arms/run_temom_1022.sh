#!/bin/bash
# te×动量日型跳过臂 (2026-10-22 晚, 单臂; 排在 ahday 之后串行).
#
# 依据 (daytype_1022 只读探针, .arms/logs/daytype_*.log):
#   new2 实际入场 203 笔 / 129 日. 日型 = target_exposure>=0.85 且当日非持仓 top2 的
#   mom_60d 均值 > 0.10 → 45 笔 (22%): w5 26.7% / w20 46.7% / r20 +0.34%;
#   其余 158 笔 w5≈51.9%. 6/6 年 w5 同向 (te_hi×mom_hi 版). te 与 AH 独立:
#   非AH内 te_hi 45.3 vs te_lo 55.7; te_hi内 AH 21.4 vs 非AH 45.3.
#   te 单变量跳过砍 114/203 笔 (56%) 太粗 ⇒ 只做组合日型.
#
# 钩子: portfolio.py `QUANT_TEMOM_SKIP=1` (+ TEMOM_TE=0.85 / TEMOM_MOM=0.10).
# 起点: base_tr1022 产物态 + yaml new_entries_cap=2 ⇒ 对照 new2 (1,098,168.62).
# 判据: 胜率优先 (w5/w20/tW), 收益护栏 (NAV/Sharpe/MDD); NAV 跌超 150k 噪声地板
#       或 Sharpe 明显降 ⇒ 不采纳. 臂产物存 .temom, 结束还原生产锚.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
TAG=temom
LOG=$LOGD/temom_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
ANCHOR_TAG=new2
ANCHOR_NAV=1098168.6234241817
ANCHOR_MD5=b9655c937edd6377574b0acd2ba59a6d

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

echo "[$(date '+%F %T')] temom 启动, 等待其它 bt / ahday 退出…" >> $LOG
while pgrep -f 'bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py|run_ahday_1022\.sh' > /dev/null 2>&1; do sleep 20; done
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

echo "[$(date '+%F %T')] === $TAG: QUANT_TEMOM_SKIP=1 TE=0.85 MOM=0.10 (yaml cap2) ===" >> $LOG
env QUANT_TEMOM_SKIP=1 QUANT_TEMOM_TE=0.85 QUANT_TEMOM_MOM=0.10 \
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
NTRIG=$(grep -c "\[TEMOM-Day\]" "$LOGD/bt_$TAG.log" 2>/dev/null || true)
NSEL=$(($(wc -l < "$RVD/portfolio_selections.csv") - 1))
echo "[$(date '+%F %T')] $TAG NAV=$NAV md5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1 | cut -c1-8) " \
     "TEMOM触发=$NTRIG nSel=$NSEL (new2: NAV=$ANCHOR_NAV nSel=204)" >> $LOG
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$TAG" 2>/dev/null; done
"$PY" "$SCRATCH/datastate.py" rec temom_1022 "$TAG" "$NAV" \
  "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" \
  "QUANT_TEMOM_SKIP=1 TE=0.85 MOM=0.10 + yaml cap2; 触发=$NTRIG nSel=$NSEL" >> $LOG 2>&1

if [ "$NTRIG" -eq 0 ]; then
  echo "[$(date '+%F %T')] ⛔ TEMOM-Day 从未触发 — 臂无效 (检查 mom_60d/target_exposure)" >> $LOG; exit 8
fi
echo "[$(date '+%F %T')] ✅ $TAG 完成, 产物存 .$TAG; 生产还原到 anchor new2" >> $LOG
