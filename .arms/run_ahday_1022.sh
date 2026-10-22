#!/bin/bash
# AH日型跳过臂 (2026-10-22 晚, 单臂).
#
# 依据 (池级 + 实现层, 只读探针):
#   new2 实际入场落在「当日非持仓前2名含 AH股」的 32 个调仓日共 38 笔:
#     w5 28.9% / w20 34.2% / r20 -5.17%
#   其余 165 笔: w5 50.3% / w20 64.8% / r20 +4.56%; 6/6年 w20 同向.
#   个股级 QUANT_AH_SKIP 臂失败 (补位同坏: new2ah NAV -70k) ⇒ 信号是日型, 不是个股.
#
# 钩子: portfolio.py `QUANT_AH_DAY_SKIP=1` — 当日排序后非持仓前2名含 AH股时,
#       跳过全部新入场 (含 bp2 保底槽); 持仓保留/加仓不受影响. 默认关闭.
# 起点: base_tr1022 产物态 + 当前 yaml (new_entries_cap=2) ⇒ 对照 new2 (1,098,168.62).
# 判据: 胜率优先 (w5/w20/tW), 收益护栏 (NAV/Sharpe/MDD); 臂产物存 .ahday, 结束还原生产锚.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
TAG=ahday
LOG=$LOGD/ahday_1022.log
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

echo "[$(date '+%F %T')] ahday 启动, 等待其它 bt 退出…" >> $LOG
while pgrep -f 'bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py' > /dev/null 2>&1; do sleep 20; done
sleep 5

# 基线快照检查
for f in $PROD; do
  [ -f "$RVD/$f.$BASE_TAG" ] || { echo "⛔ 缺基线快照 $f.$BASE_TAG" >> $LOG; exit 6; }
done
BASE_NAV_CHK=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
if [ "$BASE_NAV_CHK" != "984331.4340947575" ]; then
  echo "⛔ 基线快照 NAV=$BASE_NAV_CHK != 984331.4340947575; 中止." >> $LOG; exit 7
fi

# 从基线产物态出发
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done

SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

echo "[$(date '+%F %T')] === $TAG: QUANT_AH_DAY_SKIP=1 (yaml new_entries_cap=2) ===" >> $LOG
env QUANT_AH_DAY_SKIP=1 QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_$TAG.log" 2>&1
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
NSEL=$(($(wc -l < "$RVD/portfolio_selections.csv") - 1))
echo "[$(date '+%F %T')] $TAG NAV=$NAV md5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1 | cut -c1-8) " \
     "AH-Day触发=$NAHD nSel=$NSEL (new2: NAV=$ANCHOR_NAV nSel=204)" >> $LOG
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$TAG" 2>/dev/null; done
"$PY" "$SCRATCH/datastate.py" rec ahday_1022 "$TAG" "$NAV" \
  "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" \
  "QUANT_AH_DAY_SKIP=1 + yaml new_entries_cap=2; AH-Day触发=$NAHD nSel=$NSEL" >> $LOG 2>&1

if [ "$NAHD" -eq 0 ]; then
  echo "[$(date '+%F %T')] ⛔ AH-Day 从未触发 — 臂无效 (检查 industry 标签)" >> $LOG; exit 8
fi
echo "[$(date '+%F %T')] ✅ $TAG 完成, 产物存 .$TAG; 生产还原到 anchor new2" >> $LOG
