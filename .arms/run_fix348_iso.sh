#!/bin/bash
# #348 iso 臂 (2026-10-23): 2元闸口径修复 VWAP → none.csv 真实收盘 (与实盘对齐).
#   step0: bt_execution.py → bt_execution_iso.py (patch_348.py 同源补丁), 裸跑
#     (QUANT_ALT_NO_AUTOREFRESH=1). 预期≈零账本影响 (probe: 2,157/6.88M 翻转格,
#     锚点 selections 交集 0, nr<=2 交集 0; 唯一 rank<=2 格 600226@2021-09-30 非决策日).
#   双出口:
#     逐位=anchor_1022 ⇒ 零影响证明 (exit 0)
#     有差 ⇒ 产物存 .fix348_iso 标签 + datastate rec (exit 10 = 待判读, 非失败)
#   期望日志: "2元闸口径(#348): 真实收盘 N 格 / 兜底VWAP M 格"
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/fix348_iso_1023.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
ANCHOR_TAG=anchor_1022
CTRL_NAV=1321718.0546206068
CTRL_MD5=cf222d55d4a81fcdf9d84fab2e531fc2
ISO=bt_execution_iso.py
ARMTAG=fix348_iso

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

echo "[$(date '+%F %T')] fix348 iso 启动, 等待其它 bt 退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py/ {f=1} END{exit !f}'; do sleep 20; done
sleep 5

BASE_NAV_CHK=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
if [ "$BASE_NAV_CHK" != "984331.4340947575" ]; then
  echo "⛔ 基线快照 NAV=$BASE_NAV_CHK != 984331.4340947575; 中止." >> $LOG; exit 7
fi

# step0a: 建 iso 副本并施加 #348 补丁
cp -p bt_execution.py "$ISO"
"$PY" "$SCRATCH/patch_348.py" "$ISO" >> $LOG 2>&1 || { echo "⛔ 补丁失败" >> $LOG; rm -f "$ISO"; exit 5; }
grep -q "已含 #348" "$LOG" 2>/dev/null && { echo "⛔ 生产已打过补丁? 中止" >> $LOG; rm -f "$ISO"; exit 6; }

# step0b: 裸跑 (无 QUANT_* 钩子)
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG"); SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)
echo "[$(date '+%F %T')] === iso 裸跑 (#348 补丁, 无 QUANT_* 钩子) ===" >> $LOG
env QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u "$ISO" > "$LOGD/bt_fix348_iso.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "⛔ iso rc=$rc" >> $LOG; rm -f "$ISO"; exit $rc; fi
grep -q "复用已有信号" "$LOGD/bt_fix348_iso.log" || { echo "⛔ iso 未复用信号" >> $LOG; rm -f "$ISO"; exit 3; }
[ "$(md5sum "$SIG" | cut -d' ' -f1)" = "$SIGMD50" ] && [ "$(stat -c %Y "$SIG")" = "$SIGMT0" ] \
  || { echo "⛔ iso 信号CSV变化" >> $LOG; rm -f "$ISO"; exit 4; }
grep "2元闸口径(#348)" "$LOGD/bt_fix348_iso.log" >> $LOG || { echo "⚠ 未找到覆盖度日志行" >> $LOG; }
grep -c "raw收盘读取失败" "$LOGD/bt_fix348_iso.log" | sed 's/^/  raw收盘读取失败行数: /' >> $LOG

NAV0=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
MD50=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$NAV0" = "$CTRL_NAV" ] && [ "$MD50" = "$CTRL_MD5" ]; then
  echo "[$(date '+%F %T')] ✅ #348 iso 身份闸过: 逐位=anchor_1022 ($NAV0) — 零影响证明" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec fix348_iso fix348_iso "$NAV0" "$MD50" \
    "2元闸真实收盘对齐实盘 iso 逐位=锚 (零影响)" >> $LOG 2>&1
  rm -f "$ISO"
  echo "[$(date '+%F %T')] ✅ fix348 iso 完成 (逐位), 生产=anchor_1022 (trap), 副本已删" >> $LOG
  exit 0
fi
# 有差: 存臂产物 (trap 前先存) + 登记, exit 10 = 待判读
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$ARMTAG"; done
"$PY" "$SCRATCH/datastate.py" rec fix348_iso fix348_iso "$NAV0" "$MD50" \
  "2元闸真实收盘对齐实盘 iso 有差 (待判读: NAV=$NAV0 vs 锚 $CTRL_NAV)" >> $LOG 2>&1
echo "[$(date '+%F %T')] ⚠ #348 iso 有差: NAV=$NAV0 (锚 $CTRL_NAV), 产物已存 .$ARMTAG — exit 10 待判读" >> $LOG
rm -f "$ISO"
exit 10
