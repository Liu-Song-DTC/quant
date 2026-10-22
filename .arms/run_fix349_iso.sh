#!/bin/bash
# #349 iso 身份闸 (2026-10-23): 矩阵层宇宙修复 (剔除非个股死列) 零影响证明.
#   step0: bt_execution.py → bt_execution_iso.py (patch_349.py 同源补丁), 裸跑
#     (QUANT_ALT_NO_AUTOREFRESH=1) 必须逐位复现 anchor_1022
#     (NAV 1,321,718.0546206068 / md5 cf222d55d4a81fcdf9d84fab2e531fc2) 且信号复用.
#     iso 副本不在信号代码指纹里(指纹读原文件) ⇒ 复用门禁不受副本 mtime 影响.
#   期望日志: "矩阵宇宙: 剔除非个股序列 29 个" (sh000852/399006 + 27 裸代码;
#     sh000001 已被文件名守卫排除).
# 通过 ⇒ 同补丁施加生产 + 提交; 生产代价 = 下次每日流 80min 信号重生成 (常规).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/fix349_iso_1023.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
ANCHOR_TAG=anchor_1022
CTRL_NAV=1321718.0546206068
CTRL_MD5=cf222d55d4a81fcdf9d84fab2e531fc2
ISO=bt_execution_iso.py

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

echo "[$(date '+%F %T')] fix349 iso 启动, 等待其它 bt 退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py/ {f=1} END{exit !f}'; do sleep 20; done
sleep 5

BASE_NAV_CHK=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
if [ "$BASE_NAV_CHK" != "984331.4340947575" ]; then
  echo "⛔ 基线快照 NAV=$BASE_NAV_CHK != 984331.4340947575; 中止." >> $LOG; exit 7
fi

# step0a: 建 iso 副本并施加 #349 补丁
cp -p bt_execution.py "$ISO"
"$PY" "$SCRATCH/patch_349.py" "$ISO" >> $LOG 2>&1 || { echo "⛔ 补丁失败" >> $LOG; rm -f "$ISO"; exit 5; }
grep -q "已含 #349" "$LOG" 2>/dev/null && { echo "⛔ 生产已打过补丁? 中止" >> $LOG; rm -f "$ISO"; exit 6; }

# step0b: 裸跑身份闸
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG"); SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)
echo "[$(date '+%F %T')] === iso 裸跑 (#349 补丁, 无 QUANT_* 钩子) ===" >> $LOG
env QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u "$ISO" > "$LOGD/bt_fix349_iso.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "⛔ iso rc=$rc" >> $LOG; rm -f "$ISO"; exit $rc; fi
grep -q "复用已有信号" "$LOGD/bt_fix349_iso.log" || { echo "⛔ iso 未复用信号" >> $LOG; rm -f "$ISO"; exit 3; }
[ "$(md5sum "$SIG" | cut -d' ' -f1)" = "$SIGMD50" ] && [ "$(stat -c %Y "$SIG")" = "$SIGMT0" ] \
  || { echo "⛔ iso 信号CSV变化" >> $LOG; rm -f "$ISO"; exit 4; }
grep "矩阵宇宙: 剔除非个股序列" "$LOGD/bt_fix349_iso.log" >> $LOG || { echo "⚠ 未找到剔除日志行" >> $LOG; }

NAV0=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
MD50=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$NAV0" = "$CTRL_NAV" ] && [ "$MD50" = "$CTRL_MD5" ]; then
  echo "[$(date '+%F %T')] ✅ #349 iso 身份闸过: 逐位=anchor_1022 ($NAV0)" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec fix349_iso fix349_iso "$NAV0" "$MD50" \
    "矩阵层宇宙修复(剔除非个股死列) iso 零影响证明" >> $LOG 2>&1
else
  echo "[$(date '+%F %T')] ⛔ #349 iso 身份闸失败 NAV=$NAV0 md5=$MD50 (期望 $CTRL_NAV/$CTRL_MD5)" >> $LOG
  rm -f "$ISO"; exit 8
fi
rm -f "$ISO"
echo "[$(date '+%F %T')] ✅ fix349 iso 完成; 生产=anchor_1022 (trap), 副本已删" >> $LOG
