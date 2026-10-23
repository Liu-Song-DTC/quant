#!/bin/bash
# #348 phase-2 (2026-10-23): 首个 iso 链因 #349 落地后指纹 bust 必 exit 3 (强制再生).
# 本脚本接力: 等链退出 → 验证态 → 信号噪声普查 (sigcells/sigcmp) → 重跑 iso 链
# (sidecar=新代码态 27e6fc54 ⇒ 复用恢复) → P1 落盘待判读.
set -u
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/fix348_phase2.log
OLD_SIG_MD5=58fbcd1d7c76b3b83d8ed91ce8c39c8b
ANCHOR_EQ_MD5=cf222d55d4a81fcdf9d84fab2e531fc2

echo "[$(date '+%F %T')] phase2 启动, 等 iso bt 退出…" >> $LOG
while ps -eo args= | grep -E "bt_execution_iso\.py" | grep -v grep > /dev/null; do sleep 30; done
# 链脚本本身还要跑 trap/gate (秒级), 再等它退
while ps -eo args= | grep -E "run_fix348_iso\.sh" | grep -v grep > /dev/null; do sleep 10; done
sleep 5

echo "[$(date '+%F %T')] 链已退出, 验证态:" >> $LOG
NEW_SIG_MD5=$(md5sum "$RVD/backtest_signals.csv" | cut -d' ' -f1)
EQ_MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
FP=$(cat "$RVD/.signal_code_fp" 2>/dev/null)
echo "  signals md5=$NEW_SIG_MD5 (旧 $OLD_SIG_MD5)" >> $LOG
echo "  equity md5=$EQ_MD5 (锚 $ANCHOR_EQ_MD5)" >> $LOG
echo "  sidecar fp=$FP" >> $LOG
[ "$NEW_SIG_MD5" != "$OLD_SIG_MD5" ] || { echo "⛔ 新信号 md5 未变? 再生没发生" >> $LOG; exit 1; }
[ "$EQ_MD5" = "$ANCHOR_EQ_MD5" ] || echo "⚠ equity != 锚 (trap 未恢复? 手查)" >> $LOG

echo "[$(date '+%F %T')] === 信号噪声普查 (sigcells x2 + sigcmp, ~10min) ===" >> $LOG
cd /mnt/d/quant
"$PY" .arms/sigcells.py "$RVD/backtest_signals.csv.anchor_1022" .arms/sigdig/sig_anchor_1022.pkl >> $LOG 2>&1
"$PY" .arms/sigcells.py "$RVD/backtest_signals.csv" .arms/sigdig/sig_regen_1023.pkl >> $LOG 2>&1
"$PY" .arms/sigcmp.py .arms/sigdig/sig_anchor_1022.pkl .arms/sigdig/sig_regen_1023.pkl >> $LOG 2>&1

echo "[$(date '+%F %T')] === 重跑 iso 链 (复用新信号 → P1) ===" >> $LOG
bash /mnt/d/quant/.arms/run_fix348_iso.sh >> $LOG 2>&1
RC=$?
echo "[$(date '+%F %T')] iso 链重跑 rc=$RC (0=逐位零影响 / 10=有差待判读 / 其他=失败)" >> $LOG
if [ $RC -eq 10 ]; then
  echo "  P1 产物已存 .fix348_iso 标签; 下一步: P0 对照 (生产裸跑) 隔离 #348 效应" >> $LOG
fi
echo "[$(date '+%F %T')] phase2 完成" >> $LOG
