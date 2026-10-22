#!/bin/bash
# 采纳验证 (2026-10-22): yaml ah_day_skip=true + portfolio.py self.ah_day_skip 裸跑.
# 必须从 base_tr1022 产物态 + 同一信号, 复现 ahday 臂:
#   NAV 1,321,718.0546206068 / equity md5 cf222d55d4a81fcdf9d84fab2e531fc2 /
#   AH-Day 34 次 / nSel 166.
# 成功: 产物存 .anchor_1022, 生产留在新锚 (不还原 new2).
# 失败: 还原 new2 锚并 exit.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/verify_anchor1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
EXP_NAV=1321718.0546206068
EXP_MD5=cf222d55d4a81fcdf9d84fab2e531fc2

restore_new2 () {
  for f in $PROD; do [ -f "$RVD/$f.new2" ] && cp -p "$RVD/$f.new2" "$RVD/$f"; done
}

echo "[$(date '+%F %T')] verify_anchor1022 启动 (裸跑 ah_day_skip=yaml)" >> $LOG
# 当前生产必须是 new2
[ "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" = "b9655c937edd6377574b0acd2ba59a6d" ] || {
  echo "⛔ 生产 equity 不是 new2 (b9655c93); 中止." >> $LOG; exit 5; }
for f in $PROD; do cp -p "$RVD/$f.base_tr1022" "$RVD/$f" || { echo "⛔ 缺 $f.base_tr1022" >> $LOG; restore_new2; exit 6; }; done

SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

echo "[$(date '+%F %T')] === 裸跑: 无 QUANT_AH_DAY_SKIP, yaml ah_day_skip=true, cap2 ===" >> $LOG
env QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_verify_1022.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "⛔ rc=$rc (bt_verify_1022.log)" >> $LOG; restore_new2; exit $rc; fi
if ! grep -q "复用已有信号" "$LOGD/bt_verify_1022.log"; then
  echo "⛔ 未复用信号" >> $LOG; restore_new2; exit 3; fi
[ "$(stat -c %Y "$SIG")" = "$SIGMT0" ] && [ "$(md5sum "$SIG" | cut -d' ' -f1)" = "$SIGMD50" ] || {
  echo "⛔ 信号CSV变化" >> $LOG; restore_new2; exit 4; }

NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
NAHD=$(grep -c '\[AH-Day\]' "$LOGD/bt_verify_1022.log" || true)
NSEL=$(($(wc -l < "$RVD/portfolio_selections.csv") - 1))
echo "[$(date '+%F %T')] 裸跑结果 NAV=$NAV md5=$MD5 AH触发=$NAHD nSel=$NSEL" >> $LOG

if [ "$NAV" = "$EXP_NAV" ] && [ "$MD5" = "$EXP_MD5" ]; then
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.anchor_1022"; done
  echo "[$(date '+%F %T')] ✅ 逐位复现 ahday 臂; 产物存 .anchor_1022; 生产=新锚" >> $LOG
else
  echo "[$(date '+%F %T')] ⛔ 未复现 (期望 NAV=$EXP_NAV md5=$EXP_MD5); 还原 new2" >> $LOG
  restore_new2
  exit 7
fi
