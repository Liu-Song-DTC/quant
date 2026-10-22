#!/bin/bash
# 采纳验证臂 (2026-10-22): new_entries_cap=2 进 yaml 默认 ⇒ 裸跑(无env)必须逐位复现 new2.
#
# 背景: cap 剂量面 不限 984,331 / cap2 1,098,169 / cap1 720,299 (同生产信号)
#   ⇒ 内部峰在 2. 采纳 = portfolio.py self.new_entry_cap + yaml selection.new_entries_cap: 2.
#   yaml portfolio 节在信号指纹豁免清单内 (_PORTFOLIO_LAYER_YAML_SECTIONS) ⇒ 信号应复用.
# 判据: (1) bt 日志出现"复用已有信号"; (2) equity/trade/selections md5 与 .new2 族逐位相同;
#       (3) NAV == 1,098,168.62.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
TAG=adopt_new2
LOG=$LOGD/adopt_new2_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022
EXPECT_NAV=1098168.62

echo "[$(date '+%F %T')] adopt_new2 启动, 等待其它 bt 退出…" >> $LOG
while pgrep -f 'bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py' > /dev/null 2>&1; do sleep 20; done
sleep 5

# 从基线产物态出发 (与其它臂同程序)
for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done

SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

echo "[$(date '+%F %T')] === $TAG: 裸跑, 无 QUANT_NEW_CAP (yaml 默认 2) ===" >> $LOG
env -u QUANT_NEW_CAP QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_$TAG.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] ⛔ rc=$rc (bt_$TAG.log)" >> $LOG; exit $rc; fi
if ! grep -q "复用已有信号" "$LOGD/bt_$TAG.log"; then
  echo "[$(date '+%F %T')] ⛔ 未复用信号 — yaml 指纹被破坏?" >> $LOG; exit 3
fi
if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ] || \
   [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ]; then
  echo "[$(date '+%F %T')] ⛔ 信号CSV变化" >> $LOG; exit 4
fi

NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
echo "[$(date '+%F %T')] $TAG NAV=$NAV (期望 $EXPECT_NAV)" >> $LOG
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$TAG" 2>/dev/null; done

fail=0
for f in equity_curve.csv trade_realized.csv portfolio_selections.csv; do
  a=$(md5sum "$RVD/$f" | cut -d' ' -f1); b=$(md5sum "$RVD/$f.new2" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then echo "  ✅ $f md5 ${a:0:8} == new2" >> $LOG
  else echo "  ⛔ $f md5 ${a:0:8} != new2 ${b:0:8}" >> $LOG; fail=1; fi
done
"$PY" "$SCRATCH/datastate.py" rec adopt_new2 "$TAG" "$NAV" "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" "yaml new_entries_cap=2, no env" >> $LOG 2>&1
if [ $fail -ne 0 ]; then echo "[$(date '+%F %T')] ⛔ 产物与 new2 不一致" >> $LOG; exit 6; fi
if [ "$NAV" != "$EXPECT_NAV" ]; then echo "[$(date '+%F %T')] ⛔ NAV 与 new2 不同" >> $LOG; exit 7; fi
echo "[$(date '+%F %T')] ✅ 采纳验证通过: 裸跑逐位复现 new2" >> $LOG
