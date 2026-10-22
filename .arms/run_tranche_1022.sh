#!/bin/bash
# 执行层 P1: 分段建仓 tranche 组合层臂 (2026-10-22 夜, 架构面 arch_1022 后续).
#
# 背景:
#   bt_execution 已有 tranche 实现 (新入场拆首段开盘 + 第二段限价坑底加仓,
#   未成交余量下个调仓日开盘回补), yaml 默认关. 交易级证据: 限价成交笔 98% 更优,
#   3/5/8% 折价下半仓增益 +1.84~4.17pp — 从未在组合层验证, 实盘也未实现.
#   2026-10-22 新增 env 钩子 (不改 yaml backtest 节, 避免每臂 80min 信号重生成):
#     QUANT_TRANCHE=1  QUANT_TRANCHE_RATIO=r  QUANT_TRANCHE_DISCOUNT=d
#
# 前置: 今晚 run_daily_live.sh 已完成 (信号已按新 bt_execution mtime 重生成,
#       todate=今天). 本脚本必须在**实盘出单完成之后**运行, 不与 live 链并发.
#       本脚本只改 rolling_validation_results 产物, 不碰 trade_orders.json /
#       current_positions.json / live_*; 结束时还原基线产物.
#
# 预注册判据 (与 arch_1022 一致):
#   (a) id0 逐位复现当晚基线 NAV/md5; 每臂出现"复用已有信号"且信号 CSV mtime/md5 不变.
#   (b) ΔNAV ≥ +20k 才判读, 采纳线 150k (噪声地板); Sharpe/MDD 同看.
#   (c) 逐年同号, 或单年 |Δ| < 60% 总 Δ.
#   (d) 胜率: 1/5/20 日 + 月度/年度, 短长都不劣化 (McNemar 只算不一致期).
#   (e) ex-top: 剥离臂独有 top1 后仍 > 基线.
#   (f) Jaccard: tranche 改成交路径 ⇒ 选股集合可能变化 (不是纯权重轴), 记录.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/tranche_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_tr1022

echo "[$(date '+%F %T')] tranche_1022 启动, 等bt/live退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py|run_daily_live|generate_trade_orders\.py/ {f=1} END{exit !f}'; do sleep 20; done

# 前置: bt_execution.py 的 env 钩子改过 mtime ⇒ 信号必须比源码新(今晚链路已重生成).
# 若今晚是熊市空仓免算(precheck 跳过, 未重生成信号), 本脚本必须让位, 否则会触发
# 一次 80min 重生成并在"未复用信号"处中止.
SIG_PRE=/mnt/d/quant/strategy/rolling_validation_results/backtest_signals.csv
BT_SRC=/mnt/d/quant/strategy/bt_execution.py
if [ ! "$SIG_PRE" -nt "$BT_SRC" ]; then
  echo "⛔ 信号($(stat -c %y "$SIG_PRE" 2>/dev/null)) 不比 bt_execution.py($(stat -c %y "$BT_SRC")) 新: 今晚未重生成(可能是熊市空仓跳过), 本次不跑 tranche" >> $LOG
  exit 5
fi

# ---- 基线快照 (首次运行): 当晚生产产物 ----
if [ ! -f "$RVD/equity_curve.csv.$BASE_TAG" ]; then
  if ! grep -q "复用已有信号" "$LOGD/live_$(date +%Y%m%d).log" 2>/dev/null; then
    echo "⛔ 未找到当晚 live 日志的「复用已有信号」— 可能信号未重生成/链路未跑完; 先核对" >> $LOG
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$BASE_TAG"; done
  BNAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  BMD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] 基线快照 $BNAV / ${BMD5:0:8}" >> $LOG
fi
BASE_NAV=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
BASE_MD5=$(md5sum "$RVD/equity_curve.csv.$BASE_TAG" | cut -d' ' -f1)
SIG=$RVD/backtest_signals.csv
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

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
  if ! grep -q "复用已有信号" "$LOGD/bt_$tag.log"; then
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号 — 中止 (bt_$tag.log)" >> $LOG; exit 3
  fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ] || \
     [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ]; then
    echo "[$(date '+%F %T')] ⛔ $tag 信号CSV变化 — 中止" >> $LOG; exit 4
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec tranche_1022 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm id0
if [ "$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)" != "$BASE_NAV" ]; then
  echo "⛔ id0 未复现当晚基线 $BASE_NAV — 中止" >> $LOG; exit 2
fi

run_arm tr505  QUANT_TRANCHE=1 QUANT_TRANCHE_RATIO=0.5 QUANT_TRANCHE_DISCOUNT=0.05
run_arm tr508  QUANT_TRANCHE=1 QUANT_TRANCHE_RATIO=0.5 QUANT_TRANCHE_DISCOUNT=0.08
run_arm tr503  QUANT_TRANCHE=1 QUANT_TRANCHE_RATIO=0.5 QUANT_TRANCHE_DISCOUNT=0.03
run_arm tr305  QUANT_TRANCHE=1 QUANT_TRANCHE_RATIO=0.3 QUANT_TRANCHE_DISCOUNT=0.05
run_arm tr705  QUANT_TRANCHE=1 QUANT_TRANCHE_RATIO=0.7 QUANT_TRANCHE_DISCOUNT=0.05

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] tranche_1022 完毕, 产物已还原基线态" >> $LOG
