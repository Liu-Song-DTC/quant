#!/bin/bash
# 旋钮普查 census_1020c (2026-10-21 日盘): 800%时代其余"仍ON"的组合层闸复扫.
# 背景: census_1020 只覆盖 bp/mp/fec/slk/ea2 五族; 但生产还有四类在旧数据态
# 采纳、从未在修复态复验的**默认开启**闸:
#   · QUANT_ZG_FILTER=2 + QUANT_ZG_PEN=0.20  (E-ZG1b 中枢追高软罚, 9/23-24)
#   · QUANT_ML_GATE=0.15                     (K-10 ML顶桶软罚, 9/24)
#   · QUANT_AUG_GATE=1                       (G-2r 8月入场硬闸, 10/7)
#   · CLB consecutive_loss_breaker 0.50/3    (连亏减仓)
# 基线 = 当前生产态 (修复+bp3, md5 d6365c1a). 每臂 env 单变量, 其余不变.
# 判据同 census_1020 (预注册): ΔNAV≥+20k + 逐年 + 票级α; 判读用
# .arms/probe/judge_census_1020.py (改 TAGS/BASE 后跑).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1020c.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
SIG=$RVD/backtest_signals.csv
BASE_TAG=base1020bp3
DS_LABEL=census_1020c

echo "[$(date '+%F %T')] census_1020c 启动, 等 bt 退出…" >> $LOG
while pgrep -af "[b]t_execution.py" > /dev/null; do sleep 20; done

# 基线快照 = 当前生产(修复+bp3)
BASE_MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
case "$BASE_MD5" in
  d6365c1a*) ;;
  *) echo "[$(date '+%F %T')] ⚠ 生产 md5=$BASE_MD5 不是 bp3(d6365c1a), 拒绝跑臂" >> $LOG; exit 1;;
esac
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$BASE_TAG"; done
echo "[$(date '+%F %T')] 基线快照 $BASE_TAG md5=${BASE_MD5:0:8}" >> $LOG
"$PY" "$SCRATCH/datastate.py" rec "$DS_LABEL" "$BASE_TAG" \
  "$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)" "$BASE_MD5" "prod bp3 baseline" >> $LOG 2>&1
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ]; then
    echo "[$(date '+%F %T')] $tag 已有产物, 跳过" >> $LOG; return 0
  fi
  for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
  echo "[$(date '+%F %T')] === $tag: $* ===" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 "$@" "$PY" -u bt_execution.py > "$LOGD/bt_$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ⚠ $tag rc=$rc (bt_$tag.log)" >> $LOG; return $rc
  fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ]; then
    echo "  ⚠ $tag 重写了信号CSV(不应发生!)" >> $LOG
    SIGMT0=$(stat -c %Y "$SIG")
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm zg0   QUANT_ZG_FILTER=0
run_arm mlg0  QUANT_ML_GATE=0
run_arm mlg30 QUANT_ML_GATE=0.30
run_arm aug0  QUANT_AUG_GATE=0
run_arm clb1  QUANT_CLB_RED=1.0

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1020c 完毕, 产物已还原基线(bp3)态" >> $LOG
