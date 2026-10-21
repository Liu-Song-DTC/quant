#!/bin/bash
# census_1020f (2026-10-21): E-K2 定向臂 — bp2加成在FAST期抑制.
# 动机: 修复态下 bp2_score_boost 面 = {0:717,296; 0.15:711,609; 0.30:715,101;
# 0.45:585,917} = 0~0.30 平台 + 0.45 断崖. 平台三臂在2022(+10~14pp)与
# 2026(+5.5~6.3pp)一致胜过0.45, 但都过不了预注册判据的全套(bp2_0: (b)2021
# 单年77%; bp2_15: (c)票级α; bp2_30: (b)2022单年64%).
# E-K2(`bp2_boost_norm_only`)是 9/6 预置旋钮: FAST期抑制加成(n_positions小,
# 排挤代价最高). 本次首次启用 — 若2022/2026劣势来自FAST期拥挤, 该臂应捕获
# 大部分增益且机制(d)自洽. 新env钩子 QUANT_BP2_NORM_ONLY (portfolio.py,
# 非信号指纹文件; 默认空=零改动). 基线=修复+bp3 (d6365c1a).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1020f.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
SIG=$RVD/backtest_signals.csv
BASE_TAG=base1020bp3
DS_LABEL=census_1020f
TARGET_MD5=d6365c1a7a9c098eb98834638382cd07

echo "[$(date '+%F %T')] census_1020f 启动, 等bt退出…" >> $LOG
# 只认 comm=python*: pgrep -f 会匹配命令行文本含 bt_execution.py 的包装 shell(10-21死锁)
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
[ -f "$RVD/equity_curve.csv.$BASE_TAG" ] || { echo "⚠ 缺基线快照" >> $LOG; exit 1; }
CUR=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$CUR" != "$TARGET_MD5" ]; then
  if [ "$CUR" == "$(md5sum "$RVD/equity_curve.csv.$BASE_TAG" | cut -d' ' -f1)" ]; then
    for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
  else
    echo "[$(date '+%F %T')] ⚠ 生产md5=$CUR 既非bp3也非快照, 拒绝跑臂" >> $LOG; exit 1
  fi
fi
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)

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
  if [ "$tag" == "identity" ] && [ "$MD5" != "$TARGET_MD5" ]; then
    echo "[$(date '+%F %T')] ⛔ identity 未复现 d6365c1a (得 ${MD5:0:8}) — 钩子非零改动, 中止" >> $LOG
    exit 2
  fi
}

run_arm identity
run_arm ek2     QUANT_BP2_NORM_ONLY=1
run_arm ek2_15  QUANT_BP2_NORM_ONLY=1 QUANT_BP2_BOOST=0.15

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1020f 完毕, 产物已还原基线(bp3)态" >> $LOG
