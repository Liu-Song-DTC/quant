#!/bin/bash
# census_1021a (2026-10-21): 新锚 711,609 下 bp闸×bp2加成交互 + 槽位复验.
#
# 动机: bp_confirm_min=3 是在 bp2_score_boost=0.45 时代采纳的. 0.45 的断崖
# (0~0.30≈712-717k vs 0.45=586k) 可能来自 "硬闸 × 过度抬升" 的交互: 闸越松,
# 被0.45抬到池顶的bp2越多. 现在加成降到0.15, 闸的最优值必须重测.
# 同时按 #359 在"年度稳定性"目标下复验 max_positions (bp3 时代 mp4 为峰).
#
# 基线 = prod_bp2v15_1021 (NAV 711,609.4465235589 / md5 b2b6c428),
# 即 yaml bp2_score_boost=0.15 的生产态. 信号CSV逐位复用(portfolio层豁免指纹).
#
# 预注册判据(先于结果, 同 census_1020): (a) ΔNAV≥+20k; (b) 逐年同号或
# 最大单年<60%·总Δ; (c) 臂独有票 fwd10 α ≥ 基线独有票 α (n≥30); (d) 机制一致.
# 全过才采纳; 否则维持当前锚.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1021a.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_bp2v15_1021
TARGET_MD5=b2b6c42883e7c8b94a785d6884077ec3

echo "[$(date '+%F %T')] census_1021a 启动, 等bt退出…" >> $LOG
# 只认 comm=python*: pgrep -f 会匹配命令行文本含 bt_execution.py 的包装 shell(10-21死锁)
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
for f in $PROD; do
  [ -f "$RVD/$f.$BASE_TAG" ] || { echo "⚠ 缺基线快照 $f.$BASE_TAG" >> $LOG; exit 1; }
done
CUR=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
BMD5=$(md5sum "$RVD/equity_curve.csv.$BASE_TAG" | cut -d' ' -f1)
if [ "$CUR" != "$TARGET_MD5" ] && [ "$CUR" != "$BMD5" ]; then
  echo "[$(date '+%F %T')] 生产md5=$CUR 既非锚也非快照 ⇒ 从快照还原" >> $LOG
  for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
fi
SIG=$RVD/backtest_signals.csv
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
  if ! grep -q "复用已有信号" "$LOGD/bt_$tag.log"; then
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号(可能触发重生成) — 中止" >> $LOG; exit 3
  fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ]; then
    echo "[$(date '+%F %T')] ⛔ $tag 信号CSV mtime变化 — 中止" >> $LOG; exit 4
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec census_1021a "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
  if [ "$tag" == "id15" ] && [ "$MD5" != "$TARGET_MD5" ]; then
    echo "[$(date '+%F %T')] ⛔ id15 未复现 b2b6c428 (得 ${MD5:0:8}) — 中止" >> $LOG
    exit 2
  fi
}

run_arm id15
run_arm bp2gate15 QUANT_BP_CONFIRM_MIN=2
run_arm bp4gate15 QUANT_BP_CONFIRM_MIN=4
run_arm mp3_15   QUANT_MAX_POSITIONS=3
run_arm mp5_15   QUANT_MAX_POSITIONS=5

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] census_1021a 完毕, 产物已还原基线(0.15)态" >> $LOG
