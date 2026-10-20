#!/bin/bash
# 旋钮普查 census_1020 (2026-10-20夜) — 修复态(cb245b3)下重扫旧采纳旋钮
# 用户指令: "凌晨5点之前启动实盘就行，之前专心调优" + "把历史所有的调优修改，
#   都重新思考一遍，尤其是调优到800%的逻辑"
#
# 动机(800%审计): 现役组合层旋钮全部在 加性数据 + #365冻结bug 时代(≤10/15)采纳:
#   bp_confirm_min=2 (10/10, 闸被bug中和≈min=1 ⇒ 面形作废) / max_positions=4 (10/14)
#   / fast_entry_cap=1, ea2=2.0, slack=1.8 (10/15, 加性884%峰)。
#   cb245b3 修复后 bi_buy 活化 ⇒ bp_confirm 闸首次以设计语义运行, buy=True −10.6%
#   ⇒ 全栈从未在修复态校准。本普查按"从未复扫 + 与修复耦合度"排序。
#
# 臂 = 纯 portfolio 层 ⇒ 信号CSV复用(core/portfolio.py 不在 _signal_code_fingerprint,
#   yaml portfolio节豁免) ⇒ 每臂 ≈ 20-45min, 免78min重生成税。
# 基线 = 今晚纯修复态跑(22:06 启动, todate 10-20, 无env), 产物快照为 $f.base1020fix。
# 产物: $RVD/$f.$tag; 台账: datastate.py rec census_1020。
# 截止: 04:00 后不再**启动**新臂(留给判读+出单; 04:xx 在跑的臂让它跑完)。
# 用法: setsid nohup bash /mnt/d/quant/.arms/run_census_1020.sh &
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/census_1020.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
SIG=$RVD/backtest_signals.csv
DS_LABEL=census_1020

echo "[$(date '+%F %T')] 普查启动, 等待基线bt完成…" >> $LOG
T0=$(( $(date +%s) - 5400 ))   # bt 22:06 启动, 覆盖"脚本启动前已完成"的情形
while pgrep -f "bt_execution.py" > /dev/null; do sleep 60; done
sleep 45   # 产物落盘缓冲
if [ "$(stat -c %Y "$RVD/equity_curve.csv" 2>/dev/null || echo 0)" -le "$T0" ]; then
  echo "[$(date '+%F %T')] ⚠ equity_curve.csv 不新鲜(bt可能崩溃)! 不快照不跑臂, 手动处理" >> $LOG
  exit 1
fi
for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.base1020fix"; done
NAV0=$(tail -1 "$RVD/equity_curve.csv.base1020fix" | cut -d, -f2)
MD50=$(md5sum "$RVD/equity_curve.csv.base1020fix" | cut -d' ' -f1)
echo "[$(date '+%F %T')] 基线(纯修复态) NAV=$NAV0 md5=${MD50:0:8}" >> $LOG
$PY $SCRATCH/datastate.py snap base_1020fix >> $LOG 2>&1
SIGMT0=$(stat -c %Y "$SIG" 2>/dev/null || echo 0)

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ]; then
    echo "[$(date '+%F %T')] $tag 已有产物, 跳过" >> $LOG; return 0
  fi
  local now=$(date +%H%M)
  if [ "$now" -ge 400 ] && [ "$now" -le 2000 ]; then
    echo "[$(date '+%F %T')] ⚠ 过04:00截止, $tag 不启动" >> $LOG; return 0
  fi
  for f in $PROD; do cp -p "$RVD/$f.base1020fix" "$RVD/$f"; done   # 每臂从基线出发
  echo "[$(date '+%F %T')] === $tag: $* ===" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 "$@" "$PY" -u bt_execution.py > "$LOGD/bt_$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ⚠ $tag rc=$rc (bt_$tag.log)" >> $LOG; return $rc
  fi
  if [ "$(stat -c %Y "$SIG" 2>/dev/null || echo 0)" != "$SIGMT0" ]; then
    echo "  ⚠ $tag 重写了信号CSV(不应发生! 检查指纹门禁)" >> $LOG
    SIGMT0=$(stat -c %Y "$SIG")
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

# 优先级: 闸(从未复扫+修复直接消费) > 槽位(从未复扫) > 10/15三旋钮(10/16复扫过但仍是bug态)
run_arm bp1   QUANT_BP_CONFIRM_MIN=1
run_arm bp3   QUANT_BP_CONFIRM_MIN=3
run_arm mp3   QUANT_MAX_POSITIONS=3
run_arm mp5   QUANT_MAX_POSITIONS=5
run_arm fec0  QUANT_FAST_ENTRY_CAP=0
run_arm fec2  QUANT_FAST_ENTRY_CAP=2
run_arm slk15 QUANT_SLACK_CAP=1.5
run_arm ea2old QUANT_EA2_RATIO=1.15

# 收尾: 还原基线产物(判读若采纳某臂, 手动恢复该臂产物+设yaml)
for f in $PROD; do cp -p "$RVD/$f.base1020fix" "$RVD/$f"; done
echo "[$(date '+%F %T')] 普查完毕, 产物已还原基线态(NAV=$NAV0)" >> $LOG
