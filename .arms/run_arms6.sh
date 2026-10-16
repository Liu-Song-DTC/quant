#!/bin/bash
# 臂链 v6 (2026-10-16) — mp 面(max_positions)在新数据态+新信号集下复扫.
# 用法: bash /mnt/d/quant/.arms/run_arms6.sh mp
#
# 动机 — mp4 是**最陈旧的已采纳臂**: 10/14 采纳时的面 {3,4,5,6,8} 跑在
#   (a) 旧数据态(10-14), (b) 旧信号集(todate=10-14), (c) fec1/ea2/slk180 **全部未落地**.
#   而 max_positions 是三重耦合点:
#     1. slack = (max_positions − nsel)/max_positions 的**分母** ⇒ 与 slk180 直接耦合
#     2. fec1 把 FAST 日 nsel 压到 1 ⇒ 在 mp=4 下 slack=0.75(天花板 0.45); mp=3 下 slack=0.667
#     3. 槽位数本身决定每只的等权目标 1/mp
#   面形若已移位 ⇒ 直接采纳新点; 若 mp4 仍在平台中心 ⇒ 复验通过.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
SCRATCH=/mnt/d/quant/.arms
LOGD=$SCRATCH/logs
BASE_MD5=dca8678ba7ba928bb9ac272f09b6a55c
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
FACE="${1:-mp}"
LOG=$LOGD/chain_$FACE.log
# 数据态指纹(2026-10-16): 每条臂链开始时快照, 每条臂记录 (数据态, NAV, md5).
# ⇒ 下次数据刷新导致漂移时, `datastate.py diff <旧> <新>` 秒级列出变化文件, 不再靠回忆.
DS_LABEL="ds_$(date +%m%d_%H%M)_$FACE"
PY=/mnt/d/quant/.venv/bin/python

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ] && [ -f "$RVD/trade_realized.csv.$tag" ]; then
    echo "[$(date +%H:%M:%S)] --- $tag 已有产物, 跳过(还原为当前态) ---" >> $LOG
    for f in $PROD; do cp -p "$RVD/$f.$tag" "$RVD/$f" 2>/dev/null; done
    echo "  $tag NAV=$(tail -1 $RVD/equity_curve.csv | cut -d, -f2) md5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)" >> $LOG
    return 0
  fi
  echo "[$(date +%H:%M:%S)] --- $tag ($*) ---" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 "$@" /mnt/d/quant/.venv/bin/python -u bt_execution.py > $LOGD/bt_$tag.log 2>&1
  local RC=$?
  if [ $RC -ne 0 ]; then echo "  $tag 退出码=$RC !!失败" >> $LOG; tail -15 $LOGD/bt_$tag.log >> $LOG; return 1; fi
  for f in $PROD; do cp -p $RVD/$f $RVD/$f.$tag 2>/dev/null; done
  local NAV=$(tail -1 $RVD/equity_curve.csv | cut -d, -f2)
  local MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
  echo "  $tag NAV=$NAV md5=$MD5" >> $LOG
  $PY $SCRATCH/datastate.py rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$@" >> $LOG 2>&1
  awk '/=== 向量化回测结果 ===/,/净值曲线已保存/' $LOGD/bt_$tag.log | grep -aE "^(Sharpe|最大回撤|总收益|年化)" | sed 's/^/    /' >> $LOG
  grep -a "A1赢家修剪\|行业上限(\|最小持仓锁(" $LOGD/bt_$tag.log | sed 's/^/    /' >> $LOG
  return 0
}

echo "=== 臂链[$FACE] $(date +%m-%d_%H:%M:%S) 基线=anchor_1015d($BASE_MD5) ===" >> $LOG
echo "[$(date +%H:%M:%S)] 数据态快照 $DS_LABEL ..." >> $LOG
$PY $SCRATCH/datastate.py snap "$DS_LABEL" >> $LOG 2>&1
for f in $PROD; do [ -f $RVD/$f.anchor_1015d ] || cp -p $RVD/$f $RVD/$f.anchor_1015d 2>/dev/null; done

if [ "$FACE" = "mp" ]; then
  # 先复验 portfolio.py 新钩子默认路径零行为(QUANT_MAX_POSITIONS 未设 ⇒ 应逐字节 = anchor_1015d)
  run_arm c_ident
  MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
  if [ "$MD5" != "$BASE_MD5" ]; then
    echo "  ❌ 恒等失败! 期望 $BASE_MD5 实得 $MD5 — 中止" >> $LOG; exit 1
  fi
  echo "  ✅ 恒等烟测通过(QUANT_MAX_POSITIONS 钩子默认空=逐位零改动)" >> $LOG
  # ⚠ 左缘 c_mp2 = **从未探索过的点**: 旧面只跑过 {3,4,5,6,8}(mp3 是事后补的下界,
  #   起因是"6是不是太多"而非"能不能更少"). 而左坡**浅于右坡**(mp3 −27,085 vs mp5 −76,781)
  #   ⇒ 峰值可能不在 4 而在更左. 与 fec1(FAST日只取1只, +452k/+230k)同向.
  #   耦合警告: mp 同时是 slack=(mp−n)/mp 的**分母**. mp=2 时 nsel=1 → slack=0.5 →
  #   天花板 min(0.25×1.75, 0.25×1.8)=0.4375 (vs mp4 的 0.45, 仅差 2.8%, 可接受);
  #   但 mp=1 → slack=0 → 天花板塌回 0.25, 等于**整臂抹掉 slk180** ⇒ mp1 是混淆点, 不跑.
  run_arm c_mp2 QUANT_MAX_POSITIONS=2
  run_arm c_mp3 QUANT_MAX_POSITIONS=3
  run_arm c_mp5 QUANT_MAX_POSITIONS=5
  run_arm c_mp6 QUANT_MAX_POSITIONS=6
  echo "=== 臂链[$FACE] 完成 $(date +%m-%d_%H:%M:%S) ===" >> $LOG
  touch $SCRATCH/DONE_$FACE
else
  echo "未知 face: $FACE" >> $LOG; exit 2
fi
