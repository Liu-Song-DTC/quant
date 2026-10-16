#!/bin/bash
# 臂链 v10 (2026-10-16) — **复权口径**面: 加性前复权 qfq → 乘性重建总收益序列 Q.
# 用法: setsid nohup bash /mnt/d/quant/.arms/run_arms10.sh &
#
# 动机(记忆 `adjustment-method-affine-20261016`):
#   生产 backtrader_data/*_qfq.csv 是东财**加性**前复权 qfq(t)=(raw−C(t))/Π(t),
#   C = t **之后**的累计分红 ⇒ ① 越早的日期失真越大(闭式 ret_q/ret_n = 1/(1−C/(raw·Π)))
#   ② 锚在最新日 ⇒ 每次分红刷新**全史重写** ⇒ 曲线不可复现(重述).
#   实测失真: amp=σ(ret)/σ(ret_raw) 中位 1.0057 但 **>1.10 占 13.9%**, 尾部 7.2×.
#
#   正解 = **乘性重建**: 段内 ret≡raw, 除权日并入当日分红 D_d
#     J = Δqfq − a·Δraw = a·D_d   ⇒   Q(d)/Q(d−1) = (raw(d) + J/a) / raw(d−1)
#   J 只含**当日**分红, 不含未来累计 ⇒ **收益率与锚解耦** ⇒ 免疫重述.
#   (口径不能消除重述本身 — 锚仍在最新日 — 但能把**收益率**从锚里摘出来.)
#
# 数据: data/stock_data/backtrader_data_mul/ (5380 文件, 与生产**同宇宙**, 790M)
#   工具 data/rebuild_mult_qfq.py 两遍检测器(v5→union), 全市场 213s:
#     mult 595(乘性直通) | affine 4610(重建) | fallback 31(保留原 qfq) | read_fail 116 | short 26
#   守卫: 段内 C 单调违反 ≤5% + amp_Q ∈ [0.60,1.10].
#   验证 vs v1: **0 回归**, 59 只 fallback→affine, 4551 只逐位相同; amp>1.10: 680→15.
#   ⚠ read_fail/short 的 142 只是**原样拷贝**(宇宙不缩水), 2 个指数文件同.
#
# 臂:
#   adj_mul      QUANT_DATA_PATH=…/backtrader_data_mul — 乘性重建口径
#   adj_restore  QUANT_DATA_PATH 空(生产路径) — **必须复现 anchor_1015d md5**
#                (既是还原, 也是"换数据源→换回⇒逐字节回到锚点"的可复现性证明)
#
# ⚠ 两臂指纹不同(data_path 计入 _signal_code_fingerprint) ⇒ **各自**重生成信号
#   (~78min/臂), 串行 ~2h50m. 串行, 不要并行 (9/11 教训: 会 OOM).
# ⚠ 未跑臂前**不声称**"重建更好" — 四指标对比 anchor_1015d 后才可断言.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
SCRATCH=/mnt/d/quant/.arms
LOGD=$SCRATCH/logs
BASE_MD5=dca8678ba7ba928bb9ac272f09b6a55c
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
LOG=$LOGD/chain_adj.log
DS_LABEL="ds_$(date +%m%d_%H%M)_adj"
PY=/mnt/d/quant/.venv/bin/python
MUL=/mnt/d/quant/data/stock_data/backtrader_data_mul

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ] && [ -f "$RVD/trade_realized.csv.$tag" ]; then
    echo "[$(date +%H:%M:%S)] --- $tag 已有产物, 跳过(还原为当前态) ---" >> $LOG
    for f in $PROD; do cp -p "$RVD/$f.$tag" "$RVD/$f" 2>/dev/null; done
    local NAV=$(tail -1 $RVD/equity_curve.csv | cut -d, -f2)
    local MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
    echo "  $tag NAV=$NAV md5=$MD5" >> $LOG
    $PY $SCRATCH/datastate.py rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$@" >> $LOG 2>&1
    return 0
  fi
  echo "[$(date +%H:%M:%S)] --- $tag ($*) ---" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 "$@" /mnt/d/quant/.venv/bin/python -u bt_execution.py > $LOGD/bt_$tag.log 2>&1
  local RC=$?
  if [ $RC -ne 0 ]; then
    echo "  $tag 退出码=$RC !!失败" >> $LOG; tail -15 $LOGD/bt_$tag.log >> $LOG; return 1
  fi
  for f in $PROD; do cp -p $RVD/$f $RVD/$f.$tag 2>/dev/null; done
  local NAV=$(tail -1 $RVD/equity_curve.csv | cut -d, -f2)
  local MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
  echo "  $tag NAV=$NAV md5=$MD5" >> $LOG
  $PY $SCRATCH/datastate.py rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$@" >> $LOG 2>&1
  awk '/=== 向量化回测结果 ===/,/净值曲线已保存/' $LOGD/bt_$tag.log \
    | grep -aE "^(Sharpe|最大回撤|总收益|年化)" | sed 's/^/    /' >> $LOG
  return 0
}

echo "=== 臂链[adj] $(date +%m-%d_%H:%M:%S) 基线=anchor_1015d($BASE_MD5) ===" >> $LOG
echo "[$(date +%H:%M:%S)] 数据态快照 $DS_LABEL ..." >> $LOG
$PY $SCRATCH/datastate.py snap "$DS_LABEL" >> $LOG 2>&1
for f in $PROD; do [ -f $RVD/$f.anchor_1015d ] || cp -p $RVD/$f $RVD/$f.anchor_1015d 2>/dev/null; done

# ---- 臂1: 乘性重建口径 ----
run_arm adj_mul QUANT_DATA_PATH=$MUL

# ---- 臂2: 换回生产口径 (可复现性证明: 必须逐字节回到锚点) ----
run_arm adj_restore
MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
if [ "$MD5" = "$BASE_MD5" ]; then
  echo "  ✅ 换回生产口径 ⇒ 逐字节复现 anchor_1015d ⇒ 数据源钩子可逆" >> $LOG
else
  echo "  ⚠ 换回生产口径未复现锚点: 期望 $BASE_MD5 实得 $MD5 (查数据态漂移)" >> $LOG
fi

echo "=== 臂链[adj] 完成 $(date +%m-%d_%H:%M:%S) ===" >> $LOG
touch $SCRATCH/DONE_adj
