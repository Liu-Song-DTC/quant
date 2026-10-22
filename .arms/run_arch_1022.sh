#!/bin/bash
# 架构面 arch_1022 (2026-10-22): 敞口平滑系数 bracket + 实盘 stateless 诊断臂.
#
# 背景 (读码/探针, 详见 .arms/arch_1022_PLAN.md):
#   1) PortfolioConstructor 是有状态的, 但 generate_trade_orders.py 每天新建实例且
#      从不 restore/save_tracking_state ⇒ 实盘丢失 4 条状态通道:
#        · 敞口平滑 current_exposure (每日回 1.0 ⇒ 目标敞口系统性 +8.6pp)
#        · 波动率降仓 (每日 _daily_returns 空 ⇒ 永不触发, 生产 16/140 调仓日绑定)
#        · 组合止损 (peak 每日重置 ⇒ drawdown≡0, 生产 20/147 调仓日绑定)
#        · 连亏熔断/入场日期 (生产 CLB=0 从未触发; 入场日期影响 min_hold 锁)
#      生产 telemetry /tmp/exposure_tel.csv 量化: live 语义的平滑重置使
#      E 均值 0.710→0.814 (+10.4pp, 146/147 调仓日更高).
#   2) 平滑常数 0.3 来自 f0e5692 (2026-05-02 大重构, 0.5→0.3), 从未在当前锚 bracket.
#      本脚本同时扫 w ∈ {0.15,0.30,0.50,0.70,1.00}.
#
# 语义 (portfolio.py, env 默认空 ⇒ 生产逐位零改动):
#   QUANT_EXP_SMOOTH=w   E_t = (1-w)E_{t-1} + w*target; 生产 w=0.7 (=0.3*prev+0.7*target),
#                        w=1 ⇒ 不平滑 (E_t=target); w=0 ⇒ 冻结在上次敞口
#   QUANT_LIVE_SMOOTH_RESET=1  只把 current_exposure 每日重置为 1.0 (其余状态保持连续)
#   QUANT_LIVE_STATELESS=1     每次 build 把跨日状态全重置为 __init__ 初值
#                              (只保留 _last_rebalance_date, 实盘由 rebuild_state 持久化)
#
# 基线: 984,331.4340947575 / equity md5 350208d1 (todate 2026-10-21).
# 纯组合层: 信号复用 (portfolio.py 不在信号指纹内), 每臂 ~9min.
#
# 预注册判据:
#   (a) id0 必须逐位复现基线; 每臂必须出现"复用已有信号"且信号 CSV mtime/md5 不变.
#   (b) 平滑面: ΔNAV ≥ +20k 才进入判读, 采纳线 150k 噪声地板; Sharpe 同看.
#   (c) 逐年同号, 或单年 |Δ| < 60% 总 Δ.
#   (d) 胜率: 1/5/20 日 + 月度/年度, 短长都不劣化 (McNemar 只算不一致期).
#   (e) ex-top: 臂独有 top1 贡献剥离后仍 > 基线.
#   (f) 选股集合 Jaccard: 平滑面应 =1.000 (纯权重轴); livestate 可 <1 (状态改选股).
#   (g) livestate 是**诊断臂**, 不作为采纳候选; 它量化"实盘现状"与回测的差距.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/arch_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_arch1022
TARGET_NAV=984331.4340947575
TARGET_MD5=350208d1

echo "[$(date '+%F %T')] arch_1022 启动, 等bt退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done

# ---- 基线快照 (首次运行) ----
if [ ! -f "$RVD/equity_curve.csv.$BASE_TAG" ]; then
  BNAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  BMD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  if [ "$BNAV" != "$TARGET_NAV" ] || [ "${BMD5:0:8}" != "$TARGET_MD5" ]; then
    echo "⛔ 生产基线不符: NAV=$BNAV md5=${BMD5:0:8} (期望 $TARGET_NAV / $TARGET_MD5) — 中止" >> $LOG
    exit 2
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$BASE_TAG"; done
  echo "[$(date '+%F %T')] 基线快照完成 $BNAV / ${BMD5:0:8}" >> $LOG
fi
BMD5=$(md5sum "$RVD/equity_curve.csv.$BASE_TAG" | cut -d' ' -f1)
case "$BMD5" in
  $TARGET_MD5*) : ;;
  *) echo "⛔ 基线快照 md5=${BMD5:0:8} 非 $TARGET_MD5 — 中止" >> $LOG; exit 2 ;;
esac
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
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号 — 中止" >> $LOG; exit 3
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
  "$PY" "$SCRATCH/datastate.py" rec arch_1022 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm id0       # identity: 必须逐位复现 984331.4340947575 / 350208d1
run_arm sm70      QUANT_EXP_SMOOTH=0.70   # 值对照: 必须与 id0 逐位相同 (走精确分支)
run_arm sm85      QUANT_EXP_SMOOTH=0.85
run_arm sm100     QUANT_EXP_SMOOTH=1.00   # 完全不平滑
run_arm sm50      QUANT_EXP_SMOOTH=0.50
run_arm sm15      QUANT_EXP_SMOOTH=0.15   # 强平滑 (接近冻结)
run_arm smonly    QUANT_LIVE_SMOOTH_RESET=1  # 诊断: 只复现平滑重置
run_arm livestate QUANT_LIVE_STATELESS=1     # 诊断: 复现实盘全部状态丢失

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] arch_1022 完毕, 产物已还原基线态" >> $LOG
