#!/bin/bash
# rank_decay 面 (2026-10-22): 选股内名次权重衰减, 首次系统性 bracket.
#
# 语义(portfolio.py:2044): rank_w = 1 - decay * i/(n-1), i=名次.
#   decay=0 ⇒ 等权; **生产=config 0.15**(代码默认0.30只是 fallback, 勿混):
#   n=3 时 0.15 ⇒ 1.0/0.925/0.85; 0.60 ⇒ 1.0/0.70/0.40; 0.90 ⇒ 1.0/0.55/0.10.
#   Aug-21 旧面 {0.05/0.15/0.30/0.45}=336/344/339/299 曾在旧锚定 0.15;
#   10-22 新锚重扫显示单调向高 decay (rk0 −22.7k / 0.45 +9.3k / 0.60 +25.5k)
#   ⇒ 必须补上沿 0.75/0.90 收面, 不能停在 0.60.
# 生产选股规模: 156 个选股日 {n=0:9, 1:61, 2:6, 3:80} ⇒ 86 日(55%) n≥2, 该旋钮活.
# 依据: 选股α定位(10/15)证明梯度100%在 resid 不在批内名次; 名次证伪(10/13)
#   证明批内名次零胜率梯度 ⇒ 预注册方向 = 降 decay 应不劣于生产.
#   census_1020d rk0 在旧锚 585,917 上 +34,441 (+5.9%), 未采纳(低于当时地板);
#   当前新锚 984,331 下必须重扫全 bracket, 不能只测单点.
#
# 基线: 生产 984,331.4340947575 / equity md5 350208d1 (live_20261021 产物).
# 纯组合层: 信号复用(不重生成), portfolio节豁免信号指纹.
#
# 预注册判据:
#   (a) ΔNAV ≥ +20k 才进入判读; 采纳线仍 150k 噪声地板.
#   (b) 逐年同号, 或单年 |Δ| < 60% 总 Δ.
#   (c) 臂独有选股 fwd10 α ≥ 基线独有 (n≥30) — 本轴选股集合应 100% 相同,
#       若 Jacc<1 则说明并非纯权重轴, 必须重查.
#   (d) ex-top: 剥离臂独有 top1 贡献后须仍 > 基线 (10/21 彩票判据).
#   (e) 胜率: 1/5/20 日 + 月度/年度, McNemar 只算不一致期.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/rk_1022.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base_rk1022
TARGET_NAV=984331.4340947575
TARGET_MD5=350208d1

echo "[$(date '+%F %T')] rk_1022 启动, 等bt退出…" >> $LOG
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
  "$PY" "$SCRATCH/datastate.py" rec rk_1022 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm id0    # identity: 必须逐位复现 984331.4340947575 / 350208d1
run_arm rk0    QUANT_RANK_DECAY=0
run_arm rk15   QUANT_RANK_DECAY=0.15
run_arm rk45   QUANT_RANK_DECAY=0.45
run_arm rk60   QUANT_RANK_DECAY=0.60
run_arm rk75   QUANT_RANK_DECAY=0.75
run_arm rk90   QUANT_RANK_DECAY=0.90

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
echo "[$(date '+%F %T')] rk_1022 完毕, 产物已还原基线态" >> $LOG
