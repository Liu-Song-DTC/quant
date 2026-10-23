#!/bin/bash
# 融资拥挤新仓加权臂 (2026-10-23 午, QUANT_MRG_TILT ladder).
#
# 依据 (margind_1023/2/3/4 realized-first 探针, anchor_1022 账簿/池):
#   个股融资余额 5 交易日变化 mchg5>=+3% 的已实现入场 33/166 笔: tW 75.8%/均ret
#   +6.35% vs 其余 59.4%/+3.90%; corr(mchg5,ret5b)=-0.10 非动量重编码; 动量分层
#   内边际±23pp; 池级 w20 Δ+17.5 p=0.0000; FAST/NORM 双 regime 正.
#   ⚠ 未入选候选拥挤桶 w5 34.5% < 其余 46.5% ⇒ 调节变量: 只做**已入选新仓**
#   相对加权 (w×(1+tilt), 归一化后总敞口不变), 选择集合不变 (Jaccard≡1).
#   两段结构: 2021-24 拥挤 90.5% vs 41.3% (+49pp); 2025-26 50.0% vs 67.5% (反转).
#
# 钩子: portfolio.py QUANT_MRG_TILT (默认0零行为) + QUANT_MRG_THR(3.0)/WIN(5)/AGE(10).
# 起点: 生产态=anchor_1022 (NAV 1,321,718.05, equity md5 cf222d55d4a81fcdf9d84fab2e531fc2).
# 协议: ①身份闸裸跑(无env)必须逐位=anchor_1022 (证明编辑零行为) → ②ladder
#       {0.10,0.20,0.30} 每臂: 还原锚产物→复用信号跑→存 .mrgT{X}→datastate 登记.
# 判据 (预注册): 胜率优先 tW/w5/w20 升; NAV 跌超 150k 噪声地板或 Sharpe 明显降 ⇒
#       不采纳; 触发数<10 ⇒ 臂无效; 信号CSV md5 变化 ⇒ 假臂中止.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/mrg_1023.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
ANCHOR_TAG=anchor_1022
ANCHOR_NAV=1321718.0546206068
ANCHOR_MD5=cf222d55d4a81fcdf9d84fab2e531fc2

restore_anchor () {
  for f in $PROD; do
    [ -f "$RVD/$f.$ANCHOR_TAG" ] && cp -p "$RVD/$f.$ANCHOR_TAG" "$RVD/$f"
  done
}
trap restore_anchor EXIT

echo "[$(date '+%F %T')] mrg 启动, 等待其它 bt 退出…" >> $LOG
# ⚠ 只等重负载本体进程 (bt/出单/日链): 等待模式不可匹配本链脚本名 (曾自死锁)
while pgrep -f 'bt_execution(_iso)?\.py|run_daily_live\.sh|generate_trade_orders\.py' > /dev/null 2>&1; do sleep 20; done
sleep 5

for f in $PROD; do
  [ -f "$RVD/$f.$ANCHOR_TAG" ] || { echo "[$(date '+%F %T')] ⛔ 缺锚快照 $f.$ANCHOR_TAG" >> $LOG; exit 6; }
done
NAV_CHK=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
if [ "$NAV_CHK" != "$ANCHOR_NAV" ]; then
  echo "[$(date '+%F %T')] ⛔ 生产NAV=$NAV_CHK != anchor_1022 $ANCHOR_NAV; 中止." >> $LOG; exit 7
fi
SIG=$RVD/backtest_signals.csv
SIGMD50=$(md5sum "$SIG" | cut -d' ' -f1)

# ── ① 身份闸: 裸跑必须逐位=anchor (证明 portfolio.py 编辑零行为) ──
echo "[$(date '+%F %T')] === identity: 无env裸跑 (期望逐位=anchor_1022) ===" >> $LOG
QUANT_ALT_NO_AUTOREFRESH=1 "$PY" -u bt_execution.py > "$LOGD/bt_mrgid.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] ⛔ identity rc=$rc" >> $LOG; exit $rc; fi
grep -q "复用已有信号" "$LOGD/bt_mrgid.log" || { echo "[$(date '+%F %T')] ⛔ identity 未复用信号 (指纹被破坏?)" >> $LOG; exit 3; }
[ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ] && { echo "[$(date '+%F %T')] ⛔ identity 信号CSV变化" >> $LOG; exit 4; }
ID_MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$ID_MD5" != "$ANCHOR_MD5" ]; then
  echo "[$(date '+%F %T')] ⛔ identity equity md5=$ID_MD5 != anchor $ANCHOR_MD5 — 编辑泄漏, 全链中止" >> $LOG; exit 5
fi
echo "[$(date '+%F %T')] ✅ identity 逐位=anchor_1022 (md5 ${ANCHOR_MD5:0:8})" >> $LOG

# ── ② ladder ──
for TILT in 0.10 0.20 0.30; do
  TAG="mrgT$(echo $TILT | tr -d '.')"
  restore_anchor
  echo "[$(date '+%F %T')] === $TAG: QUANT_MRG_TILT=$TILT thr=3.0 win=5 ===" >> $LOG
  env QUANT_MRG_TILT=$TILT QUANT_ALT_NO_AUTOREFRESH=1 \
      "$PY" -u bt_execution.py > "$LOGD/bt_$TAG.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[$(date '+%F %T')] ⛔ $TAG rc=$rc" >> $LOG; exit $rc; fi
  grep -q "复用已有信号" "$LOGD/bt_$TAG.log" || { echo "[$(date '+%F %T')] ⛔ $TAG 未复用信号" >> $LOG; exit 3; }
  [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD50" ] && { echo "[$(date '+%F %T')] ⛔ $TAG 信号CSV变化" >> $LOG; exit 4; }
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  NTRIG=$(grep -o "融资拥挤加权(MRG_TILT=[^)]*): [0-9]*" "$LOGD/bt_$TAG.log" | grep -o '[0-9]*$' | head -1)
  echo "[$(date '+%F %T')] $TAG NAV=$NAV md5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1 | cut -c1-8) 触发=${NTRIG:-0} (anchor $ANCHOR_NAV)" >> $LOG
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$TAG" 2>/dev/null; done
  "$PY" "$SCRATCH/datastate.py" rec mrg_1023 "$TAG" "$NAV" \
    "$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)" \
    "QUANT_MRG_TILT=$TILT thr3 win5; 触发=${NTRIG:-0}; vs anchor_1022" >> $LOG 2>&1
done
echo "[$(date '+%F %T')] ✅ mrg ladder 完成, 产物 .mrgT{010,020,030}; 生产还原 anchor_1022" >> $LOG
