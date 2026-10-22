#!/usr/bin/env bash
# #366 概念历史回补 2021-2022 — 数据态臂 conceptbf_1022 (2026-10-22)
#
# 改动: data/concept_hist.pkl 前插 39 个全史概念的 2021-01-04..2022-12-30 日线
#   (17,192 行, 全部与现有 2023+ 重叠段逐位一致 Δ=0, |r|max ≤ 8.7%),
#   现有行一个不动。数据源=东财 board kline (经生产同款 relay).
# 语义: 生产 2021-2022 get_concept_heat 恒 0.5 (无历史 ⇒ 零信息);
#   回补后 heat 中位 ~0.01、98.7% 落 G3=0.4 档 ⇒ 同时是
#   (a) 因子获得截面信息 (2021-22 IC +3.5%, t=2.7)
#   (b) G3 档位从 0.6 降到 0.4 (与 2023+ 生产态一致) — 水平移动, 非纯增信息.
#   2023-01-03 因 EMA 尾随 |Δheat| 0.08; 2023-04-03 起逐位 0.
#
# 基线: 生产 984,331.4340947575 / equity md5 350208d1 / 1403日 / todate=2026-10-21.
# 本臂会重新生成信号 (~80min; concept_hist mtime 变化), 故不用复用快路径.
#
# 预注册判据:
#   (a) 机制面: 信号格变化必须集中在 2021-01..2023-01 (直接热尾随);
#       2023-04 之后的直接变化应为 0; 若有, 必须归因到 ML 重训 (特征样本变).
#   (b) 收益: ΔNAV vs 984,331; 采纳地板 150k. 同时看 CAGR/Sharpe.
#   (c) 胜率: 逐笔 + 日/5/20日 + 月度/年度; 短长都要不劣化 (用户判据).
#   (d) 逐年同号, 或单年主导必须披露.
#   (e) ex-top: 臂独有 top1 贡献剥离后仍 > 基线 (彩票判据).
#   (f) 选股 α: 臂独有票 fwd10 α ≥ 基线独有 (n≥30).
#
# 安全: armguard 隔离全部生产产物; concept_hist 用 cp -p 备份/还原(保留 mtime_ns),
#   trap 保证异常也还原.
set -u
source /mnt/d/quant/.arms/lib/armguard.sh

ROOT=/mnt/d/quant
HIST=$ROOT/data/concept_hist.pkl
BF=$ROOT/.arms/concept_hist_bf1022.pkl
BAK=$ROOT/.arms/backup/concept_hist.preBF_1022.pkl
LOGD=$ROOT/.arms/logs
LOG=$LOGD/conceptbf_1022.log

RVD=$ROOT/strategy/rolling_validation_results
mkdir -p "$ROOT/.arms/backup"
if [ ! -f "$BF" ]; then echo "⛔ 缺候选 $BF"; exit 2; fi

# 0. 等任何 bt 退出 + 校验生产基线 (不匹配就中止, 绝不拿错基线跑)
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
BNAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
BMD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
if [ "$BNAV" != "984331.4340947575" ] || [ "${BMD5:0:8}" != "350208d1" ]; then
  echo "⛔ 生产基线不符: NAV=$BNAV md5=${BMD5:0:8} (期望 984331.4340947575 / 350208d1) — 中止" > "${LOG:-$LOGD/conceptbf_1022.log}"
  exit 2
fi

# 1. 先备份生产 concept_hist (保留 mtime_ns), 校验 md5
cp -p "$HIST" "$BAK" || { echo "⛔ 备份 concept_hist 失败"; exit 2; }
M0=$(md5sum "$HIST" | cut -d' ' -f1)
M1=$(md5sum "$BAK" | cut -d' ' -f1)
T0=$(stat -c %y "$HIST" | cut -d. -f1)
if [ "$M0" != "$M1" ]; then echo "⛔ 备份 md5 不一致"; exit 2; fi
echo "[$(date '+%F %T')] concept_hist 备份 ok md5=${M0:0:8} mtime=$T0" > "$LOG"

# 2. 隔离生产产物 + 注册还原 trap
arm_begin conceptbf_1022 || exit 1
restore_hist() {
  cp -p "$BAK" "$HIST" 2>/dev/null
  local mm mt
  mm=$(md5sum "$HIST" | cut -d' ' -f1)
  mt=$(stat -c %y "$HIST" | cut -d. -f1)
  echo "[$(date '+%F %T')] concept_hist 还原 md5=${mm:0:8} mtime=$mt (应 $M0/$T0)" >> "$LOG"
}
trap 'arm_end conceptbf_1022; restore_hist' EXIT INT TERM

# 3. 换上回补数据 (mtime 变新 ⇒ 强制重生成信号)
cp "$BF" "$HIST"
echo "[$(date '+%F %T')] 已换入回补 concept_hist md5=$(md5sum "$HIST" | cut -d' ' -f1 | cut -c1-8)" >> "$LOG"

# 4. 跑臂 (todate 保持生产 2026-10-21)
cd "$ROOT/strategy" || exit 1
env QUANT_ALT_NO_AUTOREFRESH=1 /mnt/d/quant/.venv/bin/python -u bt_execution.py \
    > "$LOGD/bt_conceptbf_1022.log" 2>&1
RC=$?
echo "[$(date '+%F %T')] bt 结束 rc=$RC" >> "$LOG"
if [ $RC -eq 0 ]; then
  /mnt/d/quant/.venv/bin/python tools/navlaw.py snap conceptbf_1022 >> "$LOG" 2>&1 || true
  NAV=$(tail -1 rolling_validation_results/equity_curve.csv | cut -d, -f2)
  MD5=$(md5sum rolling_validation_results/equity_curve.csv | cut -d' ' -f1)
  echo "[$(date '+%F %T')] conceptbf_1022 NAV=$NAV md5=${MD5:0:8}" >> "$LOG"
  /mnt/d/quant/.venv/bin/python "$ROOT/.arms/datastate.py" rec conceptbf_1022 conceptbf "$NAV" "$MD5" \
      "concept_hist backfill 2021-22" >> "$LOG" 2>&1 || true
fi
exit $RC
