#!/bin/bash
# 臂链 v7 (2026-10-16) — esp(entry_speed) 面在**新数据态+新信号集**下复扫.
# 用法: bash /mnt/d/quant/.arms/run_arms7.sh esp
#
# 动机(2026-10-16, 推翻旧否决的依据重审):
#   esp 面于 10/15 晚在**旧数据态(10-14)**跑出, 当时判"不采纳", 记忆里写的是
#   "Sharpe全段单调劣化(2.1100→1.9244)+MDD单调劣化(17.99→24.71)+七点无一CI排除0+2023率先崩".
#   ⚠ 今日重读臂产物发现该叙述**引用了错误的点**:
#     2.1100→1.9244 与 2023−7.79 都是 **es=1.00**(坍缩端点)的数字, 不是近区.
#     近区真实值(bt log 实读): es=0.85 Sharpe 2.0944 / 0.90 **2.0972** / 0.95 2.0810
#     ⇒ 相对基线 2.1100 只劣化 −0.013 / −0.029, **远小于 slk180 采纳时的 −0.035**.
#   近区真实画像(esp3 面, 基线 anchor_1015c 2,461,946):
#     es=0.85 2,523,894 +0.67pp CI[−0.36,+1.84] P=88.2% MDD −18.73% 六年全正
#     es=0.90 2,643,058 **+1.89pp** CI[−0.21,+4.40] P=95.9% MDD −19.28% **六年全正**
#     es=0.95 2,667,208 **+2.14pp** CI[−0.60,+5.40] P=92.3% MDD −19.73% 5/6年
#     es=1.00 2,312,857 −1.37pp CI[−7.54,+4.68] P=33.3% MDD −24.71% 2023−7.79
#   es=0.90 的剖面**与 slk180 采纳时几乎同型**(ΔNAV +181k vs +212k / MDD +1.29pp vs +1.23pp /
#   Sharpe −0.013 vs −0.035 / 六年全正), 唯一差别是 CI 下界 −0.21 微含 0.
#   **尾载检验更是全族最强**: 剔3月 +1.34 / 剔5月 +1.24 / 剔10月 +0.56,
#   而校准列里全部已采纳臂都是 −7 ~ −11 (见 an_tba2.py CALIB).
#   ⇒ 判据: 若新数据态复现 +1.8~2.1pp 且仍六年全正 ⇒ 两个独立数据态一致 ⇒ 旧 CI 含0 属单次
#     块bootstrap抽样伪影, 可采纳; 若塌到噪声级 ⇒ 旧否决成立, 轴彻底闭合.
#   ⚠ 已知风险: es→1.00 的坍缩是**现金闸边界**(账簿被钉在98%权益), 故 0.95 是锐边;
#     0.90 距边界更远 ⇒ 即便采纳也取 0.90 不取 0.95.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
SCRATCH=/mnt/d/quant/.arms
LOGD=$SCRATCH/logs
BASE_MD5=dca8678ba7ba928bb9ac272f09b6a55c
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
FACE="${1:-esp}"
LOG=$LOGD/chain_$FACE.log
# 数据态指纹(2026-10-16): 同 run_arms6.sh — 链首快照 + 每臂记录, 漂移可秒级归因.
DS_LABEL="ds_$(date +%m%d_%H%M)_$FACE"
PY=/mnt/d/quant/.venv/bin/python

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
  if [ $RC -ne 0 ]; then echo "  $tag 退出码=$RC !!失败" >> $LOG; tail -15 $LOGD/bt_$tag.log >> $LOG; return 1; fi
  for f in $PROD; do cp -p $RVD/$f $RVD/$f.$tag 2>/dev/null; done
  local NAV=$(tail -1 $RVD/equity_curve.csv | cut -d, -f2)
  local MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
  echo "  $tag NAV=$NAV md5=$MD5" >> $LOG
  $PY $SCRATCH/datastate.py rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$@" >> $LOG 2>&1
  awk '/=== 向量化回测结果 ===/,/净值曲线已保存/' $LOGD/bt_$tag.log | grep -aE "^(Sharpe|最大回撤|总收益|年化)" | sed 's/^/    /' >> $LOG
  return 0
}

echo "=== 臂链[$FACE] $(date +%m-%d_%H:%M:%S) 基线=anchor_1015d($BASE_MD5) ===" >> $LOG
echo "[$(date +%H:%M:%S)] 数据态快照 $DS_LABEL ..." >> $LOG
$PY $SCRATCH/datastate.py snap "$DS_LABEL" >> $LOG 2>&1
for f in $PROD; do [ -f $RVD/$f.anchor_1015d ] || cp -p $RVD/$f $RVD/$f.anchor_1015d 2>/dev/null; done

if [ "$FACE" = "esp" ]; then
  # entry_speed 在 yaml 的 **portfolio 节** ⇒ 被信号指纹剥离 ⇒ 改它不触发信号重生成, 臂仅 ~9 分钟.
  run_arm d_ident
  MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
  if [ "$MD5" != "$BASE_MD5" ]; then
    echo "  ❌ 恒等失败! 期望 $BASE_MD5 实得 $MD5 — 中止" >> $LOG; exit 1
  fi
  echo "  ✅ 恒等烟测通过(复现 anchor_1015d 逐字节)" >> $LOG

  run_arm d_esp85 QUANT_ENTRY_SPEED=0.85
  run_arm d_esp90 QUANT_ENTRY_SPEED=0.90
  run_arm d_esp95 QUANT_ENTRY_SPEED=0.95

  echo "=== 臂链[$FACE] 完成 $(date +%m-%d_%H:%M:%S) ===" >> $LOG
  touch $SCRATCH/DONE_$FACE
else
  echo "未知 face: $FACE" >> $LOG; exit 2
fi
