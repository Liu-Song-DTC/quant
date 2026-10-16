#!/bin/bash
# 臂链 v9 (2026-10-16) — 概念映射**版本化**(as-of 成分股归属)面.
# 用法: setsid nohup bash /mnt/d/quant/.arms/run_arms9.sh cmhist &
#
# 动机(用户 2026-10-16 "概念PIT前视距也要修" + "稳定可复现的收益曲线"):
#   前视根因(已定位到行): core/concept_heat.py 的 get_concept_heat 是**日期无关**的 —
#     self._concept_scores (概念涨跌幅) 是 as-of 的 ✓
#     self._stock_concepts (哪只股票属于哪个概念) 是**今天的静态名单** ✗
#   ⇒ 拿今天的成分股回测历史 = 前视; 且每次刷新全史重写 = 曲线不可复现.
#
#   churn 实测(本次, git 6 版): 共同股票中"归属集合完全不变"的比例
#     5/31→9/3(4月) 13.7% | 9/3→9/4(1天) 82.6% | 9/4→9/17 69.2% | 9/17→10/4 71.6% | 10/4→10/16 73.7%
#   成员 Jaccard 中位 0.96~1.00(变动温和), 但"集合相等"是严格判定 ⇒
#   每股 12.7 个概念里增删 1 个就翻转 ⇒ 每个刷新周期约 20~30% 股票 concept_heat 改变
#   ⇒ 每 10 天一个调仓周期, 每次刷新全曲线重写. **这就是 -38.3% 那类漂移的主引擎.**
#
# 修法: data/concept_map_history.pkl (6 版: 5/31, 9/3, 9/4, 9/17, 10/4, 10/16)
#       决策日 τ ⇒ 取"截止 τ 的最近版本"名单, 与 concept_inception 的成立日 gate 同构.
#       钩子 QUANT_CONCEPT_MAP_HIST(默认空=逐位零改动), 已计入 _signal_code_fingerprint.
#
# ⚠ 16:05 追加第二处 (更一阶): core/signal_engine.py `_get_specific_industry`
#   用 STOCK_CONCEPT_MAP[code] 决定该股套用**哪套季度因子权重** — 有成立日 gate,
#   无成分股归属 gate. 实测敏感度 26.0% 的 (股票×季度) 单元选中不同配置
#   (134,448 单元, 2026Q3 达 34.4%) ⇒ 比 concept_heat 更直接驱动评分.
#   同钩子同版本表; 缓存键已含版本(版本边界可落在季度中间, 不入键则跨 worker 不确定).
#   另: bt_execution.py `_build_concept_industry_codes` 查证为**死代码**
#   (结果作 stock_data_dict 传入 set_sector_data 被丢弃) ⇒ 无需修.
#
# 臂:
#   pit_ident    env 空 — 必须复现 anchor_1015d md5 ⇒ 证明新代码默认路径零改动
#   pit_cmhist   QUANT_CONCEPT_MAP_HIST=<history> — 版本化名单
#
# ⚠ 两臂指纹不同(hist 计入指纹) ⇒ **各自**重生成信号(~78min/臂), 串行 ~3h.
# ⚠ 串行, 不要并行 (9/11 教训: 会 OOM).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
SCRATCH=/mnt/d/quant/.arms
LOGD=$SCRATCH/logs
BASE_MD5=dca8678ba7ba928bb9ac272f09b6a55c
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
FACE="${1:-cmhist}"
LOG=$LOGD/chain_$FACE.log
DS_LABEL="ds_$(date +%m%d_%H%M)_$FACE"
PY=/mnt/d/quant/.venv/bin/python
CMHIST=/mnt/d/quant/data/concept_map_history.pkl

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

echo "=== 臂链[$FACE] $(date +%m-%d_%H:%M:%S) 基线=anchor_1015d($BASE_MD5) ===" >> $LOG
echo "[$(date +%H:%M:%S)] 数据态快照 $DS_LABEL ..." >> $LOG
$PY $SCRATCH/datastate.py snap "$DS_LABEL" >> $LOG 2>&1
for f in $PROD; do [ -f $RVD/$f.anchor_1015d ] || cp -p $RVD/$f $RVD/$f.anchor_1015d 2>/dev/null; done

if [ "$FACE" = "cmhist" ]; then
  # ---- 臂1: 恒等 (env 空 ⇒ 走生产路径) ----
  run_arm pit_ident
  MD5=$(md5sum $RVD/equity_curve.csv | cut -d' ' -f1)
  if [ "$MD5" != "$BASE_MD5" ]; then
    echo "  ❌ 恒等失败! 期望 $BASE_MD5 实得 $MD5 — 新代码非零改动, 中止" >> $LOG
    exit 1
  fi
  echo "  ✅ 恒等烟测通过(复现 anchor_1015d 逐字节) ⇒ 版本化映射默认空 = 零改动" >> $LOG

  # ---- 臂2: 版本化名单(as-of 成分股归属) ----
  run_arm pit_cmhist QUANT_CONCEPT_MAP_HIST=$CMHIST

  echo "=== 臂链[$FACE] 完成 $(date +%m-%d_%H:%M:%S) ===" >> $LOG
  touch $SCRATCH/DONE_$FACE
else
  echo "未知 face: $FACE" >> $LOG; exit 2
fi
