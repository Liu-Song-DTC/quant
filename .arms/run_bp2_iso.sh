#!/bin/bash
# bp2_boost 2022-start 隔离臂 (2026-10-21) — 消除2021账簿路径.
#
# 动机: 全史臂面 {0:717,296; 0.15:711,609; 0.30:715,101; 0.45:585,917} =
# 0~0.30平台 + 0.45断崖, 但三点均未通过预注册全套判据. 全史比较的
# 2021末比分裂(bp2_0 1.162 / bp2_15 0.990 / bp2_30 1.006) ⇒ 终值差混合了
# 2021路径与2022后的增益. 本实验: 所有臂 2022-01-03 从空账/25万起步,
# 保留完整信号CSV与指数regime暖机窗口, 直接比较"从空账出发"的5年路径.
# 这与当前实盘态(空账等regime)更接近.
#
# 实现: 复制 bt_execution.py → bt_execution_iso.py, 只改两处:
#   1) FROMDATE = QUANT_FROMDATE or yaml;
#   2) 复用信号路径下不再按FROMDATE截断指数数据 (保留regime暖机);
#   副本不参与 _signal_code_fingerprint (它哈希的是原 bt_execution.py) ⇒ 信号复用.
# 运行后删除副本; 本脚本可重建.
#
# 预注册判据 (先于结果, 2026-10-21 12:36):
#   基线=iso45(0.45). 采纳替代值须同时满足:
#   (i)   终值比 ≥ 1.05 (替代/基线);
#   (ii)  逐年 log 收益差同号 或 最大单年贡献 <60% 总差;
#   (iii) 臂独有票 fwd10 α ≥ 基线独有票 α (n≥30; n<30仅信息性);
#   (iv)  机制方向与旋钮语义一致(人工).
#   全不满足 ⇒ 维持0.45, bp2轴闭合.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/bp2_iso.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=base1020bp3
DS_LABEL=bp2_iso
SRC=bt_execution.py
ISO=bt_execution_iso.py

echo "[$(date '+%F %T')] bp2_iso 启动, 等bt退出…" >> $LOG
# 只认 comm=python*: pgrep -f 会匹配命令行文本含 bt_execution.py 的包装 shell(10-21死锁)
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
[ -f "$RVD/equity_curve.csv.$BASE_TAG" ] || { echo "⚠ 缺基线快照" >> $LOG; exit 1; }

# 构建隔离入口
$PY - "$SRC" "$ISO" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()
old1 = "FROMDATE = config.get('backtest.fromdate', None)"
new1 = ("_QF = os.environ.get('QUANT_FROMDATE') or None\n"
        "if _QF: print(f'[hook] QUANT_FROMDATE={_QF} (iso arm)')\n"
        "FROMDATE = _QF or config.get('backtest.fromdate', None)")
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)
old2 = "            if FROMDATE: _idx_df = _idx_df[_idx_df['datetime'] >= FROMDATE]"
new2 = ("            if FROMDATE and not os.environ.get('QUANT_FROMDATE'):\n"
        "                _idx_df = _idx_df[_idx_df['datetime'] >= FROMDATE]")
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)
open(dst, 'w', encoding='utf-8').write(s)
print('iso entry written')
PY
[ -f "$ISO" ] || { echo "⚠ 副本创建失败" >> $LOG; exit 1; }

SIG=$RVD/backtest_signals.csv
SIGMD5=$(md5sum "$SIG" | cut -d' ' -f1)
SIGMT=$(stat -c %Y "$SIG")
echo "[$(date '+%F %T')] 信号CSV md5=${SIGMD5:0:8} mtime=$SIGMT" >> $LOG

run_arm () {
  local tag="$1"; shift
  if [ -f "$RVD/equity_curve.csv.$tag" ]; then
    echo "[$(date '+%F %T')] $tag 已有产物, 跳过" >> $LOG; return 0
  fi
  for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
  echo "[$(date '+%F %T')] === $tag: QUANT_FROMDATE=2022-01-01 $* ===" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 QUANT_FROMDATE=2022-01-01 "$@" \
      "$PY" -u "$ISO" > "$LOGD/bt_$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ⚠ $tag rc=$rc (bt_$tag.log)" >> $LOG; return $rc
  fi
  if ! grep -q "复用已有信号" "$LOGD/bt_$tag.log"; then
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号(可能触发重生成) — 中止" >> $LOG; exit 3
  fi
  if [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD5" ]; then
    echo "[$(date '+%F %T')] ⛔ $tag 信号CSV内容变化 — 中止" >> $LOG; exit 4
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5 NR
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  NR=$(tail -1 "$RVD/regime_state.csv" | cut -d, -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8} last=$NR" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm iso45
run_arm iso30 QUANT_BP2_BOOST=0.30
run_arm iso15 QUANT_BP2_BOOST=0.15
run_arm iso0  QUANT_BP2_BOOST=0

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
rm -f "$ISO"
echo "[$(date '+%F %T')] bp2_iso 完毕, 产物已还原基线(bp3)态, 副本已删" >> $LOG
