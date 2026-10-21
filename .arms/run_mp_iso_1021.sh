#!/bin/bash
# mp iso (2026-10-21): 新锚(bp3 + bp2_boost=0.15)下槽位上限的 2022 空账隔离臂.
#
# 动机: census_1021a 全史面 {mp3:984,331; mp4:711,609(生产); mp5:?}.
# mp3 比 mp4 高 +38%, 但票级独有仅 5 笔(n<30)、权重代理和仅 +2.3% ⇒
# 净值增益高度路径化. 10/14 采纳 mp4 的依据(槽位竞争)是旧数据态, 现已
# 在新锚下被推翻 — 必须用空账 2022-start 隔离验证, 而非直接采纳.
#
# 预注册 (先于本链结果, 2026-10-21 14:57):
#   基线 = iso_mp4 (生产态, 空账 2022-01-03 起).
#   mp3 采纳须同时满足:
#   (i)   终值比 mp3/mp4 ≥ 1.05;
#   (ii)  逐年 log 收益差同号 或 最大单年贡献 < 60% 总差;
#   (iii) 臂独有票 fwd10 α ≥ 基线独有票 α (n≥30; n<30 信息性);
#   (iv)  top5 权重代理集中度不高于基线 且 机制方向一致(人工).
#   mp2 只作边界探针(非采纳候选).
#   全不满足 ⇒ 维持 mp4, 槽位轴闭合.
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/mp_iso_1021.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=prod_bp2v15_1021
DS_LABEL=mp_iso_1021
SRC=bt_execution.py
ISO=bt_execution_iso.py

echo "[$(date '+%F %T')] mp_iso_1021 启动, 等bt退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
[ -f "$RVD/equity_curve.csv.$BASE_TAG" ] || { echo "⚠ 缺基线快照 $BASE_TAG" >> $LOG; exit 1; }

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
  echo "[$(date '+%F %T')] === $tag: QUANT_FROMDATE=2022-01-01 ${*:-identity} ===" >> $LOG
  env QUANT_ALT_NO_AUTOREFRESH=1 QUANT_FROMDATE=2022-01-01 "$@" \
      "$PY" -u "$ISO" > "$LOGD/bt_$tag.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date '+%F %T')] ⚠ $tag rc=$rc (bt_$tag.log)" >> $LOG; return $rc
  fi
  if ! grep -q "复用已有信号" "$LOGD/bt_$tag.log"; then
    echo "[$(date '+%F %T')] ⛔ $tag 未复用信号 — 中止" >> $LOG; exit 3
  fi
  if [ "$(md5sum "$SIG" | cut -d' ' -f1)" != "$SIGMD5" ] || [ "$(stat -c %Y "$SIG")" != "$SIGMT" ]; then
    echo "[$(date '+%F %T')] ⛔ $tag 信号CSV变化 — 中止" >> $LOG; exit 4
  fi
  for f in $PROD; do cp -p "$RVD/$f" "$RVD/$f.$tag" 2>/dev/null; done
  local NAV MD5
  NAV=$(tail -1 "$RVD/equity_curve.csv" | cut -d, -f2)
  MD5=$(md5sum "$RVD/equity_curve.csv" | cut -d' ' -f1)
  echo "[$(date '+%F %T')] $tag NAV=$NAV md5=${MD5:0:8}" >> $LOG
  "$PY" "$SCRATCH/datastate.py" rec "$DS_LABEL" "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

# iso_mp4 ≡ 已跑过的 iso15 (bp3 + 显式 QUANT_BP2_BOOST=0.15, 生产 mp4):
# 生产 yaml 现已是 0.15, 两者逐位等价 ⇒ 直接别名, 省 9min 重跑.
if [ -f "$RVD/equity_curve.csv.iso15" ] && [ -f "$RVD/equity_curve.csv.$BASE_TAG" ]; then
  if [ ! -f "$RVD/equity_curve.csv.iso_mp4" ]; then
    _N=$(tail -1 "$RVD/equity_curve.csv.iso15" | cut -d, -f2)
    if "$PY" -c "import sys; sys.exit(0 if abs(float('$_N')-654015.0)<1000 else 1)"; then
      for f in $PROD; do cp -p "$RVD/$f.iso15" "$RVD/$f.iso_mp4"; done
      echo "[$(date '+%F %T')] iso_mp4 ← iso15 别名(NAV=$_N, bp3+0.15+mp4 等价)" >> $LOG
    else
      echo "[$(date '+%F %T')] ⚠ iso15 NAV=$_N 与预期不符, 重跑 iso_mp4" >> $LOG
      run_arm iso_mp4
    fi
  fi
else
  run_arm iso_mp4
fi
run_arm iso_mp3 QUANT_MAX_POSITIONS=3
run_arm iso_mp2 QUANT_MAX_POSITIONS=2
run_arm iso_mp5 QUANT_MAX_POSITIONS=5

for f in $PROD; do cp -p "$RVD/$f.$BASE_TAG" "$RVD/$f"; done
rm -f "$ISO"
echo "[$(date '+%F %T')] mp_iso_1021 完毕, 产物已还原基线(mp4)态, 副本已删" >> $LOG
