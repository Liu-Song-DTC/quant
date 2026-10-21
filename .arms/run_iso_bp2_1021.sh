#!/bin/bash
# iso_bp2_1021 (2026-10-21): bp2_score_boost 0.15 vs 0 的 2022 空账隔离臂.
#
# 动机: 胜率判据扩展后, 0.15 vs 0 在全史上是"NAV +12.4% 但 20日/月度胜率各 −1.4pp"
#   ⇒ 按用户规则需更强证据. winrate_pair 已证那两个百分点差只是 1 个 20日格/2 个月格
#   (p=1.00). 本臂补第二重证据: 2022 空账隔离下 0.15 的 NAV 增益是否仍在.
#   参考 mp4 时代: iso {0:544,308 / 0.15:654,015} (+20%).
# 基线 = iso_mp3 (902,016, 空账 2022-01-03 起, 已存在).
set -u
cd /mnt/d/quant/strategy
RVD=/mnt/d/quant/strategy/rolling_validation_results
LOGD=/mnt/d/quant/.arms/logs
SCRATCH=/mnt/d/quant/.arms
PY=/mnt/d/quant/.venv/bin/python
LOG=$LOGD/iso_bp2_1021.log
PROD="equity_curve.csv trade_realized.csv portfolio_selections.csv backtest_report.json regime_state.csv"
BASE_TAG=iso_mp3
PROD_TAG=prod_mp3_1021
SRC=bt_execution.py
ISO=bt_execution_iso.py

echo "[$(date '+%F %T')] iso_bp2_1021 启动, 等bt退出…" >> $LOG
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done
for f in $PROD; do
  [ -f "$RVD/$f.$BASE_TAG" ] || { echo "⚠ 缺基线快照 $f.$BASE_TAG" >> $LOG; exit 1; }
done
_N=$(tail -1 "$RVD/equity_curve.csv.$BASE_TAG" | cut -d, -f2)
"$PY" -c "import sys; sys.exit(0 if abs(float('$_N')-902016.0)<1000 else 1)" || {
  echo "⛔ iso_mp3 基线 NAV=$_N 与 902,016 不符 — 中止" >> $LOG; exit 2; }

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
  "$PY" "$SCRATCH/datastate.py" rec iso_bp2_1021 "$tag" "$NAV" "$MD5" "$*" >> $LOG 2>&1
}

run_arm iso_bp2z QUANT_BP2_BOOST=0

for f in $PROD; do cp -p "$RVD/$f.$PROD_TAG" "$RVD/$f"; done
rm -f "$ISO"
echo "[$(date '+%F %T')] iso_bp2_1021 完毕, 产物已还原生产基线(mp3), 副本已删" >> $LOG
