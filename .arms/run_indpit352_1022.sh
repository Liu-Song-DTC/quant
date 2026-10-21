#!/bin/bash
# #352 行业归属 PIT 配对臂 (2026-10-22)
#
# 背景: core/signal_engine.py 回测路径把 industry/specific_industry 锚在
#   _mid_date = dates[min(n-1, max(60, n//2))], 对整段历史用同一个行业名。
#   探针实测: 9.9% 的 (股票,日期) 行业名不同, 涉及 61.1% 股票; 且 _mid_date
#   约 2026-12 上旬越过 2024-01-01 季界 => 届时批量改写全部历史 (定时炸弹)。
#
# 本臂: QUANT_INDUSTRY_PIT=1, 同数据态 (todate=2026-10-21), 唯一改动是逐bar PIT。
# 基线: 生产 signal md5 94bca0cc / equity md5 350208d1 / NAV 984,331.43。
#
# 产物隔离: 生产产物 rename 为 *.pre352 (同盘零额外空间; 信号 3.1G, 磁盘仅 5G 空余),
#   臂产物写到原路径, 小文件复制到 armout. 臂结束后由分析脚本决定 restore/adopt.
#   ⚠ 若臂失败, 本脚本自动把 *.pre352 改回原名.
set -u
cd /mnt/d/quant/strategy
PY=/mnt/d/quant/.venv/bin/python
RVD=/mnt/d/quant/strategy/rolling_validation_results
A=/mnt/d/quant/.arms/armout/indpit352_1022
LOG=$A/run.log
mkdir -p "$A"
PROD="backtest_signals.csv equity_curve.csv portfolio_selections.csv trade_realized.csv regime_state.csv backtest_report.json"

echo "[$(date '+%F %T')] indpit352 等待 bt_execution 退出..." > "$A/chain.log"
while ps -eo comm=,args= | awk '$1 ~ /^python/ && /bt_execution(_iso)?\.py/ {f=1} END{exit !f}'; do sleep 20; done

# ---- 快照 (rename) ----
for f in $PROD; do
  if [ -f "$RVD/$f" ]; then
    if [ -f "$RVD/$f.pre352" ]; then
      echo "[$(date '+%F %T')] ABORT: $f.pre352 已存在" >> "$A/chain.log"
      exit 1
    fi
    mv "$RVD/$f" "$RVD/$f.pre352"
  fi
done
md5sum "$RVD/backtest_signals.csv.pre352" > "$A/pre_signal_md5.txt"
md5sum "$RVD/equity_curve.csv.pre352" > "$A/pre_equity_md5.txt"
echo "[$(date '+%F %T')] 快照完成: $(cat "$A/pre_signal_md5.txt")" >> "$A/chain.log"

# ---- 跑臂 ----
export QUANT_INDUSTRY_PIT=1
echo "[$(date '+%F %T')] 启动 bt_execution.py (QUANT_INDUSTRY_PIT=1)" >> "$A/chain.log"
$PY -u bt_execution.py > "$LOG" 2>&1
rc=$?

if [ $rc -ne 0 ]; then
  echo "[$(date '+%F %T')] 臂失败 rc=$rc — 恢复生产产物" >> "$A/chain.log"
  for f in $PROD; do
    [ -f "$RVD/$f" ] && rm -f "$RVD/$f"
    [ -f "$RVD/$f.pre352" ] && mv "$RVD/$f.pre352" "$RVD/$f"
  done
  echo "rc=$rc" > "$A/meta.txt"
  exit $rc
fi

# ---- 校验: 必须重生成信号, 不能复用 ----
if grep -q "复用已有信号" "$LOG"; then
  echo "[$(date '+%F %T')] ABORT: 复用了旧信号, 不是 PIT 臂 — 恢复生产产物" >> "$A/chain.log"
  for f in $PROD; do
    [ -f "$RVD/$f" ] && rm -f "$RVD/$f"
    [ -f "$RVD/$f.pre352" ] && mv "$RVD/$f.pre352" "$RVD/$f"
  done
  echo "reused=1" > "$A/meta.txt"
  exit 4
fi

# ---- 收集小产物 (不复制 3.1G 信号) ----
md5sum "$RVD/backtest_signals.csv" > "$A/post_signal_md5.txt"
for f in equity_curve.csv portfolio_selections.csv trade_realized.csv regime_state.csv backtest_report.json; do
  [ -f "$RVD/$f" ] && cp "$RVD/$f" "$A/$f"
done
{
  echo "rc=0"
  echo "env=QUANT_INDUSTRY_PIT=1"
  echo "pre=$(cat "$A/pre_signal_md5.txt")"
  echo "post=$(cat "$A/post_signal_md5.txt")"
  echo "finished=$(date '+%F %T')"
} > "$A/meta.txt"
echo "[$(date '+%F %T')] 臂完成: $(cat "$A/post_signal_md5.txt")" >> "$A/chain.log"
