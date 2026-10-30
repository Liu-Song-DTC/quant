#!/bin/bash
# 刷新**校验段** (新鲜度门禁 + 历史格哨兵)
# 与取数段分离(2026-10-28): 校验失败 ⇒ 只重跑本段, 不重复下载(取数幂等标记见下载器)。
# 用法: bash data/refresh_verify.sh
set -e
set -o pipefail
cd /mnt/d/quant
if [ "$(readlink /proc/self/fd/9 2>/dev/null)" != "$PWD/.daily_chain.lock" ]; then
    exec 9>.daily_chain.lock
fi
flock -n 9 || { echo '数据维护或每日链正在运行，拒绝并发刷新'; exit 3; }
PY=/mnt/d/quant/.venv/bin/python
mkdir -p /mnt/d/quant/logs
REFRESH_LOG=${REFRESH_LOG:-/mnt/d/quant/logs/refresh_$(date +%Y%m%d_%H%M%S).log}
echo "完整日志: $REFRESH_LOG"

echo "=== 新鲜度门禁 (取数已完成, 校验段) ==="
$PY strategy/check_data_freshness.py || {
  echo "!!! 门禁未通过, 请检查上面输出 (K线增量/bt重建失败?)"
  exit 1
}

echo "=== 另类数据观测版本校验 (历史快照不被更新覆盖) ==="
$PY -B data/verify_alt_observations.py

echo "=== 9/10 历史格哨兵 (append层零改写验证, #344 常驻化 2026-10-23) ==="
# 刷新后立即 diff 最近快照: append 层任何历史格改写/消失 ⇒ rc=2 大声报警+旗标落盘。
# 不拦链: 供应商重述属数据现实(需登记台账), 本管线bug由 navlaw 真闸在出单前拦 NAV 级。
# 干净与否都滚基线(违反日旗标已立, 新基线=今日事实), 明日 diff 对今日。
_SNAP_DIR=/mnt/d/quant/.arms/histcells
_LATEST=$(ls -t $_SNAP_DIR/*.pkl 2>/dev/null | head -1)
if [ -n "$_LATEST" ]; then
  _LBL=$(basename "$_LATEST" .pkl)
  set +e
  $PY -u strategy/tools/histcells.py diff "$_LBL" 2>&1 | tee -a "$REFRESH_LOG"
  _HRC=${PIPESTATUS[0]}
  set -e
  if [ "$_HRC" -eq 2 ]; then
    touch "$_SNAP_DIR/VIOLATION_$(date +%Y%m%d_%H%M%S)"
    echo "!!! 历史格哨兵: append 层出现改写/消失 (上方分层明细 + 登记台账 .arms/ledger.csv)"
    echo "    处置: 供应商重述→登记台账+照常出单; 本管线bug→停链排查后再出单"
  elif [ "$_HRC" -ne 0 ]; then
    echo "!!! 历史格哨兵无法完成扫描(rc=$_HRC)，拒绝滚存基线和出单"
    exit "$_HRC"
  fi
  $PY -u strategy/tools/histcells.py snap "roll_$(date +%Y%m%d)" 2>&1 | tee -a "$REFRESH_LOG" | tail -3 || {
    echo "!!! 哨兵基线滚存失败，保留上一份基线并停止出单"
    exit 1
  }
  ls -t $_SNAP_DIR/roll_*.pkl 2>/dev/null | tail -n +3 | xargs -r rm -f
else
  echo "  [跳过] 无基线快照 (首次基线: $PY strategy/tools/histcells.py snap base_<date>)"
fi

echo "=== 全部新鲜, 可以出单: cd strategy && $PY generate_trade_orders.py ==="
