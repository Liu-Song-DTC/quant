#!/bin/bash
# 实盘选股前一键刷新全部数据 (串行, 禁并行防OOM)
# 用法: bash data/refresh_all.sh
# 说明: K线(backtrader_data)在Windows侧用 xtquant 下载器更新 (data/xtquant_bt_daily.bat)
set -e
cd /mnt/d/quant
PY=.venv/bin/python
TS=$(date +%m%d)

echo "=== 1/5 基本面 (最近3报告期) ==="
$PY -u data/refresh_fundamental_recent.py 2>&1 | tail -5

echo "=== 2/5 另类数据 (龙虎榜/融资融券/北向/减持/解禁/业绩预告) ==="
$PY -u data/refresh_altdata_0902.py 2>&1 | tail -12

echo "=== 3/5 概念历史 (60板块日涨跌, 东财K线集群限流时自动轮换重试) ==="
$PY -u data/refresh_concept_hist_0903.py 2>&1 | tail -6

echo "=== 4/5 概念映射 (股票→概念, 5/31旧快照已被2026-09-03全量替代) ==="
$PY -u data/refresh_concept_map_0903.py 2>&1 | tail -6

echo "=== 5/5 qfq负价修复 (Windows侧每次重跑--bt-only会覆盖sina修复, 必须刷新后自动重跑) ==="
$PY -u data/repair_qfq_negatives.py 2>&1 | tail -4

echo ""
echo "=== 刷新完成, 跑新鲜度门禁 ==="
$PY strategy/check_data_freshness.py || {
  echo "!!! 门禁未通过, 请检查上面输出 (Windows侧K线更新了吗?)"
  exit 1
}
echo "=== 全部新鲜, 可以出单: cd strategy && $PY generate_trade_orders.py ==="
