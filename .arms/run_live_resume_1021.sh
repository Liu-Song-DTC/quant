#!/bin/bash
# 2026-10-21 实盘 2/4–4/4 续跑。
# 背景: refresh_all.sh 1/8 步全部成功, 仅末步门禁因 601326 当日停牌误报 FAIL;
#   已用新浪+腾讯双源核实停牌(今开/高/低/量全0, 现价=昨收), 按既有机制补
#   data/stock_data/kline_stale_allowlist.txt, 门禁复检 PASS。绝不使用 --skip-freshness。
set -e
cd /mnt/d/quant
PY=/mnt/d/quant/.venv/bin/python
TODAY=$(date +%F)
LOG=/mnt/d/quant/.arms/logs/live_${TODAY//-/}_resume.log
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] === resume: 数据新鲜度门禁复检 (不带 --skip-freshness) ==="
$PY strategy/check_data_freshness.py

echo "[$(date '+%F %T')] === 2/4 更新 yaml todate -> $TODAY ==="
sed -i "s/^  todate: .*/  todate: '$TODAY'/" strategy/config/factor_config.yaml
grep -q "^  todate: '$TODAY'" strategy/config/factor_config.yaml \
  || { echo "todate 更新失败!"; exit 1; }
grep -n '^  todate' strategy/config/factor_config.yaml

echo "[$(date '+%F %T')] === 3/4 信号重生成 (bt_execution, ~80分钟) ==="
cd /mnt/d/quant/strategy
../.venv/bin/python -u bt_execution.py > "logs/live_${TODAY//-/}.log" 2>&1
echo "[$(date '+%F %T')] bt 完成 -> logs/live_${TODAY//-/}.log"

echo "[$(date '+%F %T')] === 3.5/4 回测富报告 ==="
../.venv/bin/python -u analysis/backtest_report.py \
  | tee "logs/report_${TODAY//-/}.log"

echo "[$(date '+%F %T')] === 3.6/4 净值铁律哨兵 (只报不拦) ==="
../.venv/bin/python -u tools/navlaw.py watch --tail 10 \
  | tee "logs/navlaw_${TODAY//-/}.log" || true

echo "[$(date '+%F %T')] === 4/4 出单 (自带门禁复检) ==="
../.venv/bin/python -u generate_trade_orders.py --date "$TODAY" \
  | tee "logs/orders_${TODAY//-/}.log"
echo "[$(date '+%F %T')] === 完成: strategy/trade_orders.json ==="
