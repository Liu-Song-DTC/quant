#!/bin/bash
# 每日实盘出单一键流程 (交易日晚间运行, 建议 17:30 后)
#
#   前置: Windows侧 16:00 计划任务 "xtquant_bt_daily" 自动更新K线 (需 QMT 客户端登录)
#   本脚本:
#     1. data/refresh_all.sh — 另类数据/概念/基本面/qfq修复 + 数据新鲜度门禁 (失败即停)
#     2. yaml todate → 今天
#     3. bt_execution.py — 信号重生成 (~80分钟, 数据/配置未变时自动跳过)
#     4. generate_trade_orders.py — 出单 (自带门禁复检, 输出 trade_orders.json)
#
#   用法: bash strategy/run_daily_live.sh
#   周末/节假日: 门禁预期交易日≠今天会拦, 届时用 --expected 手动分步跑
#   出单产物: strategy/trade_orders.json (给 miniqmt 做盘中高抛低吸)
set -e
cd /mnt/d/quant
PY=.venv/bin/python
TODAY=$(date +%F)
DOW=$(date +%u)

[ "$DOW" -le 5 ] || { echo "今天周$DOW 不是交易日, 退出"; exit 1; }
[ "$(date +%H%M)" -ge 1730 ] || { echo "请在 17:30 后运行 (当日K线/融资融券等数据尚未齐)"; exit 1; }

echo "===== 1/4 数据刷新 + 新鲜度门禁 (失败即停) ====="
bash data/refresh_all.sh

echo "===== 2/4 更新 yaml todate → $TODAY ====="
sed -i "s/^  todate: .*/  todate: '$TODAY'/" strategy/config/factor_config.yaml
grep -q "^  todate: '$TODAY'" strategy/config/factor_config.yaml \
  || { echo "todate 更新失败!"; exit 1; }
grep -n "^  todate" strategy/config/factor_config.yaml

echo "===== 3/4 信号重生成 (bt_execution, ~80分钟) ====="
cd strategy
../.venv/bin/python -u bt_execution.py > "logs/live_${TODAY//-/}.log" 2>&1
echo "信号重生成完成 → logs/live_${TODAY//-/}.log"

echo "===== 3.5/4 回测富报告 ====="
../.venv/bin/python -u analysis/backtest_report.py \
  | tee "logs/report_${TODAY//-/}.log"

echo "===== 4/4 出单 ====="
../.venv/bin/python -u generate_trade_orders.py --date "$TODAY" \
  | tee "logs/orders_${TODAY//-/}.log"
echo ""
echo "=== 完成: 查看 strategy/trade_orders.json, 次日开盘按单执行 ==="
