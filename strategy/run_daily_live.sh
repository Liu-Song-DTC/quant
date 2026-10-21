#!/bin/bash
# 每日实盘出单一键流程 (交易日晚间运行, 建议 17:30 后)
#
#   前置: Windows侧 16:00 计划任务 "xtquant_bt_daily" 自动更新K线 (需 QMT 客户端登录)
#   本脚本:
#     1. data/refresh_all.sh — 另类数据/概念/基本面/qfq修复 + 数据新鲜度门禁 (失败即停)
#     2. yaml todate → 今天
#     3. bt_execution.py — 信号重生成 (~80分钟, 数据/配置未变时自动跳过)
#     3.6 tools/navlaw.py — 净值铁律哨兵: 存档今天的产物, 与上次比"共有历史段"
#          是否逐位不变 (判据 f(t;T)≡f(t;t))。只报不拦。
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

# 1.5/4 熊市空仓免算前置闸 (2026-10-22)
# 熊市双确认(trend<0 且 bear_risk) + 持仓为空 ⇒ build() 在调仓闸前早退,
# desired_value={} 且无仓可卖 ⇒ 0 单**确定**。跳过 ~80min 信号生成/回测,
# 写空 trade_orders.json 并追加 0 收益净值行。新鲜度门禁已在上一步跑过。
# 退出码: 0=跳过 / 1=继续完整管线 / 其他=错误即停。
echo "===== 1.5/4 熊市空仓免算前置闸 ====="
set +e
$PY strategy/live_no_trade_precheck.py --date "$TODAY" \
  | tee "strategy/logs/precheck_${TODAY//-/}.log"
_pre_rc=${PIPESTATUS[0]}
set -e
case $_pre_rc in
  0) echo "熊市+空仓: 已跳过信号生成/回测, 空订单已写入"; exit 0 ;;
  1) echo "前置闸不触发: 继续完整管线" ;;
  *) echo "前置闸错误 rc=$_pre_rc"; exit $_pre_rc ;;
esac

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

echo "===== 3.6/4 净值铁律哨兵 (回测收益能不能复现) ====="
# 每天推进 todate 一天 + 重跑回测 = 天然的铁律实验。但 equity_curve.csv 是原地
# 覆盖的, 不存档就等于**从来没测过**。这里存一份, 再与上一次比:
# 共有日期区间的**历史段**上, 净值(连续)与选股集合(离散)是否逐位不变。
# 只报不拦 —— 拦了就没单出, 代价更大; 但违反必须当天查清并记账。
../.venv/bin/python -u tools/navlaw.py watch --tail 10 \
  | tee "logs/navlaw_${TODAY//-/}.log" || true

echo "===== 4/4 出单 ====="
../.venv/bin/python -u generate_trade_orders.py --date "$TODAY" \
  | tee "logs/orders_${TODAY//-/}.log"
echo ""
echo "=== 完成: 查看 strategy/trade_orders.json, 次日开盘按单执行 ==="
