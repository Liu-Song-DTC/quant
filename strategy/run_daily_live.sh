#!/bin/bash
# 每日实盘出单一键流程 (交易日晚间运行, 建议 17:30 后)
#
#   前置: Windows侧 16:00 计划任务 "xtquant_bt_daily" 自动更新K线 (需 QMT 客户端登录)
#   本脚本:
#     1. data/refresh_all.sh — 另类数据/概念/基本面/qfq修复 + 数据新鲜度门禁 (失败即停)
#     2. yaml todate → 今天
#     3. bt_execution.py — 信号重生成 (~80分钟, 数据/配置未变时自动跳过)
#     3.6 tools/navlaw.py — 净值铁律哨兵(真闸): 存档今天的产物, 与上次比"共有
#          历史段"是否逐位不变 (判据 f(t;T)≡f(t;t))。违反⇒拦截出单 (可显式放行)。
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

# 守卫报警外发 (P0, 2026-10-24): 链内任何非预期失败 → 微信报警。
# 未配置密钥时 notify_wechat.py 自行跳过(rc0), 陷阱常驻无副作用。
QPY=/mnt/d/quant/.venv/bin/python
QLOGS=/mnt/d/quant/logs
chain_alert() {  # $1=标题 $2=内容
  set +e
  "$QPY" -u /mnt/d/quant/strategy/notify_wechat.py --alert "$1" "$2" \
    >> "$QLOGS/alert_${TODAY//-/}.log" 2>&1
  set -e
}
trap 'chain_alert "实盘链失败" "run_daily_live.sh 第 $LINENO 行附近失败 (rc=$?), $(date "+%F %T")。日志: strategy/logs/ $(tail -2 $QLOGS/alert_${TODAY//-/}.log 2>/dev/null)"' ERR

[ "$DOW" -le 5 ] || { echo "今天周$DOW 不是交易日, 退出"; exit 1; }
[ "$(date +%H%M)" -ge 1730 ] || { echo "请在 17:30 后运行 (当日K线/融资融券等数据尚未齐)"; exit 1; }

echo "===== 1/4 数据刷新 + 新鲜度门禁 (失败即停) ====="
bash data/refresh_all.sh

# 1.2/4 数据每日快照 (P0, 2026-10-24): refresh 后的当日最终态 → snapshots/daily/
# 供应商重述时按日期回到任意一天对比。失败不拦链但日志报警 (set +e)。
set +e
$PY strategy/snapshot_data.py 2>&1 | tee "logs/snapshot_${TODAY//-/}.log"
set -e

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
  *) chain_alert "熊市前置闸错误" "live_no_trade_precheck.py rc=$_pre_rc, 出单链中止"; echo "前置闸错误 rc=$_pre_rc"; exit $_pre_rc ;;
esac

echo "===== 2/4 更新 yaml todate → $TODAY ====="
sed -i "s/^  todate: .*/  todate: '$TODAY'/" strategy/config/factor_config.yaml
grep -q "^  todate: '$TODAY'" strategy/config/factor_config.yaml \
  || { chain_alert "todate更新失败" "factor_config.yaml todate 未能更新到 $TODAY, 出单链中止"; echo "todate 更新失败!"; exit 1; }
grep -n "^  todate" strategy/config/factor_config.yaml

echo "===== 3/4 信号重生成 (bt_execution, ~80分钟) ====="
cd strategy
../.venv/bin/python -u bt_execution.py > "logs/live_${TODAY//-/}.log" 2>&1
echo "信号重生成完成 → logs/live_${TODAY//-/}.log"

echo "===== 3.5/4 回测富报告 ====="
../.venv/bin/python -u analysis/backtest_report.py \
  | tee "logs/report_${TODAY//-/}.log"

echo "===== 3.6/4 净值铁律哨兵 — 真闸 (#346, 2026-10-23) ====="
# 每天推进 todate 一天 + 重跑回测 = 天然的铁律实验。但 equity_curve.csv 是原地
# 覆盖的, 不存档就等于**从来没测过**。这里存一份, 再与上一次比:
# 共有日期区间的**历史段**上, 净值(连续)与选股集合(离散)是否逐位不变。
# 违反 ⇒ 拦截出单 (对不上=其中一个错了, 不带病出单)。正当重锚/已登记修复
# 用环境变量显式放行 (原因记入当日 navlaw 日志):
#   NAVLAW_OVERRIDE="<原因>"    放行今日, 明日继续报
#   NAVLAW_REBASELINE="<原因>"  重锚: 立新基线, 不再复报
set +e
../.venv/bin/python -u tools/navlaw.py watch --tail 10 --strict \
  | tee "logs/navlaw_${TODAY//-/}.log"
_navlaw_rc=${PIPESTATUS[0]}
set -e
if [ "$_navlaw_rc" -ne 0 ]; then
  echo "⛔ 净值铁律违反 (rc=$_navlaw_rc), 已拦截出单。排查/放行方式见上方 navlaw 输出。"
  chain_alert "navlaw净值铁律违反" "rc=$_navlaw_rc, 已拦截出单。处置: docs/runbook.md §navlaw。日志: strategy/logs/navlaw_${TODAY//-/}.log"
  exit "$_navlaw_rc"
fi

echo "===== 4/4 出单 ====="
../.venv/bin/python -u generate_trade_orders.py --date "$TODAY" \
  | tee "logs/orders_${TODAY//-/}.log"

echo "===== 5/5 微信推送 ====="
# 出单结果→微信 (P0, 2026-10-24)。未配置密钥=跳过; 发送失败不拦链但日志报警。
set +e
../.venv/bin/python -u notify_wechat.py --file trade_orders.json \
  2>&1 | tee "logs/notify_${TODAY//-/}.log"
set -e
echo ""
echo "=== 完成: 查看 strategy/trade_orders.json, 次日开盘按单执行 ==="
