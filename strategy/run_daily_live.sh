#!/bin/bash
# 每日实盘出单一键流程 (交易日晚间运行, 建议 17:30 后)
#
#   数据前置: 无 — K线由第1步 akshare/efinance 中继下载 (2026-09-30 起;
#   QMT 无数据角色, 仅按 PROTOCOL.md 做下单执行器/回报器)
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
#   出单产物: /mnt/d/quant/trade_orders.json (计划；实际持仓只按券商回报更新)
set -eE -o pipefail
cd /mnt/d/quant
exec 9>.daily_chain.lock
flock -n 9 || { echo '已有每日链正在运行，拒绝并发刷新/回测/出单'; exit 3; }
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

ALLOW_WEEKEND=0; FORCE_FULL=0; PREOPEN=0
for _a in "$@"; do
  case "$_a" in
    --allow-weekend) ALLOW_WEEKEND=1 ;;
    --force-full) FORCE_FULL=1 ;;
    --preopen) PREOPEN=1 ;;
    *) echo "未知参数: $_a (支持 --allow-weekend --force-full --preopen)"; exit 1 ;;
  esac
done
if [ "$ALLOW_WEEKEND" = "1" ]; then
  echo "[weekend 放行] 官方周末重跑/冻结确认模式 — 新鲜度门禁照常, 不减配"
else
  [ "$DOW" -le 5 ] || { echo "今天周$DOW 不是交易日, 退出 (官方周末重跑用 --allow-weekend)"; exit 1; }
fi
if [ "$PREOPEN" = "1" ]; then
  # 开盘前出单 (2026-10-26 用户裁定"今天周一还未开盘, 照样出名单"):
  # 数据=上一交易日收盘(已定稿), 新鲜度门禁本来就按"17:30前=期望上一交易日"判
  # (check_data_freshness.expected_trading_day) ⇒ 口径一致, 不减配。
  # 只允许开盘前跑; 开盘后请走常规晚间流程(当日数据齐了再跑)。
  [ "$(date +%H%M)" -lt 0930 ] || { echo "--preopen 只允许 09:30 前使用 (现在 $(date +%H%M))"; exit 1; }
  echo "[--preopen] 开盘前出单模式: 数据=上一交易日收盘, todate→今天, 订单按今日开盘执行"
elif [ "$(date +%H%M)" -ge 1730 ]; then
  :
else
  echo "请在 17:30 后运行 (当日K线/融资融券等数据尚未齐); 开盘前出名单用 --preopen"; exit 1
fi

# ===== 0/4 出单区洁净闸 (研究/出单物理隔离, 2026-10-24) =====
# 研究区 = /mnt/d/quant-r (git worktree, data 共享只读): 一切臂/探针/实验在那里跑。
# 出单区唯一变更入口 = git main 提交 (晋升)。此处校验工作区只带白名单脏:
#   data/** (刷新产物) | .arms/** (研究台账) | factor_config.yaml 仅 todate 行。
# 白名单外任何未提交改动 ⇒ 拒绝出单 (不带病出单: 研究残码混进生产=静默口径漂移)。
_gdirty=$(git status --porcelain | grep -vE '^\?\?' \
  | grep -vE '^.. (data/|\.arms/|strategy/config/factor_config\.yaml$)' || true)
_yd=$(git diff HEAD -- strategy/config/factor_config.yaml \
  | grep -E '^[+-][^+-]' | grep -vE '^[+-][[:space:]]*todate:' || true)
if [ -n "$_yd" ]; then
  _gdirty="${_gdirty}
strategy/config/factor_config.yaml 含非 todate 改动: $(echo "$_yd" | head -3)"
fi
if [ -n "$_gdirty" ]; then
  echo "⛔ 出单区工作区携带白名单外未提交改动 — 拒绝出单:"
  echo "$_gdirty" | head -10
  echo "   研究改动请在 /mnt/d/quant-r (研究区) 跑; 进生产 = main 提交 (docs/runbook.md §隔离)"
  chain_alert "出单区洁净闸拦截" "白名单外未提交改动, 已拒绝出单。晋升=main 提交; 研究跑 /mnt/d/quant-r。改动: $(echo "$_gdirty" | head -3 | tr '\n' ';')"
  exit 3
fi
_gunk=$(git status --porcelain | grep -E '^\?\?' \
  | grep -vE '^\?\? (\.arms/|logs/|strategy/logs/|strategy/reports/|data/snapshots/|docs/|\.claude/)' || true)
if [ -n "$_gunk" ]; then
  echo "[洁净闸][WARN] 白名单外未跟踪文件 $(echo "$_gunk" | wc -l) 个 (不拦链, 清单见隔离审计): $(echo "$_gunk" | head -3 | tr '\n' ' ')"
fi
echo "[洁净闸] 工作区干净 (白名单外零改动) ✓"

echo "===== 1/4 数据刷新 + 新鲜度门禁 (失败即停) ====="
bash data/refresh_all.sh

# 1.2/4 数据每日快照 (P0, 2026-10-24): refresh 后的当日最终态 → snapshots/daily/
# 供应商重述时按日期回到任意一天对比。失败不拦链但日志报警 (set +e)。
set +e
$PY strategy/snapshot_data.py 2>&1 | tee "logs/snapshot_${TODAY//-/}.log"
set -e

# 1.3/4 生产目录白名单审计 (#364 制度化, 2026-10-24): 未跟踪文件出现在生产目录
# = 研究产物误留。报告不拦链, 清单见 logs/isolation_*.md。
set +e
$PY strategy/tools/prod_isolation.py 2>&1 | tee "logs/isolation_${TODAY//-/}.log"
set -e

# 1.5/4 熊市空仓免算前置闸 (2026-10-22)
# 熊市双确认(trend<0 且 bear_risk) + 持仓为空 ⇒ build() 在调仓闸前早退,
# desired_value={} 且无仓可卖 ⇒ 0 单**确定**。跳过 ~80min 信号生成/回测,
# 写空 trade_orders.json 并推进实际账户状态；不改历史回测净值。新鲜度门禁已在上一步跑过。
# 退出码: 0=跳过 / 1=继续完整管线 / 其他=错误即停。
echo "===== 1.5/4 熊市空仓免算前置闸 ====="
if [ "$FORCE_FULL" = "1" ]; then
  echo "[--force-full] 跳过前置闸, 完整管线照跑 (注意: 熊市+空仓时订单仍将为空, 策略本身不进场)"
else
set +e
$PY strategy/live_no_trade_precheck.py --date "$TODAY" \
  | tee "strategy/logs/precheck_${TODAY//-/}.log"
_pre_rc=${PIPESTATUS[0]}
set -e
case $_pre_rc in
  0) echo "熊市+空仓: 已跳过信号生成/回测, 空订单+熊市观察名单已写入"
     # 跳过日也推送 (2026-10-24, 用户裁定"供自行判断"): 空单+观察名单+守卫灯
     set +e
     $PY strategy/notify_wechat.py --file /mnt/d/quant/trade_orders.json \
       2>&1 | tee "logs/notify_${TODAY//-/}.log"
     set -e
     exit 0 ;;
  1) echo "前置闸不触发: 继续完整管线" ;;
  *) chain_alert "熊市前置闸错误" "live_no_trade_precheck.py rc=$_pre_rc, 出单链中止"; echo "前置闸错误 rc=$_pre_rc"; exit $_pre_rc ;;
esac
fi

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

echo "===== 4.5/5 风险画像 (周日生成, 三层裁定: 只报实盘台账净值) ====="
if [ "$(date +%a)" = "Sun" ]; then
  set +e
  ../.venv/bin/python -u risk_profile.py 2>&1 | tee "logs/riskprofile_${TODAY//-/}.log"
  set -e
else
  echo "(非周日跳过)"
fi

echo "===== 5/5 微信推送 ====="
# 出单结果→微信 (P0, 2026-10-24)。未配置密钥=跳过; 发送失败不拦链但日志报警。
set +e
../.venv/bin/python -u notify_wechat.py --file /mnt/d/quant/trade_orders.json \
  2>&1 | tee "logs/notify_${TODAY//-/}.log"
set -e
echo ""
echo "=== 完成: 查看 strategy/trade_orders.json, 次日开盘按单执行 ==="
