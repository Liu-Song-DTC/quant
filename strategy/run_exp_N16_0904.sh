#!/bin/bash
# E-N16(2026-09-09): yjyg靴子落地优先槽实验 — 9/4态双态验证腿 (明日执行)
# 用法: bash strategy/run_exp_N16_0904.sh            # anchor+smoke+slots2 全链
#       bash strategy/run_exp_N16_0904.sh anchor-only  # 仅anchor
# 严格串行(不要并行, 会oom)。链: todate→9/4 → 删信号CSV(强制regen 78min)
#   → anchor跑 → smoke臂(warm) → slots2臂(warm) → 开关回滚 + todate→9/9
# 前置: 无(自包含); 勿在14:30后启动(保护当日18:00实盘流 — 本链只在实盘流
# 完成后的晚间/次日早晨跑)。
set -u
cd /mnt/d/quant
MODE="${1:-full}"
VENV=.venv/bin/python
YAML=strategy/config/factor_config.yaml
RVR=strategy/rolling_validation_results
SIG="$RVR/backtest_signals.csv"
LOGDIR=strategy/logs

set_todate() {  # $1=目标日期
    $VENV - <<EOF
import re
p='$YAML'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(todate:\s*)['\"].*?['\"]", r"\1'$1'", s, count=1)
assert s2!=s, "todate未找到"
open(p,'w',encoding='utf-8').write(s2)
print("[todate] → $1")
EOF
}

echo "=== E-N16 9/4态腿开始 $(date '+%H:%M:%S') mode=$MODE ==="
# 结束时恢复: 开关关 + todate回9/9
restore_all() {
    $VENV - <<'EOF'
import re
p='/mnt/d/quant/strategy/config/factor_config.yaml'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled:) true", r"\1 false", s, count=1)
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled: false\n      window_days: 30\n      slots:) 2", r"\1 1", s2, count=1)
s2=re.sub(r"(todate:\s*)['\"].*?['\"]", r"\1'2026-09-09'", s2, count=1)
open(p,'w',encoding='utf-8').write(s2)
print("[restore] yjyg=false slots=1 todate=2026-09-09")
EOF
}
trap restore_all EXIT

# 1) 9/4态: todate→9/4 + 删信号CSV (fp失配+无CSV双保险, 强制regen)
set_todate 2026-09-04
rm -f "$SIG"
echo "[regen] 信号CSV已删, 78min全量重生成开始"

# 2) anchor: yjyg关, 基线9/4态
$VENV strategy/bt_execution.py > "$LOGDIR/bt_execution_N16_0904_anchor.log" 2>&1
RC=$?
echo "=== anchor结束 rc=$RC $(date '+%H:%M:%S') ==="
grep -E "最终净值|Sharpe:|最大回撤:|^  20(2[1-6]):" "$LOGDIR/bt_execution_N16_0904_anchor.log" | tail -12
if [ "$MODE" = "anchor-only" ] || [ $RC -ne 0 ]; then exit $RC; fi

# 3) 备份anchor产物
cp "$RVR/portfolio_selections.csv" "$RVR/portfolio_selections.bak0904.csv"
cp "$RVR/trade_realized.csv" "$RVR/trade_realized.bak0904.csv"
cp "$RVR/equity_curve.csv" "$RVR/equity_curve.bak0904.csv" 2>/dev/null

# 4) smoke臂 (warm)
$VENV - <<'EOF'
import re
p='/mnt/d/quant/strategy/config/factor_config.yaml'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled:) false", r"\1 true", s, count=1)
assert s2!=s
open(p,'w',encoding='utf-8').write(s2)
EOF
$VENV strategy/bt_execution.py > "$LOGDIR/bt_execution_N16_0904_smoke.log" 2>&1
echo "=== smoke臂结束 rc=$? $(date '+%H:%M:%S') ==="
grep -E "最终净值|Sharpe:|最大回撤:|^  20(2[1-6]):" "$LOGDIR/bt_execution_N16_0904_smoke.log" | tail -12
grep -E "yjyg优先槽" "$LOGDIR/bt_execution_N16_0904_smoke.log" | head -10

# 5) slots2臂 (warm)
$VENV - <<'EOF'
import re
p='/mnt/d/quant/strategy/config/factor_config.yaml'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled: true\n      window_days: 30\n      slots:) 1", r"\1 2", s, count=1)
assert s2!=s
open(p,'w',encoding='utf-8').write(s2)
EOF
$VENV strategy/bt_execution.py > "$LOGDIR/bt_execution_N16_0904_slots2.log" 2>&1
echo "=== slots2臂结束 rc=$? $(date '+%H:%M:%S') ==="
grep -E "最终净值|Sharpe:|最大回撤:|^  20(2[1-6]):" "$LOGDIR/bt_execution_N16_0904_slots2.log" | tail -12
grep -E "yjyg优先槽" "$LOGDIR/bt_execution_N16_0904_slots2.log" | head -10
echo "=== E-N16 9/4态腿全链完成 $(date '+%H:%M:%S') ==="
