#!/bin/bash
# E-N16(2026-09-09): yjyg靴子落地优先槽实验 — 9/9态臂跑器
# 前置: 18:00实盘流已完成(9/9态信号+sidecar fp), 由调度者先核对anchor指标。
# 用法: bash strategy/run_exp_N16_0909.sh smoke   (enabled=true, slots=1)
#       bash strategy/run_exp_N16_0909.sh slots2  (enabled=true, slots=2)
# 严格串行(不要并行, 会oom)。每臂: 翻开关→跑bt_execution→提指标→翻回开关。
set -u
cd /mnt/d/quant
ARM="${1:-smoke}"
VENV=.venv/bin/python
YAML=strategy/config/factor_config.yaml
RVR=strategy/rolling_validation_results
LOG=strategy/logs/bt_execution_N16_${ARM}.log

echo "=== E-N16 臂: $ARM 开始 $(date '+%H:%M:%S') ==="

# 1) todate核对: 必须=9/9 (与信号CSV生成态一致, 否则fp失配触发78min重生成)
TODATE=$($VENV -c "import yaml;print(yaml.safe_load(open('$YAML',encoding='utf-8'))['backtest']['todate'])")
if [ "$TODATE" != "2026-09-09" ]; then
    echo "[fix] yaml todate=$TODATE → 2026-09-09"
    $VENV - <<'EOF'
import re
p='/mnt/d/quant/strategy/config/factor_config.yaml'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(todate:\s*)['\"].*?['\"]", r"\1'2026-09-09'", s, count=1)
assert s2!=s
open(p,'w',encoding='utf-8').write(s2)
EOF
fi

# 2) anchor产物备份(仅一次; 信号CSV不备份 — 臂跑warm不重写信号)
if [ ! -f "$RVR/portfolio_selections.bak0909.csv" ]; then
    cp "$RVR/portfolio_selections.csv" "$RVR/portfolio_selections.bak0909.csv"
    cp "$RVR/trade_realized.csv" "$RVR/trade_realized.bak0909.csv"
    cp "$RVR/equity_curve.csv" "$RVR/equity_curve.bak0909.csv" 2>/dev/null
    echo "[backup] anchor产物 → *.bak0909.csv"
fi

# 3) 翻开关 (yaml正文编辑, 保注释; 精确锚定yjyg块)
$VENV - <<EOF
import re
p='$YAML'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled:) false", r"\1 true", s, count=1)
assert s2!=s, "yjyg开关未找到"
if '$ARM' == 'slots2':
    s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled: true\n      window_days: 30\n      slots:) 1", r"\1 2", s2, count=1)
    assert "slots: 2" in s2
open(p,'w',encoding='utf-8').write(s2)
print("[flip] yjyg_fresh_neg_priority.enabled=true" + (" slots=2" if '$ARM'=='slots2' else ""))
EOF

# 失败时翻回
restore() {
    $VENV - <<'EOF'
import re
p='/mnt/d/quant/strategy/config/factor_config.yaml'
s=open(p,encoding='utf-8').read()
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled:) true", r"\1 false", s, count=1)
s2=re.sub(r"(yjyg_fresh_neg_priority:\n      enabled: false\n      window_days: 30\n      slots:) 2", r"\1 1", s2, count=1)
open(p,'w',encoding='utf-8').write(s2)
print("[restore] yjyg_fresh_neg_priority.enabled=false slots=1")
EOF
}
trap restore EXIT

# 4) 跑臂 (warm: 信号CSV复用9/9态, 仅组合层重跑)
$VENV strategy/bt_execution.py > "$LOG" 2>&1
RC=$?
echo "=== 臂跑结束 rc=$RC $(date '+%H:%M:%S') ==="

# 5) 提指标
echo "--- 指标 ---"
grep -E "最终净值|Sharpe:|最大回撤:|^  20(2[1-6]):" "$LOG" | tail -12
echo "--- yjyg机制触发计数 ---"
grep -E "yjyg优先槽|yjyg_flagged" "$LOG" | head -20
echo "--- 完 ---"
exit $RC
