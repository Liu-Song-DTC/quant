#!/bin/bash
# E-K2.5(2026-09-11): 回购类级加成 0.20 首臂 — 9/10态, 基线 979,885/291.95%/1.4094/18.33%
# 信号复用(portfolio节豁免指纹, 机制代码portfolio.py不在信号指纹列表) → 全链仅回测段 ~15-20min
# 严格串行(不要并行, 会oom)。协议: 翻yaml→机制烟测→anchor备份→bt_execution→提四指标
#        →trap翻回yaml+还原anchor产物。铁律: 四指标(NAV/收益/Sharpe/MDD)全优才考虑采纳。
set -u
cd /mnt/d/quant
VENV=.venv/bin/python
YAML=strategy/config/factor_config.yaml
RVR=strategy/rolling_validation_results
LOG=strategy/logs/bt_execution_K25_0911.log
MARK=/tmp/k25_flipped

echo "=== E-K2.5 臂: repo020 开始 $(date '+%H:%M:%S') ==="

# 1) 翻 repo_score_boost 0.0 → 0.20 (幂等)
$VENV - <<EOF
import re
p='$YAML'; s=open(p,encoding='utf-8').read()
m = re.search(r"(repo_score_boost:\s*)([\d.]+)", s)
assert m, 'repo_score_boost键未找到'
if m.group(2) == '0.20':
    print('[skip] 已是0.20')
else:
    s2 = s[:m.start()] + m.group(1) + '0.20' + s[m.end():]
    open(p,'w',encoding='utf-8').write(s2)
    print(f"[flip] repo_score_boost {m.group(2)} -> 0.20")
open('$MARK','w').write('1')
EOF

restore() {
    $VENV - <<EOF
import re
p='$YAML'; s=open(p,encoding='utf-8').read()
s2=re.sub(r"(repo_score_boost:\s*)[\d.]+", r"\g<1>0.0", s, count=1)
open(p,'w',encoding='utf-8').write(s2)
print('[restore] repo_score_boost -> 0.0')
EOF
    for f in portfolio_selections trade_realized equity_curve; do
        if [ -f "$RVR/$f.preRepo0911.csv" ]; then
            cp "$RVR/$f.preRepo0911.csv" "$RVR/$f.csv"
            echo "[restore] $f.csv <- anchor"
        fi
    done
    rm -f "$MARK"
}
trap restore EXIT

# 2) 机制烟测: PortfolioConstructor读到0.20 + flag路径用pkl真实样本验证
$VENV - <<EOF
import sys; sys.path.insert(0, '/mnt/d/quant/strategy')
import pickle
import pandas as pd
from core.portfolio import PortfolioConstructor
pc = PortfolioConstructor()
assert abs(pc.repo_score_boost - 0.20) < 1e-9, f'烟测失败: repo_score_boost={pc.repo_score_boost}'
plans = pickle.load(open('/mnt/d/quant/data/alternative_data/repurchase_plans.pkl','rb'))
plans = plans[plans['NOTICEDATE'] >= '2020-12-01']
plans = plans[pd.to_numeric(plans['JESX'], errors='coerce') < 1e8].sort_values('NOTICEDATE')
code, d = str(plans.iloc[0]['code']).zfill(6), pd.Timestamp(plans.iloc[0]['NOTICEDATE'])
f0 = pc._repo_flag(code, d)                      # 公告当日 -> True
f31 = pc._repo_flag(code, d + pd.Timedelta(days=31))  # 31d后 -> False
fneg = pc._repo_flag(code, d - pd.Timedelta(days=1))  # 公告前一日 -> False
assert f0 and not f31 and not fneg, f'烟测失败: f0={f0} f31={f31} fneg={fneg}'
print(f'[smoke] OK: boost={pc.repo_score_boost}, {code} {d.date()} flag(当日)={f0} 31d后={f31} 前日={fneg}')
EOF
RC_SMOKE=$?
if [ $RC_SMOKE -ne 0 ]; then echo "烟测失败, 终止(不跑全链)"; exit 1; fi

# 3) anchor产物备份
cp "$RVR/portfolio_selections.csv" "$RVR/portfolio_selections.preRepo0911.csv"
cp "$RVR/trade_realized.csv" "$RVR/trade_realized.preRepo0911.csv"
cp "$RVR/equity_curve.csv" "$RVR/equity_curve.preRepo0911.csv" 2>/dev/null
echo "[backup] anchor产物 -> *.preRepo0911.csv"

# 4) 回测段跑 (信号复用, 约15-20min)
$VENV strategy/bt_execution.py > "$LOG" 2>&1
RC=$?
echo "=== 臂跑结束 rc=$RC $(date '+%H:%M:%S') ==="

# 5) 提四指标
grep -E "复用已有信号|重新生成信号" "$LOG" | head -3
echo "--- 指标 (基线 979,885/291.95%/1.4094/18.33%) ---"
grep -E "最终净值|Sharpe|最大回撤" "$LOG" | tail -5
exit $RC
