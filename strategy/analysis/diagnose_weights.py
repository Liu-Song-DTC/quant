"""
诊断权重计算: 为什么每只股票权重只有0.05-0.06?
模拟portfolio.py的_build_desired_value逻辑
"""

import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

signals = pd.read_csv(os.path.join(results_dir, 'backtest_signals.csv'))
signals['date'] = pd.to_datetime(signals['date'])

# 模拟portfolio选股过程
# 取一个典型rebalance日
sample_dates = sorted(signals['date'].unique())[200:205]  # 取中间段

for date in sample_dates:
    group = signals[signals['date'] == date]
    valid = group.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue

    # 计算rank_pct
    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)

    # 过滤rank_pct > 0.5
    qualified = valid[valid['rank_pct'] > 0.5]
    if len(qualified) == 0:
        qualified = valid[valid['rank_pct'] > 0.3]

    # 按factor_value排序
    qualified = qualified.sort_values('factor_value', ascending=False)

    # 行业均衡选股
    industry_count = {}
    selected = []
    for _, row in qualified.iterrows():
        ind = row.get('industry', 'default')
        if pd.isna(ind):
            ind = 'default'
        if industry_count.get(ind, 0) >= 2:
            continue
        selected.append(row.to_dict())
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(selected) >= 10:
            break

    if not selected:
        continue

    # 模拟权重计算 (和portfolio.py一致)
    total_position = 0
    for c in selected:
        rank_weight = c['rank_pct']  # 0.5-1.0
        risk_vol = max(0.01, min(1.0, c.get('risk_vol', 0.03)))
        vol_factor = min(1.0 / risk_vol, 2.0)
        extreme_factor = 0.7 if c.get('risk_extreme', False) else 1.0
        c['position'] = rank_weight * vol_factor * extreme_factor
        total_position += c['position']

    # 归一化
    max_gross_exposure = 1.0  # 假设drawdown=0
    raw_weights = {}
    for c in selected:
        raw_weights[c['code']] = (c['position'] / total_position) * max_gross_exposure

    # 限制单只权重
    max_single_weight = 0.10
    for code in raw_weights:
        if raw_weights[code] > max_single_weight:
            raw_weights[code] = max_single_weight

    # 重新归一化
    total_weight = sum(raw_weights.values())
    if total_weight > 0:
        raw_weights = {c: w / total_weight * max_gross_exposure for c, w in raw_weights.items()}

    print(f"\n日期: {date.date()}, 选股数: {len(selected)}")
    print(f"总权重: {sum(raw_weights.values()):.4f}")
    print(f"{'Code':<12} {'Rank%':>7} {'FV':>7} {'Score':>7} {'RiskVol':>8} {'VolFac':>7} {'Position':>9} {'Weight':>8}")
    print("-" * 80)
    for c in selected:
        code = c['code']
        w = raw_weights.get(code, 0)
        risk_vol = max(0.01, min(1.0, c.get('risk_vol', 0.03)))
        vol_factor = min(1.0 / risk_vol, 2.0)
        print(f"{code:<12} {c['rank_pct']:>7.3f} {c['factor_value']:>7.3f} {c['score']:>7.3f} {risk_vol:>8.4f} {vol_factor:>7.2f} {c['position']:>9.4f} {w:>8.4f}")

# 全局分析: 所有rebalance日
print("\n" + "="*60)
print("全局分析: rank_pct和风险指标对权重的影响")
print("="*60)

# 风险指标统计
risk_vols = signals['risk_vol'].dropna()  # 这列可能不存在
if 'risk_vol' not in signals.columns:
    # 从factor_value推算: risk_vol通常来自volatility_10
    print("信号中无risk_vol列")
else:
    print(f"risk_vol分布: mean={risk_vols.mean():.4f}, median={risk_vols.median():.4f}")
    print(f"  >0.03: {(risk_vols > 0.03).mean()*100:.1f}%")
    print(f"  >0.05: {(risk_vols > 0.05).mean()*100:.1f}%")
    print(f"  >0.10: {(risk_vols > 0.10).mean()*100:.1f}%")

# 核心问题: vol_factor = min(1/risk_vol, 2.0)
# risk_vol=0.02 -> vol_factor = min(50, 2) = 2.0
# risk_vol=0.03 -> vol_factor = min(33, 2) = 2.0
# risk_vol=0.10 -> vol_factor = min(10, 2) = 2.0
# 所有股票的vol_factor都是2.0! 这意味着vol_factor没有区分度

print("""
权重计算问题诊断:

1. vol_factor = min(1/risk_vol, 2.0)
   - risk_vol通常在0.02-0.05范围
   - 1/0.02=50, 1/0.03=33, 1/0.05=20 → 全被截断到2.0
   - vol_factor对所有股票几乎都是2.0 → 无区分度!

2. rank_weight = rank_pct (0.5-1.0)
   - 选中的股票rank_pct都在0.5以上
   - 区间太窄(0.5-1.0), 区分度有限

3. position = rank_weight * vol_factor * extreme_factor
   = rank_pct(0.5-1.0) * 2.0 * 1.0 = 1.0-2.0
   - 范围太窄, 所有position接近

4. 归一化后权重 = position/total_position * exposure
   - 所有position接近 → 权重接近等权
   - 6只股票: 每只约0.167
   - 但max_single_weight=0.10截断 → 每只0.10
   - 重新归一化: 6*0.10/0.60 * 1.0 = 1.0... 不对

等等，让我重新计算...

6只股票, position各约1.5, total=9.0
raw_weight = 1.5/9.0 * 1.0 = 0.167
max_single_weight=0.10 → 截断到0.10
6只 * 0.10 = 0.60
重新归一化: 0.10/0.60 * 1.0 = 0.167
每只0.167, 总权重1.0 → 应该是100%!

但实际总权重只有37%... 说明还有其他逻辑在降低权重!

让我检查: drawdown导致max_gross_exposure降低
或: risk_extreme导致extreme_factor=0.7
或: 其他因素
""")
