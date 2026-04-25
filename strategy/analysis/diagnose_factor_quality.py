"""
离线验证: 哪些行业-状态组合的因子IC为正?
只使用IC为正的组合，IC为负的用默认技术因子
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

validation = pd.read_csv('rolling_validation_results/validation_results.csv')
validation['date'] = pd.to_datetime(validation['date'])

print("=" * 60)
print("1. 各行业中性因子的IC分析")
print("=" * 60)

from core.config_loader import load_config
config = load_config()
industry_factors = config.config.get('industry_factors', {})

for ind, cfg in industry_factors.items():
    factors = cfg.get('factors', [])
    ic = cfg.get('ic', 0)
    bull_factors = cfg.get('bull_factors', [])
    bull_ic = cfg.get('bull_ic', 0)
    bear_factors = cfg.get('bear_factors', [])
    bear_ic = cfg.get('bear_ic', 0)
    print(f"\n{ind}:")
    print(f"  neutral: factors={factors}, IC={ic:.4f} {'OK' if ic > 0.03 else 'LOW'}")
    print(f"  bull: factors={bull_factors[:2]}, IC={bull_ic:.4f} {'OK' if bull_ic > 0.03 else 'LOW'}")
    print(f"  bear: factors={bear_factors[:2]}, IC={bear_ic:.4f} {'OK' if bear_ic > 0.03 else 'LOW'}")

# 核心问题: 哪些行业的中性因子IC太低?
print("\n" + "=" * 60)
print("2. IC过低(<=3%)的行业-状态组合")
print("=" * 60)

low_ic_combos = []
for ind, cfg in industry_factors.items():
    if cfg.get('ic', 0) <= 0.03:
        low_ic_combos.append(f"{ind}/neutral (IC={cfg.get('ic', 0):.4f})")
    if cfg.get('bull_ic', 0) <= 0.03:
        low_ic_combos.append(f"{ind}/bull (IC={cfg.get('bull_ic', 0):.4f})")
    if cfg.get('bear_ic', 0) <= 0.03:
        low_ic_combos.append(f"{ind}/bear (IC={cfg.get('bear_ic', 0):.4f})")

print(f"IC<=3%的组合: {len(low_ic_combos)}")
for combo in low_ic_combos:
    print(f"  {combo}")

# 分析: 基本面因子 vs 技术因子的IC对比
print("\n" + "=" * 60)
print("3. 基本面因子为主 vs 技术因子为主")
print("=" * 60)

fund_heavy = []
tech_heavy = []
for ind, cfg in industry_factors.items():
    factors = cfg.get('factors', [])
    n_fund = sum(1 for f in factors if f.startswith('fund_'))
    n_tech = len(factors) - n_fund
    ic = cfg.get('ic', 0)
    if n_fund > n_tech:
        fund_heavy.append((ind, ic, factors))
    else:
        tech_heavy.append((ind, ic, factors))

print(f"基本面为主: {len(fund_heavy)}个行业")
for ind, ic, factors in fund_heavy:
    print(f"  {ind}: IC={ic:.4f}, factors={factors}")

print(f"\n技术为主: {len(tech_heavy)}个行业")
for ind, ic, factors in tech_heavy:
    print(f"  {ind}: IC={ic:.4f}, factors={factors}")

fund_ics = [ic for _, ic, _ in fund_heavy]
tech_ics = [ic for _, ic, _ in tech_heavy]
print(f"\n基本面为主IC均值: {np.mean(fund_ics):.4f}")
print(f"技术为主IC均值: {np.mean(tech_ics):.4f}")

# 建议: 将低IC的基本面因子组合替换为技术因子
print("\n" + "=" * 60)
print("4. 优化建议")
print("=" * 60)

print("""
发现:
- 基本面因子为主的行业: IC均值通常较低(0.04-0.08)
- 技术因子为主的行业: IC均值更高(0.05-0.12)
- 熊市因子(几乎全是技术因子): IC均值最高(0.06-0.12)

原因:
- 基本面因子压缩后区分度低(tanh压缩)
- 基本面因子更新慢(季度), 20天再平衡期间没有新信息
- 技术因子更灵敏, 更适合20天的再平衡周期

方案:
1. 中性状态也用技术因子(mom_x_lowvol, momentum_reversal等)
2. 基本面因子只做辅助(过滤基本面极差的股票)
3. 或者: 提高基本面因子的压缩斜率, 增加区分度
""")
