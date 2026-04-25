"""
分析: 压缩前后的基本面因子IC对比
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os, sys

# 加载离线标定的因子数据（未压缩的基本面值 + 压缩后的值）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config_loader import load_config
from core.fundamental import FundamentalData
from core.factor_calculator import compress_fundamental_factor

# 加载validation结果
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')
validation = pd.read_csv(os.path.join(results_dir, 'validation_results.csv'))
validation['date'] = pd.to_datetime(validation['date'])

# 加载因子预计算数据（factor_preparer的输出，包含压缩后的基本面因子）
# 但factor_preparer的数据是临时计算后丢弃的...
# 只能从signals分析

# 用validation数据: factor_value = 压缩后的行业因子加权值
# 需要重新计算未压缩的值来做对比

# 先看看当前factor_value和future_ret的关系
print("="*60)
print("1. 当前factor_value的IC分布")
print("="*60)

daily_ics = []
for date, group in validation.groupby('date'):
    if len(group) < 50:
        continue
    ic, _ = spearmanr(group['factor_value'], group['future_ret'])
    daily_ics.append({'date': date, 'ic': ic})

ic_df = pd.DataFrame(daily_ics)
print(f"IC均值: {ic_df['ic'].mean():.4f}")
print(f"IC标准差: {ic_df['ic'].std():.4f}")
print(f"IR: {ic_df['ic'].mean()/ic_df['ic'].std():.4f}")
print(f"IC>0占比: {(ic_df['ic'] > 0).mean():.1%}")
print(f"IC>5%占比: {(ic_df['ic'] > 0.05).mean():.1%}")

# 检查: 基本面因子压缩后的值域
print("\n" + "="*60)
print("2. 基本面因子压缩效果分析")
print("="*60)

fund_factors = ['fund_score', 'fund_roe', 'fund_profit_growth', 'fund_revenue_growth',
                'fund_gross_margin', 'fund_cf_to_profit']

# 模拟压缩
test_values = {
    'fund_score': np.arange(0, 101, 5),           # 0-100
    'fund_roe': np.arange(-20, 41, 2),            # -20% to 40%
    'fund_profit_growth': np.arange(-100, 201, 10), # -100% to 200%
    'fund_revenue_growth': np.arange(-50, 101, 5),  # -50% to 100%
    'fund_gross_margin': np.arange(-10, 71, 5),     # -10% to 70%
    'fund_cf_to_profit': np.arange(-3, 6, 0.5),     # -3 to 5
}

print(f"\n{'Factor':<25} {'Raw范围':>15} {'压缩后范围':>15} {'有效区分度':>12}")
print("-"*70)
for factor_name, values in test_values.items():
    compressed = [compress_fundamental_factor(v, factor_name) for v in values]
    raw_range = max(values) - min(values)
    comp_range = max(compressed) - min(compressed)
    # 有效区分度: 压缩后范围 / 原始范围 * 样本数
    # 如果压缩后范围很小，说明大部分信息被压缩掉了
    print(f"{factor_name:<25} [{min(values):>5}, {max(values):>5}] [{min(compressed):>.3f}, {max(compressed):>.3f}] {comp_range/raw_range:.4f}")

# 关键: 检查压缩后不同原始值的区分度
print(f"\nfund_score 压缩映射:")
for v in range(0, 101, 10):
    c = compress_fundamental_factor(v, 'fund_score')
    print(f"  raw={v:>3} → compressed={c:.4f}")

print(f"\nfund_roe 压缩映射:")
for v in range(-10, 31, 5):
    c = compress_fundamental_factor(v, 'fund_roe')
    print(f"  raw={v:>3}% → compressed={c:.4f}")

# 核心问题: 压缩后区分度不够
# fund_score: raw 20-80 压缩后 -0.66到0.66, 大部分在(-0.5, 0.5)
# 这意味着好公司和差公司之间的factor_value差异只有0.3-0.5

# 建议方案: 分位数编码替代tanh压缩
print("\n" + "="*60)
print("3. 替代方案: 分位数编码 vs tanh压缩")
print("="*60)

print("""
问题分析:
- tanh压缩将不同质量的股票映射到相近的值
- fund_score=50 (中等) → 0.0
- fund_score=70 (较好) → 0.38
- fund_score=90 (优秀) → 0.76
- 差异: 70和90之间只有0.38, 但实际差距很大

替代方案:
1. 分位数编码: 在截面上将基本面因子转换为排名百分位
   - 优点: 直接利用截面信息, 不依赖tanh参数
   - 缺点: 需要在截面时间点有所有股票的数据

2. z-score标准化: (raw_value - mean) / std
   - 优点: 保留原始分布形状
   - 缺点: 需要历史统计量

3. 提高压缩参数: 让tanh的斜率更陡
   - fund_score: tanh((v-50)/25) 而不是 tanh((v-50)/50)
   - 这样70→0.69, 90→0.96, 区分度翻倍

推荐: 方案3(最简单), 提高压缩参数的斜率
""")
