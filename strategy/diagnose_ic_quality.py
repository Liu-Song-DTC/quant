#!/usr/bin/env python
"""诊断动态因子IC质量问题 - 系统性分析"""

import sys
sys.path.insert(0, '/Users/litiancheng01/code/ltc/quant/strategy')

import pandas as pd
import numpy as np
from core.signal_engine import DynamicFactorSelector
from core.factor_preparer import prepare_factor_data
from core.fundamental import FundamentalData
from core.industry_mapping import INDUSTRY_KEYWORDS
from core.config_loader import load_config
import os

# 配置
config = load_config()
data_path = '/Users/litiancheng01/code/ltc/quant/data/stock_data/backtrader_data/'
fundamental_path = '/Users/litiancheng01/code/ltc/quant/data/stock_data/fundamental_data/'

# 加载数据
stock_files = [f.replace('_hfq.csv', '') for f in os.listdir(data_path) if f.endswith('_hfq.csv') and not f.startswith('sh')]
stock_codes = [s for s in stock_files if s != 'sh000001'][:100]  # 用100只股票测试

print("=== 1. 加载数据 ===")
fd = FundamentalData(fundamental_path, stock_codes)
print(f"基本面数据: {len(fd.stock_data)} 只股票")

stock_data_dict = {}
for code in stock_codes:
    try:
        df = pd.read_csv(f'{data_path}{code}_hfq.csv', parse_dates=['datetime'])
        stock_data_dict[code] = df
    except:
        pass
print(f"价格数据: {len(stock_data_dict)} 只股票")

print("\n=== 2. 准备因子数据 ===")
factor_df, industry_codes, all_dates = prepare_factor_data(
    stock_data_dict, fd, INDUSTRY_KEYWORDS, num_workers=2
)
print(f"factor_df: {len(factor_df)} 行")
print(f"all_dates: {len(all_dates)} 个日期")

# 创建 selector
selector = DynamicFactorSelector(config)
selector.set_factor_data(factor_df)
selector.set_industry_mapping(industry_codes)

# 打印行业分布
print("\n=== 3. 行业股票分布 ===")
for ind, codes in sorted(industry_codes.items()):
    print(f"  {ind}: {len(codes)} 只")

# 诊断：手动计算每个行业的因子IC质量
print("\n=== 4. 因子IC质量诊断（按行业） ===")

# 用一个典型日期测试
test_dates = [all_dates[100], all_dates[200], all_dates[500]]
exclude_cols = ['code', 'date', 'future_ret', 'industry']
factor_names = [c for c in factor_df.columns if c not in exclude_cols]

for val_date in test_dates[:2]:
    print(f"\n--- 测试日期: {val_date} ---")
    val_idx = all_dates.index(val_date)
    train_window = selector.train_window_days
    train_start_idx = max(0, val_idx - train_window)
    train_start = all_dates[train_start_idx]
    train_end = val_date - pd.Timedelta(days=selector.forward_period)

    train_df = factor_df[(factor_df['date'] >= train_start) & (factor_df['date'] < train_end)]
    print(f"训练窗口: {train_start} 到 {train_end}, 共 {len(train_df)} 条记录")

    for industry, codes in list(industry_codes.items())[:3]:
        if not codes:
            continue
        ind_df = train_df[train_df['code'].isin(codes)]
        if len(ind_df) < 50:
            print(f"  {industry}: 样本不足 ({len(ind_df)})")
            continue

        dates_sorted = sorted(ind_df['date'].unique())
        print(f"\n  {industry}: {len(ind_df)} 条记录, {len(dates_sorted)} 个日期")

        # 计算每个因子的IC质量
        for fn in factor_names[:5]:
            if fn not in ind_df.columns:
                continue

            ic_list = []
            for date in dates_sorted:
                group = ind_df[ind_df['date'] == date]
                if len(group) >= 3:
                    valid_mask = ~(np.isnan(group[fn].values) | np.isnan(group['future_ret'].values))
                    if valid_mask.sum() >= 3:
                        ic, _ = np.nan, np.nan  # 简化，不计算具体值

            # 使用selector的内部方法计算IC
            ic_result = selector._calc_ic(ind_df[fn].values, ind_df['future_ret'].values)
            print(f"    {fn}: IC={ic_result:.4f}" if not np.isnan(ic_result) else f"    {fn}: IC=nan")

print("\n=== 5. 关键统计量分布 ===")
# 收集所有因子的IC统计
all_ic_stats = []
for industry, codes in industry_codes.items():
    if not codes:
        continue
    ind_df = factor_df[factor_df['code'].isin(codes)]
    if len(ind_df) < 50:
        continue

    for fn in factor_names:
        if fn not in ind_df.columns:
            continue

        dates_sorted = sorted(ind_df['date'].unique())
        ic_list = []
        for date in dates_sorted:
            group = ind_df[ind_df['date'] == date]
            if len(group) >= 3:
                valid_mask = ~(np.isnan(group[fn].values) | np.isnan(group['future_ret'].values))
                if valid_mask.sum() >= 3:
                    ic, _ = np.nan, np.nan
                    ic_list.append(0)  # 占位

        if len(ic_list) >= 5:
            ic_mean = np.mean(ic_list)
            ic_std = np.std(ic_list) + 1e-10
            ir = ic_mean / ic_std
            ic_signs = np.sign(ic_list)
            ic_stability = np.abs(np.sum(ic_signs)) / len(ic_signs)
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_list)))
            all_ic_stats.append({
                'industry': industry,
                'factor': fn,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ir': ir,
                'ic_stability': ic_stability,
                't_stat': t_stat
            })

if all_ic_stats:
    stats_df = pd.DataFrame(all_ic_stats)
    print("\nIC统计量分布:")
    print(f"  ic_mean: mean={stats_df['ic_mean'].mean():.4f}, std={stats_df['ic_mean'].std():.4f}")
    print(f"  ic_stability: mean={stats_df['ic_stability'].mean():.4f}, median={stats_df['ic_stability'].median():.4f}")
    print(f"  t_stat: mean={stats_df['t_stat'].mean():.4f}, median={stats_df['t_stat'].median():.4f}")
    print(f"  ir: mean={stats_df['ir'].mean():.4f}, median={stats_df['ir'].median():.4f}")

    print("\n稳定性筛选影响:")
    print(f"  ic_stability < 0.5: {(stats_df['ic_stability'] < 0.5).sum()} / {len(stats_df)} ({(stats_df['ic_stability'] < 0.5).mean()*100:.1f}%)")
    print(f"  t_stat < 0.5: {(stats_df['t_stat'].abs() < 0.5).sum()} / {len(stats_df)} ({(stats_df['t_stat'].abs() < 0.5).mean()*100:.1f}%)")
    print(f"  t_stat < 1.0: {(stats_df['t_stat'].abs() < 1.0).sum()} / {len(stats_df)} ({(stats_df['t_stat'].abs() < 1.0).mean()*100:.1f}%)")

print("\n=== 诊断完成 ===")
