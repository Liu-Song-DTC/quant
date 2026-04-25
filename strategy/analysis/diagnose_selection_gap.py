"""
核心诊断: 为什么实际回测只选6只，而离线模拟选10只?
检查signal_store在rebalance日的信号完整性
"""

import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'rolling_validation_results')

signals = pd.read_csv(os.path.join(results_dir, 'backtest_signals.csv'))
selections = pd.read_csv(os.path.join(results_dir, 'portfolio_selections.csv'))

signals['date'] = pd.to_datetime(signals['date'])
selections['date'] = pd.to_datetime(selections['date'])

# 关键: backtest_signals.csv存的是所有股票所有日期的信号
# 但portfolio.build()只看tradable_universe（非停牌、价格>2、成交量>100）
# 和signal_store.get(code, date)有值的股票

# 问题1: tradable_universe过滤
# bt_execution.py: _is_tradable() checks price > 2 and volume > 100
# 这会排除停牌股和低价股

# 问题2: signal_store.get(code, date) 可能返回None
# 如果code不在signal_store中（比如数据太短<60天）

# 问题3: rank_pct > 0.5 过滤
# 在实际回测中，tradable_universe可能只有200只（而非全量481只）
# 200只的rank_pct>0.5只有100只，然后行业均衡选10只
# 但如果很多股票的factor_value缺失或为NaN...

# 关键检查: 实际选股日期中，选中了哪些股票，和信号对比
print("="*60)
print("1. 实际选股 vs 信号匹配")
print("="*60)

# 看实际选股日的信号数据
sel_dates = sorted(selections['date'].unique())
print(f"选股日期数: {len(sel_dates)}")

# 看几个典型选股日
for date in sel_dates[:5]:
    sel_codes = set(selections[selections['date'] == date]['code'].tolist())
    sig_on_date = signals[signals['date'] == date]

    # 有信号的股票数
    n_with_signal = len(sig_on_date)
    # buy=True的股票数
    n_buy = sig_on_date['buy'].sum()
    # factor_value非NaN的
    n_valid_fv = sig_on_date['factor_value'].notna().sum()

    # rank_pct>0.5的（模拟计算）
    valid = sig_on_date.dropna(subset=['factor_value'])
    if len(valid) > 0:
        valid = valid.copy()
        valid['rank_pct'] = valid['factor_value'].rank(pct=True)
        n_qualified = (valid['rank_pct'] > 0.5).sum()
    else:
        n_qualified = 0

    # 选中的行业分布
    sel_on_date = selections[selections['date'] == date]

    print(f"\n{date.date()}: 信号{n_with_signal}只, buy={n_buy}, "
          f"有效fv={n_valid_fv}, rank>0.5={n_qualified}, "
          f"实际选{len(sel_on_date)}只")

    # 检查: 选中的股票是否rank_pct>0.5
    if len(valid) > 0 and len(sel_on_date) > 0:
        sel_fv = valid[valid['code'].isin(sel_codes)]
        print(f"  选中股票在valid中: {len(sel_fv)}/{len(sel_on_date)}")
        if len(sel_fv) > 0:
            print(f"  选中股票rank_pct: mean={sel_fv['rank_pct'].mean():.4f}")
            print(f"  选中股票factor_value: mean={sel_fv['factor_value'].mean():.4f}")

    # 行业分布
    if 'industry' in sel_on_date.columns:
        ind_dist = sel_on_date['industry'].value_counts()
        print(f"  行业: {dict(ind_dist)}")

# 问题: 实际选股只有6只，为什么?
# 可能: rank_pct > 0.5过滤后，行业均衡选不够10只
# 或者: factor_value分布有问题，大部分集中在0附近

print("\n" + "="*60)
print("2. factor_value截面分布 (典型日)")
print("="*60)

for date in sel_dates[50:55]:
    sig_on_date = signals[signals['date'] == date]
    valid = sig_on_date.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue

    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)

    # factor_value十分位分布
    valid['decile'] = pd.qcut(valid['factor_value'], 10, labels=False, duplicates='drop')
    decile_stats = valid.groupby('decile').agg(
        count=('code', 'count'),
        fv_mean=('factor_value', 'mean'),
        fv_min=('factor_value', 'min'),
        fv_max=('factor_value', 'max'),
    )

    sel_on_date = selections[selections['date'] == date]
    print(f"\n{date.date()}: 有效{len(valid)}只, 选中{len(sel_on_date)}只")
    print(f"factor_value十分位:")
    for d, row in decile_stats.iterrows():
        marker = " <-- SELECTED" if d >= 5 else ""
        print(f"  D{d}: count={row['count']}, fv=[{row['fv_min']:.3f}, {row['fv_max']:.3f}], mean={row['fv_mean']:.3f}{marker}")

# 关键: 检查选股中的industry空值问题
print("\n" + "="*60)
print("3. 行业信息缺失问题")
print("="*60)

# 在portfolio.py中, industry来自sig.industry
# 如果sig.industry为空字符串, 会被设为'default'
# 所有industry=''的股票被归到'default'组, industry_cap=2只选2只

sig_with_industry = signals[signals['industry'].notna() & (signals['industry'] != '')]
sig_without_industry = signals[(signals['industry'].isna()) | (signals['industry'] == '')]
print(f"有行业信息: {len(sig_with_industry)} ({100*len(sig_with_industry)/len(signals):.1f}%)")
print(f"无行业信息: {len(sig_without_industry)} ({100*len(sig_without_industry)/len(signals):.1f}%)")

# 检查行业为空的情况
if len(sig_without_industry) > 0:
    print(f"\n无行业信息的factor_value分布:")
    fv_no_ind = sig_without_industry['factor_value'].dropna()
    print(f"  mean: {fv_no_ind.mean():.4f}")
    print(f"  >0: {(fv_no_ind > 0).mean()*100:.1f}%")
    print(f"  >0.5: {(fv_no_ind > 0.5).mean()*100:.1f}%")
    print(f"\n[问题] industry=''的股票全被归到'default'组, industry_cap=2只选2只!")
    print(f"  如果有50只无行业股票的rank_pct>0.5, 只能选2只")

# 在实际选股中查看industry分布
sel_industry = selections['industry'].value_counts()
print(f"\n实际选股行业分布:")
for ind, cnt in sel_industry.items():
    print(f"  '{ind}': {cnt}")

# 检查有多少空行业
empty_ind = selections[(selections['industry'].isna()) | (selections['industry'] == '')]
print(f"\n空行业选股: {len(empty_ind)} 条")

# 关键测试: 如果去掉industry均衡限制，能选多少只?
print("\n" + "="*60)
print("4. 去掉行业均衡限制后的模拟选股")
print("="*60)

sim_no_limit = []
for date, group in signals.groupby('date'):
    valid = group.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue
    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)
    qualified = valid[valid['rank_pct'] > 0.5]
    if len(qualified) == 0:
        qualified = valid[valid['rank_pct'] > 0.3]
    top10 = qualified.nlargest(10, 'factor_value')
    sim_no_limit.extend([{'date': date, 'code': row['code']} for _, row in top10.iterrows()])

sim_nl_df = pd.DataFrame(sim_no_limit)
print(f"无行业限制选股: {sim_nl_df.groupby('date').size().mean():.1f}只/天")

# 对比: 有行业均衡限制
sim_with_limit = []
for date, group in signals.groupby('date'):
    valid = group.dropna(subset=['factor_value'])
    if len(valid) == 0:
        continue
    valid = valid.copy()
    valid['rank_pct'] = valid['factor_value'].rank(pct=True)
    qualified = valid[valid['rank_pct'] > 0.5]
    if len(qualified) == 0:
        qualified = valid[valid['rank_pct'] > 0.3]
    qualified = qualified.sort_values('factor_value', ascending=False)

    industry_count = {}
    selected = []
    for _, row in qualified.iterrows():
        ind = row.get('industry', '')
        if pd.isna(ind) or ind == '':
            ind = 'default'
        if industry_count.get(ind, 0) >= 2:
            continue
        selected.append(row)
        industry_count[ind] = industry_count.get(ind, 0) + 1
        if len(selected) >= 10:
            break

    sim_with_limit.append({'date': date, 'n_selected': len(selected)})

sim_wl_df = pd.DataFrame(sim_with_limit)
print(f"行业均衡选股: {sim_wl_df['n_selected'].mean():.1f}只/天")
print(f"  选<10只的天数: {(sim_wl_df['n_selected'] < 10).sum()}/{len(sim_wl_df)}")
print(f"  选6只的天数: {(sim_wl_df['n_selected'] == 6).sum()}/{len(sim_wl_df)}")

# 检查: 在选不够10只时，是因为行业均衡还是因为qualified不够?
few_sel = sim_wl_df[sim_wl_df['n_selected'] < 10]
print(f"\n选不够10只的日期数: {len(few_sel)}")
if len(few_sel) > 0:
    print(f"分布: {few_sel['n_selected'].value_counts().sort_index().to_dict()}")
