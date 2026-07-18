"""
计算各行业在不同市场状态下的实证IC (bear / bull / neutral)

输出: 每个行业的 bear_ic, bull_ic, neutral_ic, 以及对应的样本量
用于替代config中当前全等于neutral_ic的bear_ic/bull_ic
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from core.market_regime_detector import MarketRegimeDetector

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE), 'data', 'stock_data', 'backtrader_data')

# 1. 加载验证数据(含future_ret)
val_path = os.path.join(BASE, 'rolling_validation_results', 'validation_results.csv')
print(f"加载验证数据: {val_path}")
val = pd.read_csv(val_path, parse_dates=['date'])
print(f"  行数: {len(val):,}, 日期范围: {val['date'].min()} ~ {val['date'].max()}")

# 过滤掉 industry 为空的行和 factor_value 无效的行
val = val[val['industry'].notna() & (val['industry'] != '')]
val = val[val['factor_value'].notna()]

# 2. 加载指数数据, 生成市场状态
idx_path = os.path.join(DATA_PATH, 'sh000001_qfq.csv')
if not os.path.exists(idx_path):
    idx_path = os.path.join(DATA_PATH, '000001_qfq.csv')
print(f"加载指数: {idx_path}")
idx_df = pd.read_csv(idx_path, parse_dates=['datetime'])
idx_df = idx_df[(idx_df['datetime'] >= val['date'].min()) & (idx_df['datetime'] <= val['date'].max())]

detector = MarketRegimeDetector()
regime_df = detector.generate(idx_df)
regime_df['date'] = regime_df['datetime'].dt.date
print(f"  指数行数: {len(regime_df):,}")

# 3. 合并市场状态到验证数据
val['date_d'] = val['date'].dt.date
regime_map = regime_df.set_index('date')[['regime', 'bear_risk', 'bear_risk_fast']]
val = val.join(regime_map, on='date_d', how='left')

# 填充早期无regime数据的日期为neutral
val['regime'] = val['regime'].fillna(0).astype(int)
val['bear_risk'] = val['bear_risk'].fillna(False)

print(f"\n合并后数据: {len(val):,} 行")
print(f"  分市场: bear_risk=True: {(val['bear_risk']==True).sum():,}, "
      f"bull(regime=1): {(val['regime']==1).sum():,}, "
      f"neutral: {((val['regime']==0)&(val['bear_risk']==False)).sum():,}")

# 4. 计算各行业×市场状态的IC
results = []
industries = sorted(val['industry'].unique())
print(f"\n计算 {len(industries)} 个行业的实证IC...")

for ind in industries:
    ind_data = val[val['industry'] == ind]
    if len(ind_data) < 30:
        continue

    row = {'industry': ind, 'total_samples': len(ind_data)}

    # Bear IC: bear_risk=True
    bear_data = ind_data[ind_data['bear_risk'] == True]
    if len(bear_data) >= 30:
        fv = bear_data['factor_value'].values
        fr = bear_data['future_ret'].values
        valid = ~(np.isnan(fv) | np.isnan(fr))
        if valid.sum() >= 30:
            ic, pval = spearmanr(fv[valid], fr[valid])
            row['bear_ic'] = round(ic, 4) if not np.isnan(ic) else 0.0
            row['bear_samples'] = int(valid.sum())
            row['bear_pvalue'] = round(pval, 4) if not np.isnan(pval) else 1.0

    # Bull IC: regime=1
    bull_data = ind_data[ind_data['regime'] == 1]
    if len(bull_data) >= 30:
        fv = bull_data['factor_value'].values
        fr = bull_data['future_ret'].values
        valid = ~(np.isnan(fv) | np.isnan(fr))
        if valid.sum() >= 30:
            ic, pval = spearmanr(fv[valid], fr[valid])
            row['bull_ic'] = round(ic, 4) if not np.isnan(ic) else 0.0
            row['bull_samples'] = int(valid.sum())
            row['bull_pvalue'] = round(pval, 4) if not np.isnan(pval) else 1.0

    # Neutral IC: neither bear_risk nor bull
    neutral_data = ind_data[(ind_data['bear_risk'] == False) & (ind_data['regime'] != 1)]
    if len(neutral_data) >= 30:
        fv = neutral_data['factor_value'].values
        fr = neutral_data['future_ret'].values
        valid = ~(np.isnan(fv) | np.isnan(fr))
        if valid.sum() >= 30:
            ic, pval = spearmanr(fv[valid], fr[valid])
            row['neutral_ic'] = round(ic, 4) if not np.isnan(ic) else 0.0
            row['neutral_samples'] = int(valid.sum())
            row['neutral_pvalue'] = round(pval, 4) if not np.isnan(pval) else 1.0

    # 整体IC
    fv_all = ind_data['factor_value'].values
    fr_all = ind_data['future_ret'].values
    valid_all = ~(np.isnan(fv_all) | np.isnan(fr_all))
    if valid_all.sum() >= 30:
        ic_all, _ = spearmanr(fv_all[valid_all], fr_all[valid_all])
        row['overall_ic'] = round(ic_all, 4) if not np.isnan(ic_all) else 0.0

    results.append(row)

# 5. 输出
df_result = pd.DataFrame(results)
df_result = df_result.set_index('industry')

# 填充默认值
for col in ['bear_ic', 'bull_ic', 'neutral_ic', 'overall_ic']:
    if col in df_result.columns:
        df_result[col] = df_result[col].fillna(0.0)

# 计算bear_ic - neutral_ic差异
df_result['bear_vs_neutral'] = df_result.get('bear_ic', 0) - df_result.get('neutral_ic', 0)
df_result['bull_vs_neutral'] = df_result.get('bull_ic', 0) - df_result.get('neutral_ic', 0)

print(f"\n=== 实证IC统计 ({len(df_result)} 个行业) ===")
print(f"\n整体:")
for col in ['bear_ic', 'bull_ic', 'neutral_ic', 'overall_ic']:
    if col in df_result.columns:
        vals = df_result[col][df_result[col] != 0]
        print(f"  {col}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"positive={((vals>0).sum()/len(vals)*100):.1f}% n={len(vals)}")

print(f"\nbear_ic > neutral_ic 的行业: {(df_result['bear_vs_neutral'] > 0.01).sum()}")
print(f"bear_ic < neutral_ic 的行业: {(df_result['bear_vs_neutral'] < -0.01).sum()}")
print(f"bear_ic ≈ neutral_ic 的行业: {(df_result['bear_vs_neutral'].abs() <= 0.01).sum()}")

# 显示差异最大的行业
print(f"\nBear IC 显著高于 Neutral 的行业 (Top 15):")
top_bear = df_result.nlargest(15, 'bear_vs_neutral')
for idx, row in top_bear.iterrows():
    b = row.get('bear_ic', 0)
    n = row.get('neutral_ic', 0)
    print(f"  {idx}: bear={b:.4f} neutral={n:.4f} diff={row['bear_vs_neutral']:+.4f} "
          f"(n_bear={row.get('bear_samples',0)}, n_neutral={row.get('neutral_samples',0)})")

print(f"\nBear IC 显著低于 Neutral 的行业 (Bottom 15):")
bot_bear = df_result.nsmallest(15, 'bear_vs_neutral')
for idx, row in bot_bear.iterrows():
    b = row.get('bear_ic', 0)
    n = row.get('neutral_ic', 0)
    print(f"  {idx}: bear={b:.4f} neutral={n:.4f} diff={row['bear_vs_neutral']:+.4f} "
          f"(n_bear={row.get('bear_samples',0)}, n_neutral={row.get('neutral_samples',0)})")

# 保存
out_path = os.path.join(BASE, 'rolling_validation_results', 'empirical_industry_ic.csv')
df_result.to_csv(out_path)
print(f"\n结果已保存: {out_path}")
print(f"可直接用于更新 factor_config.yaml 中的 industry_factors.{industry}.bear_ic")
