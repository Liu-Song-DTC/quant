"""
找出熊市IC为正的因子 — approach C: 添加熊市专属因子
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, 'cache')
VAL_DIR = os.path.join(BASE, 'rolling_validation_results')

# 1. 加载因子数据
factor_path = os.path.join(CACHE, 'factor_df_2812s_1258d_987d4d51.parquet')
print(f"加载因子: {factor_path}")
fdf = pd.read_parquet(factor_path)
print(f"  行数: {len(fdf):,}, 列数: {len(fdf.columns)}")

# 2. 加载市场状态
from core.market_regime_detector import MarketRegimeDetector
DATA_DIR = os.path.join(os.path.dirname(BASE), 'data', 'stock_data', 'backtrader_data')
idx_path = os.path.join(DATA_DIR, 'sh000001_qfq.csv')
if not os.path.exists(idx_path):
    idx_path = os.path.join(DATA_DIR, '000001_qfq.csv')
print(f"加载指数: {idx_path}")
idx_df = pd.read_csv(idx_path, parse_dates=['datetime'])
idx_df = idx_df[(idx_df['datetime'] >= fdf['date'].min()) & (idx_df['datetime'] <= fdf['date'].max())]
detector = MarketRegimeDetector()
regime_df = detector.generate(idx_df)
regime_df['date'] = regime_df['datetime'].dt.date
regime_map = regime_df.set_index('date')[['bear_risk', 'bear_risk_fast', 'regime']]
print(f"  指数行数: {len(regime_df)}")

# 3. 合并: fdf + regime
fdf['date_d'] = fdf['date'].dt.date
merged = fdf.join(regime_map, on='date_d', how='left')
merged['bear_risk'] = merged['bear_risk'].fillna(False)
merged['bear_risk_fast'] = merged['bear_risk_fast'].fillna(False)
merged['regime'] = merged['regime'].fillna(0).astype(int)
print(f"合并regime后: {len(merged):,}")

# 5. 计算每个因子的分市场IC
# 排除非因子列
skip_cols = {'code', 'date', 'date_d', 'industry', 'future_ret', 'bear_risk', 'bear_risk_fast', 'regime'}
factor_cols = [c for c in merged.columns if c not in skip_cols]

print(f"\n计算 {len(factor_cols)} 个因子的分市场IC...")

results = []
for col in factor_cols:
    data = merged[['future_ret', 'bear_risk', 'bear_risk_fast', 'regime', col]].dropna(subset=[col, 'future_ret'])
    if len(data) < 1000:
        continue

    # 整体IC
    valid = ~(np.isnan(data[col]) | np.isnan(data['future_ret']))
    if valid.sum() < 100:
        continue

    # Bear IC: bear_risk=True
    bear = data[data['bear_risk'] == True]
    bear_ic = np.nan
    if len(bear) >= 100:
        bv = bear[[col, 'future_ret']].dropna()
        if len(bv) >= 100:
            bear_ic, _ = spearmanr(bv[col], bv['future_ret'])

    # Bull IC: regime=1
    bull = data[data['regime'] == 1]
    bull_ic = np.nan
    if len(bull) >= 100:
        bv = bull[[col, 'future_ret']].dropna()
        if len(bv) >= 100:
            bull_ic, _ = spearmanr(bv[col], bv['future_ret'])

    # Neutral IC: neither bear nor bull
    neutral = data[(data['bear_risk'] == False) & (data['regime'] != 1)]
    neutral_ic = np.nan
    if len(neutral) >= 100:
        bv = neutral[[col, 'future_ret']].dropna()
        if len(bv) >= 100:
            neutral_ic, _ = spearmanr(bv[col], bv['future_ret'])

    # 整体IC
    bv_all = data[[col, 'future_ret']].dropna()
    overall_ic, _ = spearmanr(bv_all[col], bv_all['future_ret']) if len(bv_all) >= 100 else (np.nan, 1)

    results.append({
        'factor': col,
        'bear_ic': round(bear_ic, 4) if not np.isnan(bear_ic) else np.nan,
        'bull_ic': round(bull_ic, 4) if not np.isnan(bull_ic) else np.nan,
        'neutral_ic': round(neutral_ic, 4) if not np.isnan(neutral_ic) else np.nan,
        'overall_ic': round(overall_ic, 4) if not np.isnan(overall_ic) else np.nan,
        'bear_samples': len(bear),
        'bull_samples': len(bull),
        'neutral_samples': len(neutral),
    })

df_r = pd.DataFrame(results)
df_r['bear_vs_neutral'] = df_r['bear_ic'] - df_r['neutral_ic']

# 排序: bear_ic降序
df_r = df_r.sort_values('bear_ic', ascending=False)

print(f"\n=== 熊市IC TOP 20 ===")
for _, row in df_r.head(20).iterrows():
    print(f"  {row['factor']:30s}  bear={row['bear_ic']:+.4f}  neutral={row['neutral_ic']:+.4f}  "
          f"bull={row['bull_ic']:+.4f}  n={row['bear_samples']:,}")

print(f"\n=== 熊市IC BOTTOM 10 ===")
for _, row in df_r.tail(10).iterrows():
    print(f"  {row['factor']:30s}  bear={row['bear_ic']:+.4f}  neutral={row['neutral_ic']:+.4f}  "
          f"bull={row['bull_ic']:+.4f}")

# 统计
bear_positive = df_r[df_r['bear_ic'] > 0]
print(f"\n熊市IC>0的因子: {len(bear_positive)}/{len(df_r)}")
print(f"熊市IC均值: {df_r['bear_ic'].mean():.4f} (neutral: {df_r['neutral_ic'].mean():.4f})")

# 保存
out_path = os.path.join(VAL_DIR, 'factor_bear_ic.csv')
df_r.to_csv(out_path, index=False)
print(f"\n已保存: {out_path}")
