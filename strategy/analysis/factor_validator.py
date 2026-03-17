"""
因子挖掘与验证 - 使用统一因子库
基于 core/factors.py 中的因子定义进行验证
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from multiprocessing import Pool

# 加载配置
script_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(strategy_dir)
sys.path.insert(0, strategy_dir)

from core.config_loader import load_config
from core.factors import calc_all_factors_for_validation

config = load_config(os.path.join(project_root, 'config/factor_config.yaml'))
detailed_industries = config.config.get('detailed_industries', {})


def process_stock(args):
    """处理单只股票的所有时间点"""
    code, df, sample_dates, fd = args
    stock_dates = sorted(df.index.tolist())

    results = []
    for sample_date in sample_dates:
        valid_dates = [d for d in stock_dates if d <= sample_date]
        if len(valid_dates) < 120:
            continue

        eval_date = valid_dates[-1]
        idx = stock_dates.index(eval_date)

        if idx < 120:
            continue

        history = df.iloc[:idx+1].iloc[-120:]
        if len(history) < 60:
            continue

        # 使用因子库的统一函数计算所有因子（含基本面）
        factors = calc_all_factors_for_validation(
            history['close'].values,
            history['high'].values if 'high' in history.columns else history['close'].values,
            history['low'].values if 'low' in history.columns else history['close'].values,
            history['volume'].values if 'volume' in history.columns else np.ones(len(history)),
            fundamental_data=fd,
            code=code,
            eval_date=eval_date
        )

        row = {'code': code, 'date': eval_date}
        for fn, vals in factors.items():
            # 处理数组和单个值
            if hasattr(vals, '__len__') and len(vals) > 0:
                val = vals[-1]
            else:
                val = vals
            if val is not None and not np.isnan(val):
                row[fn] = float(val)

        if idx + 20 < len(df):
            future_price = df.iloc[idx + 20]['close']
            current_price = df.iloc[idx]['close']
            if current_price > 0:
                row['future_ret'] = (future_price - current_price) / current_price
                results.append(row)

    return results


def precompute_all_factors(stock_data, fd, num_workers=8):
    """预计算所有因子"""
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    common_dates = sorted(all_dates)[120:-20]
    sample_dates = common_dates[::5]

    print(f"预计算因子 ({len(stock_data)} 只股票, {len(sample_dates)} 个时间点)...")

    args_list = [(code, df, sample_dates, fd) for code, df in stock_data.items()]

    all_factor_data = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap(process_stock, args_list, chunksize=10), total=len(args_list), desc="计算因子"):
            all_factor_data.extend(res)

    return pd.DataFrame(all_factor_data)


def validate_factor_vectorized(factor_name, factor_df, min_stocks=10):
    """向量化验证单个因子"""
    valid = factor_df.dropna(subset=[factor_name, 'future_ret'])
    if len(valid) < min_stocks:
        return None

    ic_series = []
    for date, group in valid.groupby('date'):
        if len(group) >= min_stocks:
            fv = group[factor_name].values
            fr = group['future_ret'].values
            valid_mask = ~(np.isnan(fv) | np.isnan(fr))
            if valid_mask.sum() >= min_stocks // 2:
                ic, _ = stats.spearmanr(fv[valid_mask], fr[valid_mask])
                if not np.isnan(ic):
                    ic_series.append(ic)

    if not ic_series:
        return None

    return {
        'factor': factor_name,
        'ic_mean': np.mean(ic_series),
        'ic_std': np.std(ic_series),
        'ir': np.mean(ic_series) / (np.std(ic_series) + 1e-10),
        'win_rate': np.mean([1 if i > 0 else 0 for i in ic_series])
    }


if __name__ == '__main__':
    print("=" * 60)
    print("因子验证 - 使用统一因子库（含基本面）")
    print("=" * 60)

    DATA_PATH = os.path.join(project_root, 'data/stock_data/backtrader_data/')
    FUND_PATH = os.path.join(project_root, 'data/stock_data/fundamental_data/')

    # 加载数据
    print("\n加载数据...")
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv']

    stock_data = {}
    for f in files:
        code = f.replace('_qfq.csv', '')
        df = pd.read_csv(os.path.join(DATA_PATH, f), parse_dates=['datetime']).set_index('datetime').sort_index()
        if len(df) >= 200:
            stock_data[code] = df

    print(f"加载 {len(stock_data)} 只股票")

    # 加载行业和基本面数据
    from core.fundamental import FundamentalData
    fd = FundamentalData(FUND_PATH, list(stock_data.keys()))

    industry_stocks = {cat: [] for cat in detailed_industries.keys()}

    common_dates = sorted(list(set().union(*[set(df.index) for df in stock_data.values()])))

    for code in stock_data:
        try:
            ind = fd.get_industry(code, common_dates[100])
            for cat, keywords in detailed_industries.items():
                if any(kw in ind for kw in keywords):
                    industry_stocks[cat].append(code)
                    break
        except:
            pass

    print("\n行业分布:")
    for k, v in sorted(industry_stocks.items(), key=lambda x: -len(x[1])):
        if len(v) > 0:
            print(f"  {k}: {len(v)}")

    # 计算因子（含基本面）
    factor_df = precompute_all_factors(stock_data, fd, num_workers=8)

    # 添加行业
    factor_df['industry'] = factor_df['code'].map(
        {code: cat for cat, codes in industry_stocks.items() for code in codes}
    )

    # 因子列表
    exclude_cols = ['code', 'date', 'future_ret', 'industry']
    factor_names = [c for c in factor_df.columns if c not in exclude_cols]
    print(f"\n{len(factor_names)} 个因子待验证")

    # 分行业验证
    results = {}
    for cat, codes in industry_stocks.items():
        if len(codes) < 5:
            continue

        print(f"\n=== {cat} ({len(codes)}只) ===")
        cat_df = factor_df[factor_df['code'].isin(codes)]

        if len(cat_df) == 0:
            continue

        min_stocks = max(5, min(15, len(codes) // 2))

        cat_results = []
        for fn in tqdm(factor_names, desc=cat):
            r = validate_factor_vectorized(fn, cat_df, min_stocks=min_stocks)
            if r:
                cat_results.append(r)

        cat_results.sort(key=lambda x: x['ir'], reverse=True)
        results[cat] = cat_results

        if cat_results:
            print("Top5: {}".format([(r['factor'], 'IC={:.3f}'.format(r['ic_mean'])) for r in cat_results[:5]]))

    # 保存结果
    results_dir = os.path.join(project_root, 'strategy/factor_validation_results')
    os.makedirs(results_dir, exist_ok=True)
    all_rows = []
    for cat, rs in results.items():
        for r in rs:
            all_rows.append({'industry': cat, **r})

    if all_rows:
        pd.DataFrame(all_rows).sort_values('ir', ascending=False).to_csv(
            os.path.join(results_dir, 'all_results.csv'), index=False
        )
        print(f"\n结果已保存到 {results_dir}/all_results.csv")

    # 汇总
    print("\n" + "=" * 60)
    print("各行业Top3因子汇总")
    print("=" * 60)
    for cat, rs in results.items():
        print(f"\n{cat}:")
        for r in rs[:3]:
            print(f"  {r['factor']}: IC={r['ic_mean']:.4f}, IR={r['ir']:.4f}, 胜率={r['win_rate']:.1%}")
