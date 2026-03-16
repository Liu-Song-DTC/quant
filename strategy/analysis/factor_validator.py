"""
因子挖掘与滚动验证 - 高效版
使用向量化操作 + 多进程并行
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 加载配置
script_dir = os.path.dirname(os.path.abspath(__file__))
strategy_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(strategy_dir)
sys.path.insert(0, strategy_dir)

from core.config_loader import load_config
config = load_config(os.path.join(project_root, 'config/factor_config.yaml'))
industry_category = config.get('industry_category', {})
detailed_industries = config.config.get('detailed_industries', {})

INDUSTRY_MAP = {}
for cat, inds in industry_category.items():
    for i in inds:
        INDUSTRY_MAP[i] = cat


def calc_all_factors_for_stock(df):
    """为一只股票计算所有因子（向量化）"""
    c = df['close'].values
    h = df['high'].values
    l = df['low'].values
    v = df['volume'].values
    o = df.get('open', df['close']).values
    n = len(df)

    factors = {}

    # ===== 基础因子 =====

    # 收益率
    for p in [3, 5, 8, 10, 20, 30, 60]:
        if n > p:
            factors[f'ret_{p}'] = c / np.roll(c, p) - 1

    # 动量
    for p in [5, 10, 20]:
        if n > p:
            factors[f'mom_{p}'] = c / np.roll(c, p) - 1

    # 价格 vs MA
    for s in [5, 10, 20, 30, 60]:
        if n > s:
            ma = pd.Series(c).rolling(s).mean().values
            factors[f'price_ma_{s}'] = c / ma - 1

    # MA交叉
    if n > 60:
        ma5, ma10, ma20, ma30, ma60 = [pd.Series(c).rolling(s).mean().values for s in [5, 10, 20, 30, 60]]
        factors['ma_cross_5_10'] = (ma5 - ma10) / (ma10 + 1e-10)
        factors['ma_cross_5_20'] = (ma5 - ma20) / (ma20 + 1e-10)
        factors['ma_cross_10_20'] = (ma10 - ma20) / (ma20 + 1e-10)
        factors['ma_cross_10_60'] = (ma10 - ma60) / (ma60 + 1e-10)
        factors['ma_cross_20_60'] = (ma20 - ma60) / (ma60 + 1e-10)

        # 价格位置
        hp, lp = pd.Series(h).rolling(20).max().values, pd.Series(l).rolling(20).min().values
        factors['price_pos_20'] = (c - lp) / (hp - lp + 1e-10)

        # 最高/最低价位置
        factors['high_pos'] = (h - lp) / (hp - lp + 1e-10)
        factors['low_pos'] = (c - lp) / (hp - lp + 1e-10)

    # 波动率
    r = pd.Series(c).pct_change().values
    for p in [5, 10, 20, 30, 60]:
        if n > p:
            factors[f'volatility_{p}'] = pd.Series(r).rolling(p).std().values

    # RSI
    d = np.diff(c, prepend=c[0])
    g, lss = np.where(d > 0, d, 0), np.where(d < 0, -d, 0)
    for p in [6, 10, 14, 20]:
        if n > p:
            ag, al = pd.Series(g).rolling(p).mean().values, pd.Series(lss).rolling(p).mean().values
            factors[f'rsi_{p}'] = 100 - (100 / (1 + ag/(al+1e-10)))

    # 成交量
    for p in [5, 10, 20]:
        if n > p:
            vm = pd.Series(v).rolling(p).mean().values
            factors[f'vol_ma_{p}'] = v / (vm + 1e-10)

    # 布林带
    for p in [10, 20]:
        if n > p:
            ma = pd.Series(c).rolling(p).mean().values
            std = pd.Series(c).rolling(p).std().values
            factors[f'bb_pos_{p}'] = (c - (ma - 2*std)) / (4*std + 1e-10)
            factors[f'bb_width_{p}'] = 4*std / (ma + 1e-10)

    # ATR (平均真实波幅)
    for p in [10, 14, 20]:
        if n > p:
            tr1 = h - l
            tr2 = np.abs(h - np.roll(c, 1))
            tr3 = np.abs(l - np.roll(c, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            factors[f'atr_{p}'] = pd.Series(tr).rolling(p).mean().values / (c + 1e-10)

    # 威廉姆斯%R
    for p in [14, 20]:
        if n > p:
            hh = pd.Series(h).rolling(p).max().values
            ll = pd.Series(l).rolling(p).min().values
            factors[f'williams_r_{p}'] = -100 * (hh - c) / (hh - ll + 1e-10)

    # 成交量变化率
    for p in [5, 10, 20]:
        if n > p:
            factors[f'vol_change_{p}'] = v / (np.roll(v, p) + 1e-10) - 1

    # 收益波动率比
    for p in [10, 20]:
        if n > p and f'mom_{p}' in factors and f'volatility_{p}' in factors:
            factors[f'ret_vol_ratio_{p}'] = factors[f'mom_{p}'] / (factors[f'volatility_{p}'] + 1e-10)

    # ===== 复合因子 =====
    # 动量差（动量加速/减速）
    for p1, p2 in [(3, 5), (5, 10), (5, 20), (10, 20), (20, 60)]:
        if f'ret_{p1}' in factors and f'ret_{p2}' in factors:
            factors[f'mom_diff_{p1}_{p2}'] = factors[f'ret_{p1}'] - factors[f'ret_{p2}']

    # 低波动 + 高位置
    if 'price_pos_20' in factors and 'volatility_10' in factors:
        factors['low_vol_high_pos'] = -factors['volatility_10'] + factors['price_pos_20']
        factors['low_vol_high_pos_20'] = -factors['volatility_20'] + factors['price_pos_20']

    # 动量 * 低波动
    for mp in [5, 10, 20]:
        for vp in [10, 20]:
            if f'mom_{mp}' in factors and f'volatility_{vp}' in factors:
                factors[f'mom_x_lowvol_{mp}_{vp}'] = factors[f'mom_{mp}'] * (-factors[f'volatility_{vp}'])

    # RSI超卖 + 动量反弹
    for rp in [10, 14, 20]:
        for mp in [5, 10, 20]:
            if f'rsi_{rp}' in factors and f'mom_{mp}' in factors:
                # RSI低于30可能反弹，动量向上可能延续
                factors[f'rsi_mom_{rp}_{mp}'] = (50 - factors[f'rsi_{rp}']) + factors[f'mom_{mp}']

    # MA多头排列强度
    if n > 60:
        ma5, ma10, ma20, ma60 = [pd.Series(c).rolling(s).mean().values for s in [5, 10, 20, 60]]
        factors['ma_golden_count'] = ((ma5 > ma10).astype(float) +
                                       (ma10 > ma20).astype(float) +
                                       (ma20 > ma60).astype(float))

    # 价格突破
    for p in [20, 60]:
        if n > p:
            hh = pd.Series(h).rolling(p).max().values
            factors[f'price_break_{p}'] = c / (hh + 1e-10) - 1

    # 缩量反弹
    if 'mom_10' in factors and 'vol_ma_10' in factors:
        factors['vol_shrink_up'] = factors['mom_10'] * (-factors['vol_ma_10'] + 1)

    # 强势股回调
    if 'price_pos_20' in factors and 'mom_20' in factors:
        factors['strong_pullback'] = factors['price_pos_20'] * (1 - factors['mom_20'])

    return factors


def precompute_all_factors(stock_data):
    """预计算所有股票的所有因子（严格滚动窗口：每只股票使用自己的交易日）"""
    # 获取全局日期范围（用于确定采样点）
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index.tolist())
    common_dates = sorted(all_dates)[120:-20]
    sample_dates = common_dates[::5]  # 每5天一个采样点

    print(f"预计算因子 ({len(stock_data)} 只股票, {len(sample_dates)} 个验证时间点)...")

    all_factor_data = []

    for code, df in tqdm(stock_data.items(), desc="计算因子"):
        # 股票自己的交易日
        stock_dates = sorted(df.index.tolist())

        # 找出股票在采样日期附近可交易的日期
        for sample_date in sample_dates:
            # 找到离采样日期最近且已发生的交易日
            valid_dates = [d for d in stock_dates if d <= sample_date]
            if len(valid_dates) < 120:
                continue

            eval_date = valid_dates[-1]  # 使用最近的交易日
            idx = stock_dates.index(eval_date)

            # 确保有足够历史
            if idx < 120:
                continue

            # 取评估日期前的120天数据计算因子
            history = df.iloc[:idx+1].iloc[-120:]

            if len(history) < 60:
                continue

            # 计算因子
            factors = calc_all_factors_for_stock(history)

            # 取因子最新值
            row = {'code': code, 'date': eval_date}
            for fn, vals in factors.items():
                if len(vals) > 0 and not np.isnan(vals[-1]):
                    row[fn] = vals[-1]

            # 计算20天后的收益
            if idx + 20 < len(df):
                future_price = df.iloc[idx + 20]['close']
                current_price = df.iloc[idx]['close']
                if current_price > 0:
                    row['future_ret'] = (future_price - current_price) / current_price
                    all_factor_data.append(row)

    return pd.DataFrame(all_factor_data)


def validate_factor_vectorized(factor_name, factor_df, min_stocks=30):
    """向量化验证单个因子"""
    # 过滤有效数据
    valid = factor_df.dropna(subset=[factor_name, 'future_ret'])
    if len(valid) < min_stocks:
        return None

    # 按日期分组计算IC
    ic_series = []
    for date, group in valid.groupby('date'):
        if len(group) >= min_stocks:
            fv = group[factor_name].values
            fr = group['future_ret'].values
            valid_mask = ~(np.isnan(fv) | np.isnan(fr))
            if valid_mask.sum() >= 10:
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
    print("高效因子验证 - 滚动时间序列验证")
    print("=" * 60)

    DATA_PATH = os.path.join(project_root, 'data/stock_data/backtrader_data/')
    FUND_PATH = os.path.join(project_root, 'data/stock_data/fundamental_data/')

    # 加载数据
    print("\n加载数据...")
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv']

    stock_data = {}
    all_dates = set()

    for f in files:
        code = f.replace('_qfq.csv', '')
        df = pd.read_csv(os.path.join(DATA_PATH, f), parse_dates=['datetime']).set_index('datetime').sort_index()
        if len(df) >= 200:
            stock_data[code] = df
            all_dates |= set(df.index)

    common_dates = sorted(list(all_dates))
    print(f"加载 {len(stock_data)} 只股票, {len(common_dates)} 交易日")

    # 加载行业 - 使用细分行业
    from core.fundamental import FundamentalData
    fd = FundamentalData(FUND_PATH, list(stock_data.keys()))

    # 细分行业分类（从配置读取）
    industry_groups = detailed_industries if detailed_industries else {}

    # 将细分行业归类
    industry_stocks = {}
    for cat in industry_groups.keys():
        industry_stocks[cat] = []

    for code in stock_data:
        try:
            ind = fd.get_industry(code, common_dates[100])
            assigned = False
            for cat, keywords in industry_groups.items():
                if any(kw in ind for kw in keywords):
                    industry_stocks[cat].append(code)
                    assigned = True
                    break
        except:
            pass

    # 不合并分类，验证时过滤掉股票数太少的

    print("\n行业分布:")
    for k, v in sorted(industry_stocks.items(), key=lambda x: -len(x[1])):
        if len(v) > 0:
            print(f"  {k}: {len(v)}")

    # 预计算所有因子（严格滚动窗口，每只股票用自己的交易日）
    factor_df = precompute_all_factors(stock_data)

    # 添加行业信息
    factor_df['industry'] = factor_df['code'].map(
        {code: cat for cat, codes in industry_stocks.items() for code in codes}
    )

    # 获取所有因子名
    exclude_cols = ['code', 'date', 'future_ret', 'industry']
    factor_names = [c for c in factor_df.columns if c not in exclude_cols]
    print(f"\n{len(factor_names)} 个因子待验证")

    # 分行业验证
    results = {}
    for cat, codes in industry_stocks.items():
        if len(codes) < 10:
            continue

        print(f"\n=== {cat} ({len(codes)}只) ===")
        cat_df = factor_df[factor_df['code'].isin(codes)]

        if len(cat_df) == 0:
            continue

        cat_results = []
        for fn in tqdm(factor_names, desc=cat):
            r = validate_factor_vectorized(fn, cat_df, min_stocks=15)
            if r:
                cat_results.append(r)

        cat_results.sort(key=lambda x: x['ir'], reverse=True)
        results[cat] = cat_results

        if cat_results:
            print("Top5: {}".format([(r['factor'], 'IC={:.3f}'.format(r['ic_mean'])) for r in cat_results[:5]]))

    # 保存结果
    results_dir = os.path.join(project_root, 'factor_validation_results')
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

    # 打印汇总
    print("\n" + "=" * 60)
    print("各行业Top3因子汇总")
    print("=" * 60)
    for cat, rs in results.items():
        print(f"\n{cat}:")
        for r in rs[:3]:
            print(f"  {r['factor']}: IC={r['ic_mean']:.4f}, IR={r['ir']:.4f}, 胜率={r['win_rate']:.1%}")
