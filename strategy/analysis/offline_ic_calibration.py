#!/usr/bin/env python
"""
离线因子标定脚本 - 从原始 OHLCV 数据计算因子，进行滚动 IC 验证

流程:
1. 加载股票 OHLCV 数据和技术指标
2. 加载基本面数据
3. 对候选因子组合进行滚动 IC 验证
4. 输出推荐配置
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STRATEGY_DIR = os.path.dirname(SCRIPT_DIR)
QUANT_DIR = STRATEGY_DIR
sys.path.insert(0, STRATEGY_DIR)

from core.factor_calculator import calculate_indicators, compute_composite_factors


def get_candidate_factors():
    """获取候选因子列表"""
    return [
        'mom_5', 'mom_10', 'mom_20', 'mom_30',
        'volatility_5', 'volatility_10', 'volatility_20',
        'rsi_6', 'rsi_8', 'rsi_10', 'rsi_14',
        'bb_width_20',
        'volume_ratio',
    ]


def get_industry_stocks(fundamental_df, industry_keywords):
    """获取行业对应的股票"""
    industry_stocks = {}
    for _, row in fundamental_df.iterrows():
        industry = row.get('industry', '')
        code = row.get('code') or row.get('股票代码', '')
        if not industry or not code:
            continue
        for kw in industry_keywords:
            if kw in str(industry):
                if industry not in industry_stocks:
                    industry_stocks[industry] = set()
                industry_stocks[industry].add(str(code).zfill(6))
                break
    return industry_stocks


def load_data():
    """加载数据和因子"""
    print("加载数据...")

    # 加载价格数据
    price_dir = os.path.join(QUANT_DIR, 'data/stock_data/backtrader_data')
    all_prices = []
    for f in os.listdir(price_dir):
        if not f.endswith('_qfq.csv'):
            continue
        df = pd.read_csv(os.path.join(price_dir, f))
        df['code'] = f.replace('_qfq.csv', '')
        all_prices.append(df)

    price_df = pd.concat(all_prices, ignore_index=True)
    price_df['date'] = pd.to_datetime(price_df['date'])
    price_df = price_df.sort_values(['code', 'date'])
    print(f"价格数据: {len(price_df):,} 条, {price_df['code'].nunique()} 只股票")

    # 加载基本面数据
    fund_dir = os.path.join(QUANT_DIR, 'data/stock_data/fundamental_data')
    all_fund = []
    for f in os.listdir(fund_dir):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(fund_dir, f))
        code = f.replace('.csv', '')
        df['code'] = code.zfill(6) if len(code) < 6 else code
        all_fund.append(df)

    fund_df = pd.concat(all_fund, ignore_index=True)
    print(f"基本面数据: {len(fund_df):,} 条, {fund_df['code'].nunique()} 只股票")

    return price_df, fund_df


def compute_factors_for_stock(df_stock):
    """计算单只股票的技术因子"""
    if len(df_stock) < 60:
        return None

    df_stock = df_stock.sort_values('date').tail(500)

    close = df_stock['close'].values.astype(float)
    high = df_stock.get('high', close).values.astype(float)
    low = df_stock.get('low', close).values.astype(float)
    volume = df_stock.get('volume', np.ones(len(close))).values.astype(float)

    indicators = calculate_indicators(close, high, low, volume)
    indicators.update(compute_composite_factors(indicators))

    result = pd.DataFrame({
        'date': df_stock['date'].values,
        'code': df_stock['code'].values[0],
    })

    for k, v in indicators.items():
        if isinstance(v, np.ndarray) and len(v) == len(result):
            result[k] = v

    # 计算未来收益 (20日)
    result = result.sort_values('date')
    result['future_ret'] = result['close'].pct_change(20).shift(-20)

    return result.dropna(subset=['future_ret'])


def compute_fundamental_factor(code, fund_df, date, factor_name):
    """获取基本面因子值"""
    stock_fund = fund_df[fund_df['code'] == code]
    if len(stock_fund) == 0:
        return None

    # 找最接近日期的数据
    stock_fund = stock_fund.copy()
    stock_fund['报告期'] = pd.to_datetime(stock_fund['报告期'].astype(str), errors='coerce')
    stock_fund = stock_fund[stock_fund['报告期'] <= date].sort_values('报告期').tail(1)

    if len(stock_fund) == 0:
        return None

    row = stock_fund.iloc[0]

    if factor_name == 'fund_score':
        # 综合评分: 用 ROE * 营收增长率近似
        roe = row.get('净资产收益率', 0)
        rev_growth = row.get('营业总收入-同比增长', 0)
        if pd.isna(roe) or pd.isna(rev_growth):
            return None
        return float(roe) * 0.6 + float(rev_growth) * 0.4
    elif factor_name == 'fund_profit_growth':
        val = row.get('净利润-同比增长', 0)
        return float(val) if not pd.isna(val) else None
    elif factor_name == 'fund_revenue_growth':
        val = row.get('营业总收入-同比增长', 0)
        return float(val) if not pd.isna(val) else None
    elif factor_name == 'fund_roe':
        val = row.get('净资产收益率', 0)
        return float(val) if not pd.isna(val) else None
    elif factor_name == 'fund_cf_to_profit':
        profit = row.get('净利润-净利润', 0)
        cf = row.get('每股经营现金流量', 0)
        if pd.isna(profit) or pd.isna(cf) or float(profit) == 0:
            return None
        return float(cf) / float(profit) if not pd.isna(cf) else None
    elif factor_name == 'fund_gross_margin':
        val = row.get('销售毛利率', 0)
        return float(val) if not pd.isna(val) else None

    return None


def apply_compression(factor_name, raw_value):
    """应用压缩"""
    if pd.isna(raw_value):
        return None

    if factor_name == 'fund_score':
        raw_clipped = max(-100, min(100, raw_value))
        return np.tanh((raw_clipped - 50) / 50)
    elif factor_name == 'fund_profit_growth':
        raw_clipped = max(-100, min(100, raw_value))
        return np.tanh(raw_clipped)
    elif factor_name == 'fund_roe':
        raw_clipped = max(-50, min(50, raw_value))
        return np.tanh((raw_clipped - 10) / 20)
    elif factor_name == 'fund_revenue_growth':
        raw_clipped = max(-100, min(100, raw_value))
        return np.tanh(raw_clipped)
    elif factor_name == 'fund_gross_margin':
        raw_clipped = max(-20, min(80, raw_value))
        return np.tanh((raw_clipped - 30) / 30)
    else:
        # 技术因子 clip 到合理范围
        return max(-10, min(10, raw_value))


def compute_rolling_ic(factor_values, future_years, min_periods=50):
    """计算滚动年度 IC"""
    yearly_ics = []

    for year in sorted(future_years['year'].unique()):
        year_fv = factor_values[factor_values['year'] == year]
        year_fr = future_years[future_years['year'] == year]

        merged = pd.merge(year_fv, year_fr, on=['code', 'year'])
        if len(merged) >= min_periods:
            try:
                ic, _ = spearmanr(merged['factor_value'], merged['future_ret'])
                if not np.isnan(ic):
                    yearly_ics.append({'year': year, 'ic': ic, 'n': len(merged)})
            except:
                pass

    if len(yearly_ics) < 3:
        return None

    ic_values = [y['ic'] for y in yearly_ics]
    return {
        'mean_ic': np.mean(ic_values),
        'std_ic': np.std(ic_values),
        'ir': np.mean(ic_values) / (np.std(ic_values) + 1e-10),
        'n_years': len(yearly_ics),
        'yearly': yearly_ics
    }


def analyze_industry(industry_name, price_df, fund_df, candidate_fund_factors):
    """分析某行业的最佳因子组合"""
    print(f"\n{'='*60}")
    print(f"行业: {industry_name}")
    print('='*60)

    # 获取该行业的股票
    from core.industry_mapping import INDUSTRY_KEYWORDS
    keywords = INDUSTRY_KEYWORDS.get(industry_name, [])

    industry_stocks = set()
    for _, row in fund_df.iterrows():
        industry = str(row.get('industry', ''))
        code = str(row.get('code', '')).zfill(6)
        for kw in keywords:
            if kw in industry:
                industry_stocks.add(code)
                break

    if not industry_stocks:
        print(f"  无股票")
        return None

    print(f"  股票数: {len(industry_stocks)}")

    # 筛选价格数据
    industry_prices = price_df[price_df['code'].isin(industry_stocks)]

    # 计算技术因子
    print(f"  计算技术因子...")
    all_tech_factors = []
    for code in list(industry_stocks)[:100]:  # 限制数量
        stock_prices = industry_prices[industry_prices['code'] == code]
        tech_factors = compute_factors_for_stock(stock_prices)
        if tech_factors is not None and len(tech_factors) > 0:
            all_tech_factors.append(tech_factors)

    if not all_tech_factors:
        print(f"  无法计算技术因子")
        return None

    tech_df = pd.concat(all_tech_factors, ignore_index=True)
    tech_df['year'] = tech_df['date'].dt.year
    print(f"  技术因子数据: {len(tech_df)} 条")

    # 计算基本面因子
    print(f"  计算基本面因子...")
    fund_records = []
    for _, row in tech_df.iterrows():
        code = row['code']
        date = row['date']
        year = row['year']

        for ff in candidate_fund_factors:
            fund_val = compute_fundamental_factor(code, fund_df, date, ff)
            if fund_val is not None:
                compressed = apply_compression(ff, fund_val)
                if compressed is not None:
                    fund_records.append({
                        'code': code,
                        'date': date,
                        'year': year,
                        f'fund_{ff}': compressed
                    })

    fund_df_out = pd.DataFrame(fund_records) if fund_records else pd.DataFrame()

    # 合并技术因子和基本面因子
    if len(fund_df_out) > 0:
        merged = pd.merge(tech_df, fund_df_out, on=['code', 'date', 'year'], how='left')
    else:
        merged = tech_df

    # 测试因子组合
    candidate_factors = candidate_fund_factors + get_candidate_factors()
    available = [f for f in candidate_factors if f in merged.columns]

    results = []

    # 单因子
    for f in available:
        if f not in merged.columns:
            continue
        test_df = merged[['code', 'year', f, 'future_ret']].dropna()
        if len(test_df) < 100:
            continue
        ic_result = compute_rolling_ic(
            test_df[['code', 'year']].assign(factor_value=test_df[f]),
            test_df[['code', 'year', 'future_ret']].assign(year=test_df['year'])
        )
        if ic_result and ic_result['mean_ic'] > 0.01:
            results.append({
                'factors': [f],
                'mean_ic': ic_result['mean_ic'],
                'std_ic': ic_result['std_ic'],
                'ir': ic_result['ir'],
                'n_years': ic_result['n_years']
            })

    # 2因子组合
    for combo in combinations(available, 2):
        f1, f2 = combo
        if f1 not in merged.columns or f2 not in merged.columns:
            continue
        merged['combo'] = (merged[f1] + merged[f2]) / 2
        test_df = merged[['code', 'year', 'combo', 'future_ret']].dropna()
        if len(test_df) < 100:
            continue
        ic_result = compute_rolling_ic(
            test_df[['code', 'year']].assign(factor_value=test_df['combo']),
            test_df[['code', 'year', 'future_ret']].assign(year=test_df['year'])
        )
        if ic_result and ic_result['mean_ic'] > 0.01:
            results.append({
                'factors': [f1, f2],
                'mean_ic': ic_result['mean_ic'],
                'std_ic': ic_result['std_ic'],
                'ir': ic_result['ir'],
                'n_years': ic_result['n_years']
            })

    if not results:
        print("  无有效组合")
        return None

    results.sort(key=lambda x: -x['mean_ic'])

    print(f"\n  Top-10 组合:")
    print(f"  {'因子':<40} {'IC':>8} {'IR':>6} {'年数':>4}")
    print(f"  {'-'*65}")
    for r in results[:10]:
        print(f"  {'+'.join(r['factors']):<40} {r['mean_ic']:>+8.4f} {r['ir']:>6.2f} {r['n_years']:>4}")

    return results


def main():
    print("="*80)
    print("离线因子标定 - 滚动 IC 验证")
    print("="*80)

    price_df, fund_df = load_data()

    # 候选基本面因子
    candidate_fund_factors = ['fund_score', 'fund_profit_growth', 'fund_revenue_growth', 'fund_roe', 'fund_gross_margin']

    # 各行业
    from core.industry_mapping import INDUSTRY_KEYWORDS

    all_results = {}
    for industry in INDUSTRY_KEYWORDS.keys():
        try:
            results = analyze_industry(industry, price_df, fund_df, candidate_fund_factors)
            if results:
                all_results[industry] = results
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 输出汇总
    print("\n" + "="*80)
    print("汇总 - 各行业推荐因子组合")
    print("="*80)

    for industry, results in sorted(all_results.items()):
        if results:
            best = results[0]
            print(f"\n{industry}:")
            print(f"  推荐: {'+'.join(best['factors'])} (IC={best['mean_ic']:+.4f}, IR={best['ir']:.2f})")


if __name__ == '__main__':
    main()