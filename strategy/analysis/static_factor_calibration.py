#!/usr/bin/env python
"""
静态因子离线标定 - 基于回测因子库重新标定各行业最优因子

设计原则:
1. 使用与backtest_factors完全相同的因子列表
2. 使用与signal_engine完全相同的因子计算逻辑
3. 截面标准化后再计算IC（消除极端值影响）
4. 输出可直接用于factor_config.yaml的配置
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
QUANT_DIR = os.path.dirname(STRATEGY_DIR)
DATA_DIR = os.path.join(QUANT_DIR, 'data')  # 数据目录
sys.path.insert(0, STRATEGY_DIR)

from core.config_loader import load_config
from core.factor_calculator import calculate_indicators, compute_composite_factors, get_default_params
from core.industry_mapping import INDUSTRY_KEYWORDS

# 使用INDUSTRY_KEYWORDS作为行业定义
DETAILED_INDUSTRIES = INDUSTRY_KEYWORDS


def get_backtest_factors():
    """获取配置文件中定义的回测因子列表"""
    config = load_config()
    return config.get('backtest_factors', [])


def compute_all_factors(close, high=None, low=None, volume=None):
    """计算所有因子 - 与factor_calculator一致"""
    n = len(close)
    if high is None:
        high = close
    if low is None:
        low = close
    if volume is None:
        volume = np.ones(n)

    params = get_default_params()
    indicators = calculate_indicators(close, high, low, volume, params)
    composite = compute_composite_factors(indicators, n - 1)

    # 合并结果
    all_factors = {}
    for k, v in indicators.items():
        if isinstance(v, np.ndarray) and len(v) == n:
            all_factors[k] = v[-1] if not np.isnan(v[-1]) else None
    for k, v in composite.items():
        if v is not None and not np.isnan(v):
            all_factors[k] = v

    return all_factors


def cross_section_standardize(df, factor_col):
    """截面标准化 - 按日期对因子值做z-score"""
    def zscore(x):
        if len(x) < 5 or x.std() < 1e-10:
            return x * 0
        return (x - x.mean()) / x.std()

    df = df.copy()
    df[factor_col] = df.groupby('date')[factor_col].transform(zscore)
    return df


def compute_ic(factor_values, future_rets):
    """计算IC (Spearman相关系数)"""
    valid = pd.DataFrame({
        'factor': factor_values,
        'ret': future_rets
    }).dropna()

    if len(valid) < 20:
        return np.nan

    try:
        ic, pval = spearmanr(valid['factor'], valid['ret'])
        return ic
    except:
        return np.nan


def load_stock_data(data_dir):
    """加载所有股票数据"""
    print("加载股票数据...")
    price_dir = os.path.join(data_dir, 'stock_data/backtrader_data')

    if not os.path.exists(price_dir):
        print(f"  错误: 目录不存在! {price_dir}")
        return {}

    all_data = {}
    files = [f for f in os.listdir(price_dir) if f.endswith('_qfq.csv')]
    print(f"  找到 {len(files)} 个文件")

    for f in files:
        code = f.replace('_qfq.csv', '')
        try:
            df = pd.read_csv(os.path.join(price_dir, f))
            # 兼容 datetime 和 date 两种列名
            if 'datetime' in df.columns:
                df['date'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                continue

            df = df.sort_values('date')
            if len(df) >= 120:  # 至少120天数据
                all_data[code] = df
        except:
            continue

    print(f"  加载 {len(all_data)} 只股票")
    return all_data


def load_fundamental_data(data_dir):
    """加载基本面数据"""
    print("加载基本面数据...")
    fund_dir = os.path.join(data_dir, 'stock_data/fundamental_data')

    all_fund = {}
    files = [f for f in os.listdir(fund_dir) if f.endswith('.csv')]

    for f in files:
        code = f.replace('.csv', '').zfill(6)
        try:
            df = pd.read_csv(os.path.join(fund_dir, f))
            df['code'] = code
            all_fund[code] = df
        except:
            continue

    print(f"加载 {len(all_fund)} 只股票的基本面数据")
    return all_fund


def get_industry_mapping(fund_data):
    """从基本面数据构建行业映射"""
    industry_stocks = {ind: set() for ind in DETAILED_INDUSTRIES.keys()}

    for code, df in fund_data.items():
        if 'industry' not in df.columns or len(df) == 0:
            continue

        industry = str(df['industry'].iloc[0])
        for ind_name, keywords in DETAILED_INDUSTRIES.items():
            for kw in keywords:
                if kw in industry:
                    industry_stocks[ind_name].add(code)
                    break

    return industry_stocks


def compute_factor_df(stock_data, fund_data, factor_list, forward_period=20):
    """计算所有股票的因子值"""
    print("计算因子值...")

    all_records = []

    for code, price_df in stock_data.items():
        if len(price_df) < 150:
            continue

        close = price_df['close'].values
        high = price_df['high'].values if 'high' in price_df.columns else close
        low = price_df['low'].values if 'low' in price_df.columns else close
        volume = price_df['volume'].values if 'volume' in price_df.columns else np.ones(len(close))
        dates = price_df['date'].values

        # 计算未来收益
        future_ret = np.full(len(close), np.nan)
        for i in range(len(close) - forward_period):
            if close[i] > 0:
                future_ret[i] = (close[i + forward_period] - close[i]) / close[i]

        # 计算因子（滚动窗口）
        for i in range(60, len(close)):  # 从第60天开始
            factors = compute_all_factors(close[:i+1], high[:i+1], low[:i+1], volume[:i+1])
            if factors:
                record = {
                    'code': code,
                    'date': dates[i],
                    'future_ret': future_ret[i],
                }

                # 添加基本面因子
                if code in fund_data:
                    fund_df = fund_data[code]
                    record.update(get_fundamental_factors(fund_df, pd.Timestamp(dates[i])))

                # 添加技术因子
                for f in factor_list:
                    if f in factors and f not in record:
                        record[f] = factors[f]

                all_records.append(record)

    return pd.DataFrame(all_records)


def get_fundamental_factors(fund_df, date):
    """获取基本面因子值"""
    result = {}

    # 找最近的报告期
    if '报告期' not in fund_df.columns:
        return result

    fund_df = fund_df.copy()
    fund_df['报告期'] = pd.to_datetime(fund_df['报告期'].astype(str), errors='coerce')
    fund_df = fund_df[fund_df['报告期'] <= date].sort_values('报告期').tail(1)

    if len(fund_df) == 0:
        return result

    row = fund_df.iloc[0]

    # fund_score
    roe = row.get('净资产收益率', np.nan)
    rev_growth = row.get('营业总收入-同比增长', np.nan)
    if pd.notna(roe) and pd.notna(rev_growth):
        raw = float(roe) * 0.6 + float(rev_growth) * 0.4
        result['fund_score'] = np.tanh((np.clip(raw, -100, 100) - 50) / 50)

    # fund_profit_growth
    val = row.get('净利润-同比增长', np.nan)
    if pd.notna(val):
        result['fund_profit_growth'] = np.tanh(np.clip(float(val), -100, 100))

    # fund_revenue_growth
    val = row.get('营业总收入-同比增长', np.nan)
    if pd.notna(val):
        result['fund_revenue_growth'] = np.tanh(np.clip(float(val), -100, 100))

    # fund_roe
    val = row.get('净资产收益率', np.nan)
    if pd.notna(val):
        result['fund_roe'] = np.tanh((np.clip(float(val), -50, 50) - 10) / 20)

    # fund_gross_margin
    val = row.get('销售毛利率', np.nan)
    if pd.notna(val):
        result['fund_gross_margin'] = np.tanh((np.clip(float(val), -20, 80) - 30) / 30)

    # fund_cf_to_profit
    profit = row.get('净利润-净利润', np.nan)
    cf = row.get('每股经营现金流量', np.nan)
    if pd.notna(profit) and pd.notna(cf) and float(profit) != 0:
        result['fund_cf_to_profit'] = np.tanh(np.clip(float(cf) / float(profit), -5, 5) - 1)

    return result


def calibrate_industry(industry_name, factor_df, industry_stocks, factor_list):
    """标定单个行业的最优因子组合"""
    print(f"\n标定行业: {industry_name}")

    stocks = industry_stocks.get(industry_name, set())
    if len(stocks) < 10:
        print(f"  股票数不足: {len(stocks)}")
        return None

    # 筛选该行业数据
    ind_df = factor_df[factor_df['code'].isin(stocks)].copy()
    if len(ind_df) < 1000:
        print(f"  数据不足: {len(ind_df)}")
        return None

    print(f"  数据量: {len(ind_df):,} 条, {len(stocks)} 只股票")

    # 获取可用因子
    available_factors = [f for f in factor_list if f in ind_df.columns and ind_df[f].notna().sum() > 100]
    print(f"  可用因子: {len(available_factors)}")

    results = []

    # 单因子测试（截面标准化后）
    for f in available_factors:
        test_df = ind_df[['date', 'code', f, 'future_ret']].dropna()
        if len(test_df) < 500:
            continue

        # 截面标准化
        test_df = cross_section_standardize(test_df, f)

        # 按日期计算IC
        ic_values = test_df.groupby('date').apply(
            lambda g: compute_ic(g[f], g['future_ret'])
        ).dropna()

        if len(ic_values) < 10:
            continue

        mean_ic = ic_values.mean()
        std_ic = ic_values.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0

        if abs(mean_ic) > 0.01:  # IC绝对值>1%
            results.append({
                'factors': [f],
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'ir': ir,
                'n_dates': len(ic_values),
                'ic_positive_rate': (ic_values > 0).mean()
            })

    # 2因子组合测试
    print("  测试2因子组合...")
    for combo in combinations(available_factors, 2):
        f1, f2 = combo
        test_df = ind_df[['date', 'code', f1, f2, 'future_ret']].dropna()
        if len(test_df) < 500:
            continue

        # 等权组合
        test_df['combo'] = (test_df[f1] + test_df[f2]) / 2
        test_df = cross_section_standardize(test_df, 'combo')

        ic_values = test_df.groupby('date').apply(
            lambda g: compute_ic(g['combo'], g['future_ret'])
        ).dropna()

        if len(ic_values) < 10:
            continue

        mean_ic = ic_values.mean()
        std_ic = ic_values.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0

        if abs(mean_ic) > 0.01:
            results.append({
                'factors': [f1, f2],
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'ir': ir,
                'n_dates': len(ic_values),
                'ic_positive_rate': (ic_values > 0).mean()
            })

    # 3因子组合测试
    print("  测试3因子组合...")
    for combo in combinations(available_factors, 3):
        f1, f2, f3 = combo
        test_df = ind_df[['date', 'code', f1, f2, f3, 'future_ret']].dropna()
        if len(test_df) < 500:
            continue

        # 等权组合
        test_df['combo'] = (test_df[f1] + test_df[f2] + test_df[f3]) / 3
        test_df = cross_section_standardize(test_df, 'combo')

        ic_values = test_df.groupby('date').apply(
            lambda g: compute_ic(g['combo'], g['future_ret'])
        ).dropna()

        if len(ic_values) < 10:
            continue

        mean_ic = ic_values.mean()
        std_ic = ic_values.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0

        if abs(mean_ic) > 0.01:
            results.append({
                'factors': [f1, f2, f3],
                'mean_ic': mean_ic,
                'std_ic': std_ic,
                'ir': ir,
                'n_dates': len(ic_values),
                'ic_positive_rate': (ic_values > 0).mean()
            })

    if not results:
        print("  无有效组合")
        return None

    # 按IC排序
    results.sort(key=lambda x: (-x['mean_ic']))

    print(f"\n  Top 5 因子组合:")
    print(f"  {'因子':<50} {'IC':>8} {'IR':>6} {'IC>0%':>6}")
    print(f"  {'-'*75}")
    for r in results[:5]:
        print(f"  {'+'.join(r['factors']):<50} {r['mean_ic']:>+8.4f} {r['ir']:>6.2f} {r['ic_positive_rate']*100:>5.1f}%")

    return results


def main():
    print("=" * 80)
    print("静态因子离线标定 - 基于回测因子库")
    print("=" * 80)

    # 数据路径
    data_dir = DATA_DIR

    # 加载数据
    stock_data = load_stock_data(data_dir)
    fund_data = load_fundamental_data(data_dir)

    # 获取行业映射
    industry_stocks = get_industry_mapping(fund_data)

    # 获取因子列表
    factor_list = get_backtest_factors()
    print(f"\n因子库: {len(factor_list)} 个因子")
    print(f"  技术因子: {[f for f in factor_list if not f.startswith('fund_')]}")
    print(f"  基本面因子: {[f for f in factor_list if f.startswith('fund_')]}")

    # 计算因子值（采样部分数据加速）
    print("\n计算因子值（采样加速）...")
    sampled_stocks = dict(list(stock_data.items())[:200])  # 采样200只股票
    factor_df = compute_factor_df(sampled_stocks, fund_data, factor_list)
    print(f"因子数据: {len(factor_df):,} 条")

    # 标定各行业
    all_results = {}
    for industry in DETAILED_INDUSTRIES.keys():
        try:
            results = calibrate_industry(industry, factor_df, industry_stocks, factor_list)
            if results:
                all_results[industry] = results
        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 输出配置
    print("\n" + "=" * 80)
    print("推荐配置 - 可直接复制到 factor_config.yaml")
    print("=" * 80)

    print("\nindustry_factors:")
    for industry in DETAILED_INDUSTRIES.keys():
        if industry in all_results and all_results[industry]:
            best = all_results[industry][0]
            # 选择IC最高的组合
            factors_str = ', '.join([f"'{f}'" for f in best['factors']])
            print(f"  {industry}:")
            print(f"    factors: [{factors_str}]  # IC={best['mean_ic']:.4f}, IR={best['ir']:.2f}")
        else:
            print(f"  {industry}:")
            print(f"    factors: [fund_score, fund_profit_growth, mom_x_lowvol_20_20]  # 默认配置")

    # 保存详细结果
    output_file = os.path.join(SCRIPT_DIR, 'offline_calibration_results.csv')
    rows = []
    for industry, results in all_results.items():
        for r in results[:10]:  # 保存每个行业Top 10
            rows.append({
                'industry': industry,
                'factors': '+'.join(r['factors']),
                'n_factors': len(r['factors']),
                'mean_ic': r['mean_ic'],
                'std_ic': r['std_ic'],
                'ir': r['ir'],
                'ic_positive_rate': r['ic_positive_rate']
            })

    if rows:
        pd.DataFrame(rows).to_csv(output_file, index=False)
        print(f"\n详细结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
