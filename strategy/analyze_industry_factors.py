"""
行业因子分析脚本

按行业分组分析因子有效性，计算IC、IR等指标，找出每个行业最有效的因子。
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# 添加项目路径
BASE_DIR = '/Users/litiancheng01/code/ltc/quant'
sys.path.insert(0, BASE_DIR)

from strategy.core.factor_library import (
    calc_factor_volatility_10, calc_factor_volatility_5, calc_factor_volatility_20,
    calc_factor_rsi_6, calc_factor_rsi_8, calc_factor_rsi_10, calc_factor_rsi_14,
    calc_factor_bb_width_20, calc_factor_momentum_10, calc_factor_momentum_20
)
from strategy.core.fundamental import FundamentalData


# 配置路径
DATA_PATH = os.path.join(BASE_DIR, 'data/stock_data')
BACKTRADER_PATH = os.path.join(DATA_PATH, 'backtrader_data')
FUNDAMENTAL_PATH = os.path.join(DATA_PATH, 'fundamental_data')


def load_stock_price_data(code, ndays=60):
    """加载股票价格数据"""
    filepath = os.path.join(BACKTRADER_PATH, f'{code}_qfq.csv')
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').tail(ndays)
        if len(df) < 30:
            return None
        return df
    except Exception as e:
        return None


def get_stock_industry(code):
    """获取股票行业"""
    filepath = os.path.join(FUNDAMENTAL_PATH, f'{code}.csv')
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)
        if '所处行业' in df.columns and len(df) > 0:
            industry = df.iloc[0]['所处行业']
            if pd.notna(industry):
                return str(industry).strip()
        return None
    except Exception:
        return None


def calculate_all_factors(df):
    """计算所有技术因子"""
    close = df['close'].values
    high = df['high'].values if 'high' in df.columns else close
    low = df['low'].values if 'low' in df.columns else close
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))

    factors = {}

    # 波动率因子
    factors['volatility_10'] = calc_factor_volatility_10(close)
    factors['volatility_5'] = calc_factor_volatility_5(close)
    factors['volatility_20'] = calc_factor_volatility_20(close)

    # RSI因子
    factors['rsi_6'] = calc_factor_rsi_6(close)
    factors['rsi_8'] = calc_factor_rsi_8(close)
    factors['rsi_10'] = calc_factor_rsi_10(close)
    factors['rsi_14'] = calc_factor_rsi_14(close)

    # 布林带宽度
    factors['bb_width_20'] = calc_factor_bb_width_20(close)

    # 动量因子
    factors['momentum_10'] = calc_factor_momentum_10(close)
    factors['momentum_20'] = calc_factor_momentum_20(close)

    # 成交量相关因子
    vol_sma = pd.Series(volume).rolling(5).mean().values
    factors['volume_ratio'] = volume / (vol_sma + 1e-10)

    # 价格变化率
    factors['price_change'] = np.diff(close, prepend=close[0]) / close

    return factors


def calculate_forward_returns(close, periods=[5, 10, 20]):
    """计算未来收益率 - 从当前时刻往后看的收益

    forward_5[i] = close[i+5] / close[i] - 1
    """
    returns = {}
    for p in periods:
        ret = np.zeros_like(close, dtype=float)
        ret[:] = np.nan
        # 从当前时间往后p天的收益
        for i in range(len(close) - p):
            ret[i] = close[i + p] / close[i] - 1
        returns[f'forward_{p}'] = ret
    return returns


def calculate_ic(factor_values, forward_returns):
    """计算IC (Information Coefficient)"""
    # 去除NaN
    valid_mask = ~(np.isnan(factor_values) | np.isnan(forward_returns))
    if valid_mask.sum() < 10:
        return 0

    ic = np.corrcoef(factor_values[valid_mask], forward_returns[valid_mask])[0, 1]
    return ic if not np.isnan(ic) else 0


def calculate_ir(ic_series):
    """计算IR (Information Ratio)"""
    if len(ic_series) < 2:
        return 0

    mean_ic = np.mean(ic_series)
    std_ic = np.std(ic_series)

    if std_ic < 1e-10:
        return 0

    return mean_ic / std_ic


def get_all_stock_codes():
    """获取所有有价格数据的股票代码"""
    codes = []
    for f in os.listdir(BACKTRADER_PATH):
        if f.endswith('_qfq.csv') and f != 'sh000001_qfq.csv':
            code = f.replace('_qfq.csv', '')
            codes.append(code)
    return codes


def group_stocks_by_industry(codes):
    """按行业分组股票"""
    industry_stocks = defaultdict(list)

    for code in codes:
        industry = get_stock_industry(code)
        if industry:
            industry_stocks[industry].append(code)

    return dict(industry_stocks)


def analyze_industry(industry, stock_codes, forward_period=10):
    """分析单个行业的因子有效性"""
    print(f"\n分析行业: {industry} ({len(stock_codes)} 只股票)")

    # 收集所有日期的因子值和收益
    all_data = []

    for code in stock_codes:
        df = load_stock_price_data(code, ndays=100)
        if df is None or len(df) < 40:
            continue

        factors = calculate_all_factors(df)
        forward_returns = calculate_forward_returns(df['close'].values)

        # 获取指定周期的未来收益
        fwd_col = f'forward_{forward_period}'
        if fwd_col not in forward_returns:
            continue

        fwd_ret = forward_returns[fwd_col]

        # 取有效数据点 (只取有前向收益的数据点)
        valid_end = len(df) - forward_period
        for i in range(30, valid_end):
            row_data = {'code': code, 'date_idx': i}
            for fname, fvals in factors.items():
                if i < len(fvals) and not np.isnan(fvals[i]):
                    row_data[fname] = fvals[i]
                else:
                    row_data[fname] = np.nan

            # 前向收益
            fwd_val = fwd_ret[i]
            if not np.isnan(fwd_val):
                row_data['forward_return'] = fwd_val
                all_data.append(row_data)

    if len(all_data) < 100:
        print(f"  数据不足，跳过")
        return None

    # 转换为DataFrame
    data_df = pd.DataFrame(all_data)

    # 计算各因子的IC
    factor_names = [k for k in factors.keys() if k != 'forward_return']
    ic_results = {}

    for fname in factor_names:
        ic = calculate_ic(data_df[fname].values, data_df['forward_return'].values)
        ic_results[fname] = ic

    # 排序
    sorted_factors = sorted(ic_results.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"  有效样本: {len(data_df)}")
    print(f"  因子IC排名:")

    result = {
        'stock_count': len(stock_codes),
        'sample_count': len(data_df),
        'factor_ic': {},
        'factor_ir': {},
        'best_factors': []
    }

    for fname, ic in sorted_factors[:10]:
        print(f"    {fname}: IC={ic:.4f}")
        result['factor_ic'][fname] = round(ic, 4)

    # 取前5个最佳因子
    result['best_factors'] = [f[0] for f in sorted_factors[:5]]

    return result


def main():
    """主函数"""
    print("=" * 60)
    print("行业因子分析")
    print("=" * 60)

    # 获取所有股票代码
    codes = get_all_stock_codes()
    print(f"\n发现 {len(codes)} 只股票")

    # 按行业分组
    industry_stocks = group_stocks_by_industry(codes)
    print(f"发现 {len(industry_stocks)} 个行业")

    # 行业统计
    print("\n行业股票数量分布:")
    sorted_industries = sorted(industry_stocks.items(), key=lambda x: len(x[1]), reverse=True)
    for ind, stocks in sorted_industries:
        print(f"  {ind}: {len(stocks)} 只")

    # 分析每个行业
    results = {}
    for industry, stocks in sorted_industries:
        if len(stocks) >= 3:  # 至少3只股票
            result = analyze_industry(industry, stocks, forward_period=10)
            if result:
                results[industry] = result

    # 汇总结果
    print("\n" + "=" * 60)
    print("分析结果汇总")
    print("=" * 60)

    # 找出跨行业有效的因子
    all_factors = set()
    for r in results.values():
        all_factors.update(r['best_factors'])

    factor_industries = defaultdict(list)
    for fname in all_factors:
        for ind, r in results.items():
            if fname in r['factor_ic']:
                ic = r['factor_ic'][fname]
                if abs(ic) > 0.02:  # IC > 0.02
                    factor_industries[fname].append((ind, ic))

    # 打印跨行业有效因子
    print("\n跨行业有效因子:")
    universal_factors = []
    for fname, ind_ics in factor_industries.items():
        if len(ind_ics) >= len(results) * 0.5:  # 超过一半行业有效
            avg_ic = np.mean([ic for _, ic in ind_ics])
            universal_factors.append((fname, avg_ic, len(ind_ics)))

    universal_factors.sort(key=lambda x: abs(x[1]), reverse=True)
    for fname, avg_ic, count in universal_factors[:10]:
        print(f"  {fname}: 平均IC={avg_ic:.4f}, 有效行业数={count}")

    # 按行业打印最佳因子
    print("\n各行业最佳因子:")
    for industry in sorted(results.keys(), key=lambda x: results[x]['stock_count'], reverse=True):
        r = results[industry]
        print(f"\n{industry} ({r['stock_count']}只股票, {r['sample_count']}样本):")
        print(f"  最佳因子: {r['best_factors']}")
        print(f"  IC值: {r['factor_ic']}")

    # 保存结果到文件
    output_path = '/Users/litiancheng01/code/ltc/quant/strategy/industry_factor_results.json'
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")

    return results


if __name__ == '__main__':
    main()
